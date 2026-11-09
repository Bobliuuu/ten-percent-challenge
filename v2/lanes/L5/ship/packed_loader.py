"""Standalone loader for the L5 packed body (Qwen2.5-32B-Instruct, scalar GPTQ-style Hadamard-rotated 2/3/4-bit, rANS-coded).
Decodes body.bin straight into GPU tensors, tensor by tensor (CPU rANS decode in a worker pool, dequantise + inverse rotation on the GPU).
No dequantised checkpoint file is ever written. Needs: torch, transformers, numpy, constriction.
  tok, model = load_packed(PACK_DIR)               # bf16 model on cuda, eval mode
Format (per tensor, in index.json order): [rANS indices][rANS high bytes of fp16 group scales][raw low bytes].
Weight = unrotate((qi - (L-1)/2) * scale_group), unrotate(M) = blockwise-128 Hadamard(M) * signs; signs from a seeded generator (see signs())."""
import os, json, math, time, multiprocessing as mp
import numpy as np, torch

HB = 128; SEED = 0
QKV = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"); LIN = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")
KEYOF = {"self_attn.k_proj": "self_attn.q_proj", "self_attn.v_proj": "self_attn.q_proj", "mlp.up_proj": "mlp.gate_proj"}   # tensors that share one rotation

def signs(oli, key, C, seed=SEED):
    g = torch.Generator().manual_seed(1000 * oli + len(key) + 7919 * seed)
    return torch.randint(0, 2, (C,), generator=g, dtype=torch.float32) * 2 - 1

def hadamard(dev):
    H = torch.ones(1, 1)
    while H.shape[0] < HB: H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    return (H / math.sqrt(HB)).to(dev)

_BODY = None
def _dec(buf, h, n):
    import constriction
    p = np.maximum(h, 1).astype(np.float64); p /= p.sum()
    c = constriction.stream.stack.AnsCoder(np.frombuffer(buf, dtype=np.uint32).copy())
    return np.asarray(c.decode(constriction.stream.model.Categorical(p.astype(np.float32), perfect=False), n), dtype=np.uint8)

def _work(a):
    path, off, m = a
    global _BODY
    if _BODY is None: _BODY = np.memmap(path, dtype=np.uint8, mode="r")
    raw = _BODY[off:off + m["n1"] + m["n2"] + m["n3"]]
    R, C = m["shape"]; g = m["g"]
    qi = _dec(bytes(raw[:m["n1"]]), np.array(m["h"]), R * C)
    hi = _dec(bytes(raw[m["n1"]:m["n1"] + m["n2"]]), np.array(m["h2"]), R * C // g)
    lo = np.asarray(raw[m["n1"] + m["n2"]:])
    sc = ((hi.astype(np.uint16) << 8) | lo.astype(np.uint16)).view(np.float16)
    return qi.reshape(R, C), sc.reshape(R, C // g)

def dequant(qi, sc, L, g, sg, Hb, dev):
    """qi uint8 [R,C], sc fp16 [R,C//g] numpy -> bf16 [R,C] cuda tensor"""
    half = (L - 1) / 2; q = torch.from_numpy(qi).to(dev, non_blocking=True).float().sub_(half)
    s = torch.from_numpy(sc).to(dev, non_blocking=True).float().repeat_interleave(g, 1); R, C = q.shape
    Q = q.mul_(s); del s
    return ((Q.view(R, C // HB, HB) @ Hb).reshape(R, C) * sg.to(dev)).to(torch.bfloat16)

def load_packed(pack_dir, dev="cuda", workers=None, log=print):
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    t0 = time.time(); J = json.load(open(pack_dir + "/index.json"))["idx"]; body = pack_dir + "/body.bin"
    cfg = AutoConfig.from_pretrained(pack_dir); tok = AutoTokenizer.from_pretrained(pack_dir)
    with torch.device("meta"): model = AutoModelForCausalLM.from_config(cfg, torch_dtype=torch.bfloat16)
    model.to_empty(device=dev); model.model.rotary_emb = type(model.model.rotary_emb)(config=cfg).to(dev)   # buffers are not filled by to_empty
    other = np.load(pack_dir + "/other.npz"); P = dict(model.named_parameters()); Hb = hadamard(dev); hid = cfg.hidden_size
    for k in other.files:     # norms, q/k/v biases (fp16 in the file)
        n = "model.norm.weight" if k == "norm" else "model.layers." + k.split(".", 1)[0] + "." + k.split(".", 1)[1]
        P[n].data.copy_(torch.from_numpy(other[k]).to(dev))
    jobs = []; off = 0
    for name, m in J.items(): jobs.append((body, off, m, name)); off += m["n1"] + m["n2"] + m["n3"]
    assert os.path.getsize(body) == off, "body.bin size does not match the index"
    workers = workers or min(32, max(1, (os.cpu_count() or 4) - 1)); ctx = mp.get_context("fork")
    with ctx.Pool(workers) as pool:
        for (path, o, m, name), (qi, sc) in zip(jobs, pool.imap(_work, [(j[0], j[1], j[2]) for j in jobs], chunksize=1)):
            if name == "head": sg = signs(99999, "head", hid); tgt = "lm_head.weight"
            elif name == "emb": sg = signs(88888, "emb", hid); tgt = "model.embed_tokens.weight"
            else:
                li, ln = name.split(".", 1); sg = signs(int(li), KEYOF.get(ln, ln), qi.shape[1]); tgt = f"model.layers.{li}.{ln}.weight"
            P[tgt].data.copy_(dequant(qi, sc, m["L"], m["g"], sg, Hb, dev))
            if name in ("0.self_attn.q_proj", "emb", "head") or name.endswith(".mlp.down_proj") and int(name.split(".")[0]) % 16 == 0: log(f"[{time.time()-t0:6.0f}s] decoded {name}")
    for p in model.parameters(): p.requires_grad_(False)
    model.config.use_cache = True; model.generation_config.pad_token_id = getattr(tok, "pad_token_id", None); log(f"[{time.time()-t0:6.0f}s] packed model on {dev}")
    return tok, model.eval()
