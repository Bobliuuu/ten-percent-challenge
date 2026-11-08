"""L9 REAL packed artifact: writer + loader + generation for Qwen3-235B-A22B-Instruct-2507 compressed with routing-mass expert pruning + grouped asymmetric RTN (MSE clip).
Layout (all shipped bytes are files under ART): per layer `layerNN.safetensors` (bit-packed expert/attention integer codes, fp16 scales, bit-packed zero points, kept-expert
router rows, kept mask, norms), `global.safetensors` (4-bit embedding + head, final norm), `meta.json`, tokenizer + config files, and the code files (this file, answer.py).
Usage:  python pack235.py pack CFG.json KEPT.json ART  [HF_SNAPSHOT]   |  loader/generation: see build_model(), answer_text()"""
import os, sys, json, math, shutil, time
import torch, torch.nn as nn
from safetensors import safe_open
from safetensors.torch import save_file, load_file
NL, NE, HID, FF, NH, NKV, HD, VOC = 94, 128, 4096, 1536, 64, 4, 128, 151936

def _qparams(w, bits, g, mse):
    sh = w.shape; wg = w.float().reshape(*sh[:-1], sh[-1] // g, g); mx = wg.amax(-1, keepdim=True); mn = wg.amin(-1, keepdim=True); qm = 2 ** bits - 1
    def q(mn_, mx_):
        sc = ((mx_ - mn_) / qm).clamp(min=1e-8).half().float(); z = (-mn_ / sc).round().clamp(0, qm)
        codes = (wg / sc + z).round().clamp(0, qm); dq = (codes - z) * sc
        return dq, sc, z, codes
    if not mse: dq, sc, z, c = q(mn, mx); return c, sc, z
    best = None
    for r in (1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65):
        dq, sc, z, c = q(mn * r, mx * r); err = ((dq - wg) ** 2).sum(-1, keepdim=True)
        if best is None: best = [dq, sc, z, c, err]
        else:
            m = err < best[4]
            best = [torch.where(m, a, b) for a, b in zip((dq, sc, z, c), best[:4])] + [torch.where(m, err, best[4])]
    return best[3], best[1], best[2]

def pack_bits(x, bits):  # x uint8 [..., n]; n % (8//bits) == 0 ; bits in (2,4)
    per = 8 // bits; x = x.to(torch.uint8).reshape(*x.shape[:-1], x.shape[-1] // per, per); out = torch.zeros(x.shape[:-1], dtype=torch.uint8, device=x.device)
    for i in range(per): out |= x[..., i] << (bits * i)
    return out
def unpack_bits(p, bits):
    per = 8 // bits; m = (1 << bits) - 1; return torch.stack([(p >> (bits * i)) & m for i in range(per)], -1).reshape(*p.shape[:-1], p.shape[-1] * per)

def quant_pack(w, bits, g, mse=True):
    c, sc, z = _qparams(w, bits, g, mse); sh = w.shape
    codes = c.reshape(sh).to(torch.uint8); sc = sc.squeeze(-1).half(); z = z.squeeze(-1).to(torch.uint8)
    return dict(q=pack_bits(codes, bits).cpu(), s=sc.cpu(), z=pack_bits(z, bits).cpu())
def dequant(q, s, z, bits, g, dev=None):
    if dev: q, s, z = q.to(dev), s.to(dev), z.to(dev)
    c = unpack_bits(q, bits).float(); zz = unpack_bits(z, bits).float(); out = c.reshape(*c.shape[:-1], -1, g); out = (out - zz[..., None]) * s.float()[..., None]
    return out.reshape(*c.shape)

def pack(cfg, kept, ART, snap, dev="cuda:0"):
    eb, eg, ab, emb, hb = cfg["eb"], cfg.get("eg", 128), cfg["attn_b"], cfg["emb_b"], cfg["head_b"]; mse = cfg.get("quant") == "rtnmse"
    idx = json.load(open(os.path.join(snap, "model.safetensors.index.json")))["weight_map"]; fh = {}
    def T(k):
        f = idx[k]
        if f not in fh: fh[f] = safe_open(os.path.join(snap, f), "pt", device="cpu")
        return fh[f].get_tensor(k)
    os.makedirs(ART, exist_ok=True); t0 = time.time()
    for l in range(NL):
        ki = kept[l]; d = {}; p = f"model.layers.{l}."
        for nm in ("gate_proj", "up_proj", "down_proj"):
            w = torch.stack([T(p + f"mlp.experts.{e}.{nm}.weight") for e in ki]).to(dev); r = quant_pack(w, eb, eg, mse)
            d[f"e.{nm}.q"], d[f"e.{nm}.s"], d[f"e.{nm}.z"] = r["q"], r["s"], r["z"]
        gw = T(p + "mlp.gate.weight"); d["router_kept"] = gw[ki].contiguous()
        mask = torch.zeros(NE, dtype=torch.uint8); mask[ki] = 1; d["keep_mask"] = pack_bits(mask, 1) if False else torch.from_numpy(__import__("numpy").packbits(mask.numpy(), bitorder="little"))
        for nm in ("q_proj", "k_proj", "v_proj", "o_proj"):
            r = quant_pack(T(p + f"self_attn.{nm}.weight").to(dev), ab, 128, mse); d[f"a.{nm}.q"], d[f"a.{nm}.s"], d[f"a.{nm}.z"] = r["q"], r["s"], r["z"]
        for nm in ("input_layernorm", "post_attention_layernorm", "self_attn.q_norm", "self_attn.k_norm"): d["n." + nm] = T(p + nm + ".weight").contiguous()
        save_file(d, os.path.join(ART, f"layer{l:02d}.safetensors"))
        if l % 10 == 0: print("packed layer", l, round(time.time() - t0), flush=True)
    d = {}
    for key, nm, b in (("model.embed_tokens.weight", "embed", emb), ("lm_head.weight", "head", hb)):
        W = T(key); parts = [quant_pack(W[s:s + 20000].to(dev), b, 128, False) for s in range(0, W.shape[0], 20000)]
        for k in ("q", "s", "z"): d[f"{nm}.{k}"] = torch.cat([x[k] for x in parts], 0)
    d["norm"] = T("model.norm.weight").contiguous(); save_file(d, os.path.join(ART, "global.safetensors"))
    for f in ("config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"):
        if os.path.exists(os.path.join(snap, f)): shutil.copy(os.path.join(snap, f), os.path.join(ART, f))
    json.dump(dict(cfg=cfg, kept_per_layer=[len(k) for k in kept], eb=eb, eg=eg, attn_b=ab, emb_b=emb, head_b=hb), open(os.path.join(ART, "meta.json"), "w"))
    print("pack done", round(time.time() - t0))

def build_model(ART, gpumem="130GiB", ndev=None):
    from accelerate import init_empty_weights, infer_auto_device_map, dispatch_model
    from accelerate.utils import set_module_tensor_to_device
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import core235 as C
    meta = json.load(open(os.path.join(ART, "meta.json"))); eg, ab = meta["eg"], meta["attn_b"]
    cfg = AutoConfig.from_pretrained(ART)
    with init_empty_weights(): m = AutoModelForCausalLM.from_config(cfg, torch_dtype=torch.bfloat16)
    layers = m.model.layers; keeps = []
    G = load_file(os.path.join(ART, "global.safetensors"))
    for l in range(NL):
        d = load_file(os.path.join(ART, f"layer{l:02d}.safetensors")); mask = __import__("numpy").unpackbits(d["keep_mask"].numpy(), bitorder="little")[:NE]; ki = [e for e in range(NE) if mask[e]]; keeps.append(ki)
        for e in range(NE):
            if e not in set(ki): layers[l].mlp.experts[e] = nn.Identity()
    n = ndev or torch.cuda.device_count()
    dm = infer_auto_device_map(m, max_memory={i: gpumem for i in range(n)}, no_split_module_classes=["Qwen3MoeDecoderLayer"])
    def dev_of(name):
        best = None
        for k in dm:
            if name == k or name.startswith(k + ".") or k == "": 
                if best is None or len(k) > len(best): best = k
        return dm[best]
    def put(name, t): set_module_tensor_to_device(m, name, dev_of(name), value=t.to(torch.bfloat16) if t.dtype.is_floating_point and t.dtype != torch.bfloat16 else t)
    put("model.embed_tokens.weight", dequant(G["embed.q"], G["embed.s"], G["embed.z"], meta["emb_b"], 128, "cuda:0")); put("lm_head.weight", dequant(G["head.q"], G["head.s"], G["head.z"], meta["head_b"], 128, "cuda:0")); put("model.norm.weight", G["norm"])
    for l in range(NL):
        d = load_file(os.path.join(ART, f"layer{l:02d}.safetensors")); ki = keeps[l]; p = f"model.layers.{l}."; dv = dev_of(p + "input_layernorm.weight")
        for nm in ("gate_proj", "up_proj", "down_proj"):
            w = dequant(d[f"e.{nm}.q"], d[f"e.{nm}.s"], d[f"e.{nm}.z"], meta["eb"], eg, dv).to(torch.bfloat16)
            for j, e in enumerate(ki): put(p + f"mlp.experts.{e}.{nm}.weight", w[j].contiguous())
        gw = torch.zeros(NE, HID, dtype=torch.bfloat16); gw[ki] = d["router_kept"]; put(p + "mlp.gate.weight", gw)
        for nm in ("q_proj", "k_proj", "v_proj", "o_proj"): put(p + f"self_attn.{nm}.weight", dequant(d[f"a.{nm}.q"], d[f"a.{nm}.s"], d[f"a.{nm}.z"], ab, 128, dv).to(torch.bfloat16))
        for nm in ("input_layernorm", "post_attention_layernorm", "self_attn.q_norm", "self_attn.k_norm"): put(p + nm + ".weight", d["n." + nm])
    dispatch_model(m, device_map=dm); m.eval()
    import types
    for l, lay in enumerate(m.model.layers):
        gw = C.GateW(lay.mlp.gate); lay.mlp.gate = gw; lay.mlp.forward = types.MethodType(C.moe_fwd, lay.mlp)
        if len(keeps[l]) < NE:
            mk = torch.zeros(NE, device=lay.mlp.gate.g.weight.device); mk[[e for e in range(NE) if e not in set(keeps[l])]] = float("-inf"); gw.mask = mk
    tok = AutoTokenizer.from_pretrained(ART); tok.padding_side = "left"
    return m, tok

@torch.no_grad()
def answer_text(m, tok, texts, max_new=5, bs=64):
    outs = []
    for s in range(0, len(texts), bs):
        pr = [tok.apply_chat_template([{"role": "user", "content": t}], tokenize=False, add_generation_prompt=True) for t in texts[s:s + bs]]
        enc = tok(pr, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda:0"); L = enc.input_ids.shape[1]
        g = m.generate(**enc, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.pad_token_id)
        outs += [t.strip() for t in tok.batch_decode(g[:, L:], skip_special_tokens=True)]
    return outs

if __name__ == "__main__" and sys.argv[1] == "pack":
    cfg = json.load(open(sys.argv[2])); kept = json.load(open(sys.argv[3]))
    snap = sys.argv[5] if len(sys.argv) > 5 else None
    pack(cfg, kept, sys.argv[4], snap)
