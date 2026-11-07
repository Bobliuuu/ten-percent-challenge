"""Streaming quantiser -> TPQ1 artifact (dense decoder-only models: Qwen2/Qwen3/Llama naming).  Same maths as research/D_scale/dense_stream.py
(sign-diag x block-Hadamard128 rotation, per-group max-scale search on an L-level uniform grid, optional GPTQ with layer-input Hessians) but
(1) the integer codes are written straight into the entropy-coded container and (2) a bit-exact SIMULATED dequantised bf16 weight is produced
independently (quantiser-side `unrot(Q)`) so the loader can be checked tensor-by-tensor.   No optimiser, no gradient, dev/calibration data only.
  python quantize.py --model DIR --out ART --levels attn=3,o=3,mlp=3,down=3,emb=4,head=4 [--gptq --calib s1_calib.jsonl --ncal 128] [--sim-dir D]"""
import argparse, glob, json, math, os, re, shutil, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import loader as LD

CLS = [("self_attn.q_proj", "attn_in"), ("self_attn.k_proj", "attn_in"), ("self_attn.v_proj", "attn_in"), ("self_attn.o_proj", "o"),
       ("mlp.gate_proj", "mlp_in"), ("mlp.up_proj", "mlp_in"), ("mlp.down_proj", "down")]
GROUP_KEY = {"attn_in": "self_attn.q_proj", "o": "self_attn.o_proj", "mlp_in": "mlp.gate_proj", "down": "mlp.down_proj"}   # seed key = dense_stream's `key`
ENT_SLACK = 1.005; SCALE_BITS_EST = 11


class ST:
    def __init__(s, path):
        from safetensors import safe_open
        s.path = path; s.so = safe_open
        if os.path.exists(path + "/model.safetensors.index.json"): s.wm = json.load(open(path + "/model.safetensors.index.json"))["weight_map"]
        else:
            f = "model.safetensors"; s.wm = {k: f for k in safe_open(path + "/" + f, "pt").keys()}
        s.h = {}; s.dtypes = {}; s.shapes = {}
        for f in sorted(set(s.wm.values())):
            with open(os.path.join(path, f), "rb") as fh:
                import struct; n = struct.unpack("<Q", fh.read(8))[0]; hd = json.loads(fh.read(n)); hd.pop("__metadata__", None)
            for k, v in hd.items(): s.dtypes[k] = v["dtype"]; s.shapes[k] = v["shape"]
    def get(s, n, dev="cpu"):
        f = s.wm[n]
        if f not in s.h: s.h[f] = s.so(s.path + "/" + f, "pt", device="cpu")
        return s.h[f].get_tensor(n).to(dev)
    def has(s, n): return n in s.wm


def ent(h):
    p = h / h.sum(); p = p[p > 0]; return float(-(p * p.log2()).sum())


def signs(seed_a, key, C, dev):
    return (torch.randint(0, 2, (C,), generator=torch.Generator().manual_seed(1000 * seed_a + len(key)), dtype=torch.float32) * 2 - 1).to(dev)


class Quant:
    def __init__(s, g, mbits, dev): s.g, s.mbits, s.dev = g, mbits, dev
    def rtn(s, Wr, Lv, g=None):
        g = g or s.g; R, C = Wr.shape; half = (Lv - 1) / 2; Q = torch.empty_like(Wr); codes = torch.empty(R, C, dtype=torch.uint8, device=Wr.device); SC = torch.empty(R, C // g, dtype=torch.float16, device=Wr.device)
        ch = max(1, int(2e7 // C))
        for r in range(0, R, ch):
            w = Wr[r:r + ch]; x = w.reshape(-1, g); sc = LD.qparams(x, Lv, s.mbits); qi = torch.clamp(torch.round(x / sc + half), 0, Lv - 1)
            Q[r:r + ch] = ((qi - half) * sc).view_as(w); codes[r:r + ch] = qi.to(torch.uint8).view_as(w); SC[r:r + ch] = sc.half().view(w.shape[0], -1)
        return Q, codes, SC
    def gptq(s, W, H, Lv, damp=0.01):
        g = s.g; W = W.float().clone(); R, C = W.shape; H = H.clone(); bsz = max(128, g); half = (Lv - 1) / 2
        dead = torch.diag(H) == 0; H[dead, dead] = 1; W[:, dead] = 0
        H = H + damp * torch.mean(torch.diag(H)) * torch.eye(C, device=H.device)
        Hinv = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(H)), upper=True); del H
        Q = torch.zeros_like(W); QI = torch.zeros(R, C, dtype=torch.uint8, device=W.device); SC = torch.zeros(R, C // g, dtype=torch.float16, device=W.device); sc = None
        for i1 in range(0, C, bsz):
            i2 = min(i1 + bsz, C); W1 = W[:, i1:i2].clone(); Q1 = torch.zeros_like(W1); E1 = torch.zeros_like(W1); Hi = Hinv[i1:i2, i1:i2]
            for i in range(i2 - i1):
                if i % g == 0: sc = LD.qparams(W1[:, i:i + g], Lv, s.mbits); SC[:, (i1 + i) // g] = sc.flatten().half()
                w = W1[:, i]; d = Hi[i, i]; qi = torch.clamp(torch.round(w.unsqueeze(1) / sc + half), 0, Lv - 1); q = ((qi - half) * sc).flatten()
                QI[:, i1 + i] = qi.flatten().to(torch.uint8); Q1[:, i] = q; err = (w - q) / d; W1[:, i:] -= err.unsqueeze(1) @ Hi[i, i:].unsqueeze(0); E1[:, i] = err
            Q[:, i1:i2] = Q1; W[:, i2:] -= E1 @ Hinv[i1:i2, i2:]
        return Q, QI, SC


class Sink:
    """collects per-tensor accounting and (optionally) the simulated bf16 tensors"""
    def __init__(s, w, sim_dir, log):
        s.w, s.sim_dir, s.log = w, sim_dir, log; s.rows = []; s.sim = {}; s.sim_bytes = 0; s.shard = 0; s.sim_hash = {}; s.est_bits = 0.0
    def put_q(s, name, codes, SC, Lv, g, mbits, rot_key, sim):
        n = codes.numel(); hist = torch.bincount(codes.flatten().long(), minlength=Lv).float(); e = ent(hist)
        by = s.w.add_q(name, codes.cpu().numpy(), SC.cpu().numpy(), Lv, g, mbits, rot_key)
        est = n * (e * ENT_SLACK + SCALE_BITS_EST / g) / 8
        s.rows.append(dict(name=name, n=n, L=Lv, g=g, ent_bits=e, est_bytes=est, real_bytes=sum(by.values()), **{"real_" + k: v for k, v in by.items()})); s.est_bits += est * 8
        s.hash(name, sim)
    def hash(s, name, sim):
        s.sim_hash[name] = LD.sha_tensor(sim)
        if s.sim_dir:
            s.sim[name] = sim.cpu(); s.sim_bytes += sim.numel() * 2
            if s.sim_bytes > (4 << 30): s.flush()
    def flush(s):
        if s.sim:
            from safetensors.torch import save_file
            os.makedirs(s.sim_dir, exist_ok=True); save_file(s.sim, os.path.join(s.sim_dir, f"sim-{s.shard:04d}.safetensors"), metadata={"format": "pt"}); s.shard += 1; s.sim = {}; s.sim_bytes = 0


def quantize_model(model_dir, out_dir, levels, g=256, g_emb=128, mbits=8, gptq=False, calib=None, ncal=128, CL=512, damp=0.01, sim_dir=None, dev=None,
                   keep_files=("config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json"), log=print):
    t0 = time.time(); dev = dev or ("cuda" if torch.cuda.is_available() else "cpu"); st = ST(model_dir); os.makedirs(out_dir, exist_ok=True)
    conf = json.load(open(model_dir + "/config.json")); nl = conf["num_hidden_layers"]; hid = conf["hidden_size"]; tied = conf.get("tie_word_embeddings", False)
    w = LD.Writer(os.path.join(out_dir, LD.BLOB)); sink = Sink(w, sim_dir, log); Qz = Quant(g, mbits, dev); raw_bytes = 0; handled = set()
    base_bytes = sum(2 * int(np.prod(s)) for k, s in st.shapes.items())
    def add_raw(n):
        nonlocal raw_bytes
        x = st.get(n); w.add_raw(n, x, st.dtypes[n]); raw_bytes += x.numel() * x.element_size(); handled.add(n)
    # ---- calibration plumbing (GPTQ only)
    cur = None
    if gptq:
        from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
        tok = AutoTokenizer.from_pretrained(model_dir); cfgm = AutoConfig.from_pretrained(model_dir); cfgm._attn_implementation = "sdpa"
        with torch.device("meta"): meta = AutoModelForCausalLM.from_config(cfgm, dtype=torch.bfloat16)
        LayerCls = type(meta.model.layers[0]); RotCls = type(meta.model.rotary_emb); del meta
        texts = []
        for l in open(calib):
            it = json.loads(l); tools = [{"type": "function", "function": t} for t in it["tools"]]
            try: pr = tok.apply_chat_template([{"role": "user", "content": it["query"]}], tools=tools, add_generation_prompt=True, tokenize=False, enable_thinking=False)
            except TypeError: pr = tok.apply_chat_template([{"role": "user", "content": it["query"]}], tools=tools, add_generation_prompt=True, tokenize=False)
            ans = "\n".join("<tool_call>\n" + json.dumps(dict(name=x["name"], arguments=x["arguments"])) + "\n</tool_call>" for x in it["gold"]) or "None of the available tools applies to this request."
            texts.append(pr + ans + "<|im_end|>\n")
        rng = np.random.RandomState(0); rng.shuffle(texts); stream = []
        for t in texts:
            stream += tok.encode(t, add_special_tokens=False)
            if len(stream) >= ncal * CL: break
        stream = torch.tensor(stream[:ncal * CL]).view(-1, CL); emb_w = st.get("model.embed_tokens.weight", dev)
        cur = [torch.nn.functional.embedding(stream[i:i + 1].to(dev), emb_w) for i in range(len(stream))]; del emb_w
        try: rope = RotCls(config=cfgm, device=dev)
        except TypeError: rope = RotCls(config=cfgm).to(dev)
        pos = torch.arange(CL, device=dev)[None]; pe = rope(cur[0], pos); kw = dict(attention_mask=None, position_ids=pos, position_embeddings=pe)
        prev = torch.get_default_dtype(); torch.set_default_dtype(torch.bfloat16)
        with torch.device(dev): layer = LayerCls(cfgm, 0)
        torch.set_default_dtype(prev); layer.eval()
        def fwd(x):
            o = layer(x, **kw); return o[0] if isinstance(o, tuple) else o
    # ---- layers
    for oli in range(nl):
        present = {n: f"model.layers.{oli}.{n}.weight" for n, _ in CLS if st.has(f"model.layers.{oli}.{n}.weight")}
        Hs = {}
        if gptq:
            for k, p in layer.named_parameters(): p.data.copy_(st.get(f"model.layers.{oli}.{k}"))
            mods = {n: layer.get_submodule(n) for n in present}; nsm = {}
            def mk(key):
                def hook(m, i, o):
                    x = i[0].reshape(-1, i[0].shape[-1]).float(); nsm[key] = nsm.get(key, 0) + x.shape[0]; Hs[key] = Hs.get(key, 0) + 2 * x.T @ x
                return hook
            hds = [mods[GROUP_KEY[c]].register_forward_hook(mk(c)) for c in GROUP_KEY if GROUP_KEY[c] in mods]
            with torch.no_grad():
                for x in cur: fwd(x)
            for h in hds: h.remove()
            Hs = {k: v / nsm[k] for k, v in Hs.items()}
        for n, c in CLS:
            if n not in present: continue
            nm = present[n]; Lv = levels.get(n, levels.get(c, levels.get({"attn_in": "attn", "o": "o", "mlp_in": "mlp", "down": "down"}[c])))
            Wf = st.get(nm, dev).float(); C = Wf.shape[1]; key = GROUP_KEY[c]; rk = f"L{oli}.{c}"; sg = signs(oli, key, C, dev)
            if rk not in w.S: w.add_signs(rk, sg)
            Wr = LD.rot(Wf, sg)
            if gptq:
                H = Hs[c]; Hr = LD.rot(LD.rot(H, sg).T.contiguous(), sg).T.contiguous(); Q, codes, SC = Qz.gptq(Wr, Hr, Lv, damp); del Hr
            else: Q, codes, SC = Qz.rtn(Wr, Lv)
            sim = LD.unrot(Q, sg).to(torch.bfloat16); sink.put_q(nm, codes, SC, Lv, g, mbits, rk, sim); handled.add(nm)
            if gptq: mods[n].weight.data = sim.clone()
            del Wf, Wr, Q, codes, SC, sim
        for k in [k for k in st.wm if k.startswith(f"model.layers.{oli}.") and k not in handled]: add_raw(k)
        if gptq:
            with torch.no_grad(): cur = [fwd(x) for x in cur]
            del Hs
        if oli % 4 == 0 or oli == nl - 1: log(f"layer {oli}/{nl} {time.time() - t0:.0f}s  est_MB={sink.est_bits / 8e6:.1f} real_MB={os.path.getsize(w.path) / 1e6:.1f}")
        if dev == "cuda": torch.cuda.empty_cache()
    # ---- head (GPTQ against final-normed hidden covariance if calibrating, else RTN) and embedding (RTN)
    def quant_head_like(name, seed, tag, Lv, gg, H=None):
        Wf = st.get(name, dev).float(); sg = signs(seed, tag, hid, dev); w.add_signs(tag, sg); Wr = LD.rot(Wf, sg); q_ = Quant(gg, mbits, dev)
        if H is not None: Hr = LD.rot(LD.rot(H, sg).T.contiguous(), sg).T.contiguous(); Q, codes, SC = q_.gptq(Wr, Hr, Lv, damp)
        else: Q, codes, SC = q_.rtn(Wr, Lv, gg)
        sim = LD.unrot(Q, sg).to(torch.bfloat16); sink.put_q(name, codes, SC, Lv, gg, mbits, tag, sim); handled.add(name)
    Hh = None
    if gptq:
        nm_ = st.get("model.norm.weight", dev); Hh = 0; nh = 0
        with torch.no_grad():
            for x in cur:
                xf = x.float(); y = (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + conf["rms_norm_eps"])).to(torch.bfloat16) * nm_
                y = y.reshape(-1, hid).float(); Hh = Hh + 2 * y.T @ y; nh += y.shape[0]
        Hh = Hh / nh; del cur
    alias = tied and st.has("lm_head.weight") and torch.equal(st.get("lm_head.weight"), st.get("model.embed_tokens.weight"))
    if st.has("lm_head.weight") and not alias: quant_head_like("lm_head.weight", 99999, "head", levels["head"], g_emb, Hh)
    quant_head_like("model.embed_tokens.weight", 88888, "emb", levels["emb"], g_emb, None)
    if alias: w.add_alias("lm_head.weight", "model.embed_tokens.weight", st.dtypes["lm_head.weight"], st.shapes["lm_head.weight"]); handled.add("lm_head.weight")
    for k in sorted(set(st.wm) - handled): add_raw(k)
    sink.flush()
    # ---- non-weight files
    files = {}
    for f in keep_files:
        if os.path.exists(os.path.join(model_dir, f)): shutil.copy(os.path.join(model_dir, f), os.path.join(out_dir, f))
    man = {k: [st.dtypes[k], st.shapes[k]] for k in st.wm}; json.dump(man, open(os.path.join(out_dir, "tensor_manifest.json"), "w"), separators=(",", ":"))
    blob = w.finish(meta=dict(levels=levels, g=g, g_emb=g_emb, mbits=mbits, gptq=gptq, source=os.path.basename(model_dir.rstrip("/"))))
    json.dump(sink.sim_hash, open(os.path.join(out_dir + ".sim_hashes.json"), "w"))
    json.dump(dict(rows=sink.rows, raw_bytes=raw_bytes, base_bytes=base_bytes, est_quant_bytes=sink.est_bits / 8, blob_bytes=blob, secs=time.time() - t0), open(out_dir + ".quant_report.json", "w"))
    log(f"QUANT_DONE blob={blob/1e6:.1f}MB est_quant={sink.est_bits/8e6:.1f}MB raw={raw_bytes/1e6:.1f}MB base={base_bytes/1e9:.3f}GB {time.time()-t0:.0f}s")
    return dict(blob=blob, base_bytes=base_bytes, est_quant_bytes=sink.est_bits / 8, raw_bytes=raw_bytes)


def parse_levels(s):
    d = {}
    for kv in s.split(","): k, v = kv.split("="); d[k] = int(v)
    return d


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True); ap.add_argument("--out", required=True); ap.add_argument("--levels", default="attn=3,o=3,mlp=3,down=3,emb=4,head=4")
    ap.add_argument("--g", type=int, default=256); ap.add_argument("--g-emb", type=int, default=128); ap.add_argument("--mbits", type=int, default=8); ap.add_argument("--gptq", action="store_true")
    ap.add_argument("--calib"); ap.add_argument("--ncal", type=int, default=128); ap.add_argument("--sim-dir"); a = ap.parse_args()
    quantize_model(a.model, a.out, parse_levels(a.levels), a.g, a.g_emb, a.mbits, a.gptq, a.calib, a.ncal, sim_dir=a.sim_dir)
