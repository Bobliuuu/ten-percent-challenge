"""L3: Llama-3.1-405B-Instruct streaming GPTQ+Hadamard+ECSQ quantisation (9.6% budget) and layer-streamed label-task evaluation on ONE H200.
Adapted from research/D_scale/dense_stream.py. All state lives on Volume tenpct-l3-405b (resumable). Dev data only (C_tasks items.jsonl, calib.jsonl; s1_calib).
modal deploy l3.py ; python l3.py launch quant|eval_base|eval_quant [name]  (spawns, prints function-call id)
Volume layout: /w/bf16 (base weights), /w/out/<name>/{sensA/,ckA.pt,ckB.pt,q/L{i}.pt,q/L{i}.stat.json,emb.pt,head.pt,meta.json}, /w/res/ct_*.json"""
import modal, os, sys, json, math, time, glob
from collections import Counter

app = modal.App("l3-405b")
vol = modal.Volume.from_name("tenpct-l3-405b", version=2)
HERE = os.path.dirname(os.path.abspath(__file__))
img = (modal.Image.debian_slim(python_version="3.11")
       .pip_install("torch==2.8.0", "transformers==4.57.1", "safetensors", "numpy", "accelerate")
       .add_local_file(HERE + "/../C_tasks/items.jsonl", "/data/items.jsonl")
       .add_local_file(HERE + "/../C_tasks/calib.jsonl", "/data/calib.jsonl")
       .add_local_file(HERE + "/../D_scale/t1v2/calib_t1.jsonl", "/data/calib_t1.jsonl")
       .add_local_file(HERE + "/inj_probe_items.jsonl", "/data/inj_probe_items.jsonl")
       .add_local_file(HERE + "/inj_calib.jsonl", "/data/inj_calib.jsonl"))

MD = "/w/bf16"
HB = 128
LS = [2, 3, 4, 5, 6, 8, 12, 16]
PG = [1.0, .9, .8, .7, .62, .55, .5, .45, .4, .36, .32, .28, .25, .22, .19, .16, .13]
linn = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]
groups = [("self_attn.q_proj", ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"]), ("self_attn.o_proj", ["self_attn.o_proj"]),
          ("mlp.gate_proj", ["mlp.gate_proj", "mlp.up_proj"]), ("mlp.down_proj", ["mlp.down_proj"])]


def log(*a): print(time.strftime("%H:%M:%S"), *a, flush=True)


def common():
    import numpy as np, torch, torch.nn.functional as F
    from safetensors import safe_open
    return np, torch, F, safe_open


class ST:
    def __init__(s, path):
        from safetensors import safe_open
        s.so = safe_open; s.path = path
        idx = json.load(open(path + "/model.safetensors.index.json")); s.wm = idx["weight_map"]; s.total = idx["metadata"]["total_size"]; s.h = {}
    def get(s, n, dev="cuda"):
        f = s.wm[n]
        if f not in s.h: s.h[f] = s.so(s.path + "/" + f, "pt", device="cpu")
        return s.h[f].get_tensor(n).to(dev)
    def has(s, n): return n in s.wm


def ent(h):
    p = h / h.sum(); p = p[p > 0]; return float(-(p * p.log2()).sum())


def make_quantiser(g):
    import torch
    class Quantiser:
        def __init__(s, g):
            s.g = g; s.PGt = torch.tensor(PG).cuda()
            Hm = torch.ones(1, 1)
            while Hm.shape[0] < HB: Hm = torch.cat([torch.cat([Hm, Hm], 1), torch.cat([Hm, -Hm], 1)], 0)
            s.Hb = (Hm / math.sqrt(HB)).cuda()
        def quant(s, x, sc, Lv):
            half = (Lv - 1) / 2; t = x / sc
            qi = torch.clamp(torch.round(t + half), 0, Lv - 1); return (qi - half) * sc, qi
        def qparams(s, x, Lv):
            half = (Lv - 1) / 2; am = x.abs().amax(1, keepdim=True).clamp(min=1e-8); best = None
            for p in s.PGt:
                sc = (am * p / max(half, 0.5)).half().float().clamp(min=1e-8); q = s.quant(x, sc, Lv)[0]; e = ((q - x) ** 2).sum(1, keepdim=True)
                if best is None: best = (e, sc)
                else: m = e < best[0]; best = (torch.where(m, e, best[0]), torch.where(m, sc, best[1]))
            return best[1]
        def rot(s, M, sg): R, C = M.shape; return ((M * sg).view(R, C // HB, HB) @ s.Hb).reshape(R, C)
        def unrot(s, M, sg): R, C = M.shape; return ((M.view(R, C // HB, HB) @ s.Hb).reshape(R, C)) * sg
        def rtn(s, W, Lv, g=None):
            g = g or s.g; R, C = W.shape; out = torch.empty_like(W); hist = torch.zeros(Lv, device=W.device); ch = max(1, int(2e7 // C))
            for r in range(0, R, ch):
                w = W[r:r + ch]; x = w.reshape(-1, g); sc = s.qparams(x, Lv); q, qi = s.quant(x, sc, Lv); out[r:r + ch] = q.view_as(w)
                hist += torch.bincount(qi.long().flatten(), minlength=Lv).float()
            return out, hist
        def gptq(s, W, H, Lv, damp=0.01):
            g = s.g; W = W.float().clone(); R, C = W.shape; H = H.clone(); bsz = max(128, g)
            dead = torch.diag(H) == 0; H[dead, dead] = 1; W[:, dead] = 0
            H = H + damp * torch.mean(torch.diag(H)) * torch.eye(C, device=H.device)
            Hinv = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(H)), upper=True); del H
            Q = torch.zeros_like(W); hist = torch.zeros(Lv, device=W.device); sc = None
            QI = torch.zeros(R, C, dtype=torch.uint8, device=W.device); SC = torch.zeros(R, C // g, dtype=torch.float16, device=W.device)
            for i1 in range(0, C, bsz):
                i2 = min(i1 + bsz, C); W1 = W[:, i1:i2].clone(); Q1 = torch.zeros_like(W1); E1 = torch.zeros_like(W1); Hi = Hinv[i1:i2, i1:i2]
                for i in range(i2 - i1):
                    if i % g == 0: sc = s.qparams(W1[:, i:i + g], Lv); SC[:, (i1 + i) // g] = sc.flatten().half()
                    w = W1[:, i]; d = Hi[i, i]; q, qi = s.quant(w.unsqueeze(1), sc, Lv); q = q.flatten()
                    hist += torch.bincount(qi.long().flatten(), minlength=Lv).float(); QI[:, i1 + i] = qi.flatten().to(torch.uint8)
                    Q1[:, i] = q; err = (w - q) / d; W1[:, i:] -= err.unsqueeze(1) @ Hi[i, i:].unsqueeze(0); E1[:, i] = err
                Q[:, i1:i2] = Q1; W[:, i2:] -= E1 @ Hinv[i1:i2, i2:]
            return Q, hist, QI, SC
    return Quantiser(g)


def pack4(QI): return (QI[:, 0::2] | (QI[:, 1::2] << 4)).contiguous()


def dequant(e, g, Qz, tobf=True):
    """e = dict(L, codes (packed4), scales, sg) -> bf16 weight (fp32 unrot, matches quantisation arithmetic)"""
    import torch
    c = e["codes"].cuda(); R = c.shape[0]; C = c.shape[1] * 2; Lv = e["L"]
    q = torch.stack([c & 15, c >> 4], -1).view(R, C).float()
    Q = (q - (Lv - 1) / 2) * e["scales"].cuda().float().repeat_interleave(g, 1)
    W = Qz.unrot(Q, e["sg"].cuda().float()); del q, Q
    return W.to(torch.bfloat16)


def setup_model(Qz=None):
    import torch
    from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
    conf = AutoConfig.from_pretrained(MD); conf._attn_implementation = "sdpa"
    tok = AutoTokenizer.from_pretrained(MD)
    with torch.device("meta"): meta = AutoModelForCausalLM.from_config(conf, dtype=torch.bfloat16)
    LayerCls = type(meta.model.layers[0]); RotCls = type(meta.model.rotary_emb); del meta
    prev = torch.get_default_dtype(); torch.set_default_dtype(torch.bfloat16)
    with torch.device("cuda"): layer = LayerCls(conf, 0)
    torch.set_default_dtype(prev); layer.eval()
    rope = RotCls(config=conf, device="cuda")
    return conf, tok, layer, rope


def chat(tok, p):
    return tok.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True, tokenize=False)


def atomic_save(obj, path):
    import torch
    torch.save(obj, path + ".tmp"); os.replace(path + ".tmp", path)


class Prefetch:
    """load next layer's tensors to CPU in a thread while GPU works"""
    def __init__(s, st, names_fn):
        from concurrent.futures import ThreadPoolExecutor
        s.st = st; s.fn = names_fn; s.ex = ThreadPoolExecutor(1); s.fut = {}
    def _load(s, i): return {k: s.st.get(k, "cpu") for k in s.fn(i)}
    def want(s, i):
        if i not in s.fut: s.fut[i] = s.ex.submit(s._load, i)
    def get(s, i):
        s.want(i); r = s.fut.pop(i).result(); return r


@app.function(image=img, gpu="H200", volumes={"/w": vol}, timeout=24 * 3600, memory=98304, cpu=8, retries=modal.Retries(max_retries=10, initial_delay=5.0, backoff_coefficient=1.0))
def quant(cfg):
    np, torch, F, safe_open = common()
    t0 = time.time(); torch.manual_seed(0); vol.reload()
    name = cfg["name"]; g = cfg.get("g", 128); GB = 11; OUT = f"/w/out/{name}"; os.makedirs(OUT + "/sensA", exist_ok=True); os.makedirs(OUT + "/q", exist_ok=True)
    st = ST(MD); conf, tok, layer, rope = setup_model(); Qz = make_quantiser(g)
    nl = conf.num_hidden_layers; hid = conf.hidden_size; V = conf.vocab_size; base_bytes = st.total
    EOT = tok.convert_tokens_to_ids("<|eot_id|>")
    # ---- calibration (dev: C_tasks calib.jsonl [S1-role cal items] + s1_calib tool-call slice)
    cc = [json.loads(l) for l in open("/data/calib.jsonl")]
    if cfg.get("calib_file"): cc = [json.loads(l) for l in open(cfg["calib_file"])]
    if cfg.get("calib_tasks"): cc = [c for c in cc if c["task"] in cfg["calib_tasks"]]
    if cfg.get("sens_from") and not os.path.exists(f"{OUT}/sens_g{g}.json"):
        import shutil; shutil.copy(f"/w/out/{cfg['sens_from']}/sens_g{g}.json", f"{OUT}/sens_g{g}.json"); log("sens copied from", cfg["sens_from"])
    cal_txt = [chat(tok, c["user"]) + c["answer"] + "<|eot_id|>" for c in cc]
    if cfg.get("calib_t1", True):
        t1 = [json.loads(l) for l in open("/data/calib_t1.jsonl")]; t1c = []
        for it in t1[:cfg.get("n_t1", len(cc) // 2)]:
            tools = [{"type": "function", "function": t} for t in it["tools"]]
            pr = tok.apply_chat_template([{"role": "user", "content": it["query"]}], tools=tools, add_generation_prompt=True, tokenize=False)
            ans = "\n".join(json.dumps(dict(name=gg["name"], parameters=gg["arguments"])) for gg in it["gold"]) or "None of the available tools applies to this request."
            t1c.append(pr + ans + "<|eot_id|>")
        cal_txt = cal_txt + t1c
    rng = np.random.RandomState(cfg.get("seed", 0)); rng.shuffle(cal_txt)
    ncal = cfg.get("ncal", 128); CL = cfg.get("CL", 512); stream = []
    for s_ in cal_txt:
        stream += tok.encode(s_, add_special_tokens=False)
        if len(stream) >= ncal * CL: break
    log("calib tokens", len(stream), "need", ncal * CL)
    n_ = min(len(stream), ncal * CL) // CL * CL; stream = torch.tensor(stream[:n_]).view(-1, CL); log('calib seqs', len(stream))
    emb_w = st.get("model.embed_tokens.weight")
    inps = [torch.nn.functional.embedding(stream[i:i + 1].cuda(), emb_w) for i in range(len(stream))]
    pos = torch.arange(CL, device="cuda")[None]; pe = rope(inps[0], pos)
    kw = dict(attention_mask=None, position_ids=pos, position_embeddings=pe)
    lnames = lambda i: [f"model.layers.{i}.{k}" for k, _ in layer.named_parameters()]
    pf = Prefetch(st, lnames)
    def load_layer(i):
        d = pf.get(i)
        for k, p in layer.named_parameters(): p.data.copy_(d[f"model.layers.{i}.{k}"])
        if i + 1 < nl: pf.want(i + 1)
    def fwd(x):
        o = layer(x, **kw); return o[0] if isinstance(o, tuple) else o
    def collect(cur):
        Hs = {}; nsm = {}; mods = {n: layer.get_submodule(n) for n in linn}
        def mk(key):
            def hook(m, i, o):
                x = i[0].reshape(-1, i[0].shape[-1]).float(); nsm[key] = nsm.get(key, 0) + x.shape[0]; Hs[key] = Hs.get(key, 0) + 2 * x.T @ x
            return hook
        hds = [mods[k].register_forward_hook(mk(k)) for k, _ in groups]
        with torch.no_grad():
            for x in cur: fwd(x)
        for h in hds: h.remove()
        return mods, {k: Hs[k] / nsm[k] for k in Hs}
    def signs(oli, key, C): return (torch.randint(0, 2, (C,), generator=torch.Generator().manual_seed(1000 * oli + len(key)), dtype=torch.float32) * 2 - 1).cuda()
    def ck_load(path):
        if os.path.exists(path):
            try: return torch.load(path)
            except Exception as e: log("ck load fail", e)
        return None
    # ------------------------------------------------------------------ pass A
    spath = f"{OUT}/sens_g{g}.json"
    if os.path.exists(spath): S = json.load(open(spath))
    else:
        ck = ck_load(f"{OUT}/ckA.pt"); start = 0; cur = [x.clone() for x in inps]
        if ck: start = ck["idx"]; cur = [x.cuda() for x in ck["cur"]]; log("resume pass A at", start)
        pf.want(start); torch.set_float32_matmul_precision("high")
        for oli in range(start, nl):
            ts = time.time(); load_layer(oli); mods, Hs = collect(cur); rec = dict(tab={}, bi=None)
            with torch.no_grad():
                nxt = [fwd(x) for x in cur]
                rec["bi"] = float(np.mean([1 - F.cosine_similarity(a.float().flatten(0, 1), b.float().flatten(0, 1), dim=-1).mean().item() for a, b in zip(cur[:16], nxt[:16])]))
                for key, mem in groups:
                    H = Hs[key]; C = H.shape[0]; sg = signs(oli, key, C); Hr = Qz.rot(Qz.rot(H, sg).T.contiguous(), sg).T.contiguous()
                    for n in mem:
                        W = Qz.rot(mods[n].weight.data.float(), sg); ref = float(((W @ Hr) * W).sum()); tab = {}
                        for Lv in LS:
                            Q, h = Qz.rtn(W, Lv); Dd = W - Q; tab[Lv] = [float(((Dd @ Hr) * Dd).sum()) / ref, ent(h)]
                        rec["tab"][f"{oli}.{n}"] = dict(n=W.numel(), tab=tab)
            json.dump(rec, open(f"{OUT}/sensA/L{oli}.json", "w")); cur = nxt; del Hs; torch.cuda.empty_cache()
            if (oli + 1) % 6 == 0 and oli + 1 < nl: atomic_save(dict(idx=oli + 1, cur=[x.cpu() for x in cur]), f"{OUT}/ckA.pt")
            if (oli + 1) % 6 == 0: vol.commit()
            log("sensA", oli, "layer s", round(time.time() - ts), "total s", round(time.time() - t0))
        torch.set_float32_matmul_precision("highest")
        S = dict(tab={}, bi=[])
        for i in range(nl):
            r = json.load(open(f"{OUT}/sensA/L{i}.json")); S["tab"].update(r["tab"]); S["bi"].append(r["bi"])
        json.dump(S, open(spath, "w")); S = json.loads(json.dumps(S)); vol.commit(); del cur
        if os.path.exists(f"{OUT}/ckA.pt"): os.remove(f"{OUT}/ckA.pt")
    log("SENS_DONE")
    # ------------------------------------------------------------------ fixed bytes + allocation
    keep = list(range(nl))
    lin_total = sum(S["tab"][f"{i}.{n}"]["n"] for i in keep for n in linn)
    lin_names = {f"model.layers.{i}.{n}.weight" for i in keep for n in linn}
    other_bytes = sum(int(np.prod(safe_open(MD + "/" + f, "pt").get_slice(k).get_shape())) * 2 for k, f in st.wm.items()
                      if k not in lin_names and k not in ("model.embed_tokens.weight", "lm_head.weight"))
    fb = sum(os.path.getsize(p) for p in glob.glob(MD + "/*") if os.path.basename(p) in ("tokenizer.json", "tokenizer_config.json", "config.json", "generation_config.json", "special_tokens_map.json"))
    CODE = 150000; fixed_bytes = other_bytes + fb + CODE
    eL = cfg.get("emb_L", 4); hL = cfg.get("head_L", 4); ge = cfg.get("g_emb", 128)
    def est_ent(W, Lv):
        gen = torch.Generator().manual_seed(7); idx = torch.randperm(W.shape[0], generator=gen)[:4096]
        sg = torch.ones(W.shape[1]).cuda(); Wr = Qz.rot(W[idx].cuda().float(), sg); Q, h = Qz.rtn(Wr, Lv); return ent(h)
    head_w = st.get("lm_head.weight")
    ent_e = est_ent(emb_w, eL); ent_h = est_ent(head_w, hL)
    emb_bytes = V * hid * (ent_e * 1.005 + GB / ge) / 8; head_bytes = V * hid * (ent_h * 1.005 + GB / ge) / 8
    target = cfg["target_ratio"] * base_bytes; body_bits = (target - fixed_bytes - emb_bytes - head_bytes) * 8
    bi = np.array(S['bi']); gam = cfg.get('gamma', 0.0); lw = {i: (bi[i] / bi.mean()) ** gam for i in keep}
    items = [(f"{i}.{n}", S["tab"][f"{i}.{n}"], lw[i]) for i in keep for n in linn]
    prot = set(cfg.get("protect_layers", []))  # layers forced to at least prot_L levels
    def pick(lam):
        tot = 0; ch = {}
        for k, it, wt in items:
            best = None; li = int(k.split(".")[0])
            for Lv in LS:
                if li in prot and Lv < cfg.get("prot_L", 4): continue
                e, H_ = it["tab"][str(Lv)]; b = it["n"] * (H_ * 1.005 + GB / g); c = wt * e * it["n"] / lin_total * 1e3 + lam * b
                if best is None or c < best[0]: best = (c, Lv, b)
            ch[k] = best[1]; tot += best[2]
        return tot, ch
    lo, hi = 1e-14, 1e-3
    for _ in range(60):
        mid = math.sqrt(lo * hi); tot, ch = pick(mid)
        if tot > body_bits: lo = mid
        else: hi = mid
    tot, levels = pick(hi); log("alloc body bpw", tot / lin_total, "target", body_bits / lin_total, "fixed MB", fixed_bytes / 1e6, "emb MB", emb_bytes / 1e6, "head MB", head_bytes / 1e6)
    # ------------------------------------------------------------------ pass B
    cur = [x.clone() for x in inps]; start = 0
    ck = ck_load(f"{OUT}/ckB.pt")
    if ck: start = ck["idx"]; cur = [x.cuda() for x in ck["cur"]]; log("resume pass B at", start)
    pf.want(start)
    for oli in range(start, nl):
        ts = time.time(); load_layer(oli); mods, Hs = collect(cur); sv = {}; qb = 0.0; Lc = Counter()
        for key, mem in groups:
            H = Hs[key]; C = H.shape[0]; sg = signs(oli, key, C); Hr = Qz.rot(Qz.rot(H, sg).T.contiguous(), sg).T.contiguous()
            for n in mem:
                Lv = levels[f"{oli}.{n}"]; W = Qz.rot(mods[n].weight.data.float(), sg)
                Q, h, QI, SC = Qz.gptq(W, Hr, Lv, damp=cfg.get('damp', 0.01)); mods[n].weight.data = Qz.unrot(Q, sg).to(torch.bfloat16)
                qb += W.numel() * (ent(h) * 1.005 + GB / g); Lc[Lv] += W.numel()
                sv[n] = dict(L=Lv, codes=pack4(QI).cpu(), scales=SC.cpu(), sg=sg.to(torch.int8).cpu()); del Q, QI, SC, W
            del Hr
        torch.save(sv, f"{OUT}/q/L{oli}.pt"); json.dump(dict(qbits=qb, Lc={str(k): v for k, v in Lc.items()}), open(f"{OUT}/q/L{oli}.stat.json", "w"))
        with torch.no_grad(): cur = [fwd(x) for x in cur]
        del Hs; torch.cuda.empty_cache()
        if (oli + 1) % 4 == 0 and oli + 1 < nl: atomic_save(dict(idx=oli + 1, cur=[x.cpu() for x in cur]), f"{OUT}/ckB.pt")
        if (oli + 1) % 2 == 0: vol.commit()
        log("passB", oli, "layer s", round(time.time() - ts), "total s", round(time.time() - t0), "bpw", round(qb / sum(S["tab"][f"{oli}.{n}"]["n"] for n in linn), 3))
    vol.commit()
    stats = [json.load(open(f"{OUT}/q/L{i}.stat.json")) for i in range(nl)]
    qbits = sum(s["qbits"] for s in stats); Lcount = Counter()
    for s in stats:
        for k, v in s["Lc"].items(): Lcount[int(k)] += v
    Hh = 0; nh = 0
    with torch.no_grad():
        nm = st.get("model.norm.weight")
        for x in cur:
            xf = x.float(); y = (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + conf.rms_norm_eps)).to(torch.bfloat16) * nm
            y = y.reshape(-1, hid).float(); Hh = Hh + 2 * y.T @ y; nh += y.shape[0]
    Hh = Hh / nh; del inps, cur; torch.cuda.empty_cache()
    sg = signs(99999, "head", hid); Wh = Qz.rot(head_w.float(), sg); Hr = Qz.rot(Qz.rot(Hh, sg).T.contiguous(), sg).T.contiguous()
    Q, h, _, _ = Qz.gptq(Wh, Hr, hL); head_bits = head_w.numel() * (ent(h) * 1.005 + GB / ge)
    torch.save(Qz.unrot(Q, sg).to(torch.bfloat16).cpu(), OUT + "/head.pt"); del Q, Wh, Hr
    sg = signs(88888, "emb", hid); We = Qz.rot(emb_w.float(), sg); Wq = torch.empty_like(We); hist = torch.zeros(eL, device="cuda")
    for r in range(0, We.shape[0], 8192):
        x = We[r:r + 8192].reshape(-1, ge); sc = Qz.qparams(x, eL); q, qi = Qz.quant(x, sc, eL); Wq[r:r + 8192] = q.view(-1, hid); hist += torch.bincount(qi.long().flatten(), minlength=eL).float()
    emb_bits = emb_w.numel() * (ent(hist) * 1.005 + GB / ge); torch.save(Qz.unrot(Wq, sg).to(torch.bfloat16).cpu(), OUT + "/emb.pt")
    tot_bytes = qbits / 8 + head_bits / 8 + emb_bits / 8 + fixed_bytes
    meta_ = dict(cfg=cfg, bytes=tot_bytes, base_bytes=base_bytes, ratio=tot_bytes / base_bytes, body_bpw=qbits / lin_total, g=g, ge=ge,
                 L_hist={str(k): v / lin_total for k, v in Lcount.items()}, parts_MB=dict(body=qbits / 8e6, emb=emb_bits / 8e6, head=head_bits / 8e6, other_bf16=other_bytes / 1e6, tokenizer_cfg=fb / 1e6, code=CODE / 1e6),
                 secs_last_run=time.time() - t0)
    json.dump(meta_, open(OUT + "/meta.json", "w")); vol.commit(); log("QUANT_DONE", name, "ratio", meta_["ratio"], "bytes", tot_bytes)
    print("RESULT_JSON", json.dumps(meta_)); return meta_


@app.function(image=img, gpu="H200", volumes={"/w": vol}, timeout=24 * 3600, memory=98304, cpu=8, retries=modal.Retries(max_retries=5, initial_delay=5.0, backoff_coefficient=1.0))
def evalrun(cfg):
    """layer-streamed teacher-forced label-task eval. cfg: kind in {base, quant}, name (for quant), tasks"""
    np, torch, F, safe_open = common()
    t0 = time.time(); vol.reload(); kind = cfg["kind"]; name = cfg.get("name", "t096"); g = cfg.get("g", 128)
    os.makedirs("/w/res", exist_ok=True)
    st = ST(MD); conf, tok, layer, rope = setup_model(); Qz = make_quantiser(g); nl = conf.num_hidden_layers; hid = conf.hidden_size
    EOT = tok.convert_tokens_to_ids("<|eot_id|>"); QD = f"/w/out/{name}"
    items = [json.loads(l) for l in open("/data/items.jsonl")]
    if cfg.get("tasks"): items = [i for i in items if i["task"] in cfg["tasks"]]
    MAXL = cfg.get("maxlen", 2048); enc = []
    for it in items:
        p = tok.encode(chat(tok, it["user"]), add_special_tokens=False); a = tok.encode(it["answer"], add_special_tokens=False) + [EOT]
        full = (p + a)[-MAXL:]; enc.append((full, len(a)))
    order = np.argsort([-len(f) for f, _ in enc]); BUD = cfg.get("tok_budget", 24000); chunks = []; cur_ = []; mx = 0
    for i in order:
        L = len(enc[i][0])
        if cur_ and max(mx, L) * (len(cur_) + 1) > BUD: chunks.append(cur_); cur_ = []; mx = 0
        cur_.append(int(i)); mx = max(mx, L)
    if cur_: chunks.append(cur_)
    log("items", len(items), "chunks", len(chunks), "tokens", sum(len(f) for f, _ in enc))
    embw = st.get("model.embed_tokens.weight") if kind == "base" else torch.load(QD + "/emb.pt").cuda()
    H = []
    for ch in chunks:
        mxl = max(len(enc[i][0]) for i in ch); ids = torch.zeros(len(ch), mxl, dtype=torch.long)
        for j, i in enumerate(ch): ids[j, :len(enc[i][0])] = torch.tensor(enc[i][0])
        H.append(F.embedding(ids.cuda(), embw))
    del embw
    lnames = lambda i: [f"model.layers.{i}.{k}" for k, _ in layer.named_parameters() if kind == "base" or k not in [n + ".weight" for n in linn]]
    pf = Prefetch(st, lnames); pf.want(0)
    posc = {}
    def getpe(L):
        if L not in posc:
            pos = torch.arange(L, device="cuda")[None]; posc[L] = (pos, rope(H[0][:1, :1], pos))
        return posc[L]
    for oli in range(nl):
        ts = time.time(); d = pf.get(oli)
        if oli + 1 < nl: pf.want(oli + 1)
        for k, p in layer.named_parameters():
            if k in [n + ".weight" for n in linn] and kind == "quant": continue
            p.data.copy_(d[f"model.layers.{oli}.{k}"])
        if kind == "quant":
            sv = torch.load(f"{QD}/q/L{oli}.pt")
            for n in linn: layer.get_submodule(n).weight.data.copy_(dequant(sv[n], g, Qz))
            del sv
        with torch.no_grad():
            for ci in range(len(H)):
                L = H[ci].shape[1]; pos, pe = getpe(L)
                o = layer(H[ci], attention_mask=None, position_ids=pos, position_embeddings=pe); H[ci] = o[0] if isinstance(o, tuple) else o
        torch.cuda.empty_cache()
        if oli % 5 == 0: log("eval", kind, "layer", oli, "s", round(time.time() - ts), "total", round(time.time() - t0))
    nm = st.get("model.norm.weight"); head = st.get("lm_head.weight") if kind == "base" else torch.load(QD + "/head.pt").cuda()
    rows = [None] * len(items)
    with torch.no_grad():
        for ci, ch in enumerate(chunks):
            for j, i in enumerate(ch):
                full, na = enc[i]; Lf = len(full); x = H[ci][j, Lf - na - 1:Lf - 1].float()
                x = (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + conf.rms_norm_eps)).to(torch.bfloat16) * nm
                lg = (x @ head.T).float(); at = torch.tensor(full[Lf - na:]).cuda()
                gl = lg.gather(1, at[:, None])[:, 0]; lg2 = lg.clone(); lg2.scatter_(1, at[:, None], -1e9); oth, oi = lg2.max(1); mg = gl - oth
                it = items[i]
                rows[i] = dict(id=it["id"], task=it["task"], role=it["role"], src=it["src"], kind=it["kind"], correct=bool((mg > 0).all().item()),
                               m_first=float(mg[0]), m_min=float(mg.min()), m_all=[round(float(v), 3) for v in mg], oth=[int(v) for v in oi], ans=[int(v) for v in at])
    meta = json.load(open(QD + "/meta.json")) if kind == "quant" else {}
    out = dict(model="Meta-Llama-3.1-405B-Instruct", kind=kind, name=name, rows=rows, secs=time.time() - t0, ratio=meta.get("ratio"), bytes=meta.get("bytes"), base_bytes=st.total, cfg=cfg)
    fn = f"/w/res/ct_L405_{'bf16' if kind == 'base' else name}.json"; json.dump(out, open(fn, "w")); vol.commit()
    log("CT_DONE", kind, name, sum(r["correct"] for r in rows), len(rows), "secs", round(time.time() - t0))
    print("RESULT_JSON", json.dumps(dict(kind=kind, name=name, acc=sum(r["correct"] for r in rows) / len(rows), ratio=out["ratio"]))); return fn


@app.function(image=img, gpu="H200", volumes={"/w": vol}, timeout=24 * 3600, memory=98304, cpu=8, retries=modal.Retries(max_retries=5, initial_delay=5.0, backoff_coefficient=1.0))
def evalinj2(cfg):
    """real-INJ-exam greedy decoding via speculative argmax chain (exactly equals greedy for resolved items).
    Sequence fed = chat(query) + Label tokens (gold, capitalised) + '.'. Saves argmax ids at every position from the first label token on."""
    np, torch, F, safe_open = common()
    t0 = time.time(); vol.reload(); kind = cfg["kind"]; name = cfg.get("name", "t096"); g = cfg.get("g", 128); tag = cfg.get("tag", kind)
    os.makedirs("/w/res", exist_ok=True)
    st = ST(MD); conf, tok, layer, rope = setup_model(); Qz = make_quantiser(g); nl = conf.num_hidden_layers; QD = f"/w/out/{name}"
    items = [json.loads(l) for l in open("/data/inj_probe_items.jsonl")]
    if cfg.get("limit"): items = items[:cfg["limit"]]
    DOT = tok.encode(".", add_special_tokens=False)[0]; enc = []; items = [dict(it, var=v) for it in items for v in (0, 1)]
    for it in items:
        p = tok.encode(chat(tok, it["query"]), add_special_tokens=False); w = it["gold_label"]; w = w.capitalize() if it["var"] == 0 else w
        lab = tok.encode(w, add_special_tokens=False)
        enc.append((p + lab + [DOT], len(lab) + 2, lab))
    order = np.argsort([-len(e[0]) for e in enc]); BUD = cfg.get("tok_budget", 24000); chunks = []; cur_ = []; mx = 0
    for i in order:
        L = len(enc[i][0])
        if cur_ and max(mx, L) * (len(cur_) + 1) > BUD: chunks.append(cur_); cur_ = []; mx = 0
        cur_.append(int(i)); mx = max(mx, L)
    if cur_: chunks.append(cur_)
    log("items", len(items), "chunks", len(chunks), "tokens", sum(len(e[0]) for e in enc), "maxlen", max(len(e[0]) for e in enc))
    embw = st.get("model.embed_tokens.weight") if kind == "base" else torch.load(QD + "/emb.pt").cuda()
    H = []
    for ch in chunks:
        mxl = max(len(enc[i][0]) for i in ch); ids = torch.zeros(len(ch), mxl, dtype=torch.long)
        for j, i in enumerate(ch): ids[j, :len(enc[i][0])] = torch.tensor(enc[i][0])
        H.append(F.embedding(ids.cuda(), embw))
    del embw
    lnames = lambda i: [f"model.layers.{i}.{k}" for k, _ in layer.named_parameters() if kind == "base" or k not in [n + ".weight" for n in linn]]
    pf = Prefetch(st, lnames); pf.want(0); posc = {}
    def getpe(L):
        if L not in posc:
            pos = torch.arange(L, device="cuda")[None]; posc[L] = (pos, rope(H[0][:1, :1], pos))
        return posc[L]
    for oli in range(nl):
        ts = time.time(); d = pf.get(oli)
        if oli + 1 < nl: pf.want(oli + 1)
        for k, p in layer.named_parameters():
            if k in [n + ".weight" for n in linn] and kind == "quant": continue
            p.data.copy_(d[f"model.layers.{oli}.{k}"])
        if kind == "quant":
            sv = torch.load(f"{QD}/q/L{oli}.pt")
            for n in linn: layer.get_submodule(n).weight.data.copy_(dequant(sv[n], g, Qz))
            del sv
        with torch.no_grad():
            for ci in range(len(H)):
                L = H[ci].shape[1]; pos, pe = getpe(L)
                o = layer(H[ci], attention_mask=None, position_ids=pos, position_embeddings=pe); H[ci] = o[0] if isinstance(o, tuple) else o
        torch.cuda.empty_cache()
        if oli % 10 == 0: log("evalinj", kind, tag, "layer", oli, "s", round(time.time() - ts), "total", round(time.time() - t0))
    nm = st.get("model.norm.weight"); head = st.get("lm_head.weight") if kind == "base" else torch.load(QD + "/head.pt").cuda()
    rows = [None] * len(items)
    with torch.no_grad():
        for ci, ch in enumerate(chunks):
            for j, i in enumerate(ch):
                full, npred, lab = enc[i]; Lf = len(full); x = H[ci][j, Lf - npred:Lf].float()
                x = (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + conf.rms_norm_eps)).to(torch.bfloat16) * nm
                lg = (x @ head.T).float(); tv, ti = lg.topk(2, -1)
                it = items[i]
                rows[i] = dict(id=it["id"], var=it["var"], source=it["source"], kind=it["kind"], subset=it["subset"], gold=it["gold_label"], group=it["group"], lab=lab,
                               pred=[int(v) for v in ti[:, 0]], gap=[round(float(v), 3) for v in (tv[:, 0] - tv[:, 1])])
    out = dict(kind=kind, name=name, tag=tag, rows=rows, secs=time.time() - t0, cfg=cfg)
    fn = f"/w/res/inj_L405_{tag}.json"; json.dump(out, open(fn, "w")); vol.commit()
    log("INJ_DONE", tag, len(rows), "secs", round(time.time() - t0)); print("RESULT_JSON", json.dumps(dict(tag=tag, n=len(rows)))); return fn


@app.function(image=img, gpu="H200", volumes={"/w": vol}, timeout=6 * 3600, memory=98304, cpu=8)
def dump_eh1(cfg):
    """write integer codes/scales/L/sg for embedding and lm_head to /w/out/<name>/q/{emb_q,head_q}.pt (same layout as L{i}.pt) and verify dequant == emb.pt/head.pt bit-exactly.
    emb: deterministic RTN re-run. head: GPTQ codes recovered from head.pt (bf16 of unrot(Q)): re-rotate, 2-cluster fit of scale, fp16-ulp search, exact bf16 check per 128-group."""
    np, torch, F, safe_open = common()
    vol.reload(); name = cfg.get("name", "t096"); QD = f"/w/out/{name}"; g = 128; Qz = make_quantiser(g); L = 4; half = (L - 1) / 2
    hid = 16384
    def signs(oli, key, C): return (torch.randint(0, 2, (C,), generator=torch.Generator().manual_seed(1000 * oli + len(key)), dtype=torch.float32) * 2 - 1).cuda()
    res = {}
    # ---- embedding: re-run RTN
    st = ST(MD); emb_w = st.get("model.embed_tokens.weight"); sg = signs(88888, "emb", hid); We = Qz.rot(emb_w.float(), sg)
    ref = torch.load(QD + "/emb.pt").cuda(); codes = []; scs = []; bad = 0; nbad_el = 0
    for r in range(0, We.shape[0], 8192):
        x = We[r:r + 8192].reshape(-1, g); sc = Qz.qparams(x, L); q, qi = Qz.quant(x, sc, L)
        rr = x.shape[0] // (hid // g)
        e = dict(L=L, codes=pack4(qi.view(rr, hid).to(torch.uint8)).cpu(), scales=sc.half().view(rr, hid // g).cpu(), sg=sg.to(torch.int8).cpu())
        codes.append(e["codes"]); scs.append(e["scales"])
        dq = Qz.unrot(q.view(rr, hid), sg).to(torch.bfloat16); nbad_el += int((dq != ref[r:r + 8192]).sum())
    emb_q = dict(L=L, codes=torch.cat(codes), scales=torch.cat(scs), sg=sg.to(torch.int8).cpu()); torch.save(emb_q, QD + "/q/emb_q.pt")
    # check via the shared dequant function
    mism = 0
    for r in range(0, emb_q["codes"].shape[0], 8192):
        e = dict(L=L, codes=emb_q["codes"][r:r + 8192], scales=emb_q["scales"][r:r + 8192], sg=emb_q["sg"]); mism += int((dequant(e, g, Qz) != ref[r:r + 8192]).sum())
    res["emb"] = dict(mismatch_elements_during_quant=nbad_el, mismatch_elements_dequant=mism, shape=list(ref.shape)); log("EMB", res["emb"])
    del We, emb_w, ref
    # ---- head: recover codes
    hd = torch.load(QD + "/head.pt"); sg = signs(99999, "head", hid); codes = []; scs = []; unmatched = 0; mism = 0; ngroups = 0
    for r in range(0, hd.shape[0], 4096):
        w = hd[r:r + 4096].cuda(); R = w.shape[0]; x = Qz.rot(w.float(), sg).view(R, hid // g, g); a = x.abs()
        t = a.mean(-1, keepdim=True)
        for _ in range(8):
            hi = a > t; n2 = hi.sum(-1, keepdim=True).clamp(min=1); n1 = (~hi).sum(-1, keepdim=True).clamp(min=1)
            m2 = (a * hi).sum(-1, keepdim=True) / n2; m1 = (a * (~hi)).sum(-1, keepdim=True) / n1; t = (m1 + m2) / 2
        lvl = torch.where(a > t, torch.full_like(a, 1.5), torch.full_like(a, 0.5)); sc0 = ((a * lvl).sum(-1, keepdim=True) / (lvl * lvl).sum(-1, keepdim=True))
        sh = sc0.half(); sb = sh.view(torch.int16); best_sc = sh.clone(); ok = torch.zeros(R, hid // g, 1, dtype=torch.bool, device="cuda"); best_qi = torch.zeros(R, hid // g, g, dtype=torch.uint8, device="cuda")
        wb = w.view(R, hid // g, g)
        for k in (0, 1, -1, 2, -2, 3, -3, 4, -4):
            cs = (sb + k).view(torch.float16); csf = cs.float().clamp(min=1e-8); qi = torch.clamp(torch.round(x / csf + half), 0, L - 1); Q = (qi - half) * csf
            dq = (Q.view(R, hid // g, g) @ Qz.Hb).reshape(R, hid)  # unrot
            dq = (dq * sg).to(torch.bfloat16).view(R, hid // g, g); m = (dq == wb).all(-1, keepdim=True) & ~ok
            best_sc = torch.where(m, cs, best_sc); best_qi = torch.where(m, qi.to(torch.uint8), best_qi); ok |= m
            if bool(ok.all()): break
        # groups still not matched: keep k=0 estimate
        miss = ~ok; unmatched += int(miss.sum()); ngroups += miss.numel()
        qi0 = torch.clamp(torch.round(x / sh.float().clamp(min=1e-8) + half), 0, L - 1).to(torch.uint8); best_qi = torch.where(miss, qi0, best_qi)
        codes.append(pack4(best_qi.view(R, hid)).cpu()); scs.append(best_sc.view(R, hid // g).cpu())
        if (r // 4096) % 5 == 0: log("head rows", r, "unmatched groups so far", unmatched, "of", ngroups)
    head_q = dict(L=L, codes=torch.cat(codes), scales=torch.cat(scs), sg=sg.to(torch.int8).cpu()); torch.save(head_q, QD + "/q/head_q.pt")
    for r in range(0, head_q["codes"].shape[0], 4096):
        e = dict(L=L, codes=head_q["codes"][r:r + 4096], scales=head_q["scales"][r:r + 4096], sg=head_q["sg"]); mism += int((dequant(e, g, Qz) != hd[r:r + 4096].cuda()).sum())
    res["head"] = dict(unmatched_groups=unmatched, groups=ngroups, mismatch_elements_dequant=mism, shape=list(hd.shape)); log("HEAD", res["head"])
    vol.commit(); print("RESULT_JSON", json.dumps(res)); return res


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "launch":
        what = sys.argv[2]; nm = sys.argv[3] if len(sys.argv) > 3 else "t096"
        extra = json.loads(sys.argv[4]) if len(sys.argv) > 4 else {}
        if what == "quant": c = modal.Function.from_name("l3-405b", "quant").spawn({**dict(name=nm, target_ratio=0.096, g=128, emb_L=4, head_L=4), **extra})
        elif what == "eval_base": c = modal.Function.from_name("l3-405b", "evalrun").spawn(dict(kind="base", **extra))
        elif what == "dump_eh": c = modal.Function.from_name("l3-405b", "dump_eh1").spawn(dict(name=nm))
        elif what == "inj_base": c = modal.Function.from_name("l3-405b", "evalinj2").spawn(dict(kind="base", **extra))
        elif what == "inj_quant": c = modal.Function.from_name("l3-405b", "evalinj2").spawn(dict(kind="quant", name=nm, **extra))
        else: c = modal.Function.from_name("l3-405b", "evalrun").spawn(dict(kind="quant", name=nm, **extra))
        print("spawned", what, nm, c.object_id)
