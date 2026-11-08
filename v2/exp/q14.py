"""X03 quantiser engine (14B ladder on NEWS). Forked from entry32b/q32_local.py (same incoherence rotation + block-LDLQ + shared D=8 k-means VQ); changes:
  * writes a REAL pack (qbody.bin mixed-radix indices, codebooks, tiered/coded embedding+head, other.safetensors, tokenizer/config, decoder code) -> exp/bytes.py
  * byte target solved analytically BEFORE quantising (K for A0; KL plan for A1+); landed within the 9.90 +-0.02 window or the run aborts
  * arm flags (cfg): alloc="kl" (A1), eh="freq" (A2: log-delta rANS scales + frequency tiers), tgt (A3, pending X02), hw="margin" (A4),
    cal="v2" (A5), codebook="tcq" (A6, not admitted)
Calibration comes ONLY from v2/exp/splits/cal.jsonl (AG buckets [20,50)) plus generic wikitext-103 TRAIN text. No gradients, no weight training.
"""
import json, sys, os, time, math, re, zlib, gc, collections, hashlib, struct, shutil
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np, torch, torch.nn as nn
import packio as P
V2 = os.path.dirname(HERE); sys.path.insert(0, V2 + "/quality"); sys.path.insert(0, HERE + "/quality")
from packio import Rot
K_GRID = sorted(set(int(round(1024 * 2 ** (i / 4))) for i in range(17)))
PROJ = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]
GROUPS = [("q_proj", ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"], "self_attn.q_proj"), ("o_proj", ["self_attn.o_proj"], "self_attn.o_proj"),
          ("gate_proj", ["mlp.gate_proj", "mlp.up_proj"], "mlp.gate_proj"), ("down_proj", ["mlp.down_proj"], "mlp.down_proj")]
D = 8

def f32(x): return float(np.float32(x))
def hexf(x): return struct.pack(">f", x).hex()

# ---------------------------------------------------------------- byte model (exact for the body; verified against the written pack by bytes.py)
def body_bytes(shapes, K_of):
    return sum(P.packed_nbytes(od * cols // D, K_of[n]) for n, (od, cols) in shapes.items())
def codebook_bytes(K_of): return sum(k * D * 2 for k in set(K_of.values()))
def write_body(path, shapes, K_of, blobs, sigmas, seed):
    off = 0; ent = []
    for n in shapes:
        ent.append([n, K_of[n], hexf(sigmas[n]), off, len(blobs[n]), shapes[n][0], shapes[n][1]]); off += len(blobs[n])
    cbo = {}; o = 0
    for k in sorted(set(K_of.values())): cbo[str(k)] = [o, k * D * 2]; o += k * D * 2
    hdr = json.dumps(dict(seed=seed, D=D, codebooks=cbo, tensors=ent), separators=(",", ":"))
    return hdr, ent
def header_size_estimate(shapes, K_of, seed):
    ent = []; off = 0
    for n in shapes:
        l = P.packed_nbytes(shapes[n][0] * shapes[n][1] // D, K_of[n]); ent.append([n, K_of[n], "00000000", off, l, shapes[n][0], shapes[n][1]]); off += l
    cbo = {}; o = 0
    for k in sorted(set(K_of.values())): cbo[str(k)] = [o, k * D * 2]; o += k * D * 2
    return len(json.dumps(dict(seed=seed, D=D, codebooks=cbo, tensors=ent), separators=(",", ":")))

def solve_uniform(shapes, fixed, budget):
    """largest uniform K whose total fits the budget; then the leftover is spent by moving a pseudo-randomly spread subset of tensors to the next 1-bit-per-group K level
    (index cost is quantised in 1/8 bit per index, i.e. ~0.08 pp per uniform step; this keeps A0 'uniform to within one bit' and lands on the byte target)."""
    lo, hi = 1024, 16384
    def tot(Ko): return fixed + body_bytes(shapes, Ko) + codebook_bytes(Ko) + header_size_estimate(shapes, Ko, 0)
    U = lambda K: {n: K for n in shapes}
    if tot(U(lo)) > budget: raise RuntimeError("even K=1024 exceeds budget")
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if tot(U(mid)) <= budget: lo = mid
        else: hi = mid - 1
    Ko = U(lo); B0 = P.group_bits(lo); K1 = int(2 ** ((B0 + 1) / 8))
    while P.group_bits(K1) > B0 + 1: K1 -= 1
    order = sorted(shapes, key=lambda n: (zlib.crc32(n.encode()) , n))
    for n in order:
        trial = dict(Ko); trial[n] = K1
        if tot(trial) <= budget: Ko = trial
    return Ko, tot(Ko)
def solve_kl(shapes, fixed, budget, sens, a=2.5):
    import alloc as A
    def mk(tb):
        pl, _ = A.plan_from_sens(sens, tb, a, K_GRID); Ko = {}
        for n in shapes:
            li, nm = n.split(".", 1); li = int(li)
            g = [gn for gn, names, _ in GROUPS if nm in names][0]; Ko[n] = pl[f"L{li}.{g}"]
        return Ko
    def tot(Ko): return fixed + body_bytes(shapes, Ko) + codebook_bytes(Ko) + header_size_estimate(shapes, Ko, 0)
    lo, hi = 10 / 8, 14 / 8; best = None
    for _ in range(50):
        mid = (lo + hi) / 2; Ko = mk(mid); t = tot(Ko)
        if t <= budget: best = (Ko, t); lo = mid
        else: hi = mid
    return best

def _impl(cfg):
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from safetensors.torch import save_file
    torch.backends.cuda.matmul.allow_tf32 = False
    t0 = time.time(); dev = "cuda"; name = cfg["name"]; M = cfg["model"]; seed = int(cfg.get("seed", 0)); SEEDP = (str(seed) + ":").encode() if seed else b""
    work = cfg.get("work", "/workspace"); pack = cfg.get("pack_dir") or f"{work}/packs/{name}_s{seed}"; os.makedirs(pack, exist_ok=True); rd = cfg.get("results_dir", f"{HERE}/results/X03"); os.makedirs(rd, exist_ok=True)
    if cfg.get("tgt"): raise SystemExit("A3 (GPTAQ) is PENDING: X02's fix failed the non-harm criterion (v2/quality/gptaq_fix.md); not run in X03")
    if cfg.get("codebook") == "tcq": raise SystemExit("A6 (trellis/Leech) is not admitted by the VQ lane evidence (see ladder_table.md); not run")
    tok = AutoTokenizer.from_pretrained(M)
    model = AutoModelForCausalLM.from_pretrained(M, torch_dtype=torch.bfloat16, device_map="cuda").eval(); model.config.use_cache = False
    V = model.config.vocab_size; hd = model.config.hidden_size; tied = model.config.tie_word_embeddings; nL = model.config.num_hidden_layers
    base_params = sum(p.numel() for p in model.parameters()); fp16_bytes = 2 * base_params
    layers = model.model.layers; res = dict(name=name, cfg=cfg, base_params=base_params)
    sd = {k: v for k, v in model.state_dict().items()}
    shapes = {}
    for li in range(nL):
        for n in PROJ: w = layers[li].get_submodule(n).weight; shapes[f"{li}.{n}"] = (w.shape[0], w.shape[1])
    nbody = sum(a * b for a, b in shapes.values())

    # ---------------- calibration (CAL only) -------------------------------------------------------------
    cal = [json.loads(l) for l in open(HERE + "/splits/cal.jsonl")]
    by = collections.defaultdict(list)
    for c in sorted(cal, key=lambda c: hashlib.sha1(c["id"].encode()).hexdigest()): by[c["answer"]].append(c)
    n_each = cfg.get("n_task", 1500) // 4
    pool_n = n_each if cfg.get("cal", "v1") == "v1" else 2 * n_each
    pool = [c for k in sorted(by) for c in by[k][:pool_n]]
    def chat(u): return tok.apply_chat_template([{"role": "user", "content": u}], tokenize=False, add_generation_prompt=True)
    if cfg.get("cal", "v1") == "v2":      # class-balanced, self-distilled: answers = the BASE model's own greedy generation (fp weights); balanced on the GENERATED label; not shipped
        tok.padding_side = "left"; outs = []
        for b0 in range(0, len(pool), 8):
            enc = tok([chat(c["user"]) for c in pool[b0:b0 + 8]], return_tensors="pt", padding=True, truncation=True, max_length=1500).to(dev)
            with torch.no_grad(): g = model.generate(**enc, max_new_tokens=4, do_sample=False, use_cache=True)
            outs += [tok.decode(x[enc["input_ids"].shape[1]:], skip_special_tokens=True).strip() for x in g]
        gl = collections.defaultdict(list)
        for c, o in zip(pool, outs):
            if o in ("World", "Sports", "Business", "Technology"): gl[o].append(dict(c, answer=o))
        m_each = min(n_each, min(len(v) for v in gl.values())); items = [c for k in sorted(gl) for c in gl[k][:m_each]]
        print("selfdist classes", {k: len(v) for k, v in gl.items()}, "per class", m_each, flush=True); res["cal_v2_counts"] = {k: len(v) for k, v in gl.items()}
    else: items = [c for k in sorted(by) for c in by[k][:n_each]]
    AW = float(cfg.get("aw", 24.0)); seqs = []
    for c in items:
        p_ = tok.encode(chat(c["user"]), add_special_tokens=False)[:1900]; a_ = tok.encode(c["answer"] + "<|im_end|>\n", add_special_tokens=False)
        seqs.append((p_ + a_, [0] * (len(p_) - 1) + [1] * (len(a_) + 1)))       # second list: 1 where the position predicts an answer token
    wiki_tok = cfg.get("wiki_tokens", 400000); wf = f"{work}/wiki_{wiki_tok}.txt"
    if not os.path.exists(wf):
        from datasets import load_dataset
        wtr = load_dataset("wikitext", "wikitext-103-raw-v1", split="train", streaming=True); acc = 0; parts = []
        for r in wtr:
            t_ = r["text"].strip()
            if len(t_) < 400: continue
            parts.append(t_.replace("\n", " ")); acc += len(t_) // 5
            if acc >= wiki_tok: break
        open(wf, "w").write("\n".join(parts))
    wiki = open(wf).read().split("\n")
    for t_ in wiki: e_ = tok.encode(t_, add_special_tokens=False)[:2048]; seqs.append((e_, [0] * len(e_)))
    rng = np.random.RandomState(0); perm = rng.permutation(len(seqs)); ncal = cfg.get("ncal", 1024); Lc = cfg.get("callen", 512); stream = []; ans = []
    for pi in perm: stream += seqs[pi][0]; ans += seqs[pi][1]
    nev = cfg.get("nsens_eval", 24) if cfg.get("mode") == "sens" else 0
    n_ = min(len(stream) // Lc, ncal + nev); stream = torch.tensor(stream[:n_ * Lc]).view(-1, Lc); ANS = torch.tensor(ans[:n_ * Lc]).view(-1, Lc).float()
    ncal = n_ - nev; ev_ids = stream[ncal:] if nev else None; stream = stream[:ncal]; ANS = ANS[:ncal]; print("calib seqs", tuple(stream.shape), flush=True)
    # token weights (A4): w = 1 on non-answer positions; answer positions: AW / (1 + base margin), margin = top1-top2 logit of the BASE at that position (forward only)
    WTS = torch.ones_like(ANS)
    if cfg.get("hw") == "margin":
        with torch.no_grad():
            for a in range(0, len(stream), 2):
                lg = model(input_ids=stream[a:a + 2].cuda()).logits.float(); t2 = lg.topk(2, -1).values; mg = (t2[..., 0] - t2[..., 1]).cpu()
                am = ANS[a:a + 2]; WTS[a:a + 2] = torch.where(am > 0, AW / (1.0 + mg), torch.ones_like(mg))
        print("margin weights: answer positions mean %.3f, n_ans_pos %d" % (float(WTS[ANS > 0].mean()), int((ANS > 0).sum())), flush=True)
    # ---------------- quantiser pieces --------------------------------------------------------------------
    cbs = {}
    def get_cb(K):
        if K in cbs: return cbs[K]
        f = f"{work}/cb_D{D}_K{K}.pt"
        if os.path.exists(f): cbs[K] = torch.load(f).to(dev); return cbs[K]
        g = torch.Generator(device=dev).manual_seed(K); N = max(K * 100, 200000); X = torch.randn(N, D, generator=g, device=dev); C = X[:K].clone()
        for it in range(30):
            sums = torch.zeros(K, D, device=dev); cnt = torch.zeros(K, device=dev); Cn = 0.5 * (C ** 2).sum(1)
            for a in range(0, N, 16384):
                xb = X[a:a + 16384]; ii = (xb @ C.T - Cn).argmax(1); sums.index_add_(0, ii, xb); cnt.index_add_(0, ii, torch.ones(len(ii), device=dev))
            e = cnt == 0; newC = sums / cnt.clamp(min=1).unsqueeze(1); newC[e] = X[torch.randint(0, N, (int(e.sum()),), device=dev, generator=g)]; C = newC
        C = C.half().float(); torch.save(C.cpu(), f); cbs[K] = C; return C       # codebooks are shipped in fp16: quantise against the fp16-rounded values
    def nn_idx(X, C, Cn):
        out = torch.empty(X.shape[0], dtype=torch.int64, device=dev); torch.backends.cuda.matmul.allow_tf32 = True
        for a in range(0, X.shape[0], 4096): out[a:a + 4096] = (X[a:a + 4096] @ C.T - Cn).argmax(1)
        torch.backends.cuda.matmul.allow_tf32 = False; return out
    def ldlq(Wn, H, C, chunk=128, damp=0.01):
        R, Cc = Wn.shape; Cn = 0.5 * (C ** 2).sum(1); H = (H + H.T) / 2; dg = torch.diag(H); dead = dg == 0; H[dead, dead] = 1; U = None
        for dtp, devc in [(torch.float32, dev), (torch.float64, "cpu")]:
            Hc = H.to(devc, dtp); md = float(torch.mean(torch.diag(Hc))); prev = 0.0
            for dm in [damp, damp * 10, damp * 100, 1.0]:
                Hc.diagonal().add_((dm - prev) * md); prev = dm
                try:
                    L_ = torch.linalg.cholesky(Hc); Hi = torch.cholesky_inverse(L_); del L_; U = torch.linalg.cholesky(Hi, upper=True).float().to(dev); del Hi; break
                except Exception as ex: print("chol retry", dtp, dm, str(ex)[:60], flush=True); torch.cuda.empty_cache()
            del Hc
            if U is not None: break
        del H; torch.cuda.empty_cache()
        W = Wn.clone(); W[:, dead] = 0; idx = torch.empty(R, Cc // D, dtype=torch.int64, device=dev)
        for c0 in range(0, Cc, chunk):
            c1 = min(c0 + chunk, Cc); Wc = W[:, c0:c1].clone(); Err = torch.zeros_like(Wc)
            for j in range(0, c1 - c0, D):
                blk = Wc[:, j:j + D]; i = nn_idx(blk, C, Cn); E = blk - C[i]; Ub = U[c0 + j:c0 + j + D, c0 + j:c0 + j + D]
                Es = torch.linalg.solve_triangular(Ub, E, upper=True, left=False); Err[:, j:j + D] = Es
                if j + D < c1 - c0: Wc[:, j + D:] -= Es @ U[c0 + j:c0 + j + D, c0 + j + D:c1]
                idx[:, (c0 + j) // D] = i
            if c1 < Cc: W[:, c1:] -= Err @ U[c0:c1, c1:]
        return idx
    decode = P.decode_tensor
    # ---------------- capture layer-0 inputs
    ins = []; kw = {}
    class Catcher(nn.Module):
        def __init__(s, m): super().__init__(); s.m = m; s.attention_type = getattr(m, "attention_type", "full_attention")
        def forward(s, x, *a, **k): ins.append(x.cpu()); kw.update(k); raise ValueError
    layers[0] = Catcher(layers[0])
    for i in range(len(stream)):
        try: model(stream[i:i + 1].cuda())
        except ValueError: pass
    layers[0] = layers[0].m; cur = ins; ins = None
    kw = {k: v for k, v in kw.items() if k not in ("past_key_values", "past_key_value", "use_cache")}
    if cfg.get("mode") == "sens": return _sens(cfg, model, layers, cur, kw, ev_ids, Rot, ldlq, get_cb, decode, SEEDP, GROUPS, res, rd, t0, dev)
    # ---------------- shipped files common to every arm ------------------------------------------------
    tok.padding_side = "right"; tok.save_pretrained(pack); model.config.save_pretrained(pack)
    for f_ in ("vocab.json", "merges.txt"):      # redundant with tokenizer.json (fast tokenizer loads from it alone): not shipped
        if os.path.exists(f"{pack}/{f_}"): os.remove(f"{pack}/{f_}")
    if model.generation_config is not None: model.generation_config.save_pretrained(pack)
    for f in ("packio.py", "decode_pack.py"): shutil.copy(f"{HERE}/{f}", f"{pack}/{f}")
    # embedding / head (weights only; independent of the body) -> exact bytes first
    t_eh = time.time(); eh = {}
    if cfg.get("eh", "A0") == "freq":
        cnt = np.zeros(V, np.int64)
        for c in cal:
            ids = tok.encode(chat(c["user"]) + c["answer"], add_special_tokens=False); cnt += np.bincount(ids, minlength=V)[:V]
        for t_ in wiki: ids = tok.encode(t_, add_special_tokens=False); cnt += np.bincount(ids, minlength=V)[:V]
        order = np.argsort(-cnt, kind="stable"); tiers = np.full(V, 1, np.int64); tiers[cnt > 0] = 2; tiers[order[:cfg.get("eh_top", 8000)]] = 3
        res["eh_tiers"] = {b: int((tiers == b).sum()) for b in (1, 2, 3)}; print("eh tiers", res["eh_tiers"], flush=True)
    else: tiers = np.full(V, 2, np.int64)
    coded = cfg.get("eh", "A0") == "freq"
    for nm_, key in (("emb", "model.embed_tokens.weight"), ("head", "lm_head.weight")):
        if nm_ == "head" and tied: continue
        e = P.quant_embhead(sd[key], "eh." + nm_, tiers, coded, dev); P.save_embhead(e, f"{pack}/{nm_}"); eh[nm_] = P.embhead_nbytes(e); del e
    print("embhead done", eh, round(time.time() - t_eh), "s", flush=True)
    qnames = {f"model.layers.{li}.{n}.weight" for li in range(nL) for n in PROJ}
    other = {k: v.detach().to(torch.bfloat16).cpu().contiguous() for k, v in sd.items() if k not in qnames and k not in ("model.embed_tokens.weight", "lm_head.weight")}
    save_file(other, f"{pack}/other.safetensors", metadata={"format": "pt"}); del sd
    fixed = sum(os.path.getsize(os.path.join(pack, f)) for f in os.listdir(pack))
    target = cfg.get("target_pct", 9.90); budget = int(fp16_bytes * target / 100.0) - int(cfg.get("slack_bytes", 2_000_000))
    if cfg.get("alloc") == "kl":
        sens = json.load(open(cfg["alloc_sens"])); K_of, tot = solve_kl(shapes, fixed, budget, sens, cfg.get("alloc_a", 2.5))
    else: K_of, tot = solve_uniform(shapes, fixed, budget)
    print("planned bytes %d (%.4f%%) budget %d; K range %d-%d; distinct K %d" % (tot, 100 * tot / fp16_bytes, budget, min(K_of.values()), max(K_of.values()), len(set(K_of.values()))), flush=True)
    res.update(planned_bytes=tot, planned_ratio_pct=100 * tot / fp16_bytes, K_min=min(K_of.values()), K_max=max(K_of.values()), K_mean_bits=float(np.mean([math.log2(k) for k in K_of.values()])))

    # ---------------- quantisation loop
    blobs = {}; sigmas = {}; st = dict(num=0.0, den=0.0); cw = {"w": None}
    for li, l in enumerate(layers):
        Hs = {}; nsm = {}
        for gname, names, hookn in GROUPS:
            m = l.get_submodule(hookn); Hs[gname] = torch.zeros(m.in_features, m.in_features, device=dev); nsm[gname] = 0.0
        def mk(gname):
            def hook(m, i, o):
                x = i[0].reshape(-1, i[0].shape[-1]).float(); torch.backends.cuda.matmul.allow_tf32 = True
                if cw["w"] is None: nsm[gname] += x.shape[0]; Hs[gname] += (x.T @ x).to(Hs[gname].device)
                else: wv = cw["w"].to(x.device).reshape(-1, 1); nsm[gname] += float(wv.sum()); Hs[gname] += (x.T @ (x * wv)).to(Hs[gname].device)
                torch.backends.cuda.matmul.allow_tf32 = False
            return hook
        hds = [l.get_submodule(hn).register_forward_hook(mk(gn)) for gn, _, hn in GROUPS]
        with torch.no_grad():
            for j, x in enumerate(cur): cw["w"] = WTS[j] if cfg.get("hw") == "margin" else None; l(x.to(dev), **kw)
        for h_ in hds: h_.remove()
        Hs = {k: v.cpu() for k, v in Hs.items()}; torch.cuda.empty_cache()
        for gname, names, hookn in GROUPS:
            Hraw = Hs[gname].to(dev) / nsm[gname]; n_in = Hraw.shape[0]; rin = Rot(n_in, zlib.crc32(SEEDP + f"{li}.{gname}.in".encode()), dev)
            Hr = rin.fwd(Hraw); Hr = rin.fwd(Hr.T.contiguous()).T.contiguous(); del Hraw
            for n in names:
                m = l.get_submodule(n); wdev = m.weight.device; W = m.weight.data.float().to(dev); od = W.shape[0]; rout = Rot(od, zlib.crc32(SEEDP + f"{li}.{n}.out".encode()), dev)
                Wr = rin.fwd(W); Wr = rout.fwd(Wr.T.contiguous()).T.contiguous(); sigma = f32(float(Wr.std()))
                key = f"{li}.{n}"; K = K_of[key]; C = get_cb(K); idx = ldlq(Wr / sigma, Hr, C, damp=cfg.get("damp", 0.01))
                Q = C[idx].reshape(od, -1) * sigma; Eq = Wr - Q; st["num"] += float(((Eq @ Hr) * Eq).sum()); st["den"] += float(((Wr @ Hr) * Wr).sum())
                m.weight.data = decode(idx, C, sigma, rin, rout).to(torch.bfloat16).to(wdev)
                blob = P.pack_idx(idx.reshape(-1), K); assert len(blob) == P.packed_nbytes(idx.numel(), K)
                if li % 6 == 0 or li == nL - 1:
                    assert torch.equal(P.unpack_idx(blob.tobytes(), idx.numel(), K, dev), idx.reshape(-1)), "pack round trip failed"
                blobs[key] = blob.tobytes(); sigmas[key] = sigma
                del W, Wr, Q, Eq, idx
            del Hr
        del Hs
        with torch.no_grad():
            for j, x in enumerate(cur): o = l(x.to(dev), **kw); cur[j] = (o[0] if isinstance(o, tuple) else o).cpu()
        torch.cuda.empty_cache(); print("layer", li, "relerr", st["num"] / st["den"], round(time.time() - t0), flush=True)
    # ---------------- write the pack
    hdr, ent = write_body(None, shapes, K_of, blobs, sigmas, seed)
    with open(f"{pack}/qbody.bin", "wb") as f:
        for n in shapes: f.write(blobs[n])
    open(f"{pack}/qbody.json", "w").write(hdr)
    with open(f"{pack}/codebooks.bin", "wb") as f:
        for k in sorted(set(K_of.values())): f.write(cbs[k].half().cpu().numpy().tobytes())
    import bytes as B
    r = B.account(pack, M, fp16=fp16_bytes, target=target); res.update(bytes_report=r, real_bytes=r["real_bytes"], ratio_pct=r["ratio_pct"], rel_proxy_err=st["num"] / st["den"], quant_secs=time.time() - t0, eh_bytes=eh, nbody=nbody)
    json.dump(res, open(f"{rd}/{name}_s{seed}_quant.json", "w"), indent=1, default=str)
    print("PACK", json.dumps(dict(real_bytes=r["real_bytes"], ratio_pct=r["ratio_pct"], planned=tot)), flush=True)
    assert abs(r["ratio_pct"] - target) <= 0.02 + 1e-9 or (target > 9.95 and r["ratio_pct"] <= target), f"byte target missed: {r['ratio_pct']:.4f}% (re-run with adjusted slack)"
    open(f"{pack}/.QUANT_DONE", "w").write("ok")
    return res

def _sens(cfg, model, layers, cur, kw, ev_ids, Rot, ldlq, get_cb, decode, SEEDP, groups, res, rd, t0, dev):
    """KL-sensitivity of each (layer, group): quantise ONLY that group at sens_K, measure KL(base || quantised) on held-out CAL sequences (forward only)."""
    sK = cfg.get("sens_K", 8192); C = get_cb(sK); sres = {}
    def fwd_logits():
        outs = []
        with torch.no_grad():
            for a in range(0, len(ev_ids), 3): outs.append(model(input_ids=ev_ids[a:a + 3].cuda()).logits.float().log_softmax(-1).cpu())
        return torch.cat(outs)
    lp0 = fwd_logits().half().cpu()
    def klnow():
        lq = fwd_logits(); tot = 0.0
        for a in range(0, len(lq), 3):
            r_ = lp0[a:a + 3].cuda().float(); tot += float((r_.exp() * (r_ - lq[a:a + 3].cuda())).sum())
        return tot / (lq.shape[0] * lq.shape[1])
    res["sens_kl_offset"] = klnow()
    for li, l in enumerate(layers):
        Hs = {}; nsm = {}
        for gname, names, hookn in groups:
            m = l.get_submodule(hookn); Hs[gname] = torch.zeros(m.in_features, m.in_features, device=dev); nsm[gname] = 0
        def mk(gname):
            def hook(m, i, o):
                x = i[0].reshape(-1, i[0].shape[-1]).float(); nsm[gname] += x.shape[0]; torch.backends.cuda.matmul.allow_tf32 = True; Hs[gname] += (x.T @ x).to(Hs[gname].device); torch.backends.cuda.matmul.allow_tf32 = False
            return hook
        hds = [l.get_submodule(hn).register_forward_hook(mk(gn)) for gn, _, hn in groups]
        with torch.no_grad():
            for j, x in enumerate(cur): o = l(x.to(dev), **kw); cur[j] = (o[0] if isinstance(o, tuple) else o).cpu()
        for h_ in hds: h_.remove()
        for gname, names, hookn in groups:
            Hraw = Hs[gname].to(dev) / nsm[gname]; n_in = Hraw.shape[0]; rin = Rot(n_in, zlib.crc32(SEEDP + f"{li}.{gname}.in".encode()), dev)
            Hr = rin.fwd(Hraw); Hr = rin.fwd(Hr.T.contiguous()).T.contiguous(); del Hraw; olds = {}
            for n in names:
                m = l.get_submodule(n); wdev = m.weight.device; olds[n] = m.weight.data.clone(); W = m.weight.data.float().to(dev); od = W.shape[0]; rout = Rot(od, zlib.crc32(SEEDP + f"{li}.{n}.out".encode()), dev)
                Wr = rin.fwd(W); Wr = rout.fwd(Wr.T.contiguous()).T.contiguous(); sigma = f32(float(Wr.std())); idx = ldlq(Wr / sigma, Hr, C)
                m.weight.data = decode(idx, C, sigma, rin, rout).to(torch.bfloat16).to(wdev)
            k_ = klnow(); sres[f"L{li}.{gname}"] = dict(kl=k_, n=sum(int(l.get_submodule(n).weight.numel()) for n in names))
            for n in names: l.get_submodule(n).weight.data = olds[n]
            del Hr
        del Hs; torch.cuda.empty_cache(); print("sens layer", li, {k: round(v["kl"], 5) for k, v in sres.items() if k.startswith(f"L{li}.")}, round(time.time() - t0), flush=True)
    res["sens"] = sres; res["sens_K"] = sK; res["cfg"] = {k: v for k, v in cfg.items()}
    json.dump(res, open(f"{rd}/sens_14b.json", "w")); return res

if __name__ == "__main__":
    cfgs = json.load(open(sys.argv[1])); cfg = [c for c in cfgs if c["name"] == sys.argv[2]][0] if isinstance(cfgs, list) else cfgs
    _impl(cfg)
