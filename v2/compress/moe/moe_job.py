"""Track C-MOE: Qwen3-30B-A3B-Instruct-2507 expert pruning (k of 128 by router stats) + Hadamard/GPTQ entropy-coded quant (simulated fake-quant,
bytes = histogram entropy + fp16 scales; no training). Fidelity vs bf16 base: (a) exact match of first 64 greedy tokens, (b) teacher-forced top-1 agreement, (c) mean KL.
modal run --detach moe_job.py::main --cfgs cfgs.json   (cfgs: list of config dicts; one GPU job per 'group' key)
"""
import modal, json, os
app = modal.App("tenpct-moe")
img = (modal.Image.debian_slim().pip_install("torch==2.6.0", "transformers==4.57.1", "numpy", "accelerate", "safetensors", "huggingface_hub", "constriction")
       .add_local_file("pack_bytes.py", "/root/pack_bytes.py"))
vol = modal.Volume.from_name("tenpct-v2-hf", create_if_missing=True)
MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507"

@app.function(image=img, volumes={"/hf": vol}, timeout=7200, cpu=8)
def download():
    os.environ["HF_HOME"] = "/hf"
    from huggingface_hub import snapshot_download
    p = snapshot_download(MODEL, allow_patterns=["*.json", "*.safetensors", "*.txt"]); vol.commit(); print(p)
    tot = 0
    for r, _, fs in os.walk(p):
        for f in fs: tot += os.path.getsize(os.path.join(r, f))
    print("bytes", tot)

@app.function(image=img, gpu=["H100", "A100-80GB"], timeout=4 * 3600, memory=110000, volumes={"/hf": vol}, max_containers=20)
def run(cfgs, tag="x"):
    import sys, time, math, hashlib, numpy as np, torch, torch.nn as nn
    os.environ["HF_HOME"] = "/hf"
    from transformers import AutoTokenizer, AutoModelForCausalLM
    t0 = time.time(); D = "/hf/moe_data"; torch.backends.cuda.matmul.allow_tf32 = False
    tok = AutoTokenizer.from_pretrained(MODEL); tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map={"": 0}).eval()
    model.config.use_cache = False
    cfg0 = model.config; NE, TK, NL = cfg0.num_experts, cfg0.num_experts_per_tok, cfg0.num_hidden_layers
    layers = model.model.layers
    print("loaded", round(time.time() - t0), torch.cuda.memory_allocated() / 1e9, flush=True)
    rd = lambda n: [json.loads(x) for x in open(f"{D}/{n}.jsonl")]
    EV = {"A": rd("evalA"), "B": rd("evalB")}; CAL = {"A": rd("calibA"), "B": rd("calibB")}
    def prompt(r): return tok.apply_chat_template(r["msgs"], add_generation_prompt=True, tokenize=False)
    def fulltext(r): return prompt(r) + (r["gold"] or "") + "<|im_end|>"
    # ---------------- patched MoE forward (stats / hessian collection / pruning)
    S = dict(stats=None, H=None, hl=-1, valid=None); keepm = [None] * NL
    def mk_fwd(li):
        def fwd(self, x):
            B, T, Dm = x.shape; xf = x.reshape(-1, Dm); logits = self.gate(xf).float()
            if keepm[li] is not None: logits = logits.masked_fill(~keepm[li].to(logits.device), float("-inf"))
            w = torch.softmax(logits, -1); tw, ti = torch.topk(w, TK, -1); tw = tw / tw.sum(-1, keepdim=True)
            out = torch.zeros_like(xf); flat = ti.reshape(-1); order = torch.argsort(flat); cnts = torch.bincount(flat, minlength=NE).tolist()
            tokidx = order // TK; ws = tw.reshape(-1)[order]; off = 0
            colH = S["H"] is not None and S["hl"] == li; st = S["stats"]
            for e, c in enumerate(cnts):
                if c == 0: continue
                ids = tokidx[off:off + c]; g = ws[off:off + c]; off += c; ex = self.experts[e]; xin = xf[ids]
                if colH:
                    inter = ex.act_fn(ex.gate_proj(xin)) * ex.up_proj(xin); h = ex.down_proj(inter)
                    Xe = xin.float(); Ie = inter.float()
                    S["H"]["gu"][e] += Xe.T @ Xe; S["H"]["d"][e] += Ie.T @ Ie; S["H"]["n"][e] += c
                else: h = ex(xin)
                if st is not None:
                    st["cnt"][li, e] += c; st["gw"][li, e] += g.float().sum(); st["reap"][li, e] += (g.float() * h.float().norm(dim=-1)).sum()
                out.index_add_(0, ids, h * g[:, None].to(h.dtype))
            return out.view(B, T, Dm), logits
        return fwd
    for li, l in enumerate(layers): l.mlp.forward = mk_fwd(li).__get__(l.mlp)
    # ---------------- calibration stream
    def stream(setname, nseq, L=512):
        rows = list(CAL[setname]); rng = np.random.RandomState(0); rng.shuffle(rows); ids = []
        for r in rows: ids += tok.encode(fulltext(r))
        need = nseq * L
        while len(ids) < need: ids += ids   # tiny sets only
        return torch.tensor(ids[:need]).view(nseq, L)
    # ---------------- generation / TF eval
    @torch.no_grad()
    def gen(rows, bs=40, maxnew=96):
        enc = [tok.encode(prompt(r))[-700:] for r in rows]; order = np.argsort([-len(e) for e in enc]); outs = [None] * len(rows)
        model.config.use_cache = True
        for b in range(0, len(rows), bs):
            idx = order[b:b + bs]; mx = max(len(enc[i]) for i in idx)
            ids = torch.full((len(idx), mx), tok.pad_token_id, dtype=torch.long); att = torch.zeros(len(idx), mx, dtype=torch.long)
            for j, i in enumerate(idx): ids[j, mx - len(enc[i]):] = torch.tensor(enc[i]); att[j, mx - len(enc[i]):] = 1
            o = model.generate(input_ids=ids.cuda(), attention_mask=att.cuda(), max_new_tokens=maxnew, do_sample=False, temperature=None, top_p=None, top_k=None, pad_token_id=tok.pad_token_id)
            for j, i in enumerate(idx):
                t = o[j, mx:].tolist()
                if tok.eos_token_id in t: t = t[:t.index(tok.eos_token_id) + 1]
                outs[i] = t
        model.config.use_cache = False; return outs
    def tf_items(rows, n, maxlen=384):
        it = []
        for r in rows:
            if not r["gold"]: continue
            p = tok.encode(prompt(r))[-250:]; a = tok.encode(r["gold"] + "<|im_end|>")
            ids = (p + a)[:maxlen]; it.append((ids, len(p)))
            if len(it) >= n: break
        return it
    TFA = tf_items(EV["A"][::1], 140) ; TFB = tf_items(EV["B"], 40)
    TOPK = 100
    @torch.no_grad()
    def tf_logits(items):
        res = []
        model.config.use_cache = False
        for ids, pl in items:
            lg = model(input_ids=torch.tensor([ids]).cuda()).logits[0, pl - 1:-1].float()
            lp = torch.log_softmax(lg, -1); v, ix = lp.topk(TOPK, -1); res.append((v.cpu(), ix.cpu()))
        return res
    def tf_compare(ref, cur, items):
        agree = n = 0; kl = 0.0
        for (rv, ri), (cv, ci), (ids, pl) in zip(ref, cur, items):
            n += rv.shape[0]; agree += int(ri[:, 0].eq(ci[:, 0]).sum())
            # KL(base||cand) over base top-K + tail; cand logprob at base's top-K tokens needs full cand dist: approximate by cand top-K lookup (missing -> floor)
            p = rv.exp(); ptail = (1 - p.sum(-1)).clamp_min(1e-9)
            q = torch.full_like(rv, -30.0)
            m = (ri.unsqueeze(-1) == ci.unsqueeze(1)); has = m.any(-1)
            cvx = (m.float() * cv.unsqueeze(1)).sum(-1); q = torch.where(has, cvx, torch.full_like(rv, float(cv[:, -1].min()) - 1.0))
            qtail = (1 - q.exp().sum(-1)).clamp_min(1e-9)
            kl += float(((p * (rv - q)).sum(-1) + ptail * (ptail.log() - qtail.log())).clamp_min(0).sum())
        return agree / n, kl / n, n
    import re
    def subcat(r):
        u = r["msgs"][-1]["content"]
        if r["cat"] == "sumcls":
            return "agnews" if u.startswith("Classify") else "sst2" if u.startswith("What is the sentiment") else "cnn_sum"
        return r["cat"]
    def calls(txt):
        out = []
        for m in re.findall(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", txt, re.S):
            try:
                j = json.loads(m); out.append(json.dumps({"n": j.get("name"), "a": j.get("arguments")}, sort_keys=True))
            except Exception:
                try:
                    import ast; j = ast.literal_eval(m); out.append(json.dumps({"n": j.get("name"), "a": j.get("arguments")}, sort_keys=True, default=str))
                except Exception: out.append("BAD")
        return sorted(out)
    def item_ok(r, t):
        txt = tok.decode(t, skip_special_tokens=True).strip(); sc = subcat(r)
        if sc in ("agnews", "sst2"): return int(txt.lower().strip(" .") == r["gold"].lower())
        if sc == "json":
            g = calls(r["gold"]); return int(len(g) > 0 and calls(txt) == g and "BAD" not in g)
        return None
    def gen_cmp(ref, cur):
        em = [int(a == b) for a, b in zip(ref, cur)]; pm = []
        for a, b in zip(ref, cur):
            k = 0
            while k < min(len(a), len(b)) and a[k] == b[k]: k += 1
            pm.append(k / max(len(a), 1) if a != b else 1.0)
        return em, pm
    def evaluate(refs, tag_):
        out = {}
        for s in ("A", "B"):
            if cfg.get("quick") and s == "B": continue
            rows = EV[s][:64] if cfg.get("quick") else EV[s]; g = gen(rows); R_ = refs["gen"][s][:len(rows)]
            for n in (8, 16, 64):
                e = [int(a[:n] == b[:n]) for a, b in zip(R_, g)]
                out[f"em{n}_{s}"] = float(np.mean(e))
                out[f"em{n}_{s}_bycat"] = {c: float(np.mean([x for x, r in zip(e, rows) if subcat(r) == c])) for c in sorted(set(subcat(r) for r in rows))}
            em_full, pm = gen_cmp(R_, g); out[f"emfull_{s}"] = float(np.mean(em_full)); out[f"prefix_{s}"] = float(np.mean(pm))
            # item-level retention of base-correct items (gold-labelled subsets)
            ret = {}
            for c in ("agnews", "sst2", "json"):
                idx = [i for i, r in enumerate(rows) if subcat(r) == c]
                if not idx: continue
                bok = [item_ok(rows[i], R_[i]) for i in idx]; cok = [item_ok(rows[i], g[i]) for i in idx]
                nb = sum(bok); kept = sum(1 for x, y in zip(bok, cok) if x and y)
                ret[c] = dict(n=len(idx), base_correct=nb, kept=kept, retention=(kept / nb if nb else None), cand_acc=sum(cok) / len(idx), base_acc=nb / len(idx))
            out[f"retention_{s}"] = ret
            # stage-0 (successive halving) numbers: first 64 prompts, 16-token match
            out[f"stage0_em16_{s}"] = float(np.mean([int(a[:16] == b[:16]) for a, b in zip(R_[:64], g[:64])]))
            out[f"_gen_{s}"] = g
        for s, items in (("A", TFA), ("B", TFB)):
            if cfg.get("quick") and s == "B": continue
            cur = tf_logits(items); a, k, n = tf_compare(refs["tf"][s], cur, items)
            out[f"top1_{s}"] = a; out[f"kl_{s}"] = k; out[f"tf_n_{s}"] = n
        return out
    # ---------------- reference (base) outputs + stats
    refp = f"{D}/base_ref.pt"
    if not os.path.exists(refp) and cfgs[0].get("group") != "smoke":
        for _ in range(360):   # wait for the smoke job to create the shared base reference
            vol.reload()
            if os.path.exists(refp): time.sleep(20); break
            time.sleep(30)
    if os.path.exists(refp): refs = torch.load(refp)
    else:
        refs = dict(gen={}, tf={})
        for s in ("A", "B"): refs["gen"][s] = gen(EV[s])
        refs["tf"]["A"] = tf_logits(TFA); refs["tf"]["B"] = tf_logits(TFB)
        # noise floor: base with different batch composition
        nf = {}
        for s in ("A", "B"):
            g2 = gen(EV[s], bs=13); em, pm = gen_cmp(refs["gen"][s], g2); nf[s] = dict(em_full=float(np.mean(em)), em64=float(np.mean([int(a[:64]==b[:64]) for a,b in zip(refs['gen'][s],g2)])), em8=float(np.mean([int(a[:8]==b[:8]) for a,b in zip(refs['gen'][s],g2)])), prefix=float(np.mean(pm)))
        refs["noise_floor"] = nf; torch.save(refs, refp); json.dump({s_: [tok.decode(t) for t in refs['gen'][s_]] for s_ in ('A','B')}, open(f'{D}/base_outputs.json','w')); vol.commit(); print("NOISE_FLOOR", nf, flush=True)
        json.dump(dict(noise_floor=nf), open(f"{D}/base_noise.json", "w"))
    def get_stats(setname):
        p = f"{D}/stats_{setname}.pt"
        vol.reload()
        if os.path.exists(p): return torch.load(p)
        S["stats"] = dict(cnt=torch.zeros(NL, NE), gw=torch.zeros(NL, NE), reap=torch.zeros(NL, NE)); x = stream(setname, 128, 512)
        with torch.no_grad():
            for i in range(0, len(x), 8): model(input_ids=x[i:i + 8].cuda())
        st = S["stats"]; S["stats"] = None; torch.save(st, p); vol.commit(); return st
    # ---------------- quantization primitives
    from pack_bytes import entropy_bits
    HB = 128
    def hada(n):
        Hm = torch.ones(1, 1)
        while Hm.shape[0] < n: Hm = torch.cat([torch.cat([Hm, Hm], 1), torch.cat([Hm, -Hm], 1)], 0)
        return (Hm / math.sqrt(n)).cuda()
    Hb = hada(HB)
    def rot_cols(M, sg):   # M[...,R,C] @ Q ; Q=diag(sg) blockdiag(Hb)
        *lead, R, C = M.shape; return ((M * sg).reshape(*lead, R, C // HB, HB) @ Hb).reshape(*lead, R, C)
    def unrot_cols(M, sg):
        *lead, R, C = M.shape; return (M.reshape(*lead, R, C // HB, HB) @ Hb).reshape(*lead, R, C) * sg
    def rot_H(Hm, sg):   # Q^T H Q  with Q = diag(sg) blockdiag(Hb)  -> consistent with W@Q
        a = rot_cols(Hm, sg)                       # H Q (rows untouched; Q acts on columns)
        return rot_cols(a.transpose(-1, -2).contiguous(), sg).transpose(-1, -2).contiguous()
    LM = 31
    def ent_of(q):   # q int tensor -> entropy bits total
        h = torch.bincount((q.long() + LM).flatten(), minlength=2 * LM + 1).float(); return h
    def quant_rtn(W, c, g):   # W[...,R,C] ; groups of g along C ; returns dq, hist
        sh = W.shape; Wg = W.reshape(*sh[:-1], sh[-1] // g, g)
        s = (c * Wg.pow(2).mean(-1, keepdim=True).sqrt()).clamp_min(1e-8).half().float()
        q = torch.clamp(torch.round(Wg / s), -LM, LM); return (q * s).reshape(sh), ent_of(q)
    def hist_entropy(h):
        p = h / h.sum(); p = p[p > 0]; return float(-(p * p.log2()).sum())
    def solve_c(Wsamp, target, g, lo=0.02, hi=6.0):
        for _ in range(14):
            mid = math.sqrt(lo * hi); _, h = quant_rtn(Wsamp, mid, g); e = hist_entropy(h)
            if e > target: lo = mid
            else: hi = mid
        return math.sqrt(lo * hi)
    @torch.no_grad()
    def gptq_batch(W, H, c, g, damp=0.01):
        """W[E,R,C] (rotated), H[E,C,C]; blockwise (block=g) GPTQ with scale = c*rms(group of updated W). returns dq W, hist"""
        E, R, C = W.shape; W = W.float().clone(); H = H.clone()
        diag = torch.diagonal(H, dim1=-2, dim2=-1); dead = diag == 0
        idx = torch.arange(C, device=W.device); H[:, idx, idx] = torch.where(dead, torch.ones_like(diag), diag)
        W = W.masked_fill(dead.unsqueeze(1), 0)
        mean = torch.diagonal(H, dim1=-2, dim2=-1).mean(-1, keepdim=True)
        H[:, idx, idx] += damp * mean
        L = torch.linalg.cholesky(H); Hinv = torch.linalg.cholesky(torch.cholesky_inverse(L), upper=True); del L
        Q = torch.zeros_like(W); hist = torch.zeros(2 * LM + 1, device=W.device); QI = torch.zeros(W.shape, dtype=torch.int8, device=W.device)
        for i1 in range(0, C, g):
            i2 = i1 + g; W1 = W[:, :, i1:i2].clone(); Q1 = torch.zeros_like(W1); Err = torch.zeros_like(W1); Hi = Hinv[:, i1:i2, i1:i2]
            s = (c * W1.pow(2).mean(-1, keepdim=True).sqrt()).clamp_min(1e-8).half().float()
            for i in range(g):
                w = W1[:, :, i]; d = Hi[:, i, i].unsqueeze(1)
                qi = torch.clamp(torch.round(w / s[:, :, 0]), -LM, LM); q = qi * s[:, :, 0]
                hist += torch.bincount((qi.long() + LM).flatten(), minlength=2 * LM + 1).float()
                Q1[:, :, i] = q; QI[:, :, i1 + i] = qi.to(torch.int8); err = (w - q) / d
                W1[:, :, i:] -= err.unsqueeze(-1) * Hi[:, i, i:].unsqueeze(1); Err[:, :, i] = err
            Q[:, :, i1:i2] = Q1
            if i2 < C: W[:, :, i2:] -= Err @ Hinv[:, i1:i2, i2:]
        gptq_batch.last_ints = QI
        return Q, hist
    # ---------------- capture first-layer inputs
    def capture(x, bs=8):
        ins = []; kw = {}
        class Cap(nn.Module):
            def __init__(s, m): super().__init__(); s.m = m
            def forward(s, h, *a, **k): ins.append(h); kw.update(k); raise ValueError
        orig = layers[0]; layers[0] = Cap(orig)
        for i in range(0, len(x), bs):
            try: model(input_ids=x[i:i + bs].cuda())
            except ValueError: pass
        layers[0] = orig; return ins, kw
    # ---------------- original weights on CPU for restore
    orig = {n: p.detach().cpu().clone() for n, p in model.named_parameters() if ("experts" in n or "self_attn" in n or "embed" in n or "lm_head" in n)}
    print("orig saved", round(time.time() - t0), flush=True)
    def restore():
        for n, p in model.named_parameters():
            if n in orig: p.data.copy_(orig[n])
    cfg_files = sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(os.environ["HF_HOME"] + "/hub/models--Qwen--Qwen3-30B-A3B-Instruct-2507/snapshots") for f in fs if f.endswith((".json", ".txt")) and "safetensors.index" not in f)
    results = []
    cfg_names = {c["name"] for c in cfgs}
    for cfg in cfgs:
        tc = time.time(); name = cfg["name"]; k = cfg.get("k", 128); cs = cfg.get("calib", "A"); nseq = cfg.get("nseq", 192)
        eb_e = cfg.get("exp_bits"); ab = cfg.get("attn_bits", 6); emb_b = cfg.get("emb_bits", 6); g = cfg.get("g", 128); rank = cfg.get("rank", "freq")
        maxlayers = cfg.get("max_layers", NL)
        restore(); Rr = dict(name=name, cfg=cfg)
        for l in layers: l.mlp.gate.weight.data = l.mlp.gate.weight.data  # router kept bf16
        st = get_stats(cs); sc = {"freq": st["cnt"], "reap": st["reap"] / st["cnt"].clamp_min(1), "gw": st["gw"]}[rank]
        stride = cfg.get("stride", 1); off = cfg.get("offset", 0)
        act = lambda li: (li - off) % stride == 0
        for li in range(NL):
            if k < NE and act(li):
                top = sc[li].topk(k)[1]; m = torch.zeros(NE, dtype=torch.bool); m[top] = True; keepm[li] = m
            else: keepm[li] = None
        bits = dict(exp=0.0, attn=0.0, emb=0.0, scales=0.0, fixed=0.0); nq = dict(exp=0, attn=0, emb=0)
        def count16(n): return n * 16
        def qplain(Wt, target, grp=64):   # unrotated RTN rows (embeddings)
            if target >= 16: bits["emb"] += Wt.numel() * 16; return Wt
            c = solve_c(Wt[:4096].float(), target, grp); dq, h = quant_rtn(Wt.float(), c, grp)
            bits["emb"] += hist_entropy(h) * Wt.numel() * 1.005; bits["scales"] += Wt.numel() / grp * 16; return dq.to(torch.bfloat16)
        emb = model.get_input_embeddings(); emb.weight.data = qplain(emb.weight.data, emb_b)
        quantize = exp_bits_ok = (eb_e is not None) or (ab < 16)
        if (eb_e is not None or ab < 16) and maxlayers > 0:
            x = stream(cs, nseq, 512); cur, kw = capture(x); del x
            corr = dict(exp=0.0, attn=0.0)
            for li in range(min(NL, maxlayers)):
                l = layers[li]; mlp = l.mlp; at = l.self_attn
                if not act(li):
                    with torch.no_grad():
                        for j, xb in enumerate(cur):
                            o = l(xb, **kw); cur[j] = o[0] if isinstance(o, tuple) else o
                    continue
                if not isinstance(cur, list): pass
                Hs = {"qkv": 0, "o": 0}; hk = []
                def mkhook(key):
                    def hook(m, i, o):
                        xx = i[0].reshape(-1, i[0].shape[-1]).float(); Hs[key] = Hs[key] + 2 * xx.T @ xx
                    return hook
                if ab < 16:
                    hk = [at.q_proj.register_forward_hook(mkhook("qkv")), at.o_proj.register_forward_hook(mkhook("o"))]
                kk = k
                if eb_e is not None:
                    S["H"] = dict(gu=torch.zeros(NE, 2048, 2048, device="cuda"), d=torch.zeros(NE, 768, 768, device="cuda"), n=torch.zeros(NE)); S["hl"] = li
                with torch.no_grad():
                    for xb in cur: l(xb, **kw)
                for h_ in hk: h_.remove()
                S["hl"] = -1
                ntok = sum(xb.shape[0] * xb.shape[1] for xb in cur)
                # --- attention
                if ab < 16:
                    for key, mods in (("qkv", [at.q_proj, at.k_proj, at.v_proj]), ("o", [at.o_proj])):
                        H = Hs[key] / ntok; C = H.shape[0]
                        sg = (torch.randint(0, 2, (C,), generator=torch.Generator().manual_seed(7000 * li + C), dtype=torch.float32) * 2 - 1).cuda()
                        Hr = rot_H(H, sg)
                        for m in mods:
                            W = m.weight.data.float(); Wr = rot_cols(W, sg)
                            c = solve_c(Wr[:2048], ab - corr["attn"], g)
                            Qr, hist = gptq_batch(Wr.unsqueeze(0), Hr.unsqueeze(0), c, g)
                            e_rtn = hist_entropy(quant_rtn(Wr[:2048], c, g)[1]); e_g = hist_entropy(hist)
                            m.weight.data = unrot_cols(Qr[0], sg).to(torch.bfloat16)
                            bits["attn"] += e_g * W.numel() * 1.005; bits["scales"] += W.numel() / g * 16; nq["attn"] += W.numel()
                    del Hs
                else:
                    for m in (at.q_proj, at.k_proj, at.v_proj, at.o_proj): bits["attn"] += m.weight.numel() * 16
                # --- experts
                keep_ids = torch.nonzero(keepm[li]).flatten().tolist() if keepm[li] is not None else list(range(NE))
                if eb_e is not None:
                    Hd = S["H"]; S["H"] = None
                    for key in ("gu", "d"):
                        nn_ = Hd["n"]
                        for e in range(NE):
                            if nn_[e] > 0: Hd[key][e] /= nn_[e]
                    kid = torch.tensor(keep_ids, device="cuda")
                    for proj_group, key, C in ((("gate_proj", "up_proj"), "gu", 2048), (("down_proj",), "d", 768)):
                        sg = (torch.randint(0, 2, (C,), generator=torch.Generator().manual_seed(9000 * li + C), dtype=torch.float32) * 2 - 1).cuda()
                        Hr = rot_H(Hd[key][kid], sg)
                        for pname in proj_group:
                            Wall = torch.stack([getattr(mlp.experts[e], pname).weight.data for e in keep_ids]).float()
                            Wr = rot_cols(Wall, sg); del Wall
                            c = solve_c(Wr[:16], eb_e - corr["exp"], g)
                            out = torch.empty_like(Wr); hp = torch.zeros(2 * LM + 1, device="cuda")
                            for a in range(0, len(keep_ids), 32):
                                Qr, hist = gptq_batch(Wr[a:a + 32], Hr[a:a + 32], c, g); out[a:a + 32] = Qr; hp += hist
                                if cfg.get("rans_check") and a == 0 and li in (10, 30) and pname != "down_proj":
                                    import constriction
                                    qi = (gptq_batch.last_ints.flatten().long() + LM).cpu().numpy().astype(np.int32)
                                    h_ = np.bincount(qi, minlength=2 * LM + 1).astype(np.float64); pr = (h_ + 1e-3) / (h_ + 1e-3).sum()
                                    mdl = constriction.stream.model.Categorical(pr.astype(np.float32), perfect=False); cd = constriction.stream.stack.AnsCoder()
                                    cd.encode_reverse(qi.astype(np.int32), mdl); real = len(cd.get_compressed()) * 32
                                    dec = constriction.stream.stack.AnsCoder(cd.get_compressed()).decode(mdl, len(qi)); assert (dec == qi).all()
                                    ent = hist_entropy(torch.tensor(h_)) * len(qi)
                                    Rr.setdefault("rans_check", []).append(dict(layer=li, proj=pname, n=len(qi), entropy_bits=ent, rans_bits=real, ratio=real / ent, bpw_rans=real / len(qi), bpw_ent=ent / len(qi)))
                            Wh = unrot_cols(out, sg).to(torch.bfloat16)
                            for j, e in enumerate(keep_ids): getattr(mlp.experts[e], pname).weight.data = Wh[j].clone()
                            n_ = Wr.numel(); bits["exp"] += hist_entropy(hp) * n_ * 1.005
                            bits["scales"] += n_ / g * 16; nq["exp"] += n_
                            e_rtn = hist_entropy(quant_rtn(Wr[:16], c, g)[1]); corr["exp"] = 0.7 * corr["exp"] + 0.3 * (hist_entropy(hp) - e_rtn)
                            del Wr, out, Wh
                        del Hr
                    del Hd
                else:
                    for e in keep_ids:
                        for pn in ("gate_proj", "up_proj", "down_proj"): bits["exp"] += getattr(mlp.experts[e], pn).weight.numel() * 16
                bits["fixed"] += 16 * (len(keep_ids) * 2048 + 2 * 2048 + 2 * 128 + 2048)   # router rows kept, 2 layernorms, q/k norm
                with torch.no_grad():
                    for j, xb in enumerate(cur):
                        o = l(xb, **kw); cur[j] = o[0] if isinstance(o, tuple) else o
                torch.cuda.empty_cache()
                if li % 6 == 0: print(name, "layer", li, round(time.time() - tc), "corr", {a: round(b, 3) for a, b in corr.items()}, "gb", round(torch.cuda.max_memory_allocated() / 1e9, 1), flush=True)
            del cur
        else:
            # no quantization of experts / attention: just account bf16
            for li in range(NL):
                kid = int(keepm[li].sum()) if keepm[li] is not None else NE
                bits["exp"] += kid * 3 * 2048 * 768 * 16; bits["attn"] += 16 * sum(p.numel() for p in layers[li].self_attn.parameters())
                bits["fixed"] += 16 * (kid * 2048 + 2 * 2048 + 2 * 128 + 2048)
        lm = model.get_output_embeddings(); 
        w = lm.weight.data
        if emb_b >= 16: bits["emb"] += w.numel() * 16
        else:
            c = solve_c(w[:4096].float(), emb_b, 64); dq, h = quant_rtn(w.float(), c, 64); lm.weight.data = dq.to(torch.bfloat16)
            bits["emb"] += hist_entropy(h) * w.numel() * 1.005; bits["scales"] += w.numel() / 64 * 16
        bits["fixed"] += 16 * 2048   # final norm
        total_bytes = sum(bits.values()) / 8 + cfg_files + 150_000   # + tokenizer/config files + code/hist tables
        Rr["bytes"] = total_bytes; Rr["ratio"] = total_bytes / 61_066_575_656; Rr["parts_MB"] = {a: b / 8e6 for a, b in bits.items()}; Rr["cfg_files"] = cfg_files
        Rr["quant_secs"] = time.time() - tc; print(name, "quantized; bytes", total_bytes, "ratio", Rr["ratio"], flush=True)
        if cfg.get("skip_eval"): results.append(Rr); print("RESULT_JSON " + json.dumps(Rr), flush=True); continue
        ev = evaluate(refs, name); Rr.update({a: b for a, b in ev.items() if not a.startswith("_")})
        Rr["secs"] = time.time() - tc; Rr["noise_floor"] = refs.get("noise_floor")
        os.makedirs("/hf/moe_res", exist_ok=True); json.dump(Rr, open(f"/hf/moe_res/{name}.json", "w")); vol.commit()
        print("RESULT_JSON " + json.dumps(Rr), flush=True); results.append(Rr)
    return results

@app.local_entrypoint()
def main(cfgs: str, tag: str = "x", dl: bool = False):
    if dl: download.remote(); return
    cs = json.load(open(cfgs)); groups = {}
    for c in cs: groups.setdefault(c.get("group", 0), []).append(c)
    hs = [run.spawn(v, tag) for v in groups.values()]
    for h in hs:
        try: r = h.get(); json.dump(r, open(f"results/{tag}_{r[0]['name']}.json", "w"), indent=1)
        except Exception as e: print("ERR", repr(e))
