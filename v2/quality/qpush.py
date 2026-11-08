"""QUALITY_PUSH: training-free sub-2-bit quantisation sweeps on Qwen2.5 bases, evaluated on REAL generation (TOOLSEL, INJ) + general MC gate.
Quantiser: random-sign + Kronecker/Hadamard incoherence on both sides, sequential block-LDLQ (Hessian from quantised-prefix activations),
D-dim Gaussian k-means codebook shared by all layers (D=1 => scalar Lloyd-Max = GPTQ-like baseline; D=8 => QuIP#-style VQ without fine-tuning).
No gradients/optimizers anywhere. Dev data only.  launch: python qpush.py cfgs.json  (spawns detached; results on Modal volume tenpct-quality:/res)"""
import modal, json, sys, os
app = modal.App("tenpct-quality")
HERE = os.path.dirname(os.path.abspath(__file__))
vol = modal.Volume.from_name("tenpct-quality", create_if_missing=True)
img = (modal.Image.debian_slim(python_version="3.11").pip_install("vllm==0.11.0", "transformers==4.57.1", "huggingface_hub<1.0", "safetensors", "numpy<2.3", "datasets<4", "accelerate")
       .add_local_dir(HERE, "/code", ignore=["__pycache__", "*.pyc", "results"]))

def _impl(cfg):
    import time, math, re, zlib, subprocess, gc, random, torch, numpy as np, torch.nn as nn
    os.environ["HF_HOME"] = "/vol/hf"
    from transformers import AutoTokenizer, AutoModelForCausalLM
    torch.backends.cuda.matmul.allow_tf32 = False
    t0 = time.time(); dev = "cuda"; name = cfg["name"]; M = cfg.get("model", "Qwen/Qwen2.5-7B-Instruct"); os.makedirs("/vol/res", exist_ok=True)
    tok = AutoTokenizer.from_pretrained(M)
    ng = torch.cuda.device_count()
    model = (AutoModelForCausalLM.from_pretrained(M, torch_dtype=torch.bfloat16, device_map="auto", max_memory=({i: v for i, v in enumerate(cfg["maxmem_list"])} if cfg.get("maxmem_list") else {i: cfg.get("maxmem", "46GiB") for i in range(ng)})) if ng > 1 else AutoModelForCausalLM.from_pretrained(M, torch_dtype=torch.bfloat16, device_map="cuda")).eval(); model.config.use_cache = False
    vol.commit()
    base_params = sum(p.numel() for p in model.parameters()); res = dict(name=name, cfg=cfg, base_params=base_params)
    # ---------------- general MC + ppl/KL (same sets/prompt as harness/validity/general_remote.py)
    LET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"; INST = "The following is a multiple choice question. Answer with the letter of the correct option."
    def build_sets(n_per, seed=0):
        from datasets import load_dataset
        rng = random.Random(seed); mk = lambda q, ch, a: dict(q=q, ch=list(ch), a=int(a)); sets = {}
        def sub(rows): rows = list(rows); rng.shuffle(rows); return rows[:n_per]
        def tl(nm, fn):
            try: sets[nm] = sub(fn())
            except Exception as e: print("dataset fail", nm, repr(e)[:200], flush=True)
        tl("mmlu", lambda: [mk(r["question"], r["choices"], r["answer"]) for r in load_dataset("cais/mmlu", "all", split="test")])
        tl("arc_challenge", lambda: [mk(r["question"], r["choices"]["text"], r["choices"]["label"].index(r["answerKey"])) for r in load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test") if r["answerKey"] in r["choices"]["label"]])
        tl("hellaswag", lambda: [mk(r["ctx"], r["endings"], r["label"]) for r in load_dataset("Rowan/hellaswag", split="validation")])
        def tqa():
            out = []
            for r in load_dataset("truthful_qa", "multiple_choice", split="validation"):
                ch = r["mc1_targets"]["choices"]
                if len(ch) <= 20: p = list(range(len(ch))); rng.shuffle(p); out.append(mk(r["question"], [ch[i] for i in p], p.index(r["mc1_targets"]["labels"].index(1))))
            return out
        tl("truthfulqa_mc1", tqa)
        try: wt = "\n\n".join(load_dataset("wikitext", "wikitext-2-raw-v1", split="test")["text"])
        except Exception as e: wt = None
        return sets, wt
    ngen = cfg.get("n_gen", 500); sets, wt = build_sets(ngen)
    ids = [tok.encode(" " + l, add_special_tokens=False)[-1] for l in LET]
    fmt = lambda r: INST + "\n\n" + r["q"].strip() + "\n" + "\n".join(f"{LET[i]}. {c.strip()}" for i, c in enumerate(r["ch"])) + "\nAnswer:"
    wids = tok(wt, return_tensors="pt").input_ids[0][:cfg.get("ppl_tokens", 40000)] if wt else None
    @torch.no_grad()
    def mc_preds(m, rows, bs=16):
        enc = [tok.encode(fmt(r), add_special_tokens=False)[-1536:] for r in rows]; order = np.argsort([-len(e) for e in enc]); out = np.zeros(len(rows), int)
        for b in range(0, len(rows), bs):
            ix = order[b:b + bs]; mx = max(len(enc[i]) for i in ix); x = torch.full((len(ix), mx), tok.pad_token_id or 0); a = torch.zeros(len(ix), mx, dtype=torch.long)
            for j, i in enumerate(ix): x[j, mx - len(enc[i]):] = torch.tensor(enc[i]); a[j, mx - len(enc[i]):] = 1
            pos = (a.cumsum(1) - 1).clamp(min=0)
            lg = m(input_ids=x.cuda(), attention_mask=a.cuda(), position_ids=pos.cuda()).logits[:, -1, :].float()
            for j, i in enumerate(ix): out[i] = int(lg[j, ids[:len(rows[i]["ch"])]].argmax())
        return out
    @torch.no_grad()
    def ppl_kl(m, ref=None, W=1024):
        nll = 0.0; n = 0; kl = 0.0; refl = []
        for k, s in enumerate(range(0, len(wids) - 1, W)):
            x = wids[s:s + W + 1][None].cuda()
            if x.shape[1] < 16: break
            l = torch.log_softmax(m(input_ids=x[:, :-1]).logits[0].float(), -1); y = x[0, 1:]; nll += -l.gather(1, y[:, None]).sum().item(); n += len(y)
            if ref is not None: r_ = ref[k].cuda().float(); kl += (r_.exp() * (r_ - l)).sum().item()
            else: refl.append(l.half().cpu())
        return nll, n, kl, refl
    def general_eval(m, tag):
        P = {k: mc_preds(m, rows) for k, rows in sets.items()}
        if tag == "base": nll, n, _, ref = ppl_kl(m); torch.save(ref, "/vol/res/ref_%s.pt" % M.replace("/", "_")); kl = 0.0
        else:
            for _w in range(720):
                try: vol.reload()
                except Exception as e_: pass
                if os.path.exists("/vol/res/ref_%s.pt" % M.replace("/", "_")) and os.path.exists(f"/vol/res/{cfg.get('base_name', 'base7b')}.json"): break
                time.sleep(10)
            ref = torch.load("/vol/res/ref_%s.pt" % M.replace("/", "_")); nll, n, kl, _ = ppl_kl(m, ref)
        return dict(preds={k: v.tolist() for k, v in P.items()}, y={k: [r["a"] for r in rows] for k, rows in sets.items()}, ppl=float(np.exp(nll / n)), kl=kl / n)
    # ---------------- quantisation
    if cfg.get("mode", "vq") in ("vq", "sens"):
        def had(p):
            Hm = torch.ones(1, 1)
            while Hm.shape[0] < p: Hm = torch.cat([torch.cat([Hm, Hm], 1), torch.cat([Hm, -Hm], 1)], 0)
            return Hm / math.sqrt(p)
        class Rot:
            def __init__(s, n, seed):
                p = n & (-n); m = n // p; g = torch.Generator().manual_seed(seed)
                s.sg = (torch.randint(0, 2, (n,), generator=g) * 2 - 1).float().to(dev)
                Q = torch.linalg.qr(torch.randn(m, m, generator=g, dtype=torch.float64))[0] if m > 1 else torch.ones(1, 1, dtype=torch.float64)
                s.Q = Q.float().to(dev); s.H = had(p).to(dev); s.m = m; s.p = p
            def fwd(s, X):
                sh = X.shape; Y = (X * s.sg).reshape(-1, s.m, s.p); Y = torch.matmul(s.Q, Y); Y = torch.matmul(Y, s.H); return Y.reshape(sh)
            def inv(s, X):
                sh = X.shape; Y = X.reshape(-1, s.m, s.p); Y = torch.matmul(s.Q.T, Y); Y = torch.matmul(Y, s.H); return Y.reshape(sh) * s.sg
        D = cfg.get("D", 8); cbs = {}
        def get_cb(K):
            if K in cbs: return cbs[K]
            f = f"/vol/res/cb_D{D}_K{K}.pt"
            if os.path.exists(f): cbs[K] = torch.load(f).to(dev); return cbs[K]
            g = torch.Generator(device=dev).manual_seed(K); N = max(K * 100, 200000); X = torch.randn(N, D, generator=g, device=dev); C = X[:K].clone()
            for it in range(30):
                sums = torch.zeros(K, D, device=dev); cnt = torch.zeros(K, device=dev); Cn = 0.5 * (C ** 2).sum(1)
                for a in range(0, N, 16384):
                    xb = X[a:a + 16384]; ii = (xb @ C.T - Cn).argmax(1); sums.index_add_(0, ii, xb); cnt.index_add_(0, ii, torch.ones(len(ii), device=dev))
                e = cnt == 0; newC = sums / cnt.clamp(min=1).unsqueeze(1); newC[e] = X[torch.randint(0, N, (int(e.sum()),), device=dev, generator=g)]; C = newC
            torch.save(C.cpu(), f); cbs[K] = C; return C
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
        def decode(idx, C, sigma, rin, rout):
            Wq = (C[idx].reshape(idx.shape[0], -1)) * sigma; Wq = rout.inv(Wq.T.contiguous()).T.contiguous(); return rin.inv(Wq)
        # calibration stream
        cc = [json.loads(l) for l in open("/code/calib_ct.jsonl")]
        def chat(u): return tok.apply_chat_template([{"role": "user", "content": u}], tokenize=False, add_generation_prompt=True)
        txt = [chat(c["user"]) + c["answer"] + "<|im_end|>\n" for c in cc]
        if cfg.get("calib") == "mix" or cfg.get("calib") is None:
            t1 = [json.loads(l) for l in open("/code/calib_t1.jsonl")][:len(cc) // 2]
            for r in t1:
                tl_ = r.get("tools", []); msgs = [{"role": "user", "content": r["query"]}]
                try:
                    s = tok.apply_chat_template(msgs, tools=[{"type": "function", "function": t} for t in tl_], tokenize=False, add_generation_prompt=True)
                    gold = r.get("gold") or []
                    ans = "".join("<tool_call>\n" + json.dumps({"name": g["name"], "arguments": g["arguments"]}) + "\n</tool_call>" for g in gold) if gold else "No tool applies."
                    txt.append(s + ans + "<|im_end|>\n")
                except Exception as e: pass
        txt = txt * cfg.get("task_rep", 1); gtok = cfg.get("gen_tokens", 0)
        if gtok:
            from datasets import load_dataset
            wtr = load_dataset("wikitext", "wikitext-103-raw-v1", split="train", streaming=True); acc = 0
            for r in wtr:
                t_ = r["text"].strip()
                if len(t_) < 400: continue
                txt.append(t_); acc += len(t_) // 5
                if acc >= gtok: break
        rng = np.random.RandomState(0); rng.shuffle(txt); ncal = cfg.get("ncal", 256); Lc = cfg.get("callen", 512); stream = []
        for s in txt: stream += tok.encode(s, add_special_tokens=False)[:2048]
        nev_s = cfg.get("nsens_eval", 24) if cfg.get("mode") == "sens" else 0
        n_ = min(len(stream) // Lc, ncal + nev_s); stream = torch.tensor(stream[:n_ * Lc]).view(-1, Lc); ncal = n_ - nev_s; ev_ids = stream[ncal:] if nev_s else None; stream = stream[:ncal]; print("calib seqs", stream.shape, flush=True)
        layers = model.model.layers; ins = []; kw = {}
        class Catcher(nn.Module):
            def __init__(s, m): super().__init__(); s.m = m; s.attention_type = getattr(m, 'attention_type', 'full_attention')
            def forward(s, x, *a, **k): ins.append(x); kw.update(k); raise ValueError
        layers[0] = Catcher(layers[0])
        for i in range(len(stream)):
            try: model(stream[i:i + 1].cuda())
            except ValueError: pass
        layers[0] = layers[0].m; cur = [x.clone() for x in ins]; del ins
        kw = {k: v for k, v in kw.items() if k not in ("past_key_values", "past_key_value", "use_cache")}
        groups = [("q_proj", ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"], "self_attn.q_proj"), ("o_proj", ["self_attn.o_proj"], "self_attn.o_proj"),
                  ("gate_proj", ["mlp.gate_proj", "mlp.up_proj"], "mlp.gate_proj"), ("down_proj", ["mlp.down_proj"], "mlp.down_proj")]
        K0 = cfg.get("K", 8192); plan = cfg.get("plan", [])
        def Kof(li, n):
            k = K0
            for rg, v in plan:
                if re.search(rg, f"L{li}.{n}"): k = v
            return k
        if cfg.get("mode") == "sens":
            sK = cfg.get("sens_K", 8192); C = get_cb(sK); sres = {}
            def fwd_logits():
                outs = []
                with torch.no_grad():
                    for a in range(0, len(ev_ids), 6): outs.append(model(input_ids=ev_ids[a:a + 6].cuda()).logits.float().log_softmax(-1))
                return torch.cat(outs)
            lp0 = fwd_logits().half().cpu(); print("sens ref logits", lp0.shape, flush=True); res["sens_kl_offset"] = None
            def klnow():
                lq = fwd_logits(); tot = 0.0
                for a in range(0, len(lq), 6):
                    r_ = lp0[a:a + 6].cuda().float(); tot += float((r_.exp() * (r_ - lq[a:a + 6])).sum())
                return tot / (lq.shape[0] * lq.shape[1])
            res["sens_kl_offset"] = klnow(); print("kl offset", res["sens_kl_offset"], flush=True)
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
                    for j, x in enumerate(cur): o = l(x, **kw); cur[j] = o[0] if isinstance(o, tuple) else o
                for h_ in hds: h_.remove()
                for gname, names, hookn in groups:
                    Hraw = Hs[gname].to(dev) / nsm[gname]; n_in = Hraw.shape[0]; rin = Rot(n_in, zlib.crc32(f"{li}.{gname}.in".encode()))
                    Hr = rin.fwd(Hraw); Hr = rin.fwd(Hr.T.contiguous()).T.contiguous(); del Hraw; olds = {}
                    for n in names:
                        m = l.get_submodule(n); wdev = m.weight.device; olds[n] = m.weight.data.clone(); W = m.weight.data.float().to(dev); od = W.shape[0]; rout = Rot(od, zlib.crc32(f"{li}.{n}.out".encode()))
                        Wr = rin.fwd(W); Wr = rout.fwd(Wr.T.contiguous()).T.contiguous(); sigma = float(Wr.std()); idx = ldlq(Wr / sigma, Hr, C)
                        m.weight.data = decode(idx, C, sigma, rin, rout).to(torch.bfloat16).to(wdev)
                    k_ = klnow(); sres[f"L{li}.{gname}"] = dict(kl=k_, n=sum(int(l.get_submodule(n).weight.numel()) for n in names))
                    for n in names: l.get_submodule(n).weight.data = olds[n]
                    del Hr
                del Hs; torch.cuda.empty_cache(); print("sens layer", li, {k: round(v["kl"], 5) for k, v in sres.items() if k.startswith(f"L{li}.")}, round(time.time() - t0), flush=True)
            res["sens"] = sres; res["sens_K"] = sK; json.dump(res, open(f"/vol/res/{name}.json", "w")); vol.commit(); print("RESULT_JSON " + json.dumps(dict(name=name, sens=sres))); return res
        st = dict(fixed_bits=0.0, ent_bits=0.0, nq=0, meta=0, num=0.0, den=0.0)
        tgt = cfg.get("tgt", False)
        if tgt: cur_fp = [x.clone() for x in cur]
        for li, l in enumerate(layers):
            Hs = {}; nsm = {}; Hf = {}; store = {}; flag = {"f": False}
            for gname, names, hookn in groups:
                m = l.get_submodule(hookn); Hs[gname] = torch.zeros(m.in_features, m.in_features, device=dev); nsm[gname] = 0
                if tgt: Hf[gname] = torch.zeros(m.in_features, m.in_features, device=dev)
            def mk(gname):
                def hook(m, i, o):
                    x = i[0].reshape(-1, i[0].shape[-1]).float(); torch.backends.cuda.matmul.allow_tf32 = True
                    if flag["f"]: Hf[gname] += (x.T @ store[gname].to(x.device)).to(Hf[gname].device)
                    else:
                        nsm[gname] += x.shape[0]; Hs[gname] += (x.T @ x).to(Hs[gname].device)
                        if tgt: store[gname] = x
                    torch.backends.cuda.matmul.allow_tf32 = False
                return hook
            hds = [l.get_submodule(hn).register_forward_hook(mk(gn)) for gn, _, hn in groups]
            with torch.no_grad():
                for j, x in enumerate(cur):
                    flag["f"] = False; l(x, **kw)
                    if tgt: flag["f"] = True; o = l(cur_fp[j], **kw); cur_fp[j] = o[0] if isinstance(o, tuple) else o
            for h_ in hds: h_.remove()
            if tgt: store.clear()
            Hs = {k: v.cpu() for k, v in Hs.items()}; Hfs = {k: v.cpu() for k, v in Hf.items()}; torch.cuda.empty_cache()
            for gname, names, hookn in groups:
                Hraw = Hs[gname].to(dev) / nsm[gname]; n_in = Hraw.shape[0]; rin = Rot(n_in, zlib.crc32(f"{li}.{gname}.in".encode()))
                Hr = rin.fwd(Hraw)
                if tgt:
                    Hq = Hraw.double(); Hq.diagonal().add_(cfg.get("damp", 0.01) * float(torch.mean(torch.diag(Hq)))); Lc = torch.linalg.cholesky(Hq); del Hq; Hfq = (Hfs[gname].to(dev) / nsm[gname]).double()
                Hr = rin.fwd(Hr.T.contiguous()).T.contiguous(); del Hraw
                for n in names:
                    m = l.get_submodule(n); wdev = m.weight.device; W = m.weight.data.float().to(dev); od = W.shape[0]; rout = Rot(od, zlib.crc32(f"{li}.{n}.out".encode()))
                    if tgt: Wt = torch.cholesky_solve((W.double() @ Hfq).T.contiguous(), Lc).T.contiguous().float(); st["tshift"] = st.get("tshift", 0.0) + float((Wt - W).norm() / W.norm()); W = Wt; del Wt
                    Wr = rin.fwd(W); Wr = rout.fwd(Wr.T.contiguous()).T.contiguous(); sigma = float(Wr.std()) * cfg.get("smult", 1.0)
                    K = Kof(li, n); C = get_cb(K); idx = ldlq(Wr / sigma, Hr, C, damp=cfg.get("damp", 0.01))
                    Q = C[idx].reshape(od, -1) * sigma; Eq = Wr - Q; st["num"] += float(((Eq @ Hr) * Eq).sum()); st["den"] += float(((Wr @ Hr) * Wr).sum())
                    m.weight.data = decode(idx, C, sigma, rin, rout).to(torch.bfloat16).to(wdev)
                    cnt = torch.bincount(idx.flatten(), minlength=K).float(); p = cnt / cnt.sum(); p = p[p > 0]; e = float(-(p * p.log2()).sum())
                    st["fixed_bits"] += idx.numel() * math.log2(K); st["ent_bits"] += idx.numel() * e; st["nq"] += W.numel(); st["meta"] += 2
                del Hr
            del Hs
            with torch.no_grad():
                for j, x in enumerate(cur): o = l(x, **kw); cur[j] = o[0] if isinstance(o, tuple) else o
            torch.cuda.empty_cache(); print("layer", li, "relerr", st["num"] / st["den"], round(time.time() - t0), flush=True)
        # bytes (7B-style: vocab-pruned 4-bit emb+head estimated for kept=vocab_keep)
        nb = st["nq"]; allp = base_params; hd = model.config.hidden_size; V = model.config.vocab_size; tied = model.config.tie_word_embeddings
        vk = cfg.get("vocab_keep", 24176); embhead_params = V * hd * (1 if tied else 2)
        other = allp - nb - embhead_params
        cbbytes = sum(k * D * 2 for k in cbs)
        fixed = other * 2 + (vk * hd * 0.5 * 1.08) * (1 if tied else 2) + 1.0e6 + 5.0e6 + cbbytes + st["meta"]
        res.update(body_bpw_fixed=st["fixed_bits"] / nb, body_bpw_ent=st["ent_bits"] / nb, rel_proxy_err=st["num"] / st["den"], codebook_bytes=cbbytes, fixed_bytes=fixed,
                   ratio_fixed=(st["fixed_bits"] / 8 + fixed) / (allp * 2), ratio_ent=(st["ent_bits"] * 1.005 / 8 + fixed) / (allp * 2), nbody=nb)
        if cfg.get("emb4", True):   # simulate 4-bit g64 RTN emb + head
            with torch.no_grad():
                for mod in [model.get_input_embeddings(), model.get_output_embeddings()]:
                    w = mod.weight.data.float().view(mod.weight.shape[0], -1, 64); mx = w.abs().amax(-1, keepdim=True).clamp(min=1e-8) / 7
                    mod.weight.data = (torch.round(w / mx).clamp(-8, 7) * mx).view_as(mod.weight).to(torch.bfloat16)
    res["quant_secs"] = time.time() - t0; print("quant done", res.get("ratio_fixed"), res.get("ratio_ent"), flush=True)
    g = general_eval(model, "base" if cfg.get("mode") == "base" else "cand"); res["general"] = g
    if cfg.get("mode") != "base":
        bg = json.load(open(f"/vol/res/{cfg.get('base_name', 'base7b')}.json"))["general"]
        res["general_summary"] = {k: dict(base=float(np.mean(np.array(bg["preds"][k]) == np.array(bg["y"][k]))), cand=float(np.mean(np.array(g["preds"][k]) == np.array(g["y"][k]))),
                                     retention=float(((np.array(g["preds"][k]) == np.array(g["y"][k])) & (np.array(bg["preds"][k]) == np.array(bg["y"][k]))).sum() / max(1, (np.array(bg["preds"][k]) == np.array(bg["y"][k])).sum()))) for k in g["preds"]}
        print("GENERAL", json.dumps(res["general_summary"]), "ppl", g["ppl"], "kl", g["kl"], flush=True)
    outd = f"/tmp/q_{name}"; model.save_pretrained(outd, safe_serialization=True); tok.save_pretrained(outd)
    layers = l = cur = kw = Catcher = None; model.model.layers = None; del model; gc.collect(); torch.cuda.empty_cache(); print('free GB after cleanup', [torch.cuda.mem_get_info(i)[0] / 2**30 for i in range(torch.cuda.device_count())], flush=True)
    for f in ("vllm_real.py", "inj_scorer.py", "toolsel_scorer.py", "inj_items.jsonl", "toolsel_items.jsonl"): pass
    os.environ["VLLM_UTIL"] = str(cfg.get("vllm_util", 0.5)); os.environ["VLLM_TP"] = str(ng); r = subprocess.run(f"python /code/vllm_real.py {outd} /tmp/real_{name} inj,toolsel", shell=True, cwd="/code"); print("vllm rc", r.returncode, flush=True)
    for t in ("inj", "toolsel"):
        f = f"/tmp/real_{name}_{t}.json"
        if os.path.exists(f): res["real_" + t] = json.load(open(f))["rows"]
    res["secs"] = time.time() - t0; json.dump(res, open(f"/vol/res/{name}.json", "w")); vol.commit()
    summ = {k: v for k, v in res.items() if k not in ("general", "real_inj", "real_toolsel")}
    for t in ("inj", "toolsel"):
        if "real_" + t in res: summ["acc_" + t] = float(np.mean([x["correct"] for x in res["real_" + t]]))
    print("RESULT_JSON " + json.dumps(summ, default=str), flush=True); return summ

@app.function(image=img, gpu=["H100", "A100-80GB"], memory=65536, timeout=4 * 3600, volumes={"/vol": vol}, max_containers=8)
def run(cfg): return _impl(cfg)

@app.function(image=img, gpu="A100-80GB:2", memory=262144, ephemeral_disk=600 * 1024, timeout=8 * 3600, volumes={"/vol": vol}, max_containers=3)
def run_big(cfg): return _impl(cfg)

@app.function(image=img, gpu="A100-80GB:4", memory=400000, ephemeral_disk=900 * 1024, timeout=10 * 3600, volumes={"/vol": vol}, max_containers=1)
def run_huge(cfg): return _impl(cfg)

if __name__ == "__main__":
    cs = json.load(open(sys.argv[1])); calls = []
    with modal.enable_output(), app.run(detach=True):
        for c in cs: calls.append((c["name"], ({"huge": run_huge, "big": run_big}.get(c.get("size"), run)).spawn(c).object_id)); print(calls[-1], flush=True)
    json.dump(calls, open(sys.argv[1] + ".calls", "w"))
