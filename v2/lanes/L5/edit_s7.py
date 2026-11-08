"""L5 SKILL7 feedback (copy of v2/fb_edit/edit.py snapshot with data_dir/old/new/marker cfg keys; run via run_fb.py on a GPU box)."""
"""fb_edit: diffuse closed-form (MEMIT/EMMET/AlphaEdit-style) feedback edits through the model's own MLP (down_proj of the last layers), no optimizer.
Modal app tenpct-fbedit. DEV data only (fb_legal/data_tox). Deploy: modal deploy edit.py ; launch: python3 launch.py cfgs.json
Method spec (cfg["methods"] = list of dicts):
  kind "memit": layers (list), lam (key-cov weight), m (margin), ridge, cols (None=dense, int=column-sparse top-r columns), init (None|"oldmean"|"nn"), beta, lm (None|"mnH")
  kind "scales": Bonsai only; edit the group scales (fp16, 0 extra bytes) of target lm_head rows by Hessian min-norm solve in the 32-dim scale space
  kind "mnH": reference readout edit (fb_legal)
Closed-form target: dx = rms * (gap+m) * u/(u.u), u = gamma*(w_t - w_comp)   (final residual change that lifts label logit over best competitor by m)
Sequential layers: layer l delivers 1/(layers left) of the CURRENT remaining dx; dW = R K^T (lam C + K K^T)^-1 via Woodbury; C = mean key outer product over task-format calibration tokens."""
import modal, json, os
app = modal.App("tenpct-l5fb")
HERE = os.path.dirname(os.path.abspath(__file__))
img = (modal.Image.debian_slim().pip_install("torch", "transformers==4.51.3", "numpy", "accelerate", "huggingface_hub")
       )
vol = modal.Volume.from_name("tenpct-l5fb", create_if_missing=True); hf = None

@app.function(image=img, gpu="A100-80GB", cpu=8, memory=65536, timeout=43200, volumes={"/r": vol, "/hf": hf}, env={"HF_HOME": "/hf"})
def run(cfg):
    import time, hashlib, random, collections, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
    from transformers import AutoTokenizer, AutoModelForCausalLM
    A = cfg; MODEL = A["model"]; NDRAW = A.get("ndraw", 10); NS = A.get("ns", [5, 12, 25]); METHODS = A["methods"]; OUT = "/r/" + A["name"]
    CONDS = A.get("conds", ["real", "wrong", "mixed_wrong", "shuffle"]); KMAX = A.get("kmax", 8); NCAL = A.get("ncal", 300)
    T0 = time.time(); log = lambda *a: print(f"[{time.time()-T0:6.0f}s]", *a, flush=True)
    dev = "cuda"; tok = AutoTokenizer.from_pretrained(MODEL); tok.padding_side = "left"
    DT = getattr(torch, A.get("dtype", "bfloat16"))
    lm = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=DT).to(dev).eval()
    for p in lm.parameters(): p.requires_grad_(False)
    if A.get("fq_bits"):   # E4 proxy: fake-quantise the body (RTN, symmetric, group 128, fp16 scales) so edits are solved on noisy hidden states; lm_head/embeddings stay fp
        qb = A["fq_bits"]; qm = 2 ** (qb - 1) - 1
        with torch.no_grad():
            for lyr in lm.model.layers:
                for mod in [lyr.self_attn.q_proj, lyr.self_attn.k_proj, lyr.self_attn.v_proj, lyr.self_attn.o_proj, lyr.mlp.gate_proj, lyr.mlp.up_proj, lyr.mlp.down_proj]:
                    w = mod.weight.data.float(); R_, C_ = w.shape; w = w.view(R_, C_ // 128, 128)
                    sc = (w.abs().amax(-1, keepdim=True) / qm).clamp(min=1e-8).half().float(); mod.weight.data = ((w / sc).round().clamp(-qm - 1, qm) * sc).view(R_, C_).to(mod.weight.dtype)
    D_h = lm.config.hidden_size; W = lm.lm_head.weight; assert A.get("allow_tied") or not lm.config.tie_word_embeddings
    NL = lm.config.num_hidden_layers; LAY = lm.model.layers; GAMMA = lm.model.norm.weight.float(); EPS = lm.config.rms_norm_eps
    OLD = ["safe", "toxic", "hate", "sexual", "violence"]; NEW = "selfharm"
    PROMPT = ('You are a content moderation system. Classify the message below into exactly one category.\n'
          '- safe: acceptable content; includes news, quotes, discussion or criticism of violence, hate, sex or crime, and ordinary strong opinions, without attacking anyone\n'
          '- toxic: insults, harassment, vulgar or demeaning language aimed at a person, or other general abuse that is not hate, sexual or violent\n'
          '- hate: attacks or demeans a person or group because of identity (race, ethnicity, religion, gender, sexuality, disability, nationality)\n'
          '- sexual: sexually explicit content, sexual solicitation or sexual harassment\n'
          '- violence: threats, incitement, or requests for / depictions of violence against people\n'
          'Reply with exactly one word: safe, toxic, hate, sexual or violence.\n\nMessage:\n"""\n{t}\n"""')
    ALLL = OLD + [NEW]; EOS = "<|im_end|>"; EOS_ID = tok.convert_tokens_to_ids(EOS)
    def chat(u):
        try: return tok.apply_chat_template([{"role": "user", "content": u}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except TypeError: return tok.apply_chat_template([{"role": "user", "content": u}], tokenize=False, add_generation_prompt=True)
    P = lambda t: tok(chat(PROMPT.replace("{t}", t)), add_special_tokens=False).input_ids
    TG = {l: tok(l + EOS, add_special_tokens=False).input_ids for l in ALLL}
    log("model", MODEL, "label tokens", {l: [tok.decode([i]) for i in v] for l, v in TG.items()})
    TASKN = A.get("task", "tox")
    DD = A.get("data_dir") or ("/root/data_tox" if TASKN == "tox" else "/root/data_news"); rows = lambda n: [json.loads(l) for l in open(f"{DD}/{n}.jsonl")]
    if TASKN == "tox":
        SETS = {n: rows(n) for n in ["neigh_eval", "other_pool"]}
        for a_, b_ in (("jb_ann", "nw_ann"), ("jb_dev", "nw_dev"), ("jb_teach", "nw_teach"), ("jb_ood", "nw_ood")): SETS[a_] = rows(b_)
        CALT = [r["text"] for r in rows("calib_text")][:NCAL]
        _h = lambda g: int(hashlib.sha256(("nsplit" + str(g)).encode()).hexdigest()[:8], 16) % 100
        SETS["neigh_eval"] = [r for r in SETS["neigh_eval"] if _h(r["grp"]) >= 35]
    else:
        PROMPT = "{t}"; OLD = A.get("old", ["World", "Sports", "Business", "Technology"]); NEW = A.get("new", "Entertainment"); ALLL = OLD + [NEW]
        TG = {l: tok(l + EOS, add_special_tokens=False).input_ids for l in ALLL}
        def nrow(r): return dict(text=r["user"], gold=r["kind"], src=r["src"], grp=r["id"])
        SETS = dict(neigh_eval=[nrow(r) for r in rows("items_dev")], other_pool=[nrow(r) for r in rows("calib")][:750],
                    jb_ann=[nrow(r) for r in rows("fb_annot_25")], jb_teach=[nrow(r) for r in rows("fb_teach_unseen_200")], jb_dev=[], jb_ood=[])
        CALT = [r["user"] for r in rows("calib")][:NCAL]
        if "neigh_srcs" in A: SETS["neigh_eval"] = [r for r in SETS["neigh_eval"] if r["src"] in A["neigh_srcs"]]
    for k, v in SETS.items():
        for r in v: r["pid"] = P(r["text"])
    # ---------------- base pass: cache residual stream entering layer NL-KMAX
    TL0 = NL - KMAX; TAIL = list(range(TL0, NL))
    @torch.no_grad()
    def base_hidden(seqs, budget=60000):
        out = [None] * len(seqs); order = sorted(range(len(seqs)), key=lambda i: len(seqs[i])); i = 0
        while i < len(order):
            j = i
            while j < len(order) and (j - i + 1) * len(seqs[order[j]]) <= budget: j += 1
            j = max(j, i + 1); idx = order[i:j]; L = max(len(seqs[k]) for k in idx)
            ids = torch.full((len(idx), L), tok.pad_token_id, dtype=torch.long); am = torch.zeros((len(idx), L), dtype=torch.long)
            for r, k in enumerate(idx): s = seqs[k]; ids[r, :len(s)] = torch.tensor(s); am[r, :len(s)] = 1
            hs = lm.model(input_ids=ids.to(dev), attention_mask=am.to(dev), output_hidden_states=True).hidden_states[TL0]
            for r, k in enumerate(idx): out[k] = hs[r, :len(seqs[k])].clone()
            i = j
        return out
    ITEMS = {}   # key -> dict(h, pos, g)
    def addset(sn, labelfn):
        S = SETS[sn]; need = [(i, l) for i, r in enumerate(S) for l in labelfn(r)]
        seqs = [S[i]["pid"] + TG[l] for i, l in need]; hs = base_hidden(seqs)
        for (i, l), h in zip(need, hs):
            pid = S[i]["pid"]; g = TG[l]; ITEMS[(sn, i, l)] = dict(h=h, pos=list(range(len(pid) - 1, len(pid) + len(g) - 1)), g=torch.tensor(g, device=dev))
    addset("jb_ann", lambda r: ALLL); addset("jb_dev", lambda r: [NEW]); addset("jb_teach", lambda r: [NEW]); addset("jb_ood", lambda r: [NEW]); addset("other_pool", lambda r: [NEW]); addset("neigh_eval", lambda r: [r["gold"]])
    log("cached items", len(ITEMS))
    # ---------------- tail runner
    class F32Lin(nn.Module):
        def __init__(s, w): super().__init__(); s.w = nn.Parameter(w.float().clone(), requires_grad=False)
        def forward(s, x): return F.linear(x.float(), s.w).to(x.dtype)
    ORIGW = {}
    for li in TAIL:
        dp = LAY[li].mlp.down_proj; ORIGW[li] = dp.weight.data.float().clone(); LAY[li].mlp.down_proj = F32Lin(dp.weight.data)
    def restore_layers():
        for li in TAIL: LAY[li].mlp.down_proj.w.data.copy_(ORIGW[li])
    CTX = dict(rec=None, store=None)
    def mkhook(li):
        def hook(mod, inp):
            r = CTX["rec"]
            if r is None or li not in r["layers"]: return
            k = inp[0]
            if r["mode"] == "cov": 
                msk = r["mask"]; kk = k[msk].float(); r["C"][li].addmm_(kk.T, kk)
                kk = k[r["mask2"]].float(); r["Ca"][li].addmm_(kk.T, kk); return
            r["keys"][li] = torch.stack([k[b, p] for b, p in r["sites"]]).float()
        return hook
    for li in TAIL: LAY[li].mlp.down_proj.register_forward_pre_hook(mkhook(li))
    @torch.no_grad()
    def tail(keys, budget=48000, rec=None, return_logits=False, pos_override=None):
        """keys: list of ITEMS keys. returns list of final residual x [npos,D] at label positions (or logits argmax if return_logits)."""
        its = [ITEMS[k] for k in keys]; out = [None] * len(its); order = sorted(range(len(its)), key=lambda i: len(its[i]["h"])); i = 0
        while i < len(order):
            j = i
            while j < len(order) and (j - i + 1) * len(its[order[j]]["h"]) <= budget: j += 1
            j = max(j, i + 1); idx = order[i:j]; L = max(len(its[k]["h"]) for k in idx)
            H = torch.zeros((len(idx), L, D_h), dtype=DT, device=dev)
            for r, k in enumerate(idx): H[r, :len(its[k]["h"])] = its[k]["h"]
            pid = torch.arange(L, device=dev)[None].expand(len(idx), -1); pe = lm.model.rotary_emb(H, pid)
            if rec is not None and rec["mode"] == "sites":
                rec["sites"] = [(r, p) for r, k in enumerate(idx) for p in (rec["want"][idx[r]] if True else [])]
            if rec is not None and rec["mode"] == "cov":
                m_ = torch.zeros((len(idx), L), dtype=torch.bool, device=dev)
                for r, k in enumerate(idx): m_[r, :len(its[k]["h"])] = True
                rec["mask"] = m_; m2 = torch.zeros_like(m_)
                for r, k in enumerate(idx): m2[r, its[k]["pos"]] = True
                rec["mask2"] = m2
            CTX["rec"] = rec
            for li in TAIL:
                o = LAY[li](H, attention_mask=None, position_ids=pid, position_embeddings=pe); H = o[0] if isinstance(o, tuple) else o
            CTX["rec"] = None
            for r, k in enumerate(idx): out[k] = H[r, its[k]["pos"]]
            i = j
        return out
    def logits_of(x): 
        xf = x.float(); h = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + EPS) * GAMMA
        return h.to(W.dtype) @ W.T
    def hfinal(x): 
        xf = x.float(); return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + EPS) * GAMMA)
    @torch.no_grad()
    def evalset(sn, idxs=None, label=None):
        S = SETS[sn]
        if not S: return np.zeros(0, dtype=np.int8)
        idxs = list(range(len(S))) if idxs is None else idxs
        keys = [(sn, i, label if label else S[i].get("gold", NEW)) for i in idxs]; xs = tail(keys); res = np.zeros(len(keys), dtype=np.int8)
        for n, (x, k) in enumerate(zip(xs, keys)): res[n] = int(bool(((logits_of(x).argmax(-1)) == ITEMS[k]["g"]).all()))
        return res
    # ---------------- calibration (task-format inputs + model's own greedy outputs), key covariance per tail layer, final-norm hidden for mnH
    @torch.no_grad()
    def gen(prompts, mx, bs=32):
        out = [None] * len(prompts); order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
        for s in range(0, len(order), bs):
            idx = order[s:s + bs]; enc = tok([prompts[i] for i in idx], return_tensors="pt", padding=True, add_special_tokens=False).to(dev)
            o = lm.generate(**enc, max_new_tokens=mx, do_sample=False, pad_token_id=tok.pad_token_id)
            for j, i in enumerate(idx): out[i] = o[j, enc.input_ids.shape[1]:].tolist()
        return out
    calP = [P(t) for t in CALT]
    # NB: gen uses the F32Lin-wrapped (still unedited) model: identical function
    _go = gen([tok.decode(p) for p in calP], 6)
    for n, (p, o) in enumerate(zip(calP, _go)):
        o = [t for t in o]; 
        if EOS_ID in o: o = o[:o.index(EOS_ID) + 1]
        _go[n] = o
    calseq = [p + o for p, o in zip(calP, _go)]; calh = base_hidden(calseq)
    for n, (p, o, h) in enumerate(zip(calP, _go, calh)): ITEMS[("cal", n, "x")] = dict(h=h, pos=list(range(len(p) - 1, len(p) + len(o) - 1)), g=torch.tensor(o, device=dev))
    CK = [("cal", n, "x") for n in range(len(calP))]
    CLAYERS = {}
    ntok = sum(len(ITEMS[k]["h"]) for k in CK); nans = sum(len(ITEMS[k]["pos"]) for k in CK); CAVG = {}; CAVA = {}
    for _g in [TAIL[i:i + 2] for i in range(0, len(TAIL), 2)]:   # 32B memory: covariances (27648^2 fp32) built 2 layers at a time, kept on CPU
        rec = dict(mode="cov", layers=set(_g), C={li: torch.zeros((LAY[li].mlp.down_proj.w.shape[1],) * 2, device=dev) for li in _g}, Ca={li: torch.zeros((LAY[li].mlp.down_proj.w.shape[1],) * 2, device=dev) for li in _g})
        tail(CK, rec=rec)
        for li in _g: CAVG[li] = (rec["C"][li] / ntok).cpu(); CAVA[li] = (rec["Ca"][li] / nans).cpu()
        del rec; torch.cuda.empty_cache()
    CAL_H = torch.cat([hfinal(x) for x in tail(CK)]).float(); log("calibration tokens", ntok, "answer positions", len(CAL_H))
    dec = lambda ids: tok.decode(ids, skip_special_tokens=True).strip()
    ann = SETS["jb_ann"]; ag = gen([chat(PROMPT.replace("{t}", r["text"])) for r in ann], 6); BASE_LAB = []
    for o in ag:
        s = dec(o); BASE_LAB.append(next((l for l in ALLL if l.lower() == s.lower().rstrip(".")), "other"))
    log("base labels on ann pool", collections.Counter(BASE_LAB))
    NAT = [i for i, b in enumerate(BASE_LAB) if b in OLD]
    # paraphrases of the annotated items (script: the model itself, base weights, greedy)
    PARA_PROMPT = "Rewrite the following message in different words, keeping the same meaning, tone and level of detail. Output only the rewritten message.\n\nMessage:\n\"\"\"\n{t}\n\"\"\""
    MARK = A.get("marker", "Article: ")
    splitp = lambda t: (t.split(MARK)[0] + MARK, t.split(MARK, 1)[1]) if (TASKN != "tox" and MARK in t) else ("", t)
    pg = gen([chat(PARA_PROMPT.replace("{t}", splitp(r["text"])[1])) for r in ann], 300, bs=16)
    SETS["para"] = [dict(text=splitp(r["text"])[0] + (dec(o) or splitp(r["text"])[1]), grp=r["grp"]) for o, r in zip(pg, ann)]
    for r in SETS["para"]: r["pid"] = P(r["text"])
    addset("para", lambda r: [NEW]); log("paraphrase sample:", SETS["para"][0]["text"][:200].replace("\n", " "))
    EV = ["jb_dev", "jb_teach", "jb_ood", "neigh_eval"]
    base = {k: evalset(k) for k in EV}
    NE_OK = np.where(base["neigh_eval"] == 1)[0].tolist()
    log("base TF acc", {k: round(float(v.mean()), 3) for k, v in base.items()}, "base-correct neighbours", len(NE_OK))
    # ---------------- sites + edit
    @torch.no_grad()
    def find_sites(items):
        xs = tail(items); S = []
        for ii, (k, x) in enumerate(zip(items, xs)):
            am = logits_of(x).argmax(-1); g = ITEMS[k]["g"]
            for j in range(len(g)):
                if int(am[j]) != int(g[j]): S.append((ii, j, int(g[j])))
        return S
    CHOL = {}
    def chol_for(li, lam, ridge, awt=0.0):
        key = (li, lam, ridge, awt)
        if key not in CHOL:
            CHOL.clear() if len(CHOL) > 2 * len(TAIL) else None
            C = CAVG[li] + awt * CAVA[li]; Am = lam * (C + ridge * C.diagonal().mean() * torch.eye(C.shape[0], device=dev)); 
            Am = torch.nan_to_num(Am, nan=0.0, posinf=0.0, neginf=0.0); jit = 0.0
            for _ in range(8):
                Lc, info = torch.linalg.cholesky_ex(Am + jit * Am.diagonal().mean() * torch.eye(Am.shape[0], device=dev))
                if int(info) == 0: break
                jit = max(jit * 10, 1e-4)
            CHOL[key] = Lc
        return CHOL[key]
    @torch.no_grad()
    def memit_edit(items, sites, mt):
        layers = sorted(mt["layers"]); lam = mt.get("lam", 10.0); mg = mt.get("m", 8.0); ridge = mt.get("ridge", 0.05); cols = mt.get("cols")
        want = collections.defaultdict(list)
        for ii, j, t in sites: want[ii].append(ITEMS[items[ii]]["pos"][j])
        # site order in recorded keys: by batch-row order -> rebuild order mapping
        for step, li in enumerate(layers):
            rem = len(layers) - step
            # record keys and final x at site positions (positions in sequence coordinates)
            keys_rec, xs_rec = {}, {}
            its = [ITEMS[k] for k in items]
            # run per item to keep ordering trivial (n<=25 items, cheap)
            Ks, Xs, Ts = [], [], []
            for ii in sorted(want):
                r_ = dict(mode="sites", layers={li}, keys={}, want={0: want[ii]})
                it = its[ii]; ITEMS[("tmp", 0, 0)] = dict(h=it["h"], pos=want[ii], g=None)
                x = tail([("tmp", 0, 0)], rec=r_)[0]; Ks.append(r_["keys"][li]); Xs.append(x)
            K = torch.cat(Ks).T.contiguous(); X = torch.cat(Xs)    # K [I, n], X [n, D]
            T = torch.tensor([t for ii, j, t in sorted(sites, key=lambda s: (s[0], s[1]))], device=dev)
            lg = logits_of(X).float(); zt = lg.gather(1, T[:, None])[:, 0]; lg2 = lg.clone(); lg2.scatter_(1, T[:, None], -1e9); zc, ci = lg2.max(1)
            gap = (zc - zt + mg).clamp(min=0)
            u = GAMMA[None] * (W[T].float() - W[ci].float()); rms = X.float().pow(2).mean(-1, keepdim=True).add(EPS).sqrt()
            dx = rms * gap[:, None] * u / u.pow(2).sum(-1, keepdim=True).clamp(min=1e-9)
            R = (dx / rem).T.contiguous()   # [D, n]
            if cols:
                sc = (K.abs() / K.norm(dim=0, keepdim=True).clamp(min=1e-6)).sum(1); J = sc.topk(cols).indices
                Jc = J.cpu(); Cs = (CAVG[li][Jc][:, Jc] + mt.get("awt", 0.0) * CAVA[li][Jc][:, Jc]).to(dev); Am = lam * (Cs + ridge * Cs.diagonal().mean() * torch.eye(len(J), device=dev)); Kj = K[J]
                Z = torch.linalg.solve(Am, Kj); S_ = torch.linalg.solve(torch.eye(K.shape[1], device=dev) + Kj.T @ Z, torch.eye(K.shape[1], device=dev))
                dW = R @ S_ @ Z.T; LAY[li].mlp.down_proj.w.data[:, J] += dW
            else:
                L_ = chol_for(li, lam, ridge, mt.get("awt", 0.0)); Z = torch.cholesky_solve(K, L_); S_ = torch.linalg.solve(torch.eye(K.shape[1], device=dev) + K.T @ Z, torch.eye(K.shape[1], device=dev))
                dW = (R @ S_) @ Z.T; LAY[li].mlp.down_proj.w.data += dW
    @torch.no_grad()
    def lm_solve(items, sites, mt):   # reference mnH readout edit (fb_legal), also used after optional gap-closing init
        mg = mt.get("m", 8.0); xs = tail(items); by = collections.defaultdict(list)
        for ii, j, t in sites: by[t].append(hfinal(xs[ii][j:j + 1])[0])
        M = (CAL_H.T @ CAL_H) / len(CAL_H); M = M + 0.05 * M.diagonal().mean() * torch.eye(D_h, device=dev); rw = {}
        for t, hs in by.items():
            Hs = torch.stack(hs); wt = W.data[t].float(); bl = (Hs.to(W.dtype) @ W.T).float(); z = Hs @ wt; bl[:, t] = -1e9; mx = bl.max(1).values
            g = (mx - z + mg).clamp(min=0); MiH = torch.linalg.solve(M, Hs.T); G = Hs @ MiH; lam = 1e-3 * G.diagonal().mean()
            rw[t] = wt + MiH @ torch.linalg.solve(G + lam * torch.eye(len(Hs), device=dev), g)
        return rw
    @torch.no_grad()
    def scales_solve(items, sites, mt):   # Bonsai: edit only the 32 fp16 group scales of the target row
        mg = mt.get("m", 8.0); xs = tail(items); by = collections.defaultdict(list); G_ = 128
        for ii, j, t in sites: by[t].append(hfinal(xs[ii][j:j + 1])[0])
        rw = {}
        for t, hs in by.items():
            Hs = torch.stack(hs); wt = W.data[t].float(); sg = torch.sign(wt); sg[sg == 0] = 1; s = wt.abs().view(-1, G_).amax(1)
            feat = lambda H: (H * sg).view(H.shape[0], -1, G_).sum(-1)   # [n, 32]
            Fa = feat(Hs); Fc = feat(CAL_H); M = (Fc.T @ Fc) / len(Fc); M = M + mt.get("ridge", 0.05) * M.diagonal().mean() * torch.eye(M.shape[0], device=dev)
            bl = (Hs.to(W.dtype) @ W.T).float(); z = Hs @ wt; bl[:, t] = -1e9; mx = bl.max(1).values; g = (mx - z + mg).clamp(min=0)
            MiF = torch.linalg.solve(M, Fa.T); Gm = Fa @ MiF; lam = 1e-3 * Gm.diagonal().mean(); ds = MiF @ torch.linalg.solve(Gm + lam * torch.eye(len(Fa), device=dev), g)
            rw[t] = ((s + ds)[:, None] * sg.view(-1, G_)).reshape(-1)
        return rw
    def init_rows(mt):
        """gap-closing prior (no annotations): move new-label token rows toward the model's own neighbouring rows"""
        rw = {}; beta = mt.get("beta", 0.5); mode = mt["init"]
        newt = [t for t in TG[NEW] if t != EOS_ID]; Wf = W.data.float()
        for t in newt:
            if mode == "oldmean": nb = Wf[[TG[l][0] for l in OLD]].mean(0)
            else:
                cs = (F.normalize(Wf, dim=1) @ F.normalize(Wf[t], dim=0)); cs[t] = -1; nb = Wf[cs.topk(8).indices].mean(0)
            rw[t] = Wf[t] + beta * (nb - Wf[t])
        return rw
    SAVED = {}
    def apply_rows(rw):
        for t, v in rw.items():
            if t not in SAVED: SAVED[t] = W.data[t].clone()
            W.data[t] = v.to(W.dtype)
    def restore():
        for t, v in SAVED.items(): W.data[t] = v
        SAVED.clear(); restore_layers()
    def apply_method(mt, items):
        kind = mt["kind"]
        if kind == "none": return 0, 0
        if mt.get("init"): apply_rows(init_rows(mt))
        sites = find_sites(items); ns = len(sites)
        if not sites: return 0, 0
        if kind == "memit":
            if mt.get("lm") == "mnH": 
                apply_rows(lm_solve(items, sites, mt)); sites = find_sites(items)
            if sites: memit_edit(items, sites, mt)
        elif kind == "mnH": apply_rows(lm_solve(items, sites, mt))
        elif kind == "scales": apply_rows(scales_solve(items, sites, mt))
        return ns, len(sites)
    def bytes_of(mt):
        if mt["kind"] == "memit":
            L = len(mt["layers"]); I = CAVG[TAIL[0]].shape[0]
            return L * (mt["cols"] * (D_h * 2 + 2) if mt.get("cols") else D_h * I * 2)
        return 0
    bits = lambda a: "".join(map(str, a.tolist()))
    grp_ann = collections.defaultdict(list)
    for i, r in enumerate(ann):
        if i in set(NAT): grp_ann[r["grp"]].append(i)
    GK = sorted(grp_ann)
    def draw_idx(n, d): r = random.Random(1000 * n + d); ks = r.sample(GK, min(n, len(GK))); return [r.choice(grp_ann[k]) for k in ks]
    other = SETS["other_pool"]
    def conds(n, d):
        ix = draw_idx(n, d); r = random.Random(7000 + 1000 * n + d); out = {}; wl = []
        for i in ix: wl.append(r.choice([l for l in OLD if l != BASE_LAB[i]]))
        out["real"] = (ix, [("jb_ann", i, NEW) for i in ix]); out["wrong"] = (ix, [("jb_ann", i, l) for i, l in zip(ix, wl)])
        n = len(ix); pos = set(r.sample(range(n), n // 2)); out["mixed_wrong"] = (ix, [("jb_ann", i, (wl[p] if p in pos else NEW)) for p, i in enumerate(ix)])
        oi = r.sample(range(len(other)), n); out["shuffle"] = (oi, [("other_pool", i, NEW) for i in oi])
        return out
    RES = dict(model=MODEL, cfg=cfg, base={k: float(v.mean()) for k, v in base.items()}, base_vec={k: bits(v) for k, v in base.items()}, base_labels=dict(collections.Counter(BASE_LAB)),
               teach_grp=[r["grp"] for r in SETS["jb_teach"]], neigh_meta=[(SETS["neigh_eval"][i]["src"], SETS["neigh_eval"][i]["gold"], SETS["neigh_eval"][i]["grp"]) for i in NE_OK], runs=[], bytes={})
    for mt in METHODS:
        tag = mt["tag"]; RES["bytes"][tag] = bytes_of(mt)
        for n in NS:
            for d in range(NDRAW):
                cs = conds(n, d)
                for cn in CONDS:
                    ixs, items = cs[cn]; restore()
                    ns, nrem = apply_method(mt, items)
                    rec_ = dict(m=tag, n=n, d=d, c=cn, nsites=ns, nrem=nrem, nrows=len(SAVED), teach=bits(evalset("jb_teach")), ood=bits(evalset("jb_ood")), dev=bits(evalset("jb_dev")),
                                neigh=bits(evalset("neigh_eval", NE_OK)))
                    if cn == "real": rec_["para"] = bits(evalset("para", cs["real"][0], NEW)); rec_["annself"] = bits(evalset("jb_ann", cs["real"][0], NEW))
                    restore(); RES["runs"].append(rec_)
            rr = [r for r in RES["runs"] if r["m"] == tag and r["n"] == n and r["c"] == "real"]; sh = [r for r in RES["runs"] if r["m"] == tag and r["n"] == n and r["c"] == "shuffle"]
            f = lambda L, k: round(float(np.mean([np.mean(list(map(int, r[k]))) for r in L])), 3) if L else -1
            log(tag, "n", n, "real teach", f(rr, "teach"), "shuffle teach", f(sh, "teach"), "neigh real", f(rr, "neigh"), "ood", f(rr, "ood"), "para", f(rr, "para"), "ann", f(rr, "annself"))
            json.dump(RES, open(OUT + ".json", "w"))
    # ---------------- validity vs REAL greedy generation under the edit
    VAL = []
    for mt in A.get("valmethods", []):
        mt = next(x for x in METHODS if x["tag"] == mt)
        for dd in range(2):
            ix, items = conds(12, 555 + dd)["real"]; restore(); apply_method(mt, items)
            for sn in ["jb_teach", "neigh_eval"]:
                pool = NE_OK if sn == "neigh_eval" else list(range(len(SETS[sn]))); idx = pool[::max(1, len(pool) // 100)][:100]
                a1 = evalset(sn, idx); go = gen([chat(PROMPT.replace("{t}", SETS[sn][i]["text"])) for i in idx], 6)
                gold = lambda i: NEW if sn == "jb_teach" else SETS[sn][i]["gold"]
                a3 = np.array([int(dec(o).lower() == gold(i).lower()) for i, o in zip(idx, go)])
                VAL.append(dict(m=mt["tag"], draw=dd, set=sn, agree=float((a1 == a3).mean()), acc_tf=float(a1.mean()), acc_gen=float(a3.mean())))
            restore()
    RES["validity"] = VAL; log("VALIDITY", VAL)
    RES["done"] = True; json.dump(RES, open(OUT + ".json", "w")); vol.commit(); log("DONE")
    return dict(done=True)
