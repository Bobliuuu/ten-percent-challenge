"""memit2: careful second attempt at a NON-HEAD closed-form feedback edit (MEMIT-style) on existing internal weights (MLP down_proj, several consecutive layers).
No lm_head / embedding change, 0 added bytes, no weight gradient training.
  keys   k = input of down_proj (act(gate)*up, d_ff dim) at the last prompt token and at each teacher-forced label position (fp32 stats)
  C      = E[k k^T] over >=100k tokens of unlabeled task-format prompts (calib_text; optional wikitext) at EACH edited layer, fp32, per-token mean
  target v* (tgt="adam"): ROME/MEMIT free delta vector on the MLP output of the LAST edited layer, 30 Adam steps on the delta ONLY (an activation, not a weight);
         (tgt="closed"): delta = alpha * gamma*(W_U[label]-mean W_U[top-5 base competitors])/rms, alpha line-searched with FORWARD passes only
  update layer l (ascending): R = (v* - current MLP out at last layer)/(L_last-l+1); keys recomputed with the already-updated model; dW = R K^T (lam*C + K K^T)^-1
Eval: teacher-forced exact match of ALL label tokens (+EOS), neighbours = base-correct neighbour rows, controls empty/wrong/mixed_wrong/shuffle (same draws as fbcmp harness).
Decisive checks: weights saved to disk and reloaded, REAL greedy generate, 4-bit RTN (g128) re-quantisation of edited matrices."""
import modal, json, os
app = modal.App("tenpct-memit2")
HERE = os.path.dirname(os.path.abspath(__file__))
img = (modal.Image.debian_slim().pip_install("torch", "transformers==4.51.3", "numpy", "accelerate", "huggingface_hub", "safetensors")
       .add_local_dir(HERE + "/../data_tox", "/root/data_tox").add_local_dir(HERE + "/../data_route", "/root/data_route"))
vol = modal.Volume.from_name("tenpct-memit2", create_if_missing=True); hf = modal.Volume.from_name("tenpct-v2-hf")


def _run(A):
    import time, hashlib, random, collections, numpy as np, torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    MODEL = A["model"]; OUT = "/r/" + A["name"]; T0 = time.time(); log = lambda *a: print(f"[{time.time()-T0:6.0f}s]", *a, flush=True)
    dev = A.get("device", "cuda"); torch.backends.cuda.matmul.allow_tf32 = False
    tok = AutoTokenizer.from_pretrained(MODEL); tok.padding_side = "left"
    DT = getattr(torch, A.get("dtype", "bfloat16"))
    lm = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=DT).to(dev).eval()
    for p in lm.parameters(): p.requires_grad_(False)
    LAY = lm.model.layers; NL = len(LAY); DM = lm.config.hidden_size; GN = lm.model.norm.weight.float(); Wl = lm.lm_head.weight
    TASK = A["task"]; DD = {"tox": "/root/data_tox", "route": "/root/data_route"}[TASK]
    rows = lambda n: [json.loads(l) for l in open(f"{DD}/{n}.jsonl")]
    if os.path.exists(DD + "/LOCK.json"):
        lock = json.load(open(DD + "/LOCK.json")); bad = [f for f, h in lock.items() if hashlib.sha256(open(f"{DD}/{f}", "rb").read()).hexdigest() != h]; assert not bad, bad; log("hash lock ok", len(lock))
    if TASK == "tox":
        OLD = ["safe", "toxic", "hate", "sexual", "violence"]; NEWS = ["selfharm"]; SYSP = None
        PROMPT = ('You are a content moderation system. Classify the message below into exactly one category.\n'
          '- safe: acceptable content; includes news, quotes, discussion or criticism of violence, hate, sex or crime, and ordinary strong opinions, without attacking anyone\n'
          '- toxic: insults, harassment, vulgar or demeaning language aimed at a person, or other general abuse that is not hate, sexual or violent\n'
          '- hate: attacks or demeans a person or group because of identity (race, ethnicity, religion, gender, sexuality, disability, nationality)\n'
          '- sexual: sexually explicit content, sexual solicitation or sexual harassment\n'
          '- violence: threats, incitement, or requests for / depictions of violence against people\n'
          'Reply with exactly one word: safe, toxic, hate, sexual or violence.\n\nMessage:\n"""\n{t}\n"""')
    else:
        OLD = ["weather", "email", "transport", "lists", "calendar", "home", "news"]; NEWS = ["recipe", "music"]
        SYSP = ("You route requests for a voice assistant to the skill that should handle them.\nSkills:\n- weather: weather conditions and forecasts (temperature, rain, snow, what to wear for the weather)\n"
          "- email: email and contacts: reading, writing, sending or managing email; looking up or saving contact details\n- transport: taxis, trains, flights, tickets, traffic and directions\n"
          "- lists: creating, reading or editing any list (shopping lists, to-do lists, other named lists)\n- calendar: calendar events, meetings, appointments and reminders\n"
          "- home: smart-home devices: lights, plugs, vacuum cleaner, coffee machine\n- news: news and current events: headlines, politics, sports results, markets, prices and rates, anything happening in the world\nReply with the skill name only.")
        PROMPT = "{t}"
    REN = A.get("rename", {})   # brand-new label NAMES the prompt never lists
    NEWS = [REN.get(n, n) for n in NEWS]; ALLL = OLD + NEWS; EOS = "<|im_end|>"; EOS_ID = tok.convert_tokens_to_ids(EOS)
    def chat(u):
        m = ([{"role": "system", "content": SYSP}] if SYSP else []) + [{"role": "user", "content": u}]
        try: return tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except TypeError: return tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
    P = lambda t: tok(chat(PROMPT.replace("{t}", t)), add_special_tokens=False).input_ids
    TG = {l: tok(l + EOS, add_special_tokens=False).input_ids for l in ALLL}
    log("model", MODEL, "label tokens", {l: [tok.decode([i]) for i in v] for l, v in TG.items()})
    SETS = {n: rows(n) for n in ["neigh_eval", "neigh_alpha", "other_pool"]}
    for a_, b_ in (("jb_ann", "nw_ann"), ("jb_dev", "nw_dev"), ("jb_teach", "nw_teach"), ("jb_ood", "nw_ood")): SETS[a_] = rows(b_)
    for k in ("jb_ann", "jb_dev", "jb_teach", "jb_ood"):
        for r in SETS[k]: r["gold"] = REN.get(r.get("gold", "selfharm"), r.get("gold", "selfharm")); r.setdefault("grp", r.get("id", str(id(r))))
    for r in SETS["other_pool"]: r.setdefault("grp", r.get("id", str(id(r))))
    _h = lambda g: int(hashlib.sha256(("nsplit" + str(g)).encode()).hexdigest()[:8], 16) % 100
    SETS["neigh_eval"] = [r for r in SETS["neigh_eval"] if _h(r["grp"]) >= 35]
    for v in SETS.values():
        for r in v: r["pid"] = P(r["text"])
    if A.get("small"):
        for k in SETS: SETS[k] = SETS[k][:A["small"]]
    CALT = [r["text"] for r in rows("calib_text")][:A.get("ncal", 10**9)]
    calP = [P(t) for t in CALT]; log("calibration prompts", len(calP), "tokens", sum(map(len, calP)))
    bits = lambda a: "".join(map(str, a.tolist()))
    # ---------------- helpers
    def mkseq(pid, label): g = TG[label]; return (pid + g, list(range(len(pid) - 1, len(pid) + len(g) - 1)), g)
    @torch.no_grad()
    def tf_ok(seqs, budget=40000):
        """seqs = [(ids, ps, g)] -> int8 array: all label tokens argmax-correct (full forward through the CURRENT weights)"""
        out = np.zeros(len(seqs), dtype=np.int8); order = sorted(range(len(seqs)), key=lambda i: len(seqs[i][0])); i = 0
        while i < len(order):
            j = i
            while j < len(order) and (j - i + 1) * len(seqs[order[j]][0]) <= budget: j += 1
            j = max(j, i + 1); idx = order[i:j]; Lm = max(len(seqs[k][0]) for k in idx)
            ids = torch.full((len(idx), Lm), tok.pad_token_id, dtype=torch.long); am = torch.zeros((len(idx), Lm), dtype=torch.long)
            for r, k in enumerate(idx): s = seqs[k][0]; ids[r, :len(s)] = torch.tensor(s); am[r, :len(s)] = 1
            h = lm.model(input_ids=ids.to(dev), attention_mask=am.to(dev)).last_hidden_state
            for r, k in enumerate(idx):
                _, ps, g = seqs[k]; out[k] = int((lm.lm_head(h[r, ps]).argmax(-1).cpu() == torch.tensor(g)).all())
            i = j
        return out
    def ev(sn, idxs=None):
        S = SETS[sn]; idxs = list(range(len(S))) if idxs is None else idxs
        return tf_ok([mkseq(S[i]["pid"], S[i]["gold"]) for i in idxs])
    @torch.no_grad()
    def gen(prompts, mx, bs=24):
        out = [None] * len(prompts); order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
        for s in range(0, len(order), bs):
            idx = order[s:s + bs]; enc = tok([prompts[i] for i in idx], return_tensors="pt", padding=True, add_special_tokens=False).to(dev)
            o = lm.generate(**enc, max_new_tokens=mx, do_sample=False, pad_token_id=tok.pad_token_id)
            for j, i in enumerate(idx): out[i] = o[j, enc.input_ids.shape[1]:].tolist()
        return out
    dec = lambda ids: tok.decode(ids, skip_special_tokens=True).strip()
    ann = SETS["jb_ann"]; ag = gen([chat(PROMPT.replace("{t}", r["text"])) for r in ann], 6); BASE_LAB = []
    for o in ag:
        s = dec(o); BASE_LAB.append(next((l for l in ALLL if l.lower() == s.lower().rstrip(".")), "other"))
    log("base labels on ann pool", collections.Counter(BASE_LAB))
    NAT = [i for i, b in enumerate(BASE_LAB) if b in OLD]
    base = {k: ev(k) for k in ["jb_dev", "jb_teach", "jb_ood", "neigh_eval", "neigh_alpha"]}
    log("base TF acc", {k: round(float(v.mean()), 3) for k, v in base.items()})
    NE_OK = np.where(base["neigh_eval"] == 1)[0].tolist(); NSUB = A.get("nsub", 400)
    NE_SUB = NE_OK[::max(1, len(NE_OK) // NSUB)] if NSUB else NE_OK; log("base-correct neighbours", len(NE_OK), "eval subset", len(NE_SUB))
    gcl = {y: collections.defaultdict(list) for y in NEWS}
    for i in NAT: gcl[ann[i]["gold"]][ann[i]["grp"]].append(i)
    def draw_idx(n, d):
        r = random.Random(1000 * n + d); out = []; k = len(NEWS)
        for q, y in enumerate(NEWS):
            GKy = sorted(gcl[y]); per = n // k + (1 if q < n % k else 0); ks = r.sample(GKy, min(per, len(GKy))); out += [r.choice(gcl[y][g_]) for g_ in ks]
        return out
    other = SETS["other_pool"]
    def conds(n, d):
        ix = draw_idx(n, d); r = random.Random(7000 + 1000 * n + d); out = {}; wl = []
        for i in ix:
            b = BASE_LAB[i]; wl.append(r.choice([l for l in OLD if l != b]))
        out["real"] = [(ann[i]["pid"], ann[i]["gold"]) for i in ix]
        out["wrong"] = [(ann[i]["pid"], l) for i, l in zip(ix, wl)]
        n = len(ix); pos = set(r.sample(range(n), n // 2)); out["mixed_wrong"] = [(ann[i]["pid"], (wl[p] if p in pos else ann[i]["gold"])) for p, i in enumerate(ix)]
        oi = r.sample(range(len(other)), n); out["shuffle"] = [(other[i]["pid"], ann[j]["gold"]) for i, j in zip(oi, ix)]
        out["empty"] = []; return out
    # ---------------- covariance of down_proj inputs (fp32, per-token mean) for every layer that is ever edited
    COV = {}; COVN = {}
    def need_cov(Ls):
        todo = [l for l in Ls if l not in COV]
        if not todo: return
        srcs = calP
        if A.get("cov_src") == "wiki":
            from datasets import load_dataset
            ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="train", streaming=True); wk = []; nt = 0
            for r in ds:
                t = r["text"].strip()
                if len(t) > 400: ids = tok(t, add_special_tokens=False).input_ids[:512]; wk.append(ids); nt += len(ids)
                if nt > A.get("cov_tokens", 150000): break
            srcs = wk; log("wiki tokens", nt)
        buf = {}; hs = []; C = {l: torch.zeros(LAY[l].mlp.down_proj.weight.shape[1], LAY[l].mlp.down_proj.weight.shape[1], device=dev) for l in todo}; cnt = 0
        for l in todo: hs.append(LAY[l].mlp.down_proj.register_forward_pre_hook(lambda m, a, l=l: buf.__setitem__(l, a[0])))
        order = sorted(range(len(srcs)), key=lambda i: len(srcs[i])); i = 0
        with torch.no_grad():
            while i < len(order):
                j = i
                while j < len(order) and (j - i + 1) * len(srcs[order[j]]) <= 16000: j += 1
                j = max(j, i + 1); idx = order[i:j]; Lm = max(len(srcs[k]) for k in idx)
                ids = torch.full((len(idx), Lm), tok.pad_token_id, dtype=torch.long); am = torch.zeros((len(idx), Lm), dtype=torch.long)
                for r, k in enumerate(idx): ids[r, :len(srcs[k])] = torch.tensor(srcs[k]); am[r, :len(srcs[k])] = 1
                lm.model(input_ids=ids.to(dev), attention_mask=am.to(dev)); msk = am.bool().to(dev); cnt += int(msk.sum())
                for l in todo: kk = buf[l][msk].float(); C[l] += kk.T @ kk
                i = j
        for h in hs: h.remove()
        for l in todo: COV[l] = C[l] / cnt; COVN[l] = cnt
        log("cov for layers", todo, "tokens", cnt)
    # ---------------- weights bookkeeping: fp32 master of edited layers; model holds the DEPLOYED (bf16 or quantised) copy
    ORIG = {}; W32 = {}
    def qrtn(w, bitsn, g=128):
        x = w.float().view(w.shape[0], -1, g); qm = 2 ** (bitsn - 1) - 1; sc = (x.abs().amax(-1, keepdim=True) / qm).clamp(min=1e-8).half().float()
        return ((x / sc).round().clamp(-qm - 1, qm) * sc).reshape(w.shape)
    def orig(l):
        if l not in ORIG: ORIG[l] = LAY[l].mlp.down_proj.weight.data.clone()
        return ORIG[l]
    def restore():
        for l, w in ORIG.items(): LAY[l].mlp.down_proj.weight.data.copy_(w)
        W32.clear()
    def deploy(l, W2, qbits=None):
        LAY[l].mlp.down_proj.weight.data.copy_((qrtn(W2, qbits) if qbits else W2).to(DT))
    # ---------------- capture
    @torch.no_grad()
    def capture(seq, kl, Llast):
        ids, ps, g = seq; buf = {}
        h1 = LAY[kl].mlp.down_proj.register_forward_pre_hook(lambda m, a: buf.__setitem__("k", a[0][0, ps].float()))
        h2 = LAY[Llast].mlp.register_forward_hook(lambda m, a, o: buf.__setitem__("o", o[0, ps].float()))
        h3 = lm.model.norm.register_forward_pre_hook(lambda m, a: buf.__setitem__("x", a[0][0, ps].float()))
        lm.model(input_ids=torch.tensor([ids]).to(dev))
        h1.remove(); h2.remove(); h3.remove(); return buf
    def inject(Llast, ps, delta):
        def hk(m, a, o):
            o2 = o.clone(); o2[0, ps] = o2[0, ps] + delta.to(o.dtype); return o2
        return LAY[Llast].mlp.register_forward_hook(hk)
    @torch.no_grad()
    def margins(seq, Llast, delta):
        ids, ps, g = seq; hk = inject(Llast, ps, delta) if delta is not None else None
        h = lm.model(input_ids=torch.tensor([ids]).to(dev)).last_hidden_state[0, ps]
        if hk: hk.remove()
        lg = lm.lm_head(h).float(); t = torch.tensor(g).to(dev); lt = lg.gather(1, t[:, None])[:, 0]; lg2 = lg.clone(); lg2.scatter_(1, t[:, None], -1e9)
        return lt - lg2.max(1).values, lg
    VSTAR = {}
    def target(seq, Llast, tgt, m, key):
        """returns delta [len(ps), DM] fp32 to ADD to the MLP output of layer Llast at the target positions"""
        ck = (key, Llast, tgt, m)
        if ck in VSTAR: return VSTAR[ck]
        ids, ps, g = seq; cap = capture(seq, Llast, Llast); T = torch.tensor(g).to(dev)
        mg0, lg0 = margins(seq, Llast, None)
        if tgt == "adam":
            nrm = cap["o"].norm(dim=-1, keepdim=True).clamp(min=1e-3) / (DM ** 0.5)   # per-position scale of a typical element
            u = torch.zeros(len(ps), DM, device=dev, requires_grad=True); opt = torch.optim.Adam([u], lr=A.get("vlr", 0.3)); wd = A.get("vwd", 0.05)
            for step in range(A.get("vsteps", 30)):
                with torch.enable_grad():
                    delta = u * nrm; hk = inject(Llast, ps, delta)
                    h = lm.model(input_ids=torch.tensor([ids]).to(dev)).last_hidden_state[0, ps]; hk.remove()
                    lg = lm.lm_head(h).float(); nll = torch.nn.functional.cross_entropy(lg, T)
                    loss = nll + wd * (u ** 2).mean(); opt.zero_grad(); loss.backward(); opt.step()
                if step >= 1 and float(nll) < A.get("vnll", 0.02): break
            d = (u * nrm).detach()
        else:   # closed form: direction through the unembedding and the final norm, alpha by forward-pass line search (no gradients at all)
            rms = ((cap["x"] ** 2).mean(-1, keepdim=True) + lm.config.rms_norm_eps).sqrt(); dirs = []
            for j in range(len(ps)):
                top = [int(t) for t in lg0[j].topk(6).indices if int(t) != g[j]][:5]
                uvec = Wl[g[j]].float() - Wl[top].float().mean(0); dirs.append(GN * uvec / rms[j])
            dirs = torch.stack(dirs); us = dirs * rms / GN; gain = (dirs * GN * us / rms).sum(-1)   # first-order logit-gap change per unit alpha
            gap = (-mg0).clamp(min=0) + m; a0 = float((gap / gain.clamp(min=1e-6)).max()); d = torch.zeros_like(dirs)
            for mult in [0.5, 1, 1.5, 2, 3, 4, 6, 8, 12, 16, 24, 32, 64, 128]:
                d = dirs * a0 * mult; mg, _ = margins(seq, Llast, d)
                if float(mg.min()) >= m: break
        VSTAR[ck] = d; return d
    def cholA(l, lam, ridge):
        C = COV[l].double(); Cn = lam * (C + ridge * C.diagonal().mean() * torch.eye(C.shape[0], device=dev, dtype=torch.float64)); return torch.linalg.cholesky(Cn)
    def memit(items, key, Ls, lam, tgt, m, ridge, gain=1.0, agg=None, qbits=None, only_fail=False):
        """items = [(pid, label)]; edits layers Ls (consecutive); returns info"""
        restore(); Llast = Ls[-1]; need_cov(Ls)
        seqs = [mkseq(p, l) for p, l in items]
        deltas = [target(s, Llast, tgt, m, (key, i)) for i, s in enumerate(seqs)]
        Z = []
        for s, d in zip(seqs, deltas): Z.append(capture(s, Llast, Llast)["o"] + d)   # absolute target of the last-layer MLP output
        info = dict(dnorm=float(torch.cat(deltas).norm(dim=-1).mean()))
        for l in Ls:
            W0 = orig(l).float(); caps = [capture(s, l, Llast) for s in seqs]
            K = torch.cat([c["k"] for c in caps]); Cur = torch.cat([c["o"] for c in caps]); Zc = torch.cat(Z)
            R = (Zc - Cur) / (Llast - l + 1)
            if only_fail:
                sel = R.norm(dim=-1) > 1e-6; K, R = K[sel], R[sel]
            if agg == "mean":   # one aggregated key / residual (Hebbian / LDA-like)
                K = K.mean(0, keepdim=True); R = R.mean(0, keepdim=True)
            Lc = cholA(l, lam, ridge); B = torch.cholesky_solve(K.T.double(), Lc).float()   # A^-1 K
            S = torch.eye(K.shape[0], device=dev) + K @ B; D = gain * (R.T @ torch.linalg.solve(S, B.T))
            W32[l] = W0 + D; deploy(l, W32[l], qbits)
            info.setdefault("dW_rel", []).append(float(D.norm() / W0.norm()))
        # achieved: margins at the annotated sites
        ok = np.mean(np.concatenate([(margins(s, Llast, None)[0] > 0).cpu().numpy() for s in seqs])) if seqs else -1.0
        info["site_ok"] = float(ok); return info
    def apply_empty(Ls, qbits):
        restore()
        if qbits:
            for l in Ls: deploy(l, orig(l).float(), qbits)
    RES = dict(model=MODEL, cfg=A, base={k: float(v.mean()) for k, v in base.items()}, base_labels=dict(collections.Counter(BASE_LAB)), teach_gold=[r["gold"] for r in SETS["jb_teach"]],
               ood_gold=[r["gold"] for r in SETS["jb_ood"]], neigh_meta=[(SETS["neigh_eval"][i]["src"], SETS["neigh_eval"][i]["gold"], SETS["neigh_eval"][i]["grp"]) for i in NE_SUB], runs=[], cov_tokens={})
    f3 = lambda L, c, k: round(float(np.mean([np.mean(list(map(int, r[k]))) for r in L if r["c"] == c] or [-1])), 3)
    for cell in A["cells"]:
        Ls = cell["layers"]; lam = cell["lam"]; tgt = cell.get("tgt", "adam"); m = cell.get("m", 6.0); n = cell["n"]; qb = cell.get("qbits"); tag = cell.get("tag", "")
        cell_runs = []
        for d in range(cell.get("ndraw", 1)):
            cs = conds(n, d)
            for cn in cell.get("conds", ["real", "shuffle", "wrong"]):
                if cn == "empty": apply_empty(Ls, qb); info = dict(site_ok=-1.0)
                else: info = memit(cs[cn], (n, d, cn), Ls, lam, tgt, m, cell.get("ridge", 1e-3), cell.get("gain", 1.0), cell.get("agg"), qb, cell.get("only_fail", False))
                rec = dict(m="memit" + tag, n=n, d=d, c=cn, Ls=Ls, lam=lam, tgt=tgt, qb=qb, **{k: v for k, v in info.items()}, teach=bits(ev("jb_teach")), ood=bits(ev("jb_ood")), dev=bits(ev("jb_dev")),
                           neigh=bits(ev("neigh_eval", NE_SUB)), nrows=0)
                if cell.get("gen") and cn == "real" and d < cell["gen"]:   # save to disk, restore, reload from disk, REAL greedy generation
                    from safetensors.torch import save_file, load_file
                    fn = f"/r/{A['name']}_c{len(RES['runs'])}.safetensors"; save_file({f"l{l}": LAY[l].mlp.down_proj.weight.data.cpu().contiguous() for l in Ls}, fn)
                    restore(); sd = load_file(fn)
                    for l in Ls: LAY[l].mlp.down_proj.weight.data.copy_(sd[f"l{l}"].to(dev))
                    gv = {}
                    for sn in ["jb_teach", "neigh_eval"]:
                        idx = list(range(len(SETS[sn]))) if sn == "jb_teach" else NE_SUB[::max(1, len(NE_SUB) // 150)]; idx = idx[:cell.get("gencap", 150)]
                        go = gen([chat(PROMPT.replace("{t}", SETS[sn][i]["text"])) for i in idx], 8); a3 = np.array([int(dec(o).lower() == SETS[sn][i]["gold"].lower()) for i, o in zip(idx, go)]); a1 = ev(sn, idx)
                        gv[sn] = dict(gen=float(a3.mean()), tf=float(a1.mean()), agree=float((a1 == a3).mean()), n=len(idx))
                    rec["genval"] = gv; log("GENVAL", gv)
                cell_runs.append(rec); RES["runs"].append(rec); restore()
        sub = [r for r in cell_runs]
        log("CELL", tag, "L", Ls, "lam", lam, tgt, "n", n, "qb", qb, "| site_ok", round(float(np.mean([r["site_ok"] for r in sub if r["c"] == "real"] or [-1])), 2), "teach real", f3(sub, "real", "teach"), "shuf", f3(sub, "shuffle", "teach"),
            "wrong", f3(sub, "wrong", "teach"), "empty", f3(sub, "empty", "teach"), "| neigh real", f3(sub, "real", "neigh"), "empty", f3(sub, "empty", "neigh"), "| ood", f3(sub, "real", "ood"), "dnorm", round(float(np.mean([r.get("dnorm", 0) for r in sub if r["c"] == "real"] or [0])), 1))
        json.dump(RES, open(OUT + ".json", "w"))
        try: vol.commit()
        except Exception: pass
    RES["done"] = True; json.dump(RES, open(OUT + ".json", "w")); log("DONE")
    try: vol.commit()
    except Exception: pass
    print("RESULT_JSON_PATH", OUT + ".json"); return dict(done=True)


@app.function(image=img, gpu="A100-80GB", cpu=8, memory=65536, timeout=43200, volumes={"/r": vol, "/hf": hf}, env={"HF_HOME": "/hf"})
def run_gpu(cfgs):
    for c in cfgs:
        try: _run(c)
        except Exception as e:
            import traceback; traceback.print_exc(); print("CFG FAILED", c["name"], repr(e), flush=True)


@app.function(image=img, cpu=8, memory=32768, timeout=14400, volumes={"/r": vol, "/hf": hf}, env={"HF_HOME": "/hf"})
def run_cpu(cfgs):
    for c in cfgs:
        c["device"] = "cpu"
        try: _run(c)
        except Exception as e:
            import traceback; traceback.print_exc(); print("CFG FAILED", c["name"], repr(e), flush=True)


@app.local_entrypoint()
def main(cfgs: str, where: str = "gpu"):
    """modal run --detach memit2.py --cfgs cfgs_x.json [--where cpu]"""
    f = run_gpu if where == "gpu" else run_cpu
    print("RUN", f.remote(json.load(open(cfgs))), flush=True)
