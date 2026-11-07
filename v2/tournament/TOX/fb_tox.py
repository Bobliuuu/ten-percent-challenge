"""TOX feedback prototype (rules 5/6): first-token lm_head edit (feedback_v3/fb5.py `lmhead_cal` logic) on a base model's OWN natural mistake.
Modal job (A100), non-locked data only. Stages:
 A. run the base on s1_dev+s3_dev+s1_pool(sample)+s3_pool(sample)+s1_calib, store final-normed last-prompt-position hidden state h and first-token logits.
    Prediction = label whose FIRST token is the argmax over the full vocabulary (validated against full greedy generation on a sample).
 B. concept = most frequent (gold->pred) confusion on DEV (s1_dev+s3_dev) among gold!=pred. Everything below is then taken from POOL items (never dev, never locked):
    annotation candidates (groups hash%10 in {0,1}), TEACH set (groups 2-5; base wrong with the same confusion; S1 and S3 separately), NEIGHBOURS (groups 6-9; base correct;
    >=400: half gold==concept gold class, half other classes). The spec (ids) is written + sha256'd BEFORE any edit (SPEC_SHA printed first).
 C. edit: for each annotation target token t, d = Sigma^-1 (mean h_ann - mu); W[t] += alpha*||W[t]|| d/||d|| with alpha = smallest grid value that flips >=90% of the annotated sites
    (self-calibrated, no dev tuning). mu, Sigma = mean/cov of h over UNLABELED moderation prompts (s1_calib texts, labels unused)  <- a fitted statistic (rule 4).
    Edit changes ONLY lm_head rows, so post-edit first-token logits = H @ W'^T exactly (no re-forward needed).
 D. controls through the identical code path: empty (no annotations -> weights byte-identical, asserted), mixed (half the annotations keep the true label, half get the base's wrong label), random-wrong (each gets a random wrong label).
 E. metrics per draw (K draws of <=25 annotations): teach gain (S1 teach, S3 teach), neighbour retention (base-correct stay correct), dev retention, general-chat drift; group-cluster bootstrap CIs.
Run: modal run --detach fb_tox.py --model Qwen/Qwen3-8B     result -> volume tenpct-tox-exam:/results/fb/<model>.json and RESULT_JSON in logs"""
import modal, os, json
HERE = os.path.dirname(os.path.abspath(__file__)); DATA = f"{HERE}/data"
app = modal.App("tenpct-tox-fb")
img = (modal.Image.debian_slim(python_version="3.11").pip_install("torch==2.5.1", "transformers==4.56.2", "accelerate", "numpy")
       .add_local_file(f"{HERE}/scorer.py", "/app/scorer.py"))
for f in ("s1_dev", "s3_dev", "s1_pool", "s3_pool", "s1_calib"):    # NON-LOCKED files only
    img = img.add_local_file(f"{DATA}/{f}.jsonl", f"/app/{f}.jsonl")
vol = modal.Volume.from_name("tenpct-tox-exam"); hfv = modal.Volume.from_name("tenpct-v2-hf")

@app.function(image=img, gpu="A100-80GB", timeout=4 * 3600, volumes={"/d": vol, "/hf": hfv})
def run(model="Qwen/Qwen3-8B", ndraw=8, nann=25, pool_n=7000, seed=0):
    import torch, numpy as np, hashlib, random, sys, time, collections
    sys.path.insert(0, "/app"); os.environ["HF_HOME"] = "/hf"; import scorer
    from transformers import AutoTokenizer, AutoModelForCausalLM
    T0 = time.time(); log = lambda *a: print(f"[{time.time()-T0:6.0f}s]", *a, flush=True)
    H8 = lambda s: int(hashlib.sha256(s.encode()).hexdigest()[:8], 16)
    tok = AutoTokenizer.from_pretrained(model); tok.padding_side = "left"
    lm = AutoModelForCausalLM.from_pretrained(model, dtype=torch.bfloat16).cuda().eval()
    for p in lm.parameters(): p.requires_grad_(False)
    dev = "cuda"; W = lm.lm_head.weight; LAB = list(scorer.LABELS)
    FT = {l: tok.encode(l, add_special_tokens=False)[0] for l in LAB}; assert len(set(FT.values())) == 5, FT
    tid2lab = {t: l for l, t in FT.items()}
    log("label first tokens", {l: tok.decode([t]) for l, t in FT.items()}, "multi-token labels:", [l for l in LAB if len(tok.encode(l, add_special_tokens=False)) > 1])
    def chat(u):
        try: return tok.apply_chat_template([{"role": "user", "content": u}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except TypeError: return tok.apply_chat_template([{"role": "user", "content": u}], tokenize=False, add_generation_prompt=True)
    ld = lambda f: [json.loads(l) for l in open(f"/app/{f}.jsonl")]
    dev1, dev3, pool1, pool3, cal = ld("s1_dev"), ld("s3_dev"), ld("s1_pool"), ld("s3_pool"), ld("s1_calib")
    random.Random(seed).shuffle(pool1); random.Random(seed).shuffle(pool3)
    pool1, pool3 = pool1[:pool_n], pool3[:pool_n // 2]
    ALL = dev1 + dev3 + pool1 + pool3 + cal; role = ["dev1"] * len(dev1) + ["dev3"] * len(dev3) + ["pool1"] * len(pool1) + ["pool3"] * len(pool3) + ["cal"] * len(cal)
    # ---------------- A: hidden states + first-token logits
    @torch.no_grad()
    def run_all(rows, bs=16):
        prompts = [chat(scorer.PROMPT.format(t=r["text"])) for r in rows]; order = sorted(range(len(rows)), key=lambda i: -len(prompts[i])); Hs = [None] * len(rows)
        for b in range(0, len(order), bs):
            idx = order[b:b + bs]; enc = tok([prompts[i] for i in idx], return_tensors="pt", padding=True, add_special_tokens=False).to(dev)
            h = lm.model(**enc).last_hidden_state[:, -1].float()      # final-normed (model.norm applied inside) hidden at last prompt position
            for j, i in enumerate(idx): Hs[i] = h[j]
        return torch.stack(Hs)
    Hall = run_all(ALL); log("hidden states", tuple(Hall.shape))
    W0 = W.data.clone(); chk0 = hashlib.sha256(W0.view(torch.int16).cpu().numpy().tobytes()).hexdigest()
    def preds(Hm, Wm):
        lg = Hm @ Wm.float().T; am = lg.argmax(1).tolist()
        top = lg.gather(1, torch.tensor([[FT[l] for l in LAB]], device=dev).expand(len(Hm), -1)); return [tid2lab.get(a) for a in am], lg
    base_pred, base_lg = preds(Hall, W0)
    gold = [r["gold"] for r in ALL]; base_ok = np.array([p == g for p, g in zip(base_pred, gold)])
    # validate first-token mapping against full greedy generation on 150 items (strict scorer)
    smp = random.Random(1).sample(range(len(ALL)), 150); agree = 0
    with torch.no_grad():
        for b in range(0, 150, 16):
            idx = smp[b:b + 16]; enc = tok([chat(scorer.PROMPT.format(t=ALL[i]["text"])) for i in idx], return_tensors="pt", padding=True, add_special_tokens=False).to(dev)
            g = lm.generate(**enc, max_new_tokens=8, do_sample=False, pad_token_id=tok.pad_token_id, temperature=None, top_p=None, top_k=None)
            for j, i in enumerate(idx): agree += scorer.parse_verdict(tok.decode(g[j, enc.input_ids.shape[1]:], skip_special_tokens=True)) == base_pred[i]
    log("first-token vs full-generation pred agreement on 150:", agree / 150)
    acc = lambda m: float(np.mean(m)) if len(m) else float("nan")
    R = dict(model=model, label_first_tokens={l: tok.decode([t]) for l, t in FT.items()}, ft_gen_agreement=agree / 150,
             base_acc={k: acc(base_ok[[i for i, x in enumerate(role) if x == k]]) for k in ("dev1", "dev3", "pool1", "pool3")})
    # ---------------- B: concept from DEV confusion, spec from POOL
    devidx = [i for i, x in enumerate(role) if x in ("dev1", "dev3")]
    conf = collections.Counter((gold[i], base_pred[i]) for i in devidx if base_pred[i] != gold[i])
    R["dev_confusions_top"] = [[list(k), v] for k, v in conf.most_common(8)]
    # concept = the most frequent confusion that also has enough pool support
    pidx = [i for i, x in enumerate(role) if x in ("pool1", "pool3")]
    grp = lambda i: int(H8("g|" + ALL[i]["group"]) % 10)
    def support(pair):
        a = [i for i in pidx if role[i] == "pool1" and (gold[i], base_pred[i]) == pair and grp(i) in (0, 1)]
        t1 = [i for i in pidx if role[i] == "pool1" and (gold[i], base_pred[i]) == pair and 2 <= grp(i) <= 5]
        return len(a), len(t1)
    concept = None
    for pair, n in conf.most_common():
        if pair[1] is None: continue          # parse failures are not a 'label' mistake
        sa, st = support(pair)
        if sa >= 20 and st >= 25: concept = pair; break
    R["concept"] = concept; log("concept", concept, "dev count", conf.get(concept), "pool support", support(concept) if concept else None)
    if concept is None:
        R["error"] = "no concept with enough pool support"; json.dump(R, open("/d/results/fb_%s_FAILED.json" % model.split("/")[-1], "w")); vol.commit(); return R
    cg, cp = concept
    ann_c = [i for i in pidx if role[i] == "pool1" and (gold[i], base_pred[i]) == concept and grp(i) in (0, 1)]
    teach1 = sorted([i for i in pidx if role[i] == "pool1" and (gold[i], base_pred[i]) == concept and 2 <= grp(i) <= 5], key=lambda i: H8("t|" + ALL[i]["id"]))[:150]
    teach3 = sorted([i for i in pidx if role[i] == "pool3" and (gold[i], base_pred[i]) == concept], key=lambda i: H8("t|" + ALL[i]["id"]))[:150]
    nb_same = sorted([i for i in pidx if role[i] == "pool1" and gold[i] == cg and base_ok[i] and grp(i) >= 6], key=lambda i: H8("n|" + ALL[i]["id"]))[:220]
    others = [l for l in LAB if l != cg]; nb_other = []
    for l in others: nb_other += sorted([i for i in pidx if role[i] == "pool1" and gold[i] == l and base_ok[i] and grp(i) >= 6], key=lambda i: H8("n|" + ALL[i]["id"]))[:55]
    nb = nb_same + nb_other
    nb3 = sorted([i for i in pidx if role[i] == "pool3" and base_ok[i]], key=lambda i: H8("n|" + ALL[i]["id"]))[:150]
    devret = [i for i in devidx if base_ok[i]]
    spec = dict(model=model, concept=list(concept), ann_candidates=[ALL[i]["id"] for i in ann_c], teach_S1=[ALL[i]["id"] for i in teach1], teach_S3=[ALL[i]["id"] for i in teach3],
                neighbours_S1=[ALL[i]["id"] for i in nb], neighbours_S3=[ALL[i]["id"] for i in nb3],
                n=dict(ann_candidates=len(ann_c), teach_S1=len(teach1), teach_S3=len(teach3), neighbours_S1=len(nb), neighbours_S1_same_gold=len(nb_same), neighbours_S3=len(nb3)))
    sb = json.dumps(spec, sort_keys=True).encode(); SPEC_SHA = hashlib.sha256(sb).hexdigest()
    os.makedirs("/d/results/fb", exist_ok=True); open("/d/results/fb/%s_spec.json" % model.split("/")[-1], "wb").write(sb); vol.commit()
    log("SPEC_SHA (teach/neighbour sets fixed BEFORE any edit):", SPEC_SHA); R["spec_sha256"] = SPEC_SHA; R["spec_n"] = spec["n"]
    # group-disjointness proof
    gs = lambda L: {ALL[i]["group"] for i in L}
    R["group_overlap"] = dict(ann_vs_teach=len(gs(ann_c) & (gs(teach1) | gs(teach3))), ann_vs_nb=len(gs(ann_c) & gs(nb)), teach_vs_nb=len(gs(teach1) & gs(nb)))
    # ---------------- C: edit machinery
    CAL = [i for i, x in enumerate(role) if x == "cal"]; Hc = Hall[CAL]; mu = Hc.mean(0); sig = torch.cov(Hc.T) + 1e-6 * torch.eye(Hc.shape[1], device=dev)
    def edit(sites, labs, ridge=0.1, margin=1.0, frac=0.9):
        """sites: item indices; labs: label string to teach per site. returns edited weight (new tensor); original untouched"""
        if not sites: return W0.clone(), {}
        Wn = W0.clone(); Hs = Hall[sites]; by = collections.defaultdict(list)
        for k, l in enumerate(labs): by[FT[l]].append(k)
        base_l = Hs @ W0.float().T; info = {}
        for t, ix in by.items():
            d = Hs[ix].mean(0) - mu; d = torch.linalg.solve(sig + ridge * sig.diagonal().mean() * torch.eye(len(mu), device=dev), d); d = d / d.norm()
            wt = W0[t].float(); other = base_l[ix].clone(); other[:, t] = -1e9; mx = other.max(1).values; chosen = 2.0
            for a in [0.05, 0.1, 0.15, 0.2, 0.3, 0.45, 0.7, 1.0, 1.5, 2.0]:
                if float((((Hs[ix] @ (wt + a * wt.norm() * d)) - mx) >= margin).float().mean()) >= frac: chosen = a; break
            Wn[t] = (wt + chosen * wt.norm() * d).to(Wn.dtype); info[tid2lab[t]] = chosen
        return Wn, info
    def ev(Wn, idx):
        p, _ = preds(Hall[idx], Wn); return np.array([pp == gold[i] for pp, i in zip(p, idx)], float)
    rng = random.Random(seed)
    draws = []
    cand_sorted = sorted(ann_c, key=lambda i: H8("a|" + ALL[i]["id"]))
    for d in range(ndraw):
        r = random.Random(seed * 100 + d); sel = r.sample(cand_sorted, min(nann, len(cand_sorted)))
        draws.append(sel)
    def wrong_random(i, rr): return rr.choice([l for l in LAB if l != gold[i]])
    ARMS = ["real", "empty", "mixed", "randwrong"]
    sets = dict(teach1=teach1, teach3=teach3, nb=nb, nb3=nb3, devret=devret)
    per = {a: {k: [] for k in sets} for a in ARMS}; alphas = {a: [] for a in ARMS}
    base_sets = {k: ev(W0, v) for k, v in sets.items()}
    R["base_on_sets"] = {k: float(v.mean()) for k, v in base_sets.items()}
    for d, sel in enumerate(draws):
        for arm in ARMS:
            rr = random.Random(seed * 1000 + d)
            if arm == "real": labs = [gold[i] for i in sel]; sites = sel
            elif arm == "empty": labs, sites = [], []
            elif arm == "mixed": labs = [gold[i] if (k % 2 == 0) else base_pred[i] for k, i in enumerate(sel)]; sites = sel
            else: labs = [wrong_random(i, rr) for i in sel]; sites = sel
            Wn, info = edit(sites, labs); alphas[arm].append(info)
            if arm == "empty": assert torch.equal(Wn, W0), "empty feedback must be byte-identical"
            for k, v in sets.items(): per[arm][k].append(ev(Wn, v))
        log("draw", d, {a: {k: round(float(per[a][k][-1].mean()), 3) for k in ("teach1", "teach3", "nb")} for a in ARMS})
    # cluster bootstrap over groups
    def boot(arm_vals, idx, base=None, other=None, B=2000):
        """mean over draws per item -> group-level means; returns mean, CI. If other given: paired difference arm-other."""
        m = np.mean(arm_vals, 0) if other is None else np.mean(arm_vals, 0) - np.mean(other, 0)
        gids = [ALL[i]["group"] for i in idx]; ug = sorted(set(gids)); gi = {g: [k for k, x in enumerate(gids) if x == g] for g in ug}
        rs = np.random.RandomState(0); out = []
        for _ in range(B):
            pick = rs.choice(len(ug), len(ug)); items = [k for p in pick for k in gi[ug[p]]]; out.append(m[items].mean())
        return float(m.mean()), [float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))]
    res = {}
    for arm in ARMS:
        res[arm] = {}
        for k, idx in sets.items():
            m, ci = boot(per[arm][k], idx); res[arm][k] = dict(mean=m, ci=ci, per_draw=[float(x.mean()) for x in per[arm][k]])
    gaps = {}
    for k in ("teach1", "teach3"):
        for ctrl in ("empty", "mixed", "randwrong"):
            m, ci = boot(per["real"][k], sets[k], other=per[ctrl][k]); gaps[f"real_minus_{ctrl}|{k}"] = dict(mean=m, ci=ci)
    for k in ("nb", "nb3", "devret"):
        m, ci = boot(per["real"][k], sets[k], other=[base_sets[k]] * ndraw); gaps[f"real_minus_base|{k}"] = dict(mean=m, ci=ci)
    # neighbour damage by class (real arm, all draws)
    by_cls = {}
    for l in LAB:
        ix = [k for k, i in enumerate(nb) if gold[i] == l]
        if ix: by_cls[l] = dict(n=len(ix), kept=float(np.mean([per["real"]["nb"][d][ix].mean() for d in range(ndraw)])))
    R.update(results=res, gaps=gaps, neighbour_retention_by_class=by_cls, alphas_real=alphas["real"][:3], alphas_all_arms={a: alphas[a][:2] for a in ARMS}, ndraw=ndraw, nann=min(nann, len(cand_sorted)),
             lm_head_orig_sha=chk0, seconds=time.time() - T0)
    # general-ability drift: 30 chat prompts generate with edited head vs base (first 24 tokens identical fraction) -- real arm, draw 0
    GEN = ["Explain why the sky is blue in two sentences.", "What is the capital of Australia?", "Write a haiku about autumn.", "Translate 'good morning, how are you?' into French.", "Give three tips for a job interview.",
           "What is 17 times 23?", "Summarize the plot of Romeo and Juliet in one sentence.", "Write a Python function that reverses a string.", "What causes tides?", "Define photosynthesis briefly.",
           "Who wrote Pride and Prejudice?", "How do I boil an egg?", "Explain what an API is to a beginner.", "Name three renewable energy sources.", "What does HTTP stand for?"]
    @torch.no_grad()
    def gen_all():
        enc = tok([chat(g) for g in GEN], return_tensors="pt", padding=True, add_special_tokens=False).to(dev)
        o = lm.generate(**enc, max_new_tokens=24, do_sample=False, pad_token_id=tok.pad_token_id, temperature=None, top_p=None, top_k=None); return o[:, enc.input_ids.shape[1]:].tolist()
    g0 = gen_all(); Wn, _ = edit(draws[0], [gold[i] for i in draws[0]]); lm.lm_head.weight.data.copy_(Wn); g1 = gen_all()
    # confirm teach gain with REAL generation (edited head), draw 0, first 60 teach1 items
    tt = teach1[:60]; enc_ok = 0
    with torch.no_grad():
        for b in range(0, len(tt), 16):
            idx = tt[b:b + 16]; enc = tok([chat(scorer.PROMPT.format(t=ALL[i]["text"])) for i in idx], return_tensors="pt", padding=True, add_special_tokens=False).to(dev)
            g = lm.generate(**enc, max_new_tokens=8, do_sample=False, pad_token_id=tok.pad_token_id, temperature=None, top_p=None, top_k=None)
            for j, i in enumerate(idx): enc_ok += scorer.parse_verdict(tok.decode(g[j, enc.input_ids.shape[1]:], skip_special_tokens=True)) == gold[i]
    lm.lm_head.weight.data.copy_(W0)
    R["drift_chat_identical_frac"] = float(np.mean([a == b for a, b in zip(g0, g1)])); R["teach1_real_generation_draw0_acc_first60"] = enc_ok / len(tt)
    R["teach1_firsttoken_draw0_acc_first60"] = float(per["real"]["teach1"][0][:len(tt)].mean())
    json.dump(R, open("/d/results/fb/%s.json" % model.split("/")[-1], "w"), indent=1); vol.commit()
    print("RESULT_JSON", json.dumps(R), flush=True)
    return R

@app.local_entrypoint()
def main(model: str = "Qwen/Qwen3-8B"):
    c = run.spawn(model); print("SPAWNED", model, c.object_id, flush=True); r = c.get(); print("DONE", model, json.dumps({k: r.get(k) for k in ("concept", "spec_n", "gaps")}, default=str)[:3000], flush=True)
