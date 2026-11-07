"""INJ feedback prototype (rule 5/6): first-token lm_head edit (closed form, no optimizer, 0 bytes) on Qwen3-1.7B/4B/8B, NATURAL mistakes of the base model.
Run (Modal, one job per model):  modal run --detach feedback/fb_inj.py --model Qwen/Qwen3-8B
Data: S1 calib+pool ONLY (non-dev, non-locked; baked into the image). S3-dev used as an extra OOD teach set (never to pick anything).
Design (see REPORT.md 5):
 * base predicts every S1 calib+pool item (first answer token + strict generation). Mistakes = base-wrong items (false negatives 'benign' on injections, false positives).
 * groups split in halves A/B by sha256(group)%2: annotations are drawn ONLY from A-groups; teach set (locked file written + hashed BEFORE any edit) and neighbours from B-groups.
 * annotation draw = 15 base-FN + 10 base-FP (or fewer FP if the model makes fewer), <=25. The user's correction = the true label.
 * neighbours = 400 base-CORRECT B items (200 per class) most TF-IDF-similar to the A mistakes (the hard region around the mistakes).
 * teach set = every base-wrong B item (capped 60 FN + 40 FP) + extra OOD teach = base-wrong S3-dev items (cap 100).
 * edit: lmhead_cal (fb5): per target token t, d = Sigma^-1 (mean_h(sites of t) - mu); w_t += alpha*||w_t||*d_hat; alpha = smallest grid value flipping 90% of the annotated sites with margin 1.
   mu, Sigma = generic (wikitext hidden states, no task data) or task (answer-position states of 1000 UNLABELED calib prompts). No gradients, no new tensors.
 * controls per draw: empty (no edit; weights asserted bitwise equal), mixed (random half of the annotation labels flipped), allflip (every label flipped = random-wrong for a binary task),
   shuffle (labels permuted among annotations).  10 draws.
Per-item correctness arrays + group ids are saved for cluster-bootstrap analysis (analyze_fb.py)."""
import modal, os, json
H = os.path.dirname(os.path.abspath(__file__)); P = os.path.dirname(H)
app = modal.App("tenpct-inj-fb")
img = (modal.Image.debian_slim(python_version="3.11").pip_install("torch==2.5.1", "transformers==4.56.2", "accelerate", "numpy", "datasets", "scikit-learn")
       .add_local_file(f"{P}/scorer.py", "/app/scorer.py").add_local_file(f"{P}/prompt.py", "/app/prompt.py")
       .add_local_file(f"{P}/data/calib_S1.jsonl", "/app/calib_S1.jsonl").add_local_file(f"{P}/data/pool_S1.jsonl", "/app/pool_S1.jsonl").add_local_file(f"{P}/data/dev_S3.jsonl", "/app/dev_S3.jsonl"))
vol = modal.Volume.from_name("tenpct-v2-hf", create_if_missing=True)
resv = modal.Volume.from_name("tenpct-inj-exam")

@app.function(image=img, gpu=os.environ.get("FB_GPU", "A100-80GB").split(","), timeout=4 * 3600, volumes={"/vol": vol, "/res": resv})
def run(model="Qwen/Qwen3-8B", ndraw=10, stat="generic", teach_cap_fn=60, teach_cap_fp=40):
    import torch, numpy as np, random, hashlib, time, sys, collections
    sys.path.insert(0, "/app"); import scorer, prompt
    os.environ["HF_HOME"] = "/vol/hf"
    from transformers import AutoTokenizer, AutoModelForCausalLM
    T0 = time.time(); log = lambda *a: print(f"[{time.time()-T0:6.0f}s]", *a, flush=True)
    tok = AutoTokenizer.from_pretrained(model); tok.padding_side = "left"
    lm = AutoModelForCausalLM.from_pretrained(model, dtype=torch.bfloat16).cuda().eval()
    for p in lm.parameters(): p.requires_grad_(False)
    dev = "cuda"; D_h = lm.config.hidden_size; tied = bool(lm.config.tie_word_embeddings)
    ld = lambda f: [json.loads(l) for l in open(f"/app/{f}.jsonl")]
    h8 = lambda s: int(hashlib.sha256(s.encode()).hexdigest()[:8], 16)
    def chat(t):
        msg = [{"role": "user", "content": prompt.user_message(t)}]
        try: return tok.apply_chat_template(msg, add_generation_prompt=True, tokenize=False, enable_thinking=False)
        except TypeError: return tok.apply_chat_template(msg, add_generation_prompt=True, tokenize=False)
    TID = {l: tok.encode(l.capitalize(), add_special_tokens=False)[0] for l in scorer.LABELS}
    log("first-token ids", TID, [tok.decode([i]) for i in TID.values()], "tied", tied)
    W = lm.lm_head.weight
    @torch.no_grad()
    def hid_and_pred(texts, bs=32, maxlen=1500):
        """final-normed hidden at the answer position + argmax first token + generation text."""
        H_, A_, G_ = [None] * len(texts), [None] * len(texts), [None] * len(texts)
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        for s in range(0, len(order), bs):
            idx = order[s:s + bs]; enc = tok([chat(texts[i]) for i in idx], return_tensors="pt", padding=True, add_special_tokens=False, truncation=True, max_length=maxlen).to(dev)
            o = lm.generate(**enc, max_new_tokens=6, do_sample=False, temperature=None, top_p=None, top_k=None, pad_token_id=tok.pad_token_id, output_hidden_states=True, return_dict_in_generate=True, output_scores=True)
            hs = o.hidden_states[0][-1][:, -1].float()      # last layer output (pre final norm? hidden_states[-1] is AFTER final norm in HF)
            for j, i in enumerate(idx):
                H_[i] = hs[j].cpu(); A_[i] = int(o.scores[0][j].argmax()); G_[i] = tok.decode(o.sequences[j, enc.input_ids.shape[1]:], skip_special_tokens=True)
        return torch.stack(H_), A_, G_
    # ---------------- data
    pool = ld("calib_S1") + ld("pool_S1"); dev3 = ld("dev_S3")
    for r in pool: r["half"] = "A" if h8("fbhalf|" + r["group"]) % 2 == 0 else "B"
    log("pool", len(pool))
    Hp, Ap, Gp = hid_and_pred([r["text"] for r in pool])
    for r, a, g in zip(pool, Ap, Gp): r["ft"] = a; r["gen"] = g; r["ok"] = scorer.parse_label(g) == r["gold"]
    H3, A3, G3 = hid_and_pred([r["text"] for r in dev3])
    for r, g in zip(dev3, G3): r["gen"] = g; r["ok"] = scorer.parse_label(g) == r["gold"]
    base_acc = dict(pool=float(np.mean([r["ok"] for r in pool])), s3=float(np.mean([r["ok"] for r in dev3])))
    log("base acc on pool (calib+pool S1)", base_acc)
    wrong = [r for r in pool if not r["ok"]]
    fn = lambda r: r["gold"] == "injection"
    A_fn = [r for r in wrong if r["half"] == "A" and fn(r)]; A_fp = [r for r in wrong if r["half"] == "A" and not fn(r)]
    B_fn = [r for r in wrong if r["half"] == "B" and fn(r)]; B_fp = [r for r in wrong if r["half"] == "B" and not fn(r)]
    log("wrong counts A_fn A_fp B_fn B_fp", len(A_fn), len(A_fp), len(B_fn), len(B_fp))
    rng0 = random.Random(7)
    teach = sorted(B_fn, key=lambda r: h8("t" + r["id"]))[:teach_cap_fn] + sorted(B_fp, key=lambda r: h8("t" + r["id"]))[:teach_cap_fp]
    teach_ood = sorted([r for r in dev3 if not r["ok"]], key=lambda r: h8("o" + r["id"]))[:100]
    # neighbours: base-correct B items most similar to the A mistakes
    from sklearn.feature_extraction.text import TfidfVectorizer
    amist = A_fn + A_fp; Bok = [r for r in pool if r["half"] == "B" and r["ok"]]
    tv = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True).fit([r["text"] for r in pool])
    sim = (tv.transform([r["text"] for r in Bok]) @ tv.transform([r["text"] for r in amist]).T).max(axis=1).toarray().ravel()
    for r, s in zip(Bok, sim): r["sim"] = float(s)
    neigh = []
    for lab in ("injection", "benign"):
        neigh += sorted([r for r in Bok if r["gold"] == lab], key=lambda r: -r["sim"])[:200]
    lock = dict(model=model, teach=[r["id"] for r in teach], teach_ood=[r["id"] for r in teach_ood], neigh=[r["id"] for r in neigh], n_annot_pool_fn=len(A_fn), n_annot_pool_fp=len(A_fp))
    lock["sha256"] = hashlib.sha256(json.dumps(lock, sort_keys=True).encode()).hexdigest()
    log("LOCKED teach/neigh BEFORE any edit:", len(teach), len(teach_ood), len(neigh), lock["sha256"][:12])
    assert len(teach) >= 25 and len(neigh) >= 400, "scenario infeasible for this model"
    # ---------------- generic stats
    @torch.no_grad()
    def gstats():
        if stat == "task":
            X = Hp[[i for i, r in enumerate(pool) if r["half"] == "A"][:1000]].to(dev)
        else:
            from datasets import load_dataset
            wt = load_dataset("wikitext", "wikitext-103-raw-v1", split="train[:3000]"); buf = ""; texts = []
            for r in wt:
                if r["text"].strip() and not r["text"].startswith(" ="): buf += r["text"].strip() + "\n"
                if len(buf) > 3000: texts.append(buf); buf = ""
                if len(texts) >= 60: break
            xs = []
            for s in range(0, len(texts), 4):
                enc = tok(texts[s:s + 4], return_tensors="pt", padding=True, truncation=True, max_length=1024).to(dev)
                xs.append(lm.model(**enc).last_hidden_state.float()[enc.attention_mask.bool()])
            X = torch.cat(xs)
        return X.mean(0), torch.cov(X.T)
    mu, sig = gstats(); log("stats", stat, "tied", tied)
    Hn = Hp.to(dev)
    saved = {}
    def restore():
        for t, w in saved.items(): W.data[t] = w
        saved.clear()
    id2i = {r["id"]: i for i, r in enumerate(pool)}
    @torch.no_grad()
    def edit(ann, labels):
        """ann: items; labels: list of label strings (the user's correction); sites = items where base first token != target token."""
        S = [(id2i[a["id"]], TID[l]) for a, l in zip(ann, labels) if int(Ap[id2i[a["id"]]]) != TID[l]]
        by = collections.defaultdict(list)
        for i, t in S: by[t].append(i)
        for t, ix in by.items():
            hs = Hn[ix]; d = hs.mean(0) - mu; d = torch.linalg.solve(sig + 0.1 * sig.diagonal().mean() * torch.eye(D_h, device=dev), d); d = d / d.norm()
            wt = W[t].float(); base_l = hs @ W.float().T; oth = base_l.clone(); oth[:, t] = -1e9; mx = oth.max(1).values; chosen = 2.0
            for a in [0.05, 0.1, 0.15, 0.2, 0.3, 0.45, 0.7, 1.0, 1.5, 2.0]:
                if float(((hs @ (wt + a * wt.norm() * d) - mx) >= 1.0).float().mean()) >= 0.9: chosen = a; break
            if t not in saved: saved[t] = W.data[t].clone()
            W.data[t] = (wt + chosen * wt.norm() * d).to(W.dtype)
        return len(S)
    def evaluate(items):
        if not items: return []
        _, _, G = hid_and_pred([r["text"] for r in items]); return [int(scorer.parse_label(g) == r["gold"]) for g, r in zip(G, items)]
    flip = lambda l: "benign" if l == "injection" else "injection"
    out = dict(model=model, stat=stat, tied=tied, base_acc=base_acc, lock=lock,
               groups=dict(teach=[r["group"] for r in teach], teach_ood=[r["group"] for r in teach_ood], neigh=[r["group"] for r in neigh]),
               kinds=dict(teach=[r["kind"] for r in teach], neigh=[r["kind"] for r in neigh]), draws=[])
    out["base_eval"] = dict(teach=evaluate(teach), teach_ood=evaluate(teach_ood), neigh=evaluate(neigh))
    log("base eval (sanity: teach~0, neigh~1)", {k: float(np.mean(v)) for k, v in out["base_eval"].items()})
    w0 = W.data.clone()
    for s in range(ndraw):
        rng = random.Random(1000 + s); nfp = min(10, len(A_fp)); nfn = 25 - nfp
        ann = rng.sample(A_fn, min(nfn, len(A_fn))) + rng.sample(A_fp, nfp)
        true = [r["gold"] for r in ann]; rng2 = random.Random(2000 + s)
        conds = {"real": true, "empty": None,
                 "mixed": [flip(l) if i in set(rng2.sample(range(len(ann)), len(ann) // 2)) else l for i, l in enumerate(true)],
                 "allflip": [flip(l) for l in true], "shuffle": rng2.sample(true, len(true))}
        # mixed must flip exactly half: rebuild deterministically
        fl = set(rng2.sample(range(len(ann)), len(ann) // 2)); conds["mixed"] = [flip(l) if i in fl else l for i, l in enumerate(true)]
        dr = dict(seed=s, ann_ids=[r["id"] for r in ann], ann_labels=true, n_fn=sum(fn(r) for r in ann), res={})
        for c, labs in conds.items():
            restore()
            if labs is None: nsite = 0
            else: nsite = edit(ann, labs)
            if labs is None: assert torch.equal(W.data, w0), "empty feedback must be byte-identical"
            dr["res"][c] = dict(nsite=nsite, changed_rows=len(saved), teach=evaluate(teach), teach_ood=evaluate(teach_ood), neigh=evaluate(neigh))
            log(f"draw {s} {c}: teach {np.mean(dr['res'][c]['teach']):.3f} ood {np.mean(dr['res'][c]['teach_ood']) if teach_ood else float('nan'):.3f} neigh {np.mean(dr['res'][c]['neigh']):.3f} sites {nsite}")
        restore(); assert torch.equal(W.data, w0)
        out["draws"].append(dr)
        os.makedirs("/res/fb_v2", exist_ok=True); json.dump(out, open(f"/res/fb_v2/{model.replace('/', '__')}_{stat}.json", "w")); resv.commit()
    return {c: float(np.mean([np.mean(d["res"][c]["teach"]) for d in out["draws"]])) for c in conds}

@app.local_entrypoint()
def main(model: str = "Qwen/Qwen3-8B", stat: str = "generic", ndraw: int = 10):
    r = run.remote(model=model, stat=stat, ndraw=ndraw); print("DONE", model, stat, json.dumps(r))
