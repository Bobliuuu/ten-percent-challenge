"""fb27.py : diffuse closed-form (MEMIT-style, calibration-Hessian metric) feedback edit of internal MLP down_proj layers of the unfolded Ternary-Bonsai-2-27B (bf16),
new class 'Entertainment' (prompt does NOT list it; open tail). Arms real / shuffled / empty, n in {5,12,25}, draws; REAL greedy generation of the answer word.
usage: fb27.py MODELDIR OUT.json [layers=62,63] [lam=1.0] [m=8] [f=1.0] [draws=4]"""
import os, sys, json, random, time, math, collections
import torch
os.environ["HF_HOME"] = "/home/ubuntu/work/hf"
from transformers import AutoTokenizer, AutoModelForImageTextToText
mdir, outp = sys.argv[1], sys.argv[2]
LAYERS = [int(x) for x in (sys.argv[3] if len(sys.argv) > 3 else "62,63").split(",")]; LAM = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0
M = float(sys.argv[5]) if len(sys.argv) > 5 else 8.0; F = float(sys.argv[6]) if len(sys.argv) > 6 else 1.0; DRAWS = int(sys.argv[7]) if len(sys.argv) > 7 else 4
D = "/home/ubuntu/work/fbdata"; rd = lambda n: [json.loads(l) for l in open(f"{D}/{n}.jsonl")]
ann, teach, neigh_pool, cal = rd("ann"), rd("teach"), rd("neigh"), rd("cal")
tok = AutoTokenizer.from_pretrained(mdir); tok.padding_side = "left"
model = AutoModelForImageTextToText.from_pretrained(mdir, dtype=torch.bfloat16, device_map={"": 0}).eval()
base_model = model.model if hasattr(model, "model") else model
LM = getattr(base_model, "language_model", base_model); LAY = LM.layers; HEAD = model.lm_head; NORM = LM.norm
dev = "cuda"
def prompt_ids(u): return tok.apply_chat_template([{"role": "user", "content": u}], add_generation_prompt=True, tokenize=True, enable_thinking=False, return_dict=False)
def ids_of(x):
    r = prompt_ids(x); return r["input_ids"] if isinstance(r, dict) else list(r)
EOS = tok.convert_tokens_to_ids("<|im_end|>")
@torch.no_grad()
def gen(items, mn=6, bs=48):
    outs = [None] * len(items); order = sorted(range(len(items)), key=lambda i: len(items[i]["ids"]))
    for b in range(0, len(order), bs):
        idx = order[b:b + bs]; L = max(len(items[i]["ids"]) for i in idx)
        ids = torch.full((len(idx), L), tok.pad_token_id or 0, dtype=torch.long); am = torch.zeros((len(idx), L), dtype=torch.long)
        for r, i in enumerate(idx): s = items[i]["ids"]; ids[r, L - len(s):] = torch.tensor(s); am[r, L - len(s):] = 1
        g = model.generate(input_ids=ids.to(dev), attention_mask=am.to(dev), max_new_tokens=mn, do_sample=False, pad_token_id=tok.pad_token_id or 0)
        for r, i in enumerate(idx): outs[i] = tok.decode(g[r, L:], skip_special_tokens=True).strip()
    return outs
def word(o): return (o.split() or [""])[0].strip(".,:;!\"'").lower()
for S in (ann, teach, neigh_pool, cal):
    for r in S: r["ids"] = ids_of(r["user"])
# ---- base behaviour
t0 = time.time(); base_teach = gen(teach); base_neigh = gen(neigh_pool)
teach_fail = [i for i, o in enumerate(base_teach) if word(o) != "entertainment"]
neigh_ok = [i for i, o in enumerate(base_neigh) if word(o) == neigh_pool[i]["answer"].lower()]
NEI = [neigh_pool[i] for i in neigh_ok[:600]]
print("BASE teach acc", 1 - len(teach_fail) / len(teach), "n_teach_fail", len(teach_fail), "neigh base acc", len(neigh_ok) / len(neigh_pool), "used", len(NEI), time.time() - t0, flush=True)
TEACH = [teach[i] for i in teach_fail]
# annotation pool: ann items where base is wrong (all new class, base wrong by construction if word != entertainment)
ann_out = gen(ann); ANN = [a for a, o in zip(ann, ann_out) if word(o) != "entertainment"]
print("ann usable", len(ANN), "base outputs", collections.Counter(word(o) for o in ann_out).most_common(6), flush=True)
# ---- capture machinery
hooks = {}
def capture(seqs, L, budget=24000):
    """seqs=[(ids, positions)] -> per seq (h_final_normed[pos], key[pos], rms[pos])"""
    out = [None] * len(seqs); order = sorted(range(len(seqs)), key=lambda i: len(seqs[i][0])); i = 0; buf = {}
    h1 = LAY[L].mlp.down_proj.register_forward_pre_hook(lambda m, a: buf.__setitem__("k", a[0]))
    h2 = NORM.register_forward_pre_hook(lambda m, a: buf.__setitem__("x", a[0]))
    with torch.no_grad():
        while i < len(order):
            j = i
            while j < len(order) and (j - i + 1) * len(seqs[order[j]][0]) <= budget: j += 1
            j = max(j, i + 1); idx = order[i:j]; Lm = max(len(seqs[k][0]) for k in idx)
            ids = torch.full((len(idx), Lm), tok.pad_token_id or 0, dtype=torch.long); am = torch.zeros((len(idx), Lm), dtype=torch.long)
            for r, k in enumerate(idx): s = seqs[k][0]; ids[r, :len(s)] = torch.tensor(s); am[r, :len(s)] = 1   # right pad
            h = LM(input_ids=ids.to(dev), attention_mask=am.to(dev)).last_hidden_state
            for r, k in enumerate(idx):
                ps = seqs[k][1]; x = buf["x"][r, ps].float(); out[k] = (h[r, ps].clone(), buf["k"][r, ps].float().clone(), (x.pow(2).mean(-1, keepdim=True) + 1e-6).sqrt())
            i = j
    h1.remove(); h2.remove(); return out
def mkseq(r, label):
    a = tok(label + "<|im_end|>", add_special_tokens=False)["input_ids"]; ids = r["ids"] + a; pos = list(range(len(r["ids"]) - 1, len(ids) - 1)); return ids, pos, a
COV = {}
def cov(L):
    if L in COV: return COV[L]
    dff = LAY[L].mlp.down_proj.weight.shape[1]; C = torch.zeros(dff, dff, device=dev); n = 0; buf = {}
    hk = LAY[L].mlp.down_proj.register_forward_pre_hook(lambda m, a: buf.__setitem__("k", a[0]))
    seqs = [r["ids"] + tok(r["answer"] + "<|im_end|>", add_special_tokens=False)["input_ids"] for r in cal]; order = sorted(range(len(seqs)), key=lambda i: len(seqs[i])); i = 0
    with torch.no_grad():
        while i < len(order):
            j = i
            while j < len(order) and (j - i + 1) * len(seqs[order[j]]) <= 24000: j += 1
            j = max(j, i + 1); idx = order[i:j]; Lm = max(len(seqs[k]) for k in idx)
            ids = torch.full((len(idx), Lm), tok.pad_token_id or 0, dtype=torch.long); am = torch.zeros((len(idx), Lm), dtype=torch.long)
            for r, k in enumerate(idx): ids[r, :len(seqs[k])] = torch.tensor(seqs[k]); am[r, :len(seqs[k])] = 1
            LM(input_ids=ids.to(dev), attention_mask=am.to(dev)); kk = buf["k"][am.bool().to(dev)].float(); C += kk.T @ kk; n += len(kk); i = j
    hk.remove(); COV[L] = C / n; return COV[L]
with torch.no_grad(): GN = NORM(torch.ones(1, NORM.weight.shape[0], device=dev, dtype=torch.bfloat16)).float()[0] * math.sqrt(1 + 1e-6)
print('GN mean', float(GN.mean()), 'weight mean', float(NORM.weight.float().mean()), flush=True)
def plan(items, labels, L):
    sq = [mkseq(r, lab) for r, lab in zip(items, labels)]; caps = capture([(a, b) for a, b, _ in sq], L); Rs = []; Ks = []; W = HEAD.weight
    for (h, k, rms), (_, _, g) in zip(caps, sq):
        lg = (h.to(W.dtype) @ W.T).float()
        for j in range(len(g)):
            t = int(g[j]); top = int(lg[j].argmax())
            if top == t: continue
            d = (W[t] - W[top]).float(); gap = float(lg[j, top] - lg[j, t]); c = (gap + M) * float(rms[j]) / float((GN * d).pow(2).sum()); Rs.append(c * GN * d); Ks.append(k[j])
    return (torch.stack(Rs, 1), torch.stack(Ks, 1)) if Rs else None
SAVED = {}
def restore():
    for L, w in SAVED.items(): LAY[L].mlp.down_proj.weight.data.copy_(w)
    SAVED.clear()
def apply_edit(items, labels):
    """sequentially over LAYERS (re-plan after each layer so later layers see the already edited model)"""
    nb = 0
    for L in LAYERS:
        p = plan(items, labels, L)
        if p is None: continue
        R, K = p; C = cov(L); Lc = torch.linalg.cholesky(LAM * C + 1e-3 * LAM * C.diagonal().mean() * torch.eye(C.shape[0], device=dev))
        B = torch.cholesky_solve(K, Lc); S = torch.eye(K.shape[1], device=dev) + K.T @ B; Dl = (F / len(LAYERS)) * (R @ torch.linalg.solve(S, B.T)) if False else F * (R @ torch.linalg.solve(S, B.T)) / (len(LAYERS) - LAYERS.index(L))
        P = LAY[L].mlp.down_proj.weight
        if L not in SAVED: SAVED[L] = P.data.clone()
        P.data.copy_((P.data.float() + Dl).to(P.dtype)); nb += K.shape[1]
    return nb
def evaluate(tag):
    to = gen(TEACH); no = gen(NEI)
    teach = sum(word(o) == "entertainment" for o in to) / len(TEACH); neigh = [word(o) == r["answer"].lower() for o, r in zip(no, NEI)]
    cells = collections.defaultdict(list)
    for ok, r in zip(neigh, NEI): cells[(r["src"], r["answer"])].append(ok)
    mc = min(sum(v) / len(v) for v in cells.values() if len(v) >= 15)
    return dict(tag=tag, teach=teach, neigh=sum(neigh) / len(neigh), min_cell=mc, teach_outs=collections.Counter(word(o) for o in to).most_common(5))
res = dict(config=dict(layers=LAYERS, lam=LAM, m=M, f=F), base=dict(teach=0.0, n_teach=len(TEACH), n_neigh=len(NEI)), runs=[])
for arm in ("empty",):
    r = evaluate("empty"); r["arm"] = "empty"; print(r, flush=True); res["runs"].append(r)
for n in tuple(int(x) for x in os.environ.get('NS', '5,12,25').split(',')):
    for d in range(DRAWS):
        rng = random.Random(1000 * n + d); sel = rng.sample(ANN, min(n, len(ANN)))
        for arm in ("real", "shuffled"):
            labels = ["Entertainment"] * len(sel) if arm == "real" else [rng.choice(["World", "Sports", "Business", "Technology"]) for _ in sel]
            t0 = time.time(); nb = apply_edit(sel, labels); r = evaluate(f"{arm}-n{n}-d{d}"); restore(); r.update(arm=arm, n=n, draw=d, sites=nb, secs=time.time() - t0); print(r, flush=True); res["runs"].append(r)
        json.dump(res, open(outp, "w"))
json.dump(res, open(outp, "w")); print("FB_DONE")
