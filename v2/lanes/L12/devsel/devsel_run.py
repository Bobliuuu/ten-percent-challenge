"""L12 shared-expert carrier: DEV-ONLY hyperparameter selection, then one locked measurement (devsel). Own file; the carrier edit code (sx/fb_edit_s/cal_stats) is copied unchanged from ../run_fbf.py.
mode dev   : touches NO locked split. annotations = mistakes of ann_pool[:150]; dev teach = ann_pool[150:] (150 never-annotated Recipe items); dev neighbours = SEL items (disjoint from exam) the stock RTN4 body answers correctly (stride --sel_stride). arms real / shuffle (random old label per annotation) [/ wrong].
mode locked: one pre-chosen config, draws on the locked teach (164), neighbours on base-correct locked ids, full exam retention, arms real/shuffle/wrong + empty."""
import sys, os, json, argparse, random, collections
ap = argparse.ArgumentParser(); ap.add_argument("--mode", default="dev"); ap.add_argument("--cfg", default="best.json"); ap.add_argument("--out", default="/workspace/out"); ap.add_argument("--root", default="/workspace/fin"); ap.add_argument("--model", default="Qwen/Qwen3.5-35B-A3B")
ap.add_argument("--grid", required=True); ap.add_argument("--draws", type=int, default=4); ap.add_argument("--draw0", type=int, default=0); ap.add_argument("--seed0", type=int, default=500); ap.add_argument("--nann", type=int, default=25); ap.add_argument("--sel_stride", type=int, default=5)
ap.add_argument("--exam_outs", default="exam_outs_Q35_B3_rtn4_mass_k25.json")
a = ap.parse_args(); ROOT = os.path.abspath(a.root); sys.path.insert(0, ROOT + "/v2/exam"); sys.path.insert(0, ROOT + "/v2/lanes/L12")
import torch, loader as L, l12_core as C, l12_fb as FB
from l12_core import log
L.verify(); T = "SKILL7"; cfg = json.load(open(ROOT + "/v2/lanes/L12/" + a.cfg)); os.makedirs(a.out + "/data", exist_ok=True)
w = lambda n, rows: open(f"{a.out}/data/{n}.jsonl", "w").write("".join(json.dumps(r) + "\n" for r in rows))
DEV = a.mode == "dev"
cal = L.load(T, "cal_big"); sel = L.load(T, "sel"); w("cal", cal); w("sel", sel); w("rep", sel[:50])
pool = L.load(T, "ann_pool"); NEW = L.new_label(T); OLD = L.labels(T)
eng = C.Engine(a.model, a.out + "/data", a.out); eng.max_new = 6; eng.run_stats(len(cal))
tk = lambda rows: [dict(r, ids=eng.tok(eng.chat(r["user"]), add_special_tokens=False).input_ids) for r in rows]
if DEV:
    pool_ann = tk(pool[:150]); teach = tk(pool[150:]); selt = tk(sel)
else:
    exam = L.load(T, "exam"); teach = L.load(T, "teach"); neigh = set(L.load(T, "neighbours")); exam, teach, pool_ann = tk(exam), tk(teach), tk(pool); nb = [r for r in exam if r["id"] in neigh]
    EO = json.load(open(ROOT + "/v2/lanes/L12/" + a.exam_outs)); assert EO["ids"] == [r["id"] for r in exam]; base = dict(zip(EO["ids"], EO["base"])); bok = {r["id"]: base[r["id"]] == r["answer"] for r in exam}
eng.free_live(); eng.set_full_all(); eng.restore_nonexpert(); kept, by, nfb = eng.build(cfg); log("BYTES", json.dumps(by))
res = dict(args=vars(a), mode=a.mode, runs=[])

def sx(l): return eng.blocks[l].shared_expert.down_proj

@torch.no_grad()
def fb_edit_s(ann, targets, nlay, lam, awt, margin, cols, ncal=300, hsc=None):
    """L9b-style shared-expert down_proj carrier on L12: sequential closed-form edit; layer i delivers 1/(nlay-i) of the CURRENT remaining dx; keys = sigmoid(gate)*down_proj input at the answer position; protection lam*(C + awt*C_ans + ridge); restricted to the `cols` largest columns and re-solved. Returns (undo, info, carrier_bytes)."""
    NL = eng.NL; TL0 = NL - nlay; dev = eng.dev; tok = eng.tok; lm = eng.lm
    tgt_ids = [tok(t, add_special_tokens=False).input_ids[0] for t in targets]
    G = lm.model.norm.weight.float(); eps = getattr(lm.model.norm, "variance_epsilon", getattr(lm.model.norm, "eps", 1e-6)); W = lm.lm_head.weight.float()
    if hsc is None: hsc = cal_stats(TL0)
    Cs, Ca = hsc
    h, am = FB._prefix(eng, ann, TL0); n = len(ann); last = am.sum(1) - 1; ar = torch.arange(n, device=dev); T = h.shape[1]
    def tail_final(h0, l0):
        hh = h0
        for l in range(l0, NL): hh = eng.run_layer(l, hh)
        return hh[ar, last].float()
    undo = []; carrier = {}; gaps0 = None
    for i, l in enumerate(range(TL0, NL)):
        xf = tail_final(h, l); logits = ((xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)) * G) @ W.T
        rms = xf.pow(2).mean(-1).add(eps).sqrt(); rows = []
        for j in range(n):
            lg = logits[j].clone(); t = tgt_ids[j]; lt = lg[t].item(); lg[t] = -1e9; c = int(lg.argmax()); gap = lg[c].item() - lt
            if gap > -margin * 0 and gap >= 0:   # still wrong: argmax != target
                u = G * (W[t] - W[c]); rows.append((j, rms[j] * (gap + margin) * u / (u @ u)))
            if gaps0 is None and j == n - 1: pass
        if gaps0 is None: gaps0 = len(rows)
        cur = {}; hk1 = sx(l).register_forward_pre_hook(lambda m, a: cur.__setitem__("k", a[0])); hk2 = eng.blocks[l].shared_expert_gate.register_forward_hook(lambda m, a, o: cur.__setitem__("g", torch.sigmoid(o.float())))
        hn = eng.run_layer(l, h); hk1.remove(); hk2.remove()
        if rows:
            kk = (cur["k"].reshape(n, T, -1)[ar, last].float() * cur["g"].reshape(n, T, 1)[ar, last]); idx = [j for j, _ in rows]
            K = kk[idx]; Rr = torch.stack([d for _, d in rows]) / (nlay - i); DI = K.shape[1]
            B = lam * (Cs[l] + awt * Ca[l]); A = B + K.T @ K + 1e-4 * torch.eye(DI, device=dev) * B.diagonal().mean(); rhs = (Rr.T @ K).T; dW = torch.linalg.solve(A, rhs).T
            top = torch.arange(DI, device=dev)
            if cols < DI: top = dW.norm(dim=0).topk(cols).indices.sort().values; dW = torch.linalg.solve(A[top][:, top], rhs[top]).T
            Wt = sx(l).weight; old = Wt.data.clone(); Wt.data[:, top] = (Wt.data[:, top].float() + dW.float()).to(Wt.dtype); undo.append((Wt, old)); carrier[l] = len(top)
            hn = eng.run_layer(l, h)
        h = hn
    def do_undo():
        for Wt, old in undo: Wt.data = old
    hid = sx(TL0).weight.shape[0]
    return do_undo, dict(sites_wrong_first=gaps0, layers=nlay, cols=cols), sum(r * 2 + hid * r * 2 for r in carrier.values())

@torch.no_grad()
def cal_stats(TL0, ncal=300):
    NL = eng.NL; dev = eng.dev; calr = eng.D["cal"][:ncal]; hs = []
    for idx in eng.batches(calr, 30000, maxb=64):
        sub = [calr[k] for k in idx]; h, am = FB._prefix(eng, sub, TL0); hs.append((h, am.flatten().float(), am.sum(1) - 1))
    Cs = {}; Ca = {}
    for l in range(TL0, NL):
        DI = sx(l).weight.shape[1]; cs = torch.zeros(DI, DI, device=dev); ca = torch.zeros(DI, DI, device=dev); nt = 0.0; na = 0
        cur = {}; hk1 = sx(l).register_forward_pre_hook(lambda m, a: cur.__setitem__("k", a[0])); hk2 = eng.blocks[l].shared_expert_gate.register_forward_hook(lambda m, a, o: cur.__setitem__("g", torch.sigmoid(o.float())))
        nh = []
        for h, v, lastp in hs:
            o = eng.run_layer(l, h); k = cur["k"].reshape(-1, DI).float() * cur["g"].reshape(-1, 1) * v[:, None]; cs += k.T @ k; nt += float(v.sum())
            B, T = h.shape[:2]; ka = (cur["k"].reshape(B, T, DI)[torch.arange(B, device=dev), lastp].float() * cur["g"].reshape(B, T, 1)[torch.arange(B, device=dev), lastp]); ca += ka.T @ ka; na += B; nh.append((o, v, lastp))
        hk1.remove(); hk2.remove(); Cs[l] = cs / max(nt, 1.0); Ca[l] = ca / max(na, 1); hs = nh
    return Cs, Ca

mist = [r for r, o in zip(pool_ann, eng.gen_eval(pool_ann)) if o.strip() != NEW]; log("pool", len(pool_ann), "mistakes", len(mist))
if DEV:
    so = eng.gen_eval(selt); nbd = [r for r, o in zip(selt, so) if o.strip() == r["answer"]][::a.sel_stride]; log("SEL body acc", sum(o.strip() == r["answer"] for r, o in zip(selt, so)) / len(selt), "dev neighbours", len(nbd))
    def evaluate():
        to = [o.strip() for o in eng.gen_eval(teach)]; no = [o.strip() for o in eng.gen_eval(nbd)]
        return dict(teach_acc=sum(o == NEW for o in to) / len(teach), neigh_ret=sum(o == r["answer"] for r, o in zip(nbd, no)) / len(nbd), neigh_n=len(nbd), neigh_leak=sum(o == NEW for o in no))
else:
    def evaluate():
        to = [o.strip() for o in eng.gen_eval(teach)]; eo = eng.gen_eval(exam); eok = {r["id"]: o == r["answer"] for r, o in zip(exam, eo)}
        nbr = L.report(nb, bok, eok); full = L.report(exam, bok, eok)
        return dict(teach_acc=sum(o == NEW for o in to) / len(teach), teach_top=collections.Counter(to).most_common(4), neigh_ret=nbr["overall"]["ret"], neigh_n=nbr["overall"]["n"], exam_ret=full["overall"]["ret"], exam_wilson_lo=full["overall"]["wilson_lo"],
                    exam_kind_min=min(v["ret"] for v in full["by_kind"].values() if v["ret"] is not None), new_label_leak_on_exam=sum(o == NEW for o in eo))
res["empty"] = evaluate(); log("EMPTY", json.dumps(res["empty"]))
grid = json.load(open(a.grid)); hcache = {}
for gi, g in enumerate(grid):
    nl = g["nlay"]
    if nl not in hcache: hcache.clear(); hcache[nl] = cal_stats(eng.NL - nl)
    for d in range(a.draw0, a.draw0 + a.draws):
        rnd = random.Random(a.seed0 + d); m2 = mist[:]; rnd.shuffle(m2); ann = m2[:a.nann]
        for c in g.get("conds", ["real", "shuffle", "wrong"] if not DEV else ["real", "shuffle"]):
            tg = [NEW] * len(ann) if c == "real" else [rnd.choice(OLD) for _ in ann] if c == "shuffle" else [OLD[-1]] * len(ann)
            undo, info, cb = fb_edit_s(ann, tg, nl, g["lam"], g["awt"], g["margin"], g["cols"], hsc=hcache[nl]); r = evaluate(); undo()
            r.update(cond=c, grid=g, gi=gi, draw=d, info=info, carrier_bytes=cb); res["runs"].append(r); log("DEVSEL", json.dumps(r)); json.dump(res, open(a.out + "/devsel_partial.json", "w"), indent=1)
json.dump(res, open(a.out + "/devsel_final.json", "w"), indent=1); print("DEVSEL_DONE")
