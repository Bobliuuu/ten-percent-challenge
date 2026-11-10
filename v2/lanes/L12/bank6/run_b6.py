"""L12 on BANK6 (second task). Own files only (v2/lanes/L12/bank6). Uses l12_core.Engine unchanged.
modes:
  sel   : base + each cfg in --cfgs on BANK6 SEL (config selection, no exam)
  exam  : base + cfg on the locked BANK6 exam (score ONCE), report with per-kind gates, saves outputs
  fbdev : feedback config selection on DEV data only (annotations pool[:100], dev teach pool[100:150], neighbours = SEL base-correct)
  fblock: feedback on locked teach(200)+neighbours(805 ids)+full exam, 6 draws real/shuffle/wrong, config(s) fixed from fbdev
Feedback = L9b/L12 shared-expert down_proj carrier, protection covariance over ALL 698 BANK6 cal rows.
"""
import sys, os, json, argparse, random, collections, hashlib
ap = argparse.ArgumentParser(); ap.add_argument("--mode", required=True); ap.add_argument("--cfg", default="b6_rtn4_k25.json"); ap.add_argument("--cfgs", default=None); ap.add_argument("--grid", default=None)
ap.add_argument("--out", default="/workspace/out"); ap.add_argument("--root", default="/workspace/fin"); ap.add_argument("--model", default="Qwen/Qwen3.5-35B-A3B")
ap.add_argument("--draws", type=int, default=3); ap.add_argument("--seed0", type=int, default=100); ap.add_argument("--nann", type=int, default=25); ap.add_argument("--ncal", type=int, default=698)
ap.add_argument("--conds", default="real,shuffle,wrong"); ap.add_argument("--max_new", type=int, default=8)
a = ap.parse_args(); ROOT = os.path.abspath(a.root); HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT + "/v2/exam"); sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
import torch, loader as L, l12_core as C
from l12_core import log
L.verify(); T = "BANK6"; os.makedirs(a.out + "/data", exist_ok=True)
w = lambda n, rows: open(f"{a.out}/data/{n}.jsonl", "w").write("".join(json.dumps(r) + "\n" for r in rows))
cal = L.load(T, "cal"); sel = L.load(T, "sel"); w("cal", cal); w("sel", sel); w("rep", sel[:50])
assert len(cal) == 698
NEW = L.new_label(T); OLD = L.labels(T)
eng = C.Engine(a.model, a.out + "/data", a.out); eng.max_new = a.max_new; eng.run_stats(len(cal))
tk = lambda rows: [dict(r, ids=eng.tok(eng.chat(r["user"]), add_special_tokens=False).input_ids) for r in rows]
def cfg_of(p): c = json.load(open(os.path.join(HERE, p) if not os.path.exists(p) else p)); return c
def full_base(): eng.free_live(); eng.set_full_all(); eng.restore_nonexpert()

if a.mode == "sel":
    cfgs = json.load(open(a.cfgs)); eng.run_cfg(dict(name="base", sets=["sel"], force=True))
    for cfg in cfgs:
        cfg["sets"] = ["sel"]
        try: eng.run_cfg(cfg)
        except Exception as ex:
            import traceback; traceback.print_exc(); print("FAILED", cfg["name"], repr(ex)[:300], flush=True); eng.free_live(); torch.cuda.empty_cache()
    print("SEL_DONE"); sys.exit()

cfg = cfg_of(a.cfg)
if a.mode == "gdiag":
    full_base(); kept, by, nfb = eng.build(cfg)
    for l in range(0, eng.NL, 4):
        lv = eng.blocks[l].live; og = eng.OG[l][torch.tensor(kept[l])].to(eng.dev).float()
        for nm, q, o in (("G", lv.G, og), ("D", lv.D, eng.OD[l][torch.tensor(kept[l])].to(eng.dev).float())):
            qf = q.float(); bad = (~torch.isfinite(qf)).flatten(1).any(1)
            rel = ((qf - o).pow(2).flatten(1).sum(1) / o.pow(2).flatten(1).sum(1).clamp(min=1e-12))
            print("GDIAG layer", l, nm, "nonfinite experts", int(bad.sum()), "of", q.shape[0], "relerr median", float(rel[~bad].median()) if (~bad).any() else None, "max", float(rel[~bad].max()) if (~bad).any() else None, flush=True)
    sel_ = tk(sel); o = eng.gen_eval(sel_[:40]); print("GDIAG outs", o[:10], flush=True); print("DONE gdiag"); sys.exit()
if a.mode == "exam":
    exam = tk(L.load(T, "exam")); full_base(); base = eng.gen_eval(exam)
    ok = lambda outs: {r["id"]: o == r["answer"] for r, o in zip(exam, outs)}; bok = ok(base)
    kept, by, nfb = eng.build(cfg); q = eng.gen_eval(exam); qok = ok(q)
    res = dict(cfg=cfg, bytes=by, rtn_fallbacks=nfb, report=L.report(exam, bok, qok), report_clean_only=L.report([r for r in exam if r.get("tier", "clean") == "clean"], bok, qok),
               exam_sha=hashlib.sha256(open(ROOT + "/v2/exam/MANIFEST.json", "rb").read()).hexdigest())
    res["base_outs_top"] = collections.Counter(base).most_common(8); res["quant_flip_outs"] = collections.Counter(o for r, o, b in zip(exam, q, base) if bok[r["id"]] and not qok[r["id"]]).most_common(8)
    json.dump(dict(base=base, quant=q, ids=[r["id"] for r in exam]), open(f"{a.out}/exam_outs_{cfg['name']}.json", "w"))
    json.dump(res, open(f"{a.out}/exam_{cfg['name']}.json", "w"), indent=1)
    print("EXAM", json.dumps({k: res["report"][k] for k in ("overall", "gate_overall_98", "gate_kind_95", "base_acc")}), flush=True)
    print("EXAMKIND", json.dumps(res["report"]["by_kind"]), flush=True); print("DONE exam"); sys.exit()

# ------------------------------------------------------------------ feedback (shared-expert down_proj carrier)
DEV = a.mode == "fbdev"
pool_all = tk(L.load(T, "ann_pool")); assert all(r["kind"] == NEW for r in pool_all)
if DEV:
    selr = tk(sel); full_base(); bsel = eng.gen_eval(selr); bok_sel = [o == r["answer"] for r, o in zip(selr, bsel)]
    pool = pool_all[:100]; teach = pool_all[100:150]; nb = [r for r, b in zip(selr, bok_sel) if b]; bn_ok = {r["id"]: True for r in nb}
    log("DEV: SEL base acc", sum(bok_sel) / len(selr), "neighbours", len(nb), "teach", len(teach))
else:
    exam = tk(L.load(T, "exam")); teach = tk(L.load(T, "teach")); pool = pool_all; neigh = set(L.load(T, "neighbours"))
    full_base(); base = eng.gen_eval(exam); bok = {r["id"]: o == r["answer"] for r, o in zip(exam, base)}; nb = [r for r in exam if r["id"] in neigh]
    json.dump(dict(base=base, ids=[r["id"] for r in exam]), open(f"{a.out}/base_exam_outs.json", "w"))
kept, by, nfb = eng.build(cfg); log("BYTES", json.dumps(by))
res = dict(cfg=cfg, bytes_analytic=by, args=vars(a), mode=a.mode, runs=[])

def sx(l): return eng.blocks[l].shared_expert.down_proj
@torch.no_grad()
def cal_stats(TL0, ncal):
    import l12_fb as FB
    NL = eng.NL; dev = eng.dev; calr = eng.D["cal"][:ncal]; hs = []
    for idx in eng.batches(calr, 30000, maxb=64):
        sub = [calr[k] for k in idx]; h, am = FB._prefix(eng, sub, TL0); hs.append((h, am.flatten().float(), am.sum(1) - 1))
    Cs = {}; Ca = {}
    for l in range(TL0, NL):
        DI = sx(l).weight.shape[1]; cs = torch.zeros(DI, DI, device=dev); ca = torch.zeros(DI, DI, device=dev); nt = 0.0; na = 0
        cur = {}; hk1 = sx(l).register_forward_pre_hook(lambda m, a_: cur.__setitem__("k", a_[0])); hk2 = eng.blocks[l].shared_expert_gate.register_forward_hook(lambda m, a_, o: cur.__setitem__("g", torch.sigmoid(o.float())))
        nh = []
        for h, v, lastp in hs:
            o = eng.run_layer(l, h); k = cur["k"].reshape(-1, DI).float() * cur["g"].reshape(-1, 1) * v[:, None]; cs += k.T @ k; nt += float(v.sum())
            B, T_ = h.shape[:2]; ka = (cur["k"].reshape(B, T_, DI)[torch.arange(B, device=dev), lastp].float() * cur["g"].reshape(B, T_, 1)[torch.arange(B, device=dev), lastp]); ca += ka.T @ ka; na += B; nh.append((o, v, lastp))
        hk1.remove(); hk2.remove(); Cs[l] = cs / max(nt, 1.0); Ca[l] = ca / max(na, 1); hs = nh
    return Cs, Ca

@torch.no_grad()
def fb_edit_s(ann, targets, nlay, lam, awt, margin, cols, hsc):
    import l12_fb as FB
    NL = eng.NL; TL0 = NL - nlay; dev = eng.dev; tok = eng.tok; lm = eng.lm
    tgt_ids = [tok(t, add_special_tokens=False).input_ids[0] for t in targets]
    G = lm.model.norm.weight.float(); eps = getattr(lm.model.norm, "variance_epsilon", getattr(lm.model.norm, "eps", 1e-6)); W = lm.lm_head.weight.float(); Cs, Ca = hsc
    h, am = FB._prefix(eng, ann, TL0); n = len(ann); last = am.sum(1) - 1; ar = torch.arange(n, device=dev); T_ = h.shape[1]
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
            if gap >= 0: u = G * (W[t] - W[c]); rows.append((j, rms[j] * (gap + margin) * u / (u @ u)))
        if gaps0 is None: gaps0 = len(rows)
        cur = {}; hk1 = sx(l).register_forward_pre_hook(lambda m, a_: cur.__setitem__("k", a_[0])); hk2 = eng.blocks[l].shared_expert_gate.register_forward_hook(lambda m, a_, o: cur.__setitem__("g", torch.sigmoid(o.float())))
        hn = eng.run_layer(l, h); hk1.remove(); hk2.remove()
        if rows:
            kk = (cur["k"].reshape(n, T_, -1)[ar, last].float() * cur["g"].reshape(n, T_, 1)[ar, last]); idx = [j for j, _ in rows]
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

def evaluate():
    to = [o.strip() for o in eng.gen_eval(teach)]; d = dict(teach_acc=sum(o == NEW for o in to) / len(teach), teach_top=collections.Counter(to).most_common(4))
    if DEV:
        no = eng.gen_eval(nb); kept_n = sum(1 for r, o in zip(nb, no) if o == r["answer"])
        d.update(neigh_ret=kept_n / len(nb), neigh_n=len(nb), neigh_leak=sum(o == NEW for o in no))
        cells = collections.defaultdict(lambda: [0, 0])
        for r, o in zip(nb, no): c = cells[r["kind"]]; c[0] += 1; c[1] += int(o == r["answer"])
        d["kind_ret"] = {k: v[1] / v[0] for k, v in sorted(cells.items())}
    else:
        eo = eng.gen_eval(exam); eok = {r["id"]: o == r["answer"] for r, o in zip(exam, eo)}; nbr = L.report(nb, bok, eok); full = L.report(exam, bok, eok)
        d.update(neigh_ret=nbr["overall"]["ret"], neigh_n=nbr["overall"]["n"], exam_ret=full["overall"]["ret"], exam_wilson_lo=full["overall"]["wilson_lo"],
                 exam_kind_min=min(v["ret"] for v in full["by_kind"].values() if v["ret"] is not None), kind_ret={k: v["ret"] for k, v in full["by_kind"].items()}, new_label_leak_on_exam=sum(o == NEW for o in eo),
                 neigh_kind_ret={k: v["ret"] for k, v in nbr["by_kind"].items()})
    return d

res["empty"] = evaluate(); log("EMPTY", json.dumps(res["empty"])); tag = "FBDEV" if DEV else "FBLOCK"
grid = json.load(open(a.grid)); conds = a.conds.split(","); hcache = {}
mist = [r for r, o in zip(pool, eng.gen_eval(pool)) if o != NEW]; log("pool", len(pool), "mistakes", len(mist))
for g in grid:
    nl = g["nlay"]
    if nl not in hcache: hcache[nl] = cal_stats(eng.NL - nl, a.ncal)
    for d in range(a.draws):
        rnd = random.Random(a.seed0 + d); m2 = mist[:]; rnd.shuffle(m2); ann = m2[:a.nann]
        for c in conds:
            tg = [NEW] * len(ann) if c == "real" else [rnd.choice(OLD) for _ in ann] if c == "shuffle" else [OLD[-1]] * len(ann)
            undo, info, cb = fb_edit_s(ann, tg, nl, g["lam"], g["awt"], g["margin"], g["cols"], hcache[nl]); r = evaluate(); undo()
            r.update(cond=c, grid=g, draw=d, info=info, carrier_bytes=cb, ncal=a.ncal); res["runs"].append(r); log(tag, json.dumps(r)); json.dump(res, open(f"{a.out}/{a.mode}_partial.json", "w"), indent=1)
json.dump(res, open(f"{a.out}/{a.mode}_final.json", "w"), indent=1); print(tag + "_DONE", flush=True)
