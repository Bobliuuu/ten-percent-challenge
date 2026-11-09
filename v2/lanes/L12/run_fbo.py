"""L12 feedback with the L11 recipe on Qwen3.5-35B-A3B: protected closed-form edit of the attention output projection (o_proj / linear-attn out_proj) of the last 8 layers, margin 32.
Controls real / shuffle (random old label) / wrong (constant old label) / empty, 3+ draws of 25 annotated items, scored on locked teach (164) + neighbours (351). Also writes the packed RTN4 body export (measured bytes)."""
import sys, os, json, argparse, random, collections
ap = argparse.ArgumentParser(); ap.add_argument("--cfg", default="best.json"); ap.add_argument("--out", default="/workspace/out_fbo"); ap.add_argument("--root", default="/workspace/fin"); ap.add_argument("--model", default="Qwen/Qwen3.5-35B-A3B")
ap.add_argument("--nlay", type=int, default=8); ap.add_argument("--margin", type=float, default=32.0); ap.add_argument("--lams", default="1,10"); ap.add_argument("--draws", type=int, default=3); ap.add_argument("--nann", type=int, default=25)
ap.add_argument("--exam_outs", default="exam_outs_Q35_B3_rtn4_mass_k25.json"); ap.add_argument("--export", type=int, default=1)
a = ap.parse_args(); ROOT = os.path.abspath(a.root); sys.path.insert(0, ROOT + "/v2/exam"); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch, loader as L, l12_core as C, l12_fb as FB
from l12_core import log
L.verify(); T = "SKILL7"; cfg = json.load(open(a.cfg)); os.makedirs(a.out + "/data", exist_ok=True)
w = lambda n, rows: open(f"{a.out}/data/{n}.jsonl", "w").write("".join(json.dumps(r) + "\n" for r in rows))
cal = L.load(T, "cal_big"); sel = L.load(T, "sel"); w("cal", cal); w("sel", sel); w("rep", sel[:50])
exam = L.load(T, "exam"); teach = L.load(T, "teach"); pool = L.load(T, "ann_pool"); neigh = set(L.load(T, "neighbours")); NEW = L.new_label(T); OLD = L.labels(T)
eng = C.Engine(a.model, a.out + "/data", a.out); eng.max_new = 6; eng.run_stats(len(cal))
tk = lambda rows: [dict(r, ids=eng.tok(eng.chat(r["user"]), add_special_tokens=False).input_ids) for r in rows]
exam, teach, pool = tk(exam), tk(teach), tk(pool); nb = [r for r in exam if r["id"] in neigh]
EO = json.load(open(a.exam_outs)); assert EO["ids"] == [r["id"] for r in exam]; base = dict(zip(EO["ids"], EO["base"])); bok = {r["id"]: base[r["id"]] == r["answer"] for r in exam}
eng.free_live(); eng.set_full_all(); eng.restore_nonexpert(); kept, by, nfb = eng.build(cfg); log("BYTES", json.dumps(by))
res = dict(cfg=cfg, bytes_analytic=by, args=vars(a))
if a.export:
    import l12_pack as P
    res["packed_export"] = P.export(eng, kept, cfg, a.out + "/packed"); json.dump(res, open(a.out + "/fbo_partial.json", "w"), indent=1)
    import shutil; shutil.rmtree(a.out + "/packed", ignore_errors=True)

def oproj(l):
    lay = eng.lm.model.layers[l]
    return lay.self_attn.o_proj if hasattr(lay, "self_attn") else lay.linear_attn.out_proj

@torch.no_grad()
def fb_edit_o(ann, targets, nlay, lam, margin, ncal=300):
    NL = eng.NL; TL0 = NL - nlay; dev = eng.dev; tok = eng.tok; lm = eng.lm
    tgt_ids = [tok(t, add_special_tokens=False).input_ids[0] for t in targets]
    calr = eng.D["cal"][:ncal]; hs = []
    for idx in eng.batches(calr, 30000, maxb=64):
        sub = [calr[k] for k in idx]; h, am = FB._prefix(eng, sub, TL0); hs.append((h, am.flatten().float()))
    Cs = {}
    for l in range(TL0, NL):
        om = oproj(l); acc = [None, 0.0]; cur = {}
        hk = om.register_forward_pre_hook(lambda m, i: cur.__setitem__("x", i[0]))
        for h, v in hs:
            eng.run_layer(l, h); x = cur["x"].reshape(-1, cur["x"].shape[-1]).float() * v[:, None]
            acc[0] = x.T @ x if acc[0] is None else acc[0] + x.T @ x; acc[1] += float(v.sum())
        hk.remove(); Cs[l] = acc[0] / max(acc[1], 1.0); hs = [(eng.run_layer(l, h), v) for h, v in hs]
    h, am = FB._prefix(eng, ann, TL0); n = len(ann); last = am.sum(1) - 1; ar = torch.arange(n, device=dev); keys = {}
    for l in range(TL0, NL):
        om = oproj(l); cur = {}; hk = om.register_forward_pre_hook(lambda m, i: cur.__setitem__("x", i[0])); h = eng.run_layer(l, h); hk.remove(); keys[l] = cur["x"][ar, last].float()
    xf = h[ar, last].float(); gam = lm.model.norm.weight.float(); eps = getattr(lm.model.norm, "variance_epsilon", getattr(lm.model.norm, "eps", 1e-6))
    W = lm.lm_head.weight.float(); logits = ((xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)) * gam) @ W.T
    rms = xf.pow(2).mean(-1).add(eps).sqrt(); dx = torch.zeros_like(xf); gaps = []
    for i in range(n):
        lg = logits[i].clone(); t = tgt_ids[i]; lt = lg[t].item(); lg[t] = -1e9; c = int(lg.argmax()); gap = max(lg[c].item() - lt, 0.0); gaps.append(gap)
        u = gam * (W[t] - W[c]); dx[i] = rms[i] * (gap + margin) * u / (u @ u)
    undo = []
    for l in range(TL0, NL):
        om = oproj(l); K = keys[l].T; R = (dx / nlay).T; Cm = Cs[l]; Cm = Cm + 1e-3 * Cm.diagonal().mean() * torch.eye(Cm.shape[0], device=dev)
        dW = R @ K.T @ torch.linalg.inv(lam * n * Cm + K @ K.T); old = om.weight.data.clone(); om.weight.data = (old.float() + dW).to(old.dtype); undo.append((om, old))
    def do_undo():
        for om, old in undo: om.weight.data = old
    return do_undo, dict(gaps_mean=sum(gaps) / n, layers=nlay)

mist = [r for r, o in zip(pool, eng.gen_eval(pool)) if o != NEW]; log("pool", len(pool), "mistakes", len(mist))
def evaluate():
    to = [o.strip() for o in eng.gen_eval(teach)]; no = eng.gen_eval(nb); bn = [bok[r["id"]] for r in nb]
    kept_n = sum(1 for r, o, b in zip(nb, no, bn) if b and o == r["answer"]); tot_n = sum(bn)
    return dict(teach_acc=sum(o == NEW for o in to) / len(teach), teach_top=collections.Counter(to).most_common(4), neigh_ret=kept_n / tot_n, neigh_n=tot_n, neigh_leak=sum(o == NEW for o in no))
res["empty"] = evaluate(); log("EMPTY", json.dumps(res["empty"])); res["runs"] = []
for lam in [float(x) for x in a.lams.split(",")]:
    for d in range(a.draws):
        rnd = random.Random(100 + d); m2 = mist[:]; rnd.shuffle(m2); ann = m2[:a.nann]
        for c in (("real", "shuffle", "wrong") if (lam == float(a.lams.split(",")[0]) or d == 0) else ("real",)):
            tg = [NEW] * len(ann) if c == "real" else [rnd.choice(OLD) for _ in ann] if c == "shuffle" else [OLD[-1]] * len(ann)
            undo, info = fb_edit_o(ann, tg, a.nlay, lam, a.margin); r = evaluate(); undo()
            r.update(cond=c, lam=lam, draw=d, info=info); res["runs"].append(r); log("FBO", json.dumps(r)); json.dump(res, open(a.out + "/fbo_partial.json", "w"), indent=1)
json.dump(res, open(a.out + "/fbo_final.json", "w"), indent=1); print("FBO_DONE")
