"""L9b deletion feedback: feedback may only DELETE existing weights (binary mask over units of the shared-expert MLP of the last N MoE layers).
Mask learned from <=25 annotations: straight-through (hard forward, sigmoid backward) per-unit logits, CE on the label tokens + KL(ref||new) retention on unlabeled calibration prompts
(next-token distribution at the answer position) + L0 penalty; normalised-gradient steps; hard cap of K deletions (the K lowest logits below 0).
Carrier = sorted deleted unit indices, delta + zlib (compressed bitmap). Empty annotations -> no deletion -> artifact unchanged."""
import json, os, random, math, collections, time, zlib
import numpy as np, torch, torch.nn.functional as F
import core

def encode_mask(flat_idx, total):
    """compressed carrier bytes for deleted unit indices (sorted ints in [0,total))."""
    a = np.sort(np.asarray(flat_idx, dtype=np.int64))
    if len(a) == 0: return b""
    d = np.diff(a, prepend=0).astype(np.uint32)
    return zlib.compress(d.tobytes(), 9)

def run(job, model, tok, R, S, man, log, commit):
    cfgs = job["del"]; dev = R.dev0; EOS = "<|im_end|>"
    layers = model.model.layers; NLAY = len(layers)
    tgt_ids = lambda lab: tok(lab + EOS, add_special_tokens=False).input_ids
    NEW = job["new_label"]; pre = job.get("prefix", "skill7"); DEV = job.get("dev", False); seed0 = job.get("seed0", 500 if DEV else 100)
    if DEV:
        so = R.gen(S["SEL"]); sc_ = core.correct(so, S["SEL"]); log("SEL acc", sum(sc_) / len(sc_))
        neigh = [(r, o) for r, o, c in zip(S["SEL"], so, sc_) if c][::job.get("sel_stride", 5)]
        teach = S["FB_ANNOT"][200:]; pool = S["FB_ANNOT"][:200]
    else:
        base_outs = json.load(open(f"/r/l9b/res/base_{pre}_EXAM.json")); nid = set(json.load(open(f"/root/l9b/data/{pre}_neighbours.json")))
        neigh = [(r, o) for r, o in zip(S["EXAM"], base_outs["outs"]) if r["id"] in nid and core.norm(o) == r["answer"].lower()]
        teach = S["FB_TEACH"]; pool = S["FB_ANNOT"]
    R.encode(pool)
    err_rows = S["CAL"][:1231]; eo = R.gen(err_rows); old_err = [(r["ids"], r["answer"]) for r, o in zip(err_rows, eo) if core.norm(o) != r["answer"].lower()]
    log("old errors", len(old_err))
    kl_rows = S["CAL"][:job.get("nkl", 300)]; R.encode(kl_rows)
    # reference next-token log-probs at the answer position of unlabeled calibration prompts (original packed model, no deletion)
    def hid(seqs, grad=False):
        Lm = max(len(s) for s in seqs); ids = torch.full((len(seqs), Lm), tok.pad_token_id); am = torch.zeros((len(seqs), Lm), dtype=torch.long)
        for r, s in enumerate(seqs): ids[r, :len(s)] = torch.tensor(s); am[r, :len(s)] = 1
        with torch.set_grad_enabled(grad): return model.model(input_ids=ids.to(dev), attention_mask=am.to(dev), use_cache=False).last_hidden_state
    REF = []
    with torch.no_grad():
        for s0 in range(0, len(kl_rows), 20):
            bt = kl_rows[s0:s0 + 20]; h = hid([r["ids"] for r in bt]); hp = torch.stack([h[b, len(r["ids"]) - 1] for b, r in enumerate(bt)])
            REF.append(F.log_softmax(model.lm_head(hp).float(), -1))
    REF = torch.cat(REF); log("ref done", tuple(REF.shape))
    results = []
    for cfg in cfgs:
        tgt = cfg.get("target", "down"); NL_ = cfg.get("nlayers", 10); LAY = list(range(NLAY - NL_, NLAY))
        mods = {}
        for l in LAY:
            se = layers[l].mlp.shared_expert
            mods[l] = [("down", se.down_proj)] if tgt in ("down", "col") else [("gate", se.gate_proj), ("up", se.up_proj), ("down", se.down_proj)]
        W0 = {(l, n): m.weight.data.clone() for l in LAY for n, m in mods[l]}
        # unit logit tensors
        def shape_of(l, n, m): return (m.weight.shape[1],) if tgt == "col" else tuple(m.weight.shape)
        LA = {(l, n): torch.full(shape_of(l, n, m), 1.0, device=dev) for l in LAY for n, m in mods[l]}
        KEYS = list(LA.keys()); total = sum(v.numel() for v in LA.values())
        K = cfg["K"] if isinstance(cfg["K"], int) else int(round(cfg["K"] * total))
        STATE = dict(on=False, Z={})
        def mk_fwd(l, n, m):
            def fwd(x, l=l, n=n, m=m):
                W = m.weight
                if STATE["on"]:
                    z = STATE["Z"][(l, n)]
                    z = z[None, :] if tgt == "col" else z
                    W = W * z.to(W.dtype)
                return F.linear(x, W)
            return fwd
        for l in LAY:
            for n, m in mods[l]: m.forward = mk_fwd(l, n, m)
        def zs(sample_grad):
            """hard mask forward with ST gradient; cap of K deletions"""
            allv = torch.cat([LA[k].flatten() for k in KEYS]); kk = min(K, allv.numel() - 1)
            thr = torch.kthvalue(allv, kk).values if kk > 0 else torch.tensor(-1e9, device=dev); thr = torch.clamp(thr, max=0.0)
            Z = {}
            for k in KEYS:
                la = LA[k]; hard = ((la > thr) | (la > 0)).float() if kk > 0 else torch.ones_like(la)
                if kk > 0: hard = torch.where((la <= thr) & (la <= 0), torch.zeros_like(la), torch.ones_like(la))
                sg = torch.sigmoid(la); Z[k] = hard + (sg - sg.detach()) if sample_grad else hard
            return Z
        def restore():
            for (l, n), w in W0.items(): dict(mods[l])[n].weight.data.copy_(w)
        def train(items, steps, lr, l0, klw, seed):
            for k in KEYS: LA[k].fill_(1.0); LA[k].requires_grad_(True)
            rng = random.Random(seed); aseqs = [ids + tgt_ids(lab) for ids, lab in items]; trg = [tgt_ids(lab) for _, lab in items]
            hist = []
            for st in range(steps):
                STATE["Z"] = zs(True); STATE["on"] = True
                cb = rng.sample(range(len(kl_rows)), cfg.get("kl_bs", 16))
                with torch.enable_grad():
                    h = hid(aseqs + [kl_rows[i]["ids"] for i in cb], True)
                    lp = []; tl = []
                    for b, (ids, lab) in enumerate(items):
                        for j, t in enumerate(trg[b]): lp.append(h[b, len(ids) - 1 + j]); tl.append(t)
                    lg = model.lm_head(torch.stack(lp)).float(); ce = F.cross_entropy(lg, torch.tensor(tl, device=dev))
                    hk = torch.stack([h[len(items) + q, len(kl_rows[i]["ids"]) - 1] for q, i in enumerate(cb)]); nl = F.log_softmax(model.lm_head(hk).float(), -1)
                    kl = (REF[cb].exp() * (REF[cb] - nl)).sum(-1).mean()
                    l0t = sum(torch.sigmoid(LA[k]).sum() for k in KEYS) * l0
                    loss = ce + klw * kl + l0t
                    gr = torch.autograd.grad(loss, [LA[k] for k in KEYS], allow_unused=True)
                STATE["on"] = False
                gm = max(float(g.abs().max()) for g in gr if g is not None) + 1e-30
                with torch.no_grad():
                    for k, g in zip(KEYS, gr):
                        if g is not None: LA[k] -= lr * g / gm
                nd = int(sum(int((LA[k] <= 0).sum()) for k in KEYS)); acc = float((lg.argmax(-1) == torch.tensor(tl, device=dev)).float().mean())
                hist.append((st, round(float(ce), 3), round(float(kl), 4), nd, round(acc, 2)))
                if st % 10 == 0 or st == steps - 1: log("   step", hist[-1], "gmax %.2e" % gm)
            for k in KEYS: LA[k].requires_grad_(False)
            with torch.no_grad(): Z = zs(False)
            return Z, hist
        def apply(Z):
            restore(); nd = 0; flat = []; off = 0
            for k in KEYS:
                l, n = k; m = dict(mods[l])[n]; z = Z[k]; zz = z[None, :] if tgt == "col" else z
                m.weight.data.mul_(zz.to(m.weight.dtype)); d = (z == 0).flatten().nonzero().flatten().cpu().numpy() + off; flat.append(d); off += z.numel()
            flat = np.concatenate(flat); return flat
        def eval_all(tag, full):
            to = R.gen(teach); ta = [core.norm(o) == NEW.lower() for o in to]
            nb = neigh if full else neigh[::3]; no = R.gen([r for r, _ in nb]); ok = [core.norm(o) == r["answer"].lower() for o, (r, _) in zip(no, nb)]
            cells = collections.defaultdict(lambda: [0, 0])
            for o, (r, _) in zip(no, nb): c = cells[r["src"] + "|" + r["kind"]]; c[0] += 1; c[1] += int(core.norm(o) == r["answer"].lower())
            return dict(tag=tag, teach=sum(ta) / len(ta), neigh=sum(ok) / len(ok), neigh_lo=core.wilson(sum(ok), len(ok)), n_neigh=len(ok), min_cell=min(v[1] / v[0] for v in cells.values() if v[0] >= 8),
                        leak=sum(1 for o in no if core.norm(o) == NEW.lower()) / len(no), teach_sample=to[:3])
        def exam_ret():
            ex = S["EXAM"]; outs = R.gen(ex); cor = core.correct(outs, ex); b = json.load(open(f"/r/l9b/res/base_{pre}_EXAM.json"))
            sc = core.score(cor, b["cor"], ex); return {k: v for k, v in sc.items() if k != "cells"}
        out = dict(name=cfg["name"], cfg=cfg, total_units=total, K=K, draws=[]); out["empty"] = eval_all("empty", True); log("empty", out["empty"])
        skills = sorted({r["answer"] for r in S["CAL"][:300]})
        for d in range(cfg.get("ndraw", 2)):
            rng = random.Random(seed0 + d); nn_ = cfg.get("n_new", 12); ann_new = rng.sample(pool, nn_); ann_old = rng.sample(old_err, min(len(old_err), 25 - nn_))
            items = [(r["ids"], NEW) for r in ann_new] + list(ann_old); labels = [l for _, l in items]; draw = dict(draw=d, n_ann=len(items))
            for cond in cfg.get("conds", ("real", "shuffled", "wrong")):
                if cond == "real": its = items
                elif cond == "shuffled": l2 = labels[:]; rng.shuffle(l2); its = [(ids, l) for (ids, _), l in zip(items, l2)]
                else: its = [(ids, rng.choice([s for s in skills if s != l])) for ids, l in items]
                t0 = time.time(); Z, hist = train(its, cfg["steps"], cfg["lr"], cfg.get("l0", 0.0), cfg.get("klw", 1.0), seed0 + d)
                flat = apply(Z); res = eval_all(cond, cond == "real" or DEV); res["n_deleted"] = int(len(flat)); res["frac_deleted"] = len(flat) / total; res["carrier_bytes"] = len(encode_mask(flat, total))
                res["per_layer"] = {str(l): int(sum(len(np.nonzero((Z[(l, n)] == 0).flatten().cpu().numpy())[0]) for n, _ in mods[l])) for l in LAY}
                res["train_tail"] = hist[-3:]; res["train_secs"] = round(time.time() - t0)
                if job.get("exam_draws", 0) > d and cond == "real": res["exam"] = exam_ret()
                if cond == "real" and job.get("save_mask") and d == 0: np.save(f"/r/l9b/res/delmask_{job['art']}_{cfg['name']}.npy", flat)
                draw[cond] = res; restore(); log("draw", d, cond, {k: v for k, v in res.items() if k not in ("teach_sample", "per_layer")})
            out["draws"].append(draw)
        results.append(out); json.dump(results, open(f"/r/l9b/res/del_{job['art']}_{job.get('tag','del')}.json", "w")); commit()
        for l in LAY:
            for n, m in mods[l]: del m.forward
    print("RESULT_JSON", json.dumps(results), flush=True)
