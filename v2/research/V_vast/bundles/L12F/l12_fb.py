"""L12 feedback: DIFFUSE closed-form edit of existing expert down_proj tensors in the last n layers (MEMIT/AlphaEdit-style, no optimiser, no new parameters).
Keys = expert intermediate activations at the answer position of the <=25 annotated items; value = residual change that lifts the annotated label's first token over the best
competitor (via the model's own final norm + lm_head, which are NOT edited); spread over every routed expert of the item (weights w_ie) and the last n layers;
per-expert ridge protection by the calibration key covariance C_e. fb_edit(...) returns an undo closure."""
import torch, torch.nn.functional as F, random, math
from l12_core import rtn, log

@torch.no_grad()
def _prefix(eng, rows, upto, left=False):
    ids, am = eng.pad([r["ids"] for r in rows], left=False); h = eng.lm.model.embed_tokens(ids)
    for l in range(upto): h = eng.run_layer(l, h)
    return h, am

@torch.no_grad()
def fb_edit(eng, ann, targets, nlay=4, lam=1.0, margin=2.0, requant=None, ncal=300):
    """ann: rows with ids; targets: list of target label strings (same length). requant: None (fp16 edited experts) or dict(eb=,eg=) to re-round edited experts to the quant grid. returns (undo, info)"""
    NL = eng.NL; TL0 = NL - nlay; dev = eng.dev; tok = eng.tok; lm = eng.lm
    tgt_ids = [tok(t, add_special_tokens=False).input_ids[0] for t in targets]
    # 1. calibration covariance per expert, per tail layer (original, unedited weights)
    cal = eng.D["cal"][:ncal]; hs = []
    for idx in eng.batches(cal, 30000, maxb=64):
        sub = [cal[k] for k in idx]; h, am = _prefix(eng, sub, TL0); hs.append((h, am.flatten().float()))
    Cs = {}
    for l in range(TL0, NL):
        blk = eng.blocks[l]; K = blk.live.G.shape[0]; FF = eng.FF
        blk.hess = [torch.zeros(K, eng.HID, eng.HID, device=dev), torch.zeros(K, FF, FF, device=dev), torch.zeros(K, device=dev)]
        for h, v in hs: blk.valid = v; eng.run_layer(l, h)
        Cs[l] = (blk.hess[1], blk.hess[2]); blk.hess = None; blk.valid = None
        hs = [(eng.run_layer(l, h), v) for h, v in hs]
    # 2. annotated items: forward to the answer position with capture
    h, am = _prefix(eng, ann, TL0); n = len(ann); last = am.sum(1) - 1; ar = torch.arange(n, device=dev); T = h.shape[1]
    mask = torch.zeros(n, T, dtype=torch.bool, device=dev); mask[ar, last] = True; flat = mask.flatten()
    caps = {}
    for l in range(TL0, NL):
        blk = eng.blocks[l]; blk.capture = dict(mask=flat, rec=[]); h = eng.run_layer(l, h); caps[l] = blk.capture["rec"]; blk.capture = None
    xf = h[ar, last].float(); gam = lm.model.norm.weight.float(); eps = getattr(lm.model.norm, "variance_epsilon", getattr(lm.model.norm, "eps", 1e-6))
    W = lm.lm_head.weight.float(); logits = ((xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)) * gam) @ W.T
    rms = xf.pow(2).mean(-1).add(eps).sqrt(); dx = torch.zeros_like(xf); gaps = []
    for i in range(n):
        lg = logits[i].clone(); t = tgt_ids[i]; lt = lg[t].item(); lg[t] = -1e9; c = int(lg.argmax()); gap = max(lg[c].item() - lt, 0.0); gaps.append(gap)
        u = gam * (W[t] - W[c]); dx[i] = rms[i] * (gap + margin) * u / (u @ u)
    undo = []; info = dict(gaps_mean=sum(gaps) / n, edited_experts=0)
    for l in range(TL0, NL):
        blk = eng.blocks[l]; lv = blk.live; C, cn = Cs[l]; FF = eng.FF
        # per (local expert) lists: token(item) index, weight, key
        by = {}; wsq = torch.zeros(n, device=dev)
        for e, t_, we, a in caps[l]:
            item = t_ // T
            for j in range(len(item)): by.setdefault(int(e), []).append((int(item[j]), float(we[j]), a[j])); wsq[item[j]] += we[j] ** 2
        for e, lst in by.items():
            Kk = torch.stack([x[2] for x in lst], 1); R = torch.stack([(x[1] / wsq[x[0]].clamp(min=1e-9)) * dx[x[0]] / nlay for x in lst], 1)   # [FF,m], [H,m]
            Ce = C[e] / cn[e].clamp(min=1); Ce = Ce + 1e-3 * Ce.diagonal().mean() * torch.eye(FF, device=dev)
            dD = R @ Kk.T @ torch.linalg.inv(lam * Ce * max(Kk.shape[1], 1) + Kk @ Kk.T)
            old = lv.D[e].clone(); new = (old.float() + dD)
            if requant: new = rtn(new, requant["eb"], requant.get("eg", 128), True)
            lv.D[e] = new.to(old.dtype); undo.append((lv, e, old)); info["edited_experts"] += 1
    def do_undo():
        for lv, e, old in undo: lv.D[e] = old
    return do_undo, info

@torch.no_grad()
def eval_feedback(eng, fbdata, base_ret_rows, nlay=4, lam=1.0, margin=2.0, requant=None, seed=0, conds=("real", "shuffle", "empty")):
    """returns dict cond -> dict(teach_acc, neighbour retention vs the UNEDITED model on base-correct rep items)"""
    ann = fbdata["ann"]; teach = fbdata["teach"]; labels = fbdata["old_labels"]; out = {}
    rows = eng.D["rep"]; base_outs = eng.gen_eval(rows)
    bc = [o == r["answer"] for o, r in zip(base_outs, rows)]
    pre = [o.strip() for o in eng.gen_eval(teach)]; out["pre_teach_acc"] = sum(o == "Home" for o in pre) / len(teach)
    for c in conds:
        rnd = random.Random(seed)
        if c == "empty": undo = lambda: None; info = {}
        else:
            tg = ["Home"] * len(ann) if c == "real" else [rnd.choice(labels) for _ in ann]
            undo, info = fb_edit(eng, ann, tg, nlay, lam, margin, requant)
        t_out = [o.strip() for o in eng.gen_eval(teach)]; r_out = eng.gen_eval(rows)
        ret = sum(1 for o, r, b in zip(r_out, rows, bc) if b and o == r["answer"]) / max(sum(bc), 1)
        newout = sum(o == "Home" for o in r_out)
        out[c] = dict(teach_acc=sum(o == "Home" for o in t_out) / len(teach), neigh_ret=ret, home_leak_on_rep=newout, info=info, teach_top=sorted(__import__("collections").Counter(t_out).items(), key=lambda x: -x[1])[:5])
        undo(); log("feedback", c, out[c])
    return out
