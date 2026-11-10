# copied verbatim from ../run_fbf.py (shared-expert carrier editing), executed in the owndata_job namespace
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
