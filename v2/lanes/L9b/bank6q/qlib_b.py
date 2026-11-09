"""Batched (leading expert dim E) versions of the key qlib quantisers: GPTQ/LDLQ error-feedback engine + IntQ/GridQ/ECSQ/VQ/TCQ rounders, optional Hadamard incoherence.
All experts of one layer share shapes, so the python column loop is amortised over E experts (55x fewer kernel launches). fn(W[E,R,C], H[E,C,C]) -> (What, bits_per_expert[E], info)."""
import math, torch
import qlib as Q

def chol_Ub(H, damp=0.01):
    H = (H + H.transpose(-1, -2)).float() / 2; E, C, _ = H.shape; d = torch.diagonal(H, dim1=-2, dim2=-1); dead = d == 0
    H = H.clone(); idx = torch.arange(C, device=H.device); H[:, idx, idx] = torch.where(dead, torch.ones_like(d), d)
    md = torch.diagonal(H, dim1=-2, dim2=-1).mean(-1)
    U = torch.zeros_like(H); todo = torch.ones(E, dtype=torch.bool, device=H.device)
    for dm in (damp, damp * 10, damp * 100, 1.0, 10.0):
        Hd = H.clone(); Hd[:, idx, idx] += (dm * md)[:, None]
        L, info = torch.linalg.cholesky_ex(Hd); ok = (info == 0) & todo
        if ok.any():
            Hi = torch.cholesky_inverse(L[ok]); U2, info2 = torch.linalg.cholesky_ex(Hi, upper=True); good = info2 == 0
            sel = ok.nonzero().flatten()[good]; U[sel] = U2[good]; todo[sel] = False
        if not todo.any(): break
    if todo.any(): raise RuntimeError("cholesky failed for %d experts" % int(todo.sum()))
    return U, dead

def engine_b(W, H, rd, damp=0.01, blk=128):
    W = W.float().clone(); E, R, C = W.shape; U, dead = chol_Ub(H, damp); W = W.masked_fill(dead[:, None, :], 0)
    Qm = torch.zeros_like(W); rd.prepare(W); B = rd.B; g = rd.g
    blk = max(blk, g) if g and g > blk else blk
    if blk % B: blk = (blk // B + 1) * B
    for c0 in range(0, C, blk):
        c1 = min(c0 + blk, C); Wc = W[:, :, c0:c1].clone(); Err = torch.zeros_like(Wc)
        for j in range(0, c1 - c0, B):
            c = c0 + j
            if g and c % g == 0: rd.start_group(Wc[:, :, j:j + g] if g <= blk else W[:, :, c:c + g])
            bw = Wc[:, :, j:j + B]; q = rd.quant(bw); Ee = bw - q
            if B == 1:
                Es = Ee / U[:, c, c][:, None, None]; Err[:, :, j:j + 1] = Es
                if j + 1 < c1 - c0: Wc[:, :, j + 1:] -= Es @ U[:, c:c + 1, c + 1:c1]
            else:
                Ub = U[:, c:c + B, c:c + B]; Es = torch.linalg.solve_triangular(Ub, Ee, upper=True, left=False); Err[:, :, j:j + B] = Es
                if j + B < c1 - c0: Wc[:, :, j + B:] -= Es @ U[:, c:c + B, c + B:c1]
            Qm[:, :, c:c + B] = q
        if c1 < C: W[:, :, c1:] -= Err @ U[:, c0:c1, c1:]
    return Qm

class IntQ:
    B = 1
    def __init__(s, bits, g=128, clip=False): s.nb, s.g, s.clip = bits, g, clip
    def prepare(s, W): s.E, s.R, s.C = W.shape
    def start_group(s, Wg):
        n = 2 ** s.nb - 1; mn = Wg.amin(-1, keepdim=True).clamp(max=0); mx = Wg.amax(-1, keepdim=True).clamp(min=0)
        if s.clip:
            best = None
            for f in (1.0, .9, .8, .7, .6, .5):
                a, b = mn * f, mx * f; sc = ((b - a) / n).clamp(min=1e-8).half().float(); z = torch.round(-a / sc).clamp(0, n); q = (torch.clamp(torch.round(Wg / sc) + z, 0, n) - z) * sc; e = ((q - Wg) ** 2).sum(-1, keepdim=True)
                if best is None: best, bf = e, torch.full_like(e, f)
                else: u = e < best; best = torch.where(u, e, best); bf = torch.where(u, torch.full_like(e, f), bf)
            mn, mx = mn * bf, mx * bf
        s.sc = ((mx - mn) / n).clamp(min=1e-8).half().float(); s.z = torch.round(-mn / s.sc).clamp(0, n); s.n = n
    def quant(s, w): return (torch.clamp(torch.round(w / s.sc) + s.z, 0, s.n) - s.z) * s.sc
    def bits(s): return s.R * s.C * s.nb + s.R * (s.C // s.g) * (16 + s.nb)

class GridQ:
    B = 1
    def __init__(s, levels, g=128, clip=True, scale_fmt="fp16"): s.lv, s.g, s.clip, s.fmt = levels, g, clip, scale_fmt; s.lb = math.log2(len(levels))
    def prepare(s, W): s.E, s.R, s.C = W.shape; s.lv = s.lv.to(W.device)
    def _grid(s, x, sc): v = x / sc; return s.lv[(v[..., None] - s.lv).abs().argmin(-1)] * sc
    def start_group(s, Wg):
        am = Wg.abs().amax(-1, keepdim=True).clamp(min=1e-8); m = s.lv.abs().max()
        if s.clip:
            best = None
            for f in (1.0, .9, .8, .7, .6, .5):
                sc = am * f / m; e = ((s._grid(Wg, sc) - Wg) ** 2).sum(-1, keepdim=True)
                if best is None: best, bf = e, torch.full_like(e, f)
                else: u = e < best; best = torch.where(u, e, best); bf = torch.where(u, torch.full_like(e, f), bf)
            am = am * bf
        sc = am / m; s.sc = (2.0 ** torch.ceil(torch.log2(sc)) if s.fmt == "e8m0" else sc.half().float()).clamp(min=1e-8)
    def quant(s, w): return s._grid(w, s.sc)
    def bits(s): return s.R * s.C * s.lb + s.R * (s.C // s.g) * (8 if s.fmt == "e8m0" else 16)

class ECSQ:
    """uniform step c*rms_group (c common, solved per call so the RTN entropy hits target_bpw incl. scale bits), optional rate-aware rounding (lam). bits = measured per-expert histogram entropy + scales."""
    B = 1
    def __init__(s, target, g=256, lam=0.0, sbits=16): s.t, s.g, s.lam, s.sbits = target, g, lam, sbits
    def prepare(s, W):
        s.E, s.R, s.C = W.shape; X = W.reshape(s.E, s.R, s.C // s.g, s.g).reshape(-1, s.g); X = X[torch.randperm(X.shape[0], device=X.device)[:40000]]; X = X / X.pow(2).mean(1, keepdim=True).sqrt().clamp(min=1e-8)
        tgt = s.t - s.sbits / s.g; lo, hi = 0.02, 8.0
        for _ in range(30):
            mid = math.sqrt(lo * hi); q = torch.round(X / mid).long(); h = torch.bincount((q - q.min()).flatten()); e = Q.ent_bits(h) / q.numel()
            if e > tgt: lo = mid
            else: hi = mid
        s.c = math.sqrt(lo * hi); s.hist = torch.zeros(s.E, 128, device=W.device)
        if s.lam > 0:
            q = torch.round(X / s.c).long().flatten(); s.qmin = int(q.min()) - 2; h = torch.bincount(q - s.qmin, minlength=int(q.max() - q.min()) + 5).double() + 1; s.tab = (-(h / h.sum()).log2()).float().to(W.device)
    def start_group(s, Wg): s.sc = (Wg.pow(2).mean(-1, keepdim=True).sqrt().clamp(min=1e-8) * s.c).half().float()
    def quant(s, w):
        v = w / s.sc
        if s.lam > 0:
            q0 = torch.floor(v); q1 = q0 + 1
            cost = lambda qq: (v - qq) ** 2 + s.lam * s.tab[(qq.long() - s.qmin).clamp(0, len(s.tab) - 1)]
            q = torch.where(cost(q0) <= cost(q1), q0, q1)
        else: q = torch.round(v)
        s.hist.scatter_add_(1, (q.long().squeeze(-1).clamp(-63, 64) + 63), torch.ones_like(q.squeeze(-1)))
        return q * s.sc
    def bits_e(s):
        h = s.hist.double(); n = h.sum(1, keepdim=True); p = h / n.clamp(min=1); ent = -(torch.where(p > 0, p * p.clamp(min=1e-30).log2(), torch.zeros_like(p))).sum(1) * n.squeeze(1)
        return ent + s.R * (s.C // s.g) * s.sbits

class VQ:
    def __init__(s, D=4, K=256, cb="gauss", stages=1): s.D, s.B, s.K, s.cbk, s.stages = D, D, K, cb, stages; s.g = None
    def prepare(s, W):
        s.E, s.R, s.C = W.shape; dev = W.device; s.sig = W.pow(2).mean((1, 2), keepdim=True).sqrt().clamp(min=1e-8)
        Ks = s.K if isinstance(s.K, (list, tuple)) else [s.K] * s.stages; s.Ks = Ks
        if s.stages == 1: s.cbs = [Q.e8_codebook(Ks[0], dev) if s.cbk == "e8" else Q.gauss_cb(Ks[0], s.D, dev)]
        else:
            res = torch.randn(200000, s.D, device=dev); s.cbs = []
            for m, Km in enumerate(Ks):
                Cm = Q.kmeans_w(res, Km, iters=10, seed=m); s.cbs.append(Cm); res = res - Cm[Q.nn_argmax(res, Cm, 0.5 * (Cm ** 2).sum(1))]
        s.cns = [0.5 * (c ** 2).sum(1) for c in s.cbs]
    def start_group(s, Wg): pass
    def quant(s, w):
        X = (w / s.sig).reshape(-1, s.D); res = X; rec = torch.zeros_like(X); idx = []
        for m in range(s.stages): ii = Q.nn_argmax(res, s.cbs[m], s.cns[m]); idx.append(ii); rec = rec + s.cbs[m][ii]; res = X - rec
        if s.stages > 1:
            for m in range(s.stages): rec = rec - s.cbs[m][idx[m]]; ii = Q.nn_argmax(X - rec, s.cbs[m], s.cns[m]); idx[m] = ii; rec = rec + s.cbs[m][ii]
        return rec.reshape(w.shape) * s.sig
    def bits(s): return s.R * s.C / s.D * sum(math.log2(k) for k in s.Ks) + 16 + (sum(k * s.D * 16 for k in s.Ks) if s.stages > 1 else 0)

class TCQ:
    def __init__(s, kb=4, L=12, V=2, dev="cuda"):
        import tcq; s.tr = tcq.Trellis(L, V, kb, dev); s.B = 16; s.g = None; s.kb, s.V = kb, V
        import os
        f = f"/r/res/tcqcb_L{L}_V{V}_kb{kb}.pt"
        if os.path.exists(f): s.tr.C = torch.load(f).to(dev)
        else:
            s.tr.train_codebook(iters=10, nseq=2048, T=128, verbose=False)
            try: torch.save(s.tr.C.cpu(), f)
            except Exception: pass
    def prepare(s, W): s.E, s.R, s.C = W.shape; s.sig = W.pow(2).mean((1, 2), keepdim=True).sqrt().clamp(min=1e-8)
    def start_group(s, Wg): pass
    def quant(s, w):
        E, R, _ = w.shape; nt = R // 16; X = (w / s.sig).reshape(E * nt, 16, 16).reshape(E * nt, 256 // s.V, s.V); out = []
        for a in range(0, E * nt, 2048): out.append(s.tr.quantize(X[a:a + 2048])[1])
        return torch.cat(out).reshape(E, R, 16) * s.sig
    def bits(s): return s.R * s.C * s.kb / s.V + 16

def rotb(W, H, seed, kind="had"):
    """W[E,R,C], H[E,C,C] -> rotated + inverse fn (same rotation for all experts)"""
    ri = Q.Rot(W.shape[2], seed, W.device, kind); ro = Q.Rot(W.shape[1], seed + 7919, W.device, kind)
    Wr = ri.fwd(W); Wr = ro.fwd(Wr.transpose(1, 2).contiguous()).transpose(1, 2).contiguous()
    Hr = ri.fwd(ri.fwd(H).transpose(1, 2).contiguous())
    def back(X): return ri.inv(ro.inv(X.transpose(1, 2).contiguous()).transpose(1, 2).contiguous())
    return Wr, Hr, back

def mb_gptq(mk, rot=None, damp=0.01):
    def fn(W, H):
        W = W.float(); H = H.float()
        if rot: Wr, Hr, back = rotb(W, H, rot[1], rot[0])
        else: Wr, Hr, back = W, H, (lambda x: x)
        rd = mk(); Qm = engine_b(Wr, Hr, rd, damp)
        bits = rd.bits_e() if hasattr(rd, "bits_e") else torch.full((W.shape[0],), float(rd.bits()), device=W.device)
        return back(Qm), bits
    return fn

def relerr_b(W, Wh, H):
    D = (W - Wh).float(); W = W.float(); return (((D @ H) * D).sum((1, 2)) / ((W @ H) * W).sum((1, 2)).clamp(min=1e-30))
