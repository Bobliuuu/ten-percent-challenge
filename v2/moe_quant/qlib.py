"""qlib: one pure-PyTorch library of weight-quantisation methods for the QUANT_SWEEP (v2/QUANT_SWEEP.md).
Every method is  fn(W[R,C], H[C,C]) -> (What, info)  with info['bits'] = ALL bits for the tensor (indices, scales, zeros, codebooks, outliers, masks, low-rank factors).
No gradient training of weights anywhere. Flags: info['gray'] lists rule-gray ingredients (learned clip params, low-rank side factors, ...).
H = X^T X (sum over calibration tokens) of the INPUT of the linear. Proxy error: relerr = tr(dW H dW^T)/tr(W H W^T)."""
import math, torch

# ---------------------------------------------------------------- helpers
def relerr(W, What, H):
    D = (W - What).float(); W = W.float()
    return float(((D @ H) * D).sum() / ((W @ H) * W).sum().clamp(min=1e-30))

def ent_bits(counts):
    c = counts.double(); n = c.sum()
    if n == 0: return 0.0
    p = c[c > 0] / n
    return float(-(p * p.log2()).sum() * n)

_HC = {}
def hadamard(p, dev):
    k = (p, str(dev))
    if k not in _HC: _HC[k] = _hadamard(p, dev)
    return _HC[k]
def _hadamard(p, dev):
    Hm = torch.ones(1, 1, device=dev)
    while Hm.shape[0] < p: Hm = torch.cat([torch.cat([Hm, Hm], 1), torch.cat([Hm, -Hm], 1)], 0)
    return Hm / math.sqrt(p)

class Rot:
    """orthogonal incoherence transform on a length-n axis: random signs, Hadamard on the power-of-2 part, random orthogonal on the odd part.
    kind='had' (fixed Hadamard + random signs), 'haar' (dense random orthogonal; slower)."""
    def __init__(s, n, seed, dev, kind="had"):
        g = torch.Generator().manual_seed(seed); s.n = n; s.kind = kind
        s.sg = (torch.randint(0, 2, (n,), generator=g) * 2 - 1).float().to(dev)
        if kind == "haar":
            s.Q = torch.linalg.qr(torch.randn(n, n, generator=g, dtype=torch.float64))[0].float().to(dev); return
        p = n & (-n); m = n // p; s.p, s.m = p, m
        s.Q = (torch.linalg.qr(torch.randn(m, m, generator=g, dtype=torch.float64))[0].float().to(dev) if m > 1 else None); s.Hd = hadamard(p, dev)
    def fwd(s, X):  # X [..., n] -> X R^T
        sh = X.shape; Y = X * s.sg
        if s.kind == "haar": return (Y.reshape(-1, s.n) @ s.Q.T).reshape(sh)
        Y = Y.reshape(-1, s.m, s.p)
        if s.Q is not None: Y = torch.matmul(s.Q, Y)
        return torch.matmul(Y, s.Hd).reshape(sh)
    def inv(s, X):
        sh = X.shape
        if s.kind == "haar": return (X.reshape(-1, s.n) @ s.Q).reshape(sh) * s.sg
        Y = X.reshape(-1, s.m, s.p); Y = torch.matmul(Y, s.Hd)
        if s.Q is not None: Y = torch.matmul(s.Q.T, Y)
        return Y.reshape(sh) * s.sg

def rotate_wh(W, H, seed, kind="had", rot_out=True):
    dev = W.device; ri = Rot(W.shape[1], seed, dev, kind); ro = Rot(W.shape[0], seed + 7919, dev, kind) if rot_out else None
    Wr = ri.fwd(W)
    if ro is not None: Wr = ro.fwd(Wr.T.contiguous()).T.contiguous()
    Hr = ri.fwd(ri.fwd(H).T.contiguous())
    def back(Wq):
        if ro is not None: Wq = ro.inv(Wq.T.contiguous()).T.contiguous()
        return ri.inv(Wq)
    return Wr, Hr, back

def chol_U(H, damp=0.01):
    """upper Cholesky factor of (H+damp I)^-1 (GPTQ). returns U, dead mask"""
    H = (H + H.T).float() / 2; dead = torch.diag(H) == 0; H = H.clone(); H[dead, dead] = 1
    md = float(torch.diag(H).mean()); prev = 0.0
    for dm in (damp, damp * 10, damp * 100, 1.0):
        H.diagonal().add_((dm - prev) * md); prev = dm
        try:
            L = torch.linalg.cholesky(H); Hi = torch.cholesky_inverse(L); U = torch.linalg.cholesky(Hi, upper=True); return U, dead
        except Exception:
            continue
    raise RuntimeError("cholesky failed")

# ---------------------------------------------------------------- GPTQ / LDLQ engine
def engine(W, H, rd, damp=0.01, blk=128, prune=None, outlier_tau=None, act_order=False, nm=None):
    """Column-sequential error-feedback rounding (GPTQ == LDLQ with this ordering). rd: rounder with .B (block width), .g (group cols or None), .prepare(W), .start_group(Wg), .quant(blk)->Q.
    prune(Wc_block, Ud_block) -> bool mask of weights to zero (SparseGPT) or None. outlier_tau: SpQR-style, keep fp16 where err^2/U_cc^2 > tau."""
    W = W.float().clone(); R, C = W.shape; dev = W.device
    perm = None
    if act_order:
        perm = torch.argsort(torch.diag(H), descending=True); W = W[:, perm]; H = H[perm][:, perm]
    U, dead = chol_U(H, damp); W[:, dead] = 0
    Q = torch.zeros_like(W); rd.prepare(W); B = rd.B; g = rd.g; n_out = 0
    blk = max(blk, g) if g and g > blk else blk
    if blk % B: blk = (blk // B + 1) * B
    for c0 in range(0, C, blk):
        c1 = min(c0 + blk, C); Wc = W[:, c0:c1].clone(); Err = torch.zeros_like(Wc); Ud = torch.diag(U)[c0:c1]
        mask = None
        if prune is not None: mask = prune(Wc, Ud)  # [R, c1-c0] bool (True = pruned), may be refreshed per m-group inside prune
        for j in range(0, c1 - c0, B):
            c = c0 + j
            if g and c % g == 0:
                rd.start_group(Wc[:, j:j + g] if g <= blk else W[:, c:c + g])
            elif g is None and c == 0: rd.start_group(W)
            bw = Wc[:, j:j + B]
            if nm is not None and (c % nm[1] == 0) and B == 1 and prune is not None:
                pass
            q = rd.quant(bw)
            if mask is not None: q = torch.where(mask[:, j:j + B], torch.zeros_like(q), q)
            if outlier_tau is not None:
                bad = ((bw - q) ** 2 / (U[c:c + B, c:c + B].diagonal() ** 2)[None]) > outlier_tau
                if mask is not None: bad &= ~mask[:, j:j + B]
                q = torch.where(bad, bw, q); n_out += int(bad.sum())
            E = bw - q
            if B == 1:
                Es = E / U[c, c]; Err[:, j:j + 1] = Es
                if j + 1 < c1 - c0: Wc[:, j + 1:] -= Es @ U[c:c + 1, c + 1:c1]
            else:
                Ub = U[c:c + B, c:c + B]; Es = torch.linalg.solve_triangular(Ub, E, upper=True, left=False); Err[:, j:j + B] = Es
                if j + B < c1 - c0: Wc[:, j + B:] -= Es @ U[c:c + B, c + B:c1]
            Q[:, c:c + B] = q
        if c1 < C: W[:, c1:] -= Err @ U[c0:c1, c1:]
    if perm is not None:
        inv = torch.argsort(perm); Q = Q[:, inv]
    return Q, n_out

# ---------------------------------------------------------------- scalar rounders
FP4 = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6.]); FP4 = torch.cat([-FP4[1:].flip(0), FP4])
def nf_levels(bits):
    """NormalFloat levels (QLoRA construction) for any bit-width: asymmetric quantiles of N(0,1), exact zero, normalised to [-1,1]."""
    n = 2 ** bits; nd = torch.distributions.Normal(0., 1.); off = 0.9677083
    pos = nd.icdf(torch.linspace(0.5 + (1 - off) / 2 * 0 + 1e-6, off, n // 2 + 1))[:-1] if False else nd.icdf(torch.linspace(0.5, off, n // 2 + 1))[1:]
    neg = -nd.icdf(torch.linspace(0.5, off, n // 2))[1:]
    v = torch.cat([neg.flip(0), torch.zeros(1), pos]); return v / v.abs().max()
def uni_levels(bits): n = 2 ** bits; return torch.linspace(-1, 1, n)
TERN = torch.tensor([-1., 0., 1.])

class ScalarQ:
    """kind: 'int' (asym min/max, zero point), 'grid' (symmetric absmax onto fixed level set), 'ecsq' (uniform step c*rms_g, ints entropy coded).
    scale_fmt: 'fp16'|'e8m0'(power of 2, MXFP)|'e4m3'(NVFP, + fp32 tensor scale). clip: mse-search shrink of group range."""
    B = 1
    def __init__(s, kind="int", bits=4, g=128, levels=None, scale_fmt="fp16", clip=False, c=None, target_bpw=None, lam=0.0, scale_bits=16):
        s.kind, s.nbits, s.g, s.levels, s.scale_fmt, s.clip, s.c, s.target, s.lam, s.sbits = kind, bits, g, levels, scale_fmt, clip, c, target_bpw, lam, scale_bits
        s.hist = None; s.ng = 0; s.dev = None
    def prepare(s, W):
        R, C = W.shape; s.R, s.C = R, C; s.dev = W.device
        if s.g is not None and C % s.g: s.g = None
        s.ng = 0; s.hist = None
        if s.levels is not None: s.levels = s.levels.to(W.device)
        if s.kind == "ecsq":
            if s.c is None and s.target is not None: s.c = s._solve_c(W)
            if s.lam > 0: s._bits_table(W)
    def _grp_rms(s, Wg): return Wg.pow(2).mean(1, keepdim=True).sqrt().clamp(min=1e-8)
    def _solve_c(s, W):
        g = s.g or W.shape[1]; X = W[:, :W.shape[1] // g * g].reshape(-1, g) if s.g else W
        X = X[torch.randperm(X.shape[0], device=X.device)[:20000]]; X = X / s._grp_rms(X) if s.g else X / X.pow(2).mean(1, keepdim=True).sqrt()
        tgt = s.target - (s.sbits / g if s.g else 0)
        lo, hi = 0.02, 8.0
        for _ in range(30):
            mid = math.sqrt(lo * hi); q = torch.round(X / mid).long(); h = torch.bincount((q - q.min()).flatten()); e = ent_bits(h) / q.numel()
            if e > tgt: lo = mid
            else: hi = mid
        return math.sqrt(lo * hi)
    def _bits_table(s, W):
        g = s.g or W.shape[1]; X = W[:, :W.shape[1] // g * g].reshape(-1, g); X = X / s._grp_rms(X); q = torch.round(X / s.c).long().flatten()
        s.qmin = int(q.min()) - 2; h = torch.bincount(q - s.qmin, minlength=int(q.max() - q.min()) + 5).double() + 1; s.tab = (-(h / h.sum()).log2()).float().to(W.device)
    def start_group(s, Wg):
        s.ng += Wg.shape[0] if s.g else Wg.shape[0]
        if s.kind == "int":
            mn, mx = Wg.min(1, keepdim=True).values.clamp(max=0), Wg.max(1, keepdim=True).values.clamp(min=0)
            n = 2 ** s.nbits - 1
            if s.clip:
                best = None
                for f in (1.0, .9, .8, .7, .6, .5):
                    a, b = mn * f, mx * f; sc = ((b - a) / n).clamp(min=1e-8); z = torch.round(-a / sc); q = (torch.clamp(torch.round(Wg / sc) + z, 0, n) - z) * sc; e = ((q - Wg) ** 2).sum(1, keepdim=True)
                    if best is None: best, bf = e, torch.full_like(e, f)
                    else: upd = e < best; best = torch.where(upd, e, best); bf = torch.where(upd, torch.full_like(e, f), bf)
                mn, mx = mn * bf, mx * bf
            s.sc = ((mx - mn) / n).clamp(min=1e-8); s.sc = s.sc.half().float(); s.z = torch.round(-mn / s.sc).clamp(0, n); s.nmax = n
        elif s.kind == "grid":
            am = Wg.abs().amax(1, keepdim=True).clamp(min=1e-8)
            lv = s.levels
            if s.clip:
                best = None
                for f in (1.0, .9, .8, .7, .6, .5):
                    sc = am * f / lv.abs().max(); q = s._grid(Wg, sc); e = ((q - Wg) ** 2).sum(1, keepdim=True)
                    if best is None: best, bf = e, torch.full_like(e, f)
                    else: upd = e < best; best = torch.where(upd, e, best); bf = torch.where(upd, torch.full_like(e, f), bf)
                am = am * bf
            sc = am / lv.abs().max()
            if s.scale_fmt == "e8m0": sc = 2.0 ** torch.ceil(torch.log2(sc))
            elif s.scale_fmt == "e4m3":
                gm = sc.max() if not hasattr(s, "tmax") else s.tmax; s.tmax = gm; r = (sc / gm * 448).clamp(min=2 ** -9)
                e = torch.floor(torch.log2(r)).clamp(min=-6); st = 2.0 ** (e - 3); r = torch.round(r / st) * st; sc = r / 448 * gm  # emulate e4m3 mantissa (3 bits)
            else: sc = sc.half().float()
            s.sc = sc.clamp(min=1e-8)
        elif s.kind == "ecsq":
            s.sc = (s._grp_rms(Wg) * s.c).half().float() if s.g else (s._grp_rms(Wg) * s.c)
            if not s.g: s.sc = Wg.pow(2).mean(1, keepdim=True).sqrt().clamp(min=1e-8) * s.c
    def _grid(s, x, sc):
        v = x / sc; lv = s.levels; idx = (v[..., None] - lv).abs().argmin(-1); return lv[idx] * sc
    def quant(s, w):
        if s.kind == "int":
            return (torch.clamp(torch.round(w / s.sc) + s.z, 0, s.nmax) - s.z) * s.sc
        if s.kind == "grid": return s._grid(w, s.sc)
        v = w / s.sc; q = torch.round(v)
        if s.lam > 0:
            q0 = torch.floor(v); q1 = q0 + 1
            def cost(qq): return (v - qq) ** 2 + s.lam * s.tab[(qq.long() - s.qmin).clamp(0, len(s.tab) - 1)]
            q = torch.where(cost(q0) <= cost(q1), q0, q1)
        qi = (q.long() - 0).flatten()
        s.hist = qi if s.hist is None else torch.cat([s.hist, qi]) if s.hist.numel() < 5e6 else s.hist  # keep sample for entropy
        s._acc(q)
        return q * s.sc
    def _acc(s, q):
        qi = q.long().flatten(); mn = int(qi.min()); mx = int(qi.max())
        if not hasattr(s, "H"): s.H = {}
        h = torch.bincount(qi - mn).cpu()
        for k, v in enumerate(h.tolist()):
            if v: s.H[mn + k] = s.H.get(mn + k, 0) + v
    def bits(s, R, C):
        ng = R * (C // s.g if s.g else 1)
        if s.kind == "int": return R * C * s.nbits + ng * (16 + s.nbits)
        if s.kind == "grid":
            lb = math.log2(len(s.levels)); sb = {"fp16": 16, "e8m0": 8, "e4m3": 8}[s.scale_fmt]; return R * C * lb + ng * sb + (32 if s.scale_fmt == "e4m3" else 0)
        h = torch.tensor(list(s.H.values()), dtype=torch.float64); return ent_bits(h) + ng * s.sbits

# ---------------------------------------------------------------- vector rounders (block of D adjacent input columns)
def kmeans_gauss(K, D, dev, seed=0, iters=25, N=None):
    g = torch.Generator(device=dev).manual_seed(seed + K); N = N or max(K * 100, 200000); X = torch.randn(N, D, generator=g, device=dev); C = X[:K].clone()
    for it in range(iters):
        sums = torch.zeros(K, D, device=dev); cnt = torch.zeros(K, device=dev); Cn = 0.5 * (C ** 2).sum(1)
        for a in range(0, N, 16384):
            xb = X[a:a + 16384]; ii = (xb @ C.T - Cn).argmax(1); sums.index_add_(0, ii, xb); cnt.index_add_(0, ii, torch.ones(len(ii), device=dev))
        e = cnt == 0; C = sums / cnt.clamp(min=1).unsqueeze(1); C[e] = X[torch.randint(0, N, (int(e.sum()),), device=dev, generator=g)]
    return C

def kmeans_w(X, K, iters=15, wts=None, seed=0):
    """weighted k-means on X[N,D] (data-fitted codebook; used by GPTVQ / SqueezeLLM / fitted RVQ)"""
    dev = X.device; N, D = X.shape; g = torch.Generator(device=dev).manual_seed(seed); C = X[torch.randperm(N, device=dev, generator=g)[:K]].clone()
    if C.shape[0] < K: C = torch.cat([C, torch.randn(K - C.shape[0], D, device=dev) * X.std()])
    w = wts if wts is not None else torch.ones(N, device=dev)
    for _ in range(iters):
        Cn = 0.5 * (C ** 2).sum(1); ii = torch.empty(N, dtype=torch.long, device=dev)
        for a in range(0, N, 16384): ii[a:a + 16384] = (X[a:a + 16384] @ C.T - Cn).argmax(1)
        sums = torch.zeros(K, D, device=dev).index_add_(0, ii, X * w[:, None]); cnt = torch.zeros(K, device=dev).index_add_(0, ii, w); ok = cnt > 0
        C[ok] = sums[ok] / cnt[ok, None]
    return C

# ---------------------------------------------------------------- e-means codebook fitter (opt-in alternative to kmeans_w)
# Minimal PyTorch port of Algorithm 1 of "E-Means: Scaling Laws for Mini-Batch K-Means" (Bessa et al., Meta FAIR, arXiv 2611.09501),
# following the reference implementation in FAISS contrib/e_means (base.py, e_means.py), Copyright (c) Meta Platforms, Inc. and affiliates,
# MIT licence. Ported: the batch and step scaling laws, cosine-annealed EMA step with a 0.1 floor, mass-weighted EMA centroid update,
# revival of unused clusters by splitting mass-sampled donors every 5 updates, closing means from the last pass. Not ported: Triton/CuTe
# kernels, spherical mode, distributed/sharded variants, max_points_per_centroid. Extension (not in the reference): optional per-row
# weights `wts` (weighted sums and counts), so it is a drop-in for kmeans_w.
_EM_BATCH_LAWS = {"throughput": (0.035134, 0.134749, 0.876473, 0.813344), "quality": (0.00439175, 0.134749, 0.876473, 0.813344)}
_EM_ALPHA_LAW = (0.8244, -0.4617, 0.9003, -0.4498, -0.1057)

def _em_pow2(v):
    if v <= 1: return 1
    lo = 2 ** int(math.floor(math.log2(v))); return lo if v / lo < 2 * lo / v else 2 * lo

def emeans_recipe(n, k, d, epochs=5.0, batch_law="auto", batch_size=None, alpha0=None):
    """(batch size B, initial EMA step alpha0, rows per pass) from the fitted laws (Eq. 6-7 of the paper, FAISS shipped coefficients)."""
    law = ("quality" if d <= 16 else "throughput") if batch_law == "auto" else batch_law
    if batch_size is None:
        c, bk, bn, bt = _EM_BATCH_LAWS[law]; B = _em_pow2(c * k ** bk * n ** bn * epochs ** bt)
        B = min(B, _em_pow2(n / 2)) if law == "throughput" else min(B, n // 2); B = max(1, min(B, n))
    else: B = min(int(batch_size), n)
    if alpha0 is None:
        c, ak, ab, an, at = _EM_ALPHA_LAW; alpha0 = min(max(c * k ** ak * B ** ab * n ** an * epochs ** at, 1e-4), 1.0)
    whole = math.floor(epochs); rem = int(round((epochs - whole) * n))
    rows = [rem] if whole == 0 else [n] * (whole - 1) + ([rem] if rem else []) + [n]
    return B, float(alpha0), rows

def emeans_w(X, K, epochs=5.0, wts=None, seed=0, batch_law="auto", batch_size=None, alpha0=None, unused_threshold=None, chunk=16384):
    """e-means on X[N,D] -> centroids [K,D] (fp32). Same contract as kmeans_w; `epochs` (passes over the data, may be fractional) is the only knob."""
    dev = X.device; N, D = X.shape; X = X.float(); gen = torch.Generator().manual_seed(seed)
    C = X[torch.randperm(N, generator=gen)[:K].to(dev)].clone()
    if C.shape[0] < K: C = torch.cat([C, torch.randn(K - C.shape[0], D, device=dev) * X.std()])
    w = wts.float() if wts is not None else None
    B, a0, rows_per_pass = emeans_recipe(N, K, D, epochs, batch_law, batch_size, alpha0)
    total = sum((r + B - 1) // B for r in rows_per_pass); tau = (0.0 if D <= 2 else 0.1) if unused_threshold is None else float(unused_threshold)
    usage = torch.full((K,), 2.0 ** -16, device=dev); step = 0; sign = torch.ones(D, device=dev); sign[1::2] = -1.0
    for ep, rows in enumerate(rows_per_pass):
        order = torch.randperm(N, generator=torch.Generator().manual_seed((seed * 1000003 + ep) % 2 ** 32)).to(dev); last = ep == len(rows_per_pass) - 1
        if last: psum = torch.zeros(K, D, device=dev); pcnt = torch.zeros(K, device=dev)
        for b0 in range(0, rows, B):
            idx = order[b0:min(b0 + B, rows)]; xb = X[idx]; Cn = 0.5 * (C ** 2).sum(1); ii = torch.empty(len(idx), dtype=torch.long, device=dev)
            for a in range(0, len(idx), chunk): ii[a:a + chunk] = (xb[a:a + chunk] @ C.T - Cn).argmax(1)       # against the batch-start snapshot
            wb = w[idx] if w is not None else torch.ones(len(idx), device=dev)
            sums = torch.zeros(K, D, device=dev).index_add_(0, ii, xb * wb[:, None]); cnt = torch.zeros(K, device=dev).index_add_(0, ii, wb)
            if last: psum += sums; pcnt += cnt
            frac = min(step, total) / max(1, total); alpha = a0 * (0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * frac)))    # Eq. 5, floor 0.1
            t = cnt > 0
            if alpha < 1: usage *= (1.0 - alpha); retained = usage[t]
            else: retained = torch.zeros_like(cnt[t])
            mass = retained + alpha * cnt[t]; mean = sums[t] / cnt[t, None]; wt = (alpha * cnt[t] / mass)[:, None]
            C[t] = torch.where(wt == 1, mean, C[t] + wt * (mean - C[t])); usage[t] = mass
            if tau > 0 and (step + 1) % 5 == 0:                                                                      # revival (Alg. 1 line 9, App. J)
                unused = usage < usage.sum() * tau / K; donors = (~unused) & (usage > 0); ns = min(int(unused.sum()), int(donors.sum()))
                if ns:
                    u = torch.rand(K, generator=gen).to(dev).clamp(min=1e-38); keys = torch.full((K,), -float("inf"), device=dev)
                    keys[donors] = usage[donors].log() - (-u[donors].log()).log(); src = keys.topk(ns).indices; tgt = unused.nonzero()[:ns, 0]
                    c0 = C[src]; C[src] = c0 * (1 - 1e-4 * sign); C[tgt] = c0 * (1 + 1e-4 * sign); usage[src] = usage[src] / 2; usage[tgt] = usage[src]
            step += 1
    ok = pcnt > 0; C[ok] = psum[ok] / pcnt[ok, None]                                                                  # closing means of the last pass
    return C

def fit_codebook(X, K, iters=15, wts=None, seed=0, method="lloyd"):
    """codebook fitter switch: method 'lloyd' = kmeans_w(iters) (default, unchanged); 'emeans' = emeans_w with epochs=iters."""
    return emeans_w(X, K, epochs=iters, wts=wts, seed=seed) if method == "emeans" else kmeans_w(X, K, iters=iters, wts=wts, seed=seed)

_CB = {}
def e8_codebook(K, dev):
    """first K points of the E8 lattice ordered by norm (D8 U D8+1/2, even coordinate sum), scaled to unit-variance Gaussian MSE optimum"""
    key = ("e8", K, str(dev))
    if key in _CB: return _CB[key]
    import itertools
    pts = []
    rng = range(-3, 4)
    base = torch.tensor(list(itertools.product(range(-3, 4), repeat=8)), dtype=torch.float32) if False else None
    # enumerate via 1D grids
    vi = torch.arange(-3, 4, dtype=torch.float32); vh = vi + 0.5
    def enum(v):
        idx = torch.cartesian_prod(*[torch.arange(len(v))] * 8) if False else None
        grids = torch.meshgrid(*([v] * 8), indexing="ij"); X = torch.stack([g.reshape(-1) for g in grids], 1); return X
    out = []
    for v in (vi, vh):
        X = enum(v); s = X.sum(1); ok = (s.round() % 2 == 0) & (X.pow(2).sum(1) <= 10.01); out.append(X[ok])
    P = torch.cat(out); n2 = P.pow(2).sum(1); order = torch.argsort(n2 + torch.rand(len(n2)) * 1e-3); P = P[order][:K]
    # scale to gaussian-optimal
    Xg = torch.randn(40000, 8, device=dev); P = P.to(dev); best = None
    for sc in torch.linspace(0.2, 1.2, 21):
        C = P * sc; e = 0
        for a in range(0, 40000, 8000):
            xb = Xg[a:a + 8000]; e += float(((xb[:, None, :] - C[None]).pow(2).sum(-1).min(1).values).sum()) if K <= 2048 else float((xb.pow(2).sum(1) - 2 * (xb @ C.T).max(1).values + 0).sum())
        if K > 2048:  # proper nearest: use argmax of x.c - |c|^2/2
            Cn = 0.5 * (C ** 2).sum(1); e = 0
            for a in range(0, 40000, 8000):
                xb = Xg[a:a + 8000]; ii = (xb @ C.T - Cn).argmax(1); e += float(((xb - C[ii]) ** 2).sum())
        if best is None or e < best[0]: best = (e, float(sc))
    _CB[key] = (P * best[1]); return _CB[key]

def gauss_cb(K, D, dev):
    key = ("g", K, D, str(dev))
    if key not in _CB: _CB[key] = kmeans_gauss(K, D, dev)
    return _CB[key]

def nn_argmax(X, C, Cn, bs=2048):
    out = torch.empty(X.shape[0], dtype=torch.long, device=X.device)
    for a in range(0, X.shape[0], bs): out[a:a + bs] = (X[a:a + bs] @ C.T - Cn).argmax(1)
    return out

class VQ:
    """D-dim VQ rounder. cb in {'gauss','e8'}; per-tensor sigma (use after rotation). stages>1 => residual multi-stage VQ (AQLM-like additive, codebooks Gaussian-trained on residuals, greedy+ICM re-encode).
    fit=True: codebooks fitted to this tensor's own vectors (data-fitted, shipped, bytes counted)."""
    def __init__(s, D=8, K=4096, cb="gauss", stages=1, fit=False, icm=1, row_scale=None, cb_bits=16, fitter="lloyd"):
        s.D, s.B, s.K, s.cbk, s.stages, s.fit, s.icm, s.rs, s.cbb, s.fitter = D, D, K, cb, stages, fit, icm, row_scale, cb_bits, fitter; s.g = None; s.dev = None
    def prepare(s, W):
        s.R, s.C = W.shape; dev = W.device; s.dev = dev
        s.sigma = float(W.pow(2).mean().sqrt()) if s.rs is None else None
        if s.rs == "row": s.sig = W.pow(2).mean(1, keepdim=True).sqrt().clamp(min=1e-8).half().float()
        Ks = s.K if isinstance(s.K, (list, tuple)) else [s.K] * s.stages; s.Ks = Ks; s.cbs = []
        Wn = W / (s.sig if s.rs == "row" else s.sigma)
        if s.cbk == "e8" and s.stages == 1 and not s.fit: s.cbs = [e8_codebook(Ks[0], dev)]
        elif not s.fit and s.stages == 1: s.cbs = [gauss_cb(Ks[0], s.D, dev)]
        else:
            Xv = Wn.reshape(-1, s.D)[:: max(1, (Wn.numel() // s.D) // 200000)] if s.fit else torch.randn(200000, s.D, device=dev)
            res = Xv
            for m, Km in enumerate(Ks):
                Cm = fit_codebook(res, Km, iters=12 if s.fit else 10, seed=m, method=s.fitter); s.cbs.append(Cm); Cn = 0.5 * (Cm ** 2).sum(1); ii = nn_argmax(res, Cm, Cn); res = res - Cm[ii]
        s.cns = [0.5 * (c ** 2).sum(1) for c in s.cbs]; s.nb = 0
    def start_group(s, Wg): pass
    def _enc(s, X):
        if s.stages == 1:
            ii = nn_argmax(X, s.cbs[0], s.cns[0]); return s.cbs[0][ii]
        idx = []; res = X; rec = torch.zeros_like(X)
        for m in range(s.stages):
            ii = nn_argmax(res, s.cbs[m], s.cns[m]); idx.append(ii); rec = rec + s.cbs[m][ii]; res = X - rec
        for _ in range(s.icm):
            for m in range(s.stages):
                rec = rec - s.cbs[m][idx[m]]; res = X - rec; idx[m] = nn_argmax(res, s.cbs[m], s.cns[m]); rec = rec + s.cbs[m][idx[m]]
        return rec
    def quant(s, w):
        sc = s.sig if s.rs == "row" else s.sigma
        X = (w / sc).reshape(w.shape[0], s.D)
        return (s._enc(X) * sc).reshape(w.shape)
    def bits(s, R, C):
        idxb = R * C / s.D * sum(math.log2(k) for k in s.Ks)
        cbb = sum(k * s.D * s.cbb for k in s.Ks) if (s.fit or s.stages > 1 or s.cbk == "gauss" and False) else 0
        return idxb + cbb + 16 + (R * 16 if s.rs == "row" else 0)

class TCQ:
    """QTIP-style bitshift trellis (repo v2/vq/tcq.py), 16x16 tiles = 256-weight sequences. per-tensor sigma (use after rotation)."""
    def __init__(s, L=12, V=2, kb=3, trellis=None, dev="cuda"):
        import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vq")); import tcq
        s.tr = trellis or tcq.Trellis(L, V, kb, dev); s.B = 16; s.g = None; s.V = V; s.kb = kb; s.L = L
    def prepare(s, W): s.R, s.C = W.shape; s.sigma = float(W.pow(2).mean().sqrt()); assert s.R % 16 == 0
    def start_group(s, Wg): pass
    def quant(s, w):
        R = w.shape[0]; nt = R // 16; X = (w / s.sigma).reshape(nt, 16, 16).reshape(nt, 256 // s.V, s.V); out = []
        for a in range(0, nt, 2048): out.append(s.tr.quantize(X[a:a + 2048])[1])
        return torch.cat(out).reshape(nt, 16, 16).reshape(R, 16) * s.sigma
    def bits(s, R, C): return R * C * s.kb / s.V + 16 + (1 << s.L) * s.V * 16 * 0

class GPTVQ:
    """GPTVQ (van Baalen 2024): D-dim codebooks fitted by Hessian-diag-weighted k-means per group of `gr` rows (8-bit codebook entries), then GPTQ-style column-block rounding of D columns."""
    def __init__(s, D=2, bits_per_w=2.0, gr=256, cb_bits=8, iters=15):
        s.D, s.B, s.bpw, s.gr, s.cbb, s.iters = D, D, bits_per_w, gr, cb_bits, iters; s.K = int(round(2 ** (bits_per_w * D))); s.g = None; s.hd = None
    def set_hess(s, H): s.hd = torch.diag(H)
    def prepare(s, W):
        R, C = W.shape; s.R, s.C = R, C; gr = min(s.gr, R); s.ngr = R // gr if R % gr == 0 else 1; gr = R // s.ngr; s.grr = gr; cbs = []
        wcol = (s.hd if s.hd is not None else torch.ones(C, device=W.device)).clone(); wv = wcol.reshape(-1, s.D).mean(1)
        for k in range(s.ngr):
            X = W[k * gr:(k + 1) * gr].reshape(gr, C // s.D, s.D).reshape(-1, s.D); w = wv[None].expand(gr, -1).reshape(-1)
            Cb = kmeans_w(X, s.K, s.iters, w); Cb = (Cb / Cb.abs().max() * 127).round() / 127 * Cb.abs().max() if s.cbb == 8 else Cb; cbs.append(Cb)
        s.cb = torch.stack(cbs); s.cn = 0.5 * (s.cb ** 2).sum(-1)
    def start_group(s, Wg): pass
    def quant(s, w):
        G, gr = s.ngr, s.grr; X = w.reshape(G, gr, s.D); ii = (torch.bmm(X, s.cb.transpose(1, 2)) - s.cn[:, None, :]).argmax(-1)
        return torch.gather(s.cb, 1, ii[..., None].expand(-1, -1, s.D)).reshape(w.shape)
    def bits(s, R, C): return R * C * math.log2(s.K) / s.D + s.ngr * s.K * s.D * s.cbb + s.ngr * 16

# ---------------------------------------------------------------- method wrappers  (all: fn(W,H) -> (What, info))
def rtn_scalar(W, H, mk):
    rd = mk(); rd.prepare(W)
    R, C = W.shape; g = rd.g or C; out = torch.empty_like(W)
    for c in range(0, C, g):
        rd.start_group(W[:, c:c + g])
        for j in range(c, min(c + g, C)): out[:, j:j + 1] = rd.quant(W[:, j:j + 1]) if rd.B == 1 else 0
    return out, rd

def run_gptq(W, H, rd, **kw):
    Q, nout = engine(W, H, rd, **kw); return Q, nout

def m_gptq(mk, rot=None, damp=0.01, act_order=False, outlier_tau=None, extra_bits=0.0, name="gptq", gray=()):
    """generic: optional rotation, GPTQ engine with rounder mk(); outliers (SpQR) counted 32 bits each"""
    def fn(W, H):
        W = W.float(); Hh = H.float()
        if rot: Wr, Hr, back = rotate_wh(W, Hh, rot[1], rot[0])
        else: Wr, Hr, back = W, Hh, (lambda x: x)
        rd = mk()
        if hasattr(rd, "set_hess"): rd.set_hess(Hr)
        Q, nout = engine(Wr, Hr, rd, damp=damp, act_order=act_order, outlier_tau=outlier_tau)
        bits = rd.bits(*W.shape) + nout * 32 + (W.shape[1] * 16 if act_order else 0)
        return back(Q), dict(bits=bits, gray=list(gray), n_out=nout)
    return fn

def m_rtn(mk, rot=None):
    def fn(W, H):
        W = W.float()
        if rot: Wr, Hr, back = rotate_wh(W, H.float(), rot[1], rot[0])
        else: Wr, back = W, (lambda x: x)
        rd = mk(); rd.prepare(Wr); R, C = Wr.shape; g = rd.g or C; out = torch.empty_like(Wr)
        if rd.B == 1:
            for c in range(0, C, g):
                rd.start_group(Wr[:, c:c + g]); out[:, c:c + g] = rd.quant(Wr[:, c:c + g]) if rd.kind != "ecsq" or True else 0
        else:
            for c in range(0, C, rd.B): out[:, c:c + rd.B] = rd.quant(Wr[:, c:c + rd.B])
        return back(out), dict(bits=rd.bits(R, C), gray=[])
    return fn

def m_hqq(bits=2, g=64, iters=20, p=0.7, beta=1.0, kappa=1.01):
    """HQQ (Badri & Shaji 2023): half-quadratic solver for zero-point with l_p<1 error, no calibration."""
    def fn(W, H):
        W = W.float(); R, C = W.shape; Wg = W.reshape(-1, g); n = 2 ** bits - 1
        mn, mx = Wg.min(1, keepdim=True).values, Wg.max(1, keepdim=True).values; sc = (mx - mn).clamp(min=1e-8) / n; z = -mn / sc; b = beta
        for _ in range(iters):
            q = torch.clamp(torch.round(Wg / sc + z), 0, n); We = (q - z) * sc; e = Wg - We
            sh = torch.sign(e) * torch.relu(e.abs() - (1 / b) * (e.abs() + 1e-8).pow(p - 1)); z = (q - (Wg - sh) / sc).mean(1, keepdim=True); b *= kappa
        q = torch.clamp(torch.round(Wg / sc + z), 0, n); What = ((q - z) * sc).reshape(R, C)
        return What, dict(bits=R * C * bits + R * C // g * 32, gray=[])
    return fn

def m_awq(base, grid=20, stat="rms", rot=None):
    """AWQ-style: per-input-channel scale s=a_j^alpha (a_j = activation RMS from diag H), alpha searched on a grid by minimising the H-weighted error of base-quantised W*s; W^ = Q(W s)/s (s folded into the previous op, free)."""
    def fn(W, H):
        W = W.float(); a = (torch.diag(H).clamp(min=1e-12)).sqrt(); a = a / a.mean(); best = None
        for k in range(grid + 1):
            al = k / grid; s = a.pow(al).clamp(min=1e-4); s = s / (s.max() * s.min()).sqrt()
            Wq, info = base((W * s), H / (s[:, None] * s[None]))
            Wh = Wq / s; e = relerr(W, Wh, H)
            if best is None or e < best[0]: best = (e, al, Wh, info)
        info = dict(best[3]); info["awq_alpha"] = best[1]; return best[2], info
    return fn

def m_smoothquant_plus(base, alphas=(0.3, 0.5, 0.7)):
    """SmoothQuant+ (weight-only): s_j = max|x_j|^a / max|w_j|^(1-a) ; absmax of x approximated by sqrt(diag H)*3 (no separate stat pass); alpha chosen by H-weighted error."""
    def fn(W, H):
        W = W.float(); xa = torch.diag(H).clamp(min=1e-12).sqrt(); wa = W.abs().amax(0).clamp(min=1e-6); best = None
        for al in alphas:
            s = (xa.pow(al) / wa.pow(1 - al)).clamp(min=1e-6); s = s / s.mean()
            Wq, info = base(W * s, H / (s[:, None] * s[None])); Wh = Wq / s; e = relerr(W, Wh, H)
            if best is None or e < best[0]: best = (e, al, Wh, info)
        info = dict(best[3]); info["sq_alpha"] = best[1]; return best[2], info
    return fn

def m_omniquant_lwc(bits=2, g=64, steps=0, sym=False, grid=True):
    """OmniQuant weight-clipping (LWC): per-group learnable gamma,beta in (0,1] scaling max/min. grid=True: coordinate grid search of (gamma,beta) per group by H-diag-weighted error (no gradient);
    steps>0: gradient (Adam, STE) on the layer-output loss tr(dW H dW^T) - LEARNED PARAMS: flagged gray."""
    def fn(W, H):
        W = W.float(); R, C = W.shape; Wg = W.reshape(R, C // g, g); n = 2 ** bits - 1; hd = torch.diag(H).reshape(1, C // g, g)
        mn0, mx0 = Wg.min(-1, keepdim=True).values.clamp(max=0), Wg.max(-1, keepdim=True).values.clamp(min=0)
        def qd(gm, bt):
            mn, mx = mn0 * bt, mx0 * gm; sc = ((mx - mn) / n).clamp(min=1e-8); z = torch.round(-mn / sc); return (torch.clamp(torch.round(Wg / sc) + z, 0, n) - z) * sc
        def err(Q): return (((Q - Wg) ** 2) * hd).sum(-1)
        gm = torch.ones_like(mn0); bt = torch.ones_like(mn0)
        if grid:
            gs = [1.0, .95, .9, .85, .8, .7, .6, .5]; best = None
            for a in gs:
                for b in gs:
                    e = err(qd(torch.full_like(mn0, a), torch.full_like(mn0, b))).unsqueeze(-1)
                    if best is None: best = e; ga = torch.full_like(mn0, a); gb = torch.full_like(mn0, b)
                    else: upd = e < best; best = torch.where(upd, e, best); ga = torch.where(upd, torch.full_like(mn0, a), ga); gb = torch.where(upd, torch.full_like(mn0, b), gb)
            gm, bt = ga, gb
        gray = ["searched-clip"]
        if steps:
            gm = gm.clone().requires_grad_(True); bt = bt.clone().requires_grad_(True); opt = torch.optim.Adam([gm, bt], lr=5e-3)
            for _ in range(steps):
                mn, mx = mn0 * bt.clamp(0.3, 1), mx0 * gm.clamp(0.3, 1); sc = ((mx - mn) / n).clamp(min=1e-8); z = (-mn / sc); z = z + (torch.round(z) - z).detach()
                v = Wg / sc + z; v = v + (torch.round(v) - v).detach(); Q = (torch.clamp(v, 0, n) - z) * sc
                loss = (((Q - Wg) ** 2) * hd).sum(); opt.zero_grad(); loss.backward(); opt.step()
            gm, bt = gm.detach().clamp(0.3, 1), bt.detach().clamp(0.3, 1); gray = ["LEARNED clip params (gradient, layer-wise)"]
        Q = qd(gm, bt); return Q.reshape(R, C), dict(bits=R * C * bits + R * C // g * (16 + bits), gray=gray)
    return fn

def m_owq(mk, frac=0.01, rot=None):
    """OWQ: keep the top-`frac` columns by sensitivity (H_cc * ||W_c - Q(W)_c||^2 for a first-pass quantisation) in fp16, quantise the rest with GPTQ."""
    def fn(W, H):
        W = W.float(); C = W.shape[1]; k = max(1, int(frac * C)); rd0 = mk(); Q0, _ = engine(W, H, rd0)
        sens = torch.diag(H) * ((W - Q0) ** 2).sum(0); keep = torch.topk(sens, k).indices; mask = torch.ones(C, dtype=torch.bool, device=W.device); mask[keep] = False
        Wm = W[:, mask]; Hm = H[mask][:, mask]; rd = mk(); Qm, _ = engine(Wm, Hm, rd, damp=0.01); out = W.clone(); out[:, mask] = Qm
        return out, dict(bits=rd.bits(*Wm.shape) + k * W.shape[0] * 16 + k * 16, gray=[])
    return fn

def m_spqr(mk, tau=None, target_frac=0.01):
    """SpQR: small-group GPTQ + fp16 unstructured outliers chosen where err^2/Ucc^2 > tau; tau auto-set so ~target_frac of weights are outliers (searched)."""
    def fn(W, H):
        W = W.float(); lo, hi = 1e-6, 1e3; N = W.numel()
        rd = mk(); Q, n0 = engine(W, H, rd, outlier_tau=1e30)
        # find tau giving target_frac via quantile of the unbounded error stat on a cheap RTN proxy
        rd = mk(); Q0, _ = engine(W, H, rd); U, _ = chol_U(H); st = ((W - Q0) ** 2 / (torch.diag(U) ** 2)[None]).flatten(); tau = float(torch.quantile(st[torch.randperm(len(st), device=st.device)[:2_000_000]], 1 - target_frac))
        rd = mk(); Q, nout = engine(W, H, rd, outlier_tau=tau)
        return Q, dict(bits=rd.bits(*W.shape) + nout * 32, gray=[], n_out=nout, frac_out=nout / N)
    return fn

def m_squeezellm(bits=2, outlier_frac=0.0045, iters=15):
    """SqueezeLLM: per-row non-uniform k-means grid (K=2^bits) weighted by sensitivity (here diag H of the input as a Fisher proxy per column) + dense-and-sparse fp16 outliers (largest |w|)."""
    def fn(W, H):
        W = W.float(); R, C = W.shape; hd = torch.diag(H).clamp(min=1e-12); k = int(outlier_frac * W.numel())
        thr = torch.topk(W.abs().flatten(), k).values[-1] if k else float("inf"); out_m = W.abs() >= thr; Wd = W.clone(); Wd[out_m] = 0
        K = 2 ** bits; Wq = torch.empty_like(W)
        # batched per-row weighted 1-D k-means
        Cc = torch.quantile(Wd, torch.linspace(0.02, 0.98, K, device=W.device), dim=1).T.contiguous() if C * R < 6e7 else torch.stack([Wd[i].sort().values[(torch.linspace(0.02, 0.98, K) * (C - 1)).long()] for i in range(R)])
        w = hd[None].expand(R, -1)
        for _ in range(iters):
            ii = (Wd[:, :, None] - Cc[:, None, :]).abs().argmin(-1)
            num = torch.zeros(R, K, device=W.device).scatter_add_(1, ii, Wd * w); den = torch.zeros(R, K, device=W.device).scatter_add_(1, ii, w); Cc = torch.where(den > 0, num / den.clamp(min=1e-12), Cc)
        ii = (Wd[:, :, None] - Cc[:, None, :]).abs().argmin(-1); Wq = torch.gather(Cc, 1, ii); Wq[out_m] = W[out_m]
        return Wq, dict(bits=R * C * bits + R * K * 16 + int(out_m.sum()) * 32, gray=[])
    return fn

# --- sparsity + quant
def _mask_unstructured(Wc, Ud, sp):
    sc = Wc ** 2 / (Ud[None] ** 2); k = int(sp * sc.numel()); thr = torch.kthvalue(sc.flatten(), max(k, 1)).values; return sc <= thr
def _mask_nm(Wc, Ud, n, m):
    sc = (Wc ** 2 / (Ud[None] ** 2)); R, Cb = sc.shape; g = sc.reshape(R, Cb // m, m); idx = torch.topk(g, n, dim=-1, largest=False).indices; mk = torch.zeros_like(g, dtype=torch.bool).scatter_(-1, idx, True); return mk.reshape(R, Cb)

def m_sparse_quant(mk, kind="sparsegpt", sp=0.5, nm=None, wanda_row=True, rot=None):
    """SparseGPT (OBS, joint with the quantiser rounding via the same error feedback) / Wanda (|w|*||x|| per-row mask) + any quantiser mk() on the survivors.
    bits: nonzero values at rounder bits + mask entropy (unstructured: H(sp); n:m: log2 C(m,n)/m)."""
    def fn(W, H):
        W = W.float(); R, C = W.shape
        if kind == "wanda":
            sc = W.abs() * torch.diag(H).clamp(min=0).sqrt()[None]
            if nm: g = sc.reshape(R, C // nm[1], nm[1]); idx = torch.topk(g, nm[0], dim=-1, largest=False).indices; mask = torch.zeros_like(g, dtype=torch.bool).scatter_(-1, idx, True).reshape(R, C)
            else: k = int(sp * C); thr = torch.kthvalue(sc, max(k, 1), dim=1, keepdim=True).values; mask = sc <= thr
            Wm = W.clone(); Wm[mask] = 0; rd = mk(); full = engine_masked(Wm, H, rd, mask)
        else:
            rd = mk()
            if nm: pr = lambda Wc, Ud: _mask_nm(Wc, Ud, *nm)
            else: pr = lambda Wc, Ud: _mask_unstructured(Wc, Ud, sp)
            full, _ = engine(W, H, rd, prune=pr, blk=128); mask = full == 0
        fr = float(mask.float().mean()); nnz = R * C - int(mask.sum())
        if nm: mb = R * C / nm[1] * math.log2(math.comb(nm[1], nm[0]))
        else: mb = R * C * (-(fr * math.log2(max(fr, 1e-9)) + (1 - fr) * math.log2(max(1 - fr, 1e-9))))
        # value bits: rounder bits count for all R*C; scale the index part by nnz fraction (zeros are not stored). approximate: bits_total_rounder*(1-fr) + scales
        vb = rd.bits(R, C) * (1 - fr) if nm is None and kind else rd.bits(R, C) * (1 - fr)
        return full, dict(bits=vb + mb, gray=[], sparsity=fr)
    return fn

def engine_masked(W, H, rd, mask):
    """GPTQ with a fixed pruning mask (Wanda + quant)."""
    W = W.float().clone(); R, C = W.shape; U, dead = chol_U(H); W[:, dead] = 0; Q = torch.zeros_like(W); rd.prepare(W); g = rd.g; blk = max(128, g or 0)
    for c0 in range(0, C, blk):
        c1 = min(c0 + blk, C); Wc = W[:, c0:c1].clone(); Err = torch.zeros_like(Wc)
        for j in range(c1 - c0):
            c = c0 + j
            if g and c % g == 0: rd.start_group(Wc[:, j:j + g] if g <= blk else W[:, c:c + g])
            elif g is None and c == 0: rd.start_group(W)
            q = rd.quant(Wc[:, j:j + 1]); q = torch.where(mask[:, c:c + 1], torch.zeros_like(q), q)
            Es = (Wc[:, j:j + 1] - q) / U[c, c]; Err[:, j:j + 1] = Es
            if j + 1 < c1 - c0: Wc[:, j + 1:] -= Es @ U[c:c + 1, c + 1:c1]
            Q[:, c:c + 1] = q
        if c1 < C: W[:, c1:] -= Err @ U[c0:c1, c1:]
    return Q

# --- low rank + quant
def m_lowrank(base, rank=32, iters=3, lr_bits=4, lr_g=32, whiten=True):
    """LQER / CALDERA-style (no fine-tuning): W ~ Q + L R. Alternating: Q = base(W - LR) ; (L,R) = rank-r SVD of (W-Q) in the H-whitened metric. L,R stored at lr_bits (RTN, group lr_g) (lr_bits=16: fp16).
    iters=1 -> LQER-like one shot. GRAY under rule 4 ('adapters of any kind'): low-rank correction is shipped computation."""
    def fn(W, H):
        W = W.float(); R, C = W.shape; Lc = None
        if whiten:
            Hh = (H + H.T) / 2; Hh = Hh + 1e-3 * Hh.diag().mean() * torch.eye(C, device=W.device); Lc = torch.linalg.cholesky(Hh)
        LR = torch.zeros_like(W); info = None
        for it in range(iters):
            Q, info = base(W - LR, H); E = W - Q
            Ew = E @ Lc if whiten else E
            U_, S_, Vh = torch.linalg.svd(Ew, full_matrices=False); r = rank
            L_ = U_[:, :r] * S_[:r].sqrt(); R_ = S_[:r].sqrt()[:, None] * Vh[:r]
            if whiten: R_ = torch.linalg.solve_triangular(Lc, R_, upper=False, left=False)
            def rq(M):
                if lr_bits >= 16: return M.half().float()
                n = 2 ** lr_bits - 1; Mg = M.reshape(-1, lr_g); mn, mx = Mg.min(1, keepdim=True).values, Mg.max(1, keepdim=True).values; sc = (mx - mn).clamp(min=1e-8) / n
                return (torch.round((Mg - mn) / sc) * sc + mn).reshape(M.shape)
            L_, R_ = rq(L_), rq(R_); LR = L_ @ R_
        What = Q + LR; lrb = (R + C) * rank * (lr_bits + (32 / lr_g if lr_bits < 16 else 0))
        info = dict(info); info["bits"] = info["bits"] + lrb; info["gray"] = list(info.get("gray", [])) + ["low-rank side factors (rule 4 adapter-like)"]; return What, info
    return fn

def m_admmq(bits=2, g=64, iters=40, rho=1.0):
    """ADMM-style Hessian-aware quantiser (in the spirit of ADMM-Q 2605.11222): min tr(dW H dW^T) s.t. Q on a group-uniform grid. Splitting W = Z, Z projected to grid, scaled dual U."""
    def fn(W, H):
        W = W.float(); R, C = W.shape; n = 2 ** bits - 1; Hn = H / H.diag().mean(); I = torch.eye(C, device=W.device)
        A = torch.linalg.inv(Hn + rho * I + 1e-3 * I)  # (H + rho I)^-1
        def proj(X, Wref):
            Xg = X.reshape(R, C // g, g); mn, mx = Wref.reshape(R, C // g, g).min(-1, keepdim=True).values, Wref.reshape(R, C // g, g).max(-1, keepdim=True).values
            sc = ((mx - mn) / n).clamp(min=1e-8); z = torch.round(-mn / sc); return ((torch.clamp(torch.round(Xg / sc) + z, 0, n) - z) * sc).reshape(R, C)
        Z = proj(W, W); Uu = torch.zeros_like(W)
        for _ in range(iters):
            X = (W @ Hn + rho * (Z - Uu)) @ A; Z = proj(X + Uu, W); Uu = Uu + X - Z
        return Z, dict(bits=R * C * bits + R * (C // g) * (16 + bits), gray=[])
    return fn
