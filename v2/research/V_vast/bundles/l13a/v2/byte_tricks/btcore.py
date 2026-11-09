"""btcore: byte-trick ablation core (L13). Fake-quant (dequantised bf16 weights) + exact bit accounting + real greedy generation eval.
No weight training. Rotation = random-sign block Hadamard on the input dim (Q-matrix, counted as code, seeds are free: derived from layer id).
Quantiser: symmetric/asym uniform grid with L levels, group g (cols) x k (rows sharing a scale), MSE-clip scale search; scale coded by a ScaleCoder.
Rounding: 'gptq' (Hessian error feedback, sequential layers) or 'rtn' (data-free). 'higgs' = Gaussian Lloyd-Max grid, data-free."""
import math, json, time, os, collections
import torch

# ------------------------------------------------------------------ entropy helpers
def H0_bits(x, nsym=None):
    """total bits of the empirical order-0 entropy of integer tensor x (the static-table rANS cost without table)."""
    x = x.flatten().long(); x = x - x.min()
    c = torch.bincount(x).double(); n = c.sum(); p = c[c > 0] / n
    return float(-(p * p.log2()).sum() * n)

def hist_bits(c):
    c = c.double(); n = c.sum(); p = c[c > 0] / n
    return float(-(p * p.log2()).sum() * n)

# ------------------------------------------------------------------ rotation
_QC = {}
def blockQ(C, seed, dev):
    k = (C, seed, str(dev))
    if k in _QC: return _QC[k]
    b = 1
    while C % (b * 2) == 0 and b < 1024: b *= 2
    Hm = torch.ones(1, 1, dtype=torch.float64)
    while Hm.shape[0] < b: Hm = torch.cat([torch.cat([Hm, Hm], 1), torch.cat([Hm, -Hm], 1)], 0)
    Hm = (Hm / math.sqrt(b)).float()
    g = torch.Generator().manual_seed(seed); sg = (torch.randint(0, 2, (C,), generator=g) * 2 - 1).float()
    Q = torch.zeros(C, C)
    for i in range(0, C, b): Q[i:i + b, i:i + b] = Hm
    Q = (Q * sg[None, :]).to(dev)          # Q = blockdiag(Hb) @ diag(sign):  x' = Q x
    if len(_QC) > 6: _QC.clear()
    _QC[k] = Q; return Q

# ------------------------------------------------------------------ scale coders
class ScaleCoder:
    """kind: fp16 | log{m} (round log2 s to 1/m octave) | log{m}rows etc. .apply(s)->(s_hat, codes or None). bits via .bits(codes)"""
    def __init__(self, kind): self.kind = kind
    def apply(self, s):
        k = self.kind
        if k == "fp16": return s.half().float(), None
        if k == "fp32": return s, None
        if k.startswith("log"):
            m = int(k[3:]); c = torch.round(torch.log2(s.clamp(min=1e-12)) * m)
            return torch.exp2(c / m), c.to(torch.int32)
        raise ValueError(k)

def scale_bits_all(codes, s_fp):
    """Bits for storing the scale table under several codings. codes [R', ng] int (log codes) or None. returns dict name->bits (total for the tensor)."""
    n = s_fp.numel(); out = {"fp16": 16.0 * n}
    if codes is None: return out
    out["raw8"] = 8.0 * n
    out["H0"] = H0_bits(codes) * 1.005
    d = codes.clone(); d[:, 1:] = codes[:, 1:] - codes[:, :-1]
    out["delta_left"] = (H0_bits(d[:, 1:]) * 1.005 + 12.0 * codes.shape[0]) if codes.shape[1] > 1 else out["raw8"]   # first col 12 bits each
    med = codes.float().median(dim=1).values.round().int()
    d2 = codes - med[:, None]
    out["rowmed"] = H0_bits(d2) * 1.005 + 10.0 * codes.shape[0]
    cm = codes.float().median(dim=0).values.round().int()   # per-group-column median (the same across rows after Hadamard)
    d3 = codes - cm[None, :]; out["colmed"] = H0_bits(d3) * 1.005 + 10.0 * codes.shape[1]
    d4 = codes - med[:, None] - (codes - med[:, None]).float().median(dim=0).values.round().int()[None, :]
    out["row+col"] = H0_bits(d4) * 1.005 + 10.0 * (codes.shape[0] + codes.shape[1])
    return out

# ------------------------------------------------------------------ grids
def gauss_lloyd_max(L, iters=200):
    """Lloyd-Max levels for N(0,1) (HIGGS-style data-free grid; fixed constants, shipped as L floats)."""
    g = torch.Generator().manual_seed(0); x = torch.randn(400000, generator=g, dtype=torch.float64)
    lv = torch.quantile(x, (torch.arange(L, dtype=torch.float64) + 0.5) / L)
    for _ in range(iters):
        th = (lv[1:] + lv[:-1]) / 2; idx = torch.bucketize(x, th)
        lv = torch.stack([x[idx == i].mean() if (idx == i).any() else lv[i] for i in range(L)])
    return lv.float()

# ------------------------------------------------------------------ quantiser state for one tensor
class Q1:
    def __init__(s, L, g, share=1, coder="fp16", asym=False, grid="uniform", clip=(1.0, .9, .8, .7, .6), theta=0.0):
        s.L, s.g, s.share, s.coder, s.asym, s.grid, s.clip, s.theta = L, g, share, ScaleCoder(coder), asym, grid, clip, theta
        s.sc_list = []; s.zp_list = []; s.hist = torch.zeros(L, dtype=torch.long)
        if grid == "gauss": s.lv = gauss_lloyd_max(L)
    def group_scale(s, Wg):
        """Wg [R, g] current weights of the group -> scale [R] (shared over `share` consecutive rows)."""
        R = Wg.shape[0]; k = s.share
        Wv = Wg.reshape(R // k, k * Wg.shape[1])
        if s.grid == "gauss":
            rms = Wv.pow(2).mean(1).sqrt().clamp(min=1e-8); best = None
            cands = torch.stack([rms * c for c in (0.8, 1.0, 1.2, 1.5)], 1)   # scale applies to the N(0,1) grid
            errs = []
            for j in range(cands.shape[1]):
                sc = cands[:, j:j + 1]; q = s._gq(Wv / sc) * sc; errs.append(((Wv - q) ** 2).sum(1))
            e = torch.stack(errs, 1); sc = cands.gather(1, e.argmin(1, keepdim=True)).squeeze(1)
        elif s.asym:
            mx, mn = Wv.max(1).values, Wv.min(1).values; sc = ((mx - mn) / (s.L - 1)).clamp(min=1e-8)
        else:
            amax = Wv.abs().max(1).values.clamp(min=1e-8); qm = (s.L - 1) / 2; cands = torch.stack([amax / qm * c for c in s.clip], 1); errs = []
            for j in range(cands.shape[1]):
                sc = cands[:, j:j + 1]; q = s._uq(Wv / sc) * sc; errs.append(((Wv - q) ** 2).sum(1))
            e = torch.stack(errs, 1); sc = cands.gather(1, e.argmin(1, keepdim=True)).squeeze(1)
        sh, codes = s.coder.apply(sc)
        s.sc_list.append((sc, codes))
        shr = sh.repeat_interleave(k)
        if s.asym:
            mn = Wv.min(1).values; zp = torch.round(-mn / sh).clamp(0, s.L - 1).repeat_interleave(k)
            s.zp_list.append(zp.to(torch.int8)); return shr, zp
        return shr, None
    def _uq(s, x):  # symmetric uniform integer grid  (L odd: 0 included; L even: half-integer offsets)
        qm = (s.L - 1) / 2
        if s.L % 2 == 1:
            r = torch.sign(x) * torch.floor(x.abs() + 0.5 - s.theta); return r.clamp(-qm, qm)
        r = torch.floor(x) + 0.5
        return r.clamp(-qm, qm)
    def _gq(s, x):
        th = (s.lv[1:] + s.lv[:-1]) / 2; return s.lv.to(x.device)[torch.bucketize(x, th.to(x.device))]
    def quant_cols(s, W, scale, zp):
        """W [R, n] -> dequantised values, and record index histogram. scale [R] (per row)."""
        sc = scale[:, None]
        if s.grid == "gauss":
            th = (s.lv[1:] + s.lv[:-1]) / 2; idx = torch.bucketize(W / sc, th.to(W.device)); q = s.lv.to(W.device)[idx] * sc
        elif s.asym:
            idx = (torch.round(W / sc) + zp[:, None]).clamp(0, s.L - 1); q = (idx - zp[:, None]) * sc
        else:
            r = s._uq(W / sc); q = r * sc; idx = (r + (s.L - 1) / 2) if s.L % 2 == 1 else (r - 0.5 + s.L / 2)
        return q, idx
    def add_hist(s, Idx):
        s.hist += torch.bincount(Idx.long().flatten(), minlength=s.L)[:s.L].cpu()
    def bits(s):
        """returns dict: idx bits (H0*1.005), scale bits by coding, zp bits."""
        sc_all = torch.cat([a.flatten() for a, _ in s.sc_list]); n = sc_all.numel()
        out = dict(idx=hist_bits(s.hist) * 1.005, nscale=n)
        if s.sc_list[0][1] is None: out["scale"] = {"fp16": 16.0 * n}
        else:
            # stack codes as [R', ng]
            codes = torch.stack([c for _, c in s.sc_list], 1); out["scale"] = scale_bits_all(codes, sc_all)
        if s.zp_list:
            z = torch.stack(s.zp_list, 1); out["zp"] = H0_bits(z) * 1.005
        return out

# ------------------------------------------------------------------ rounding engines (operate in rotated space)
@torch.no_grad()
def rtn(Wr, q1):
    R, C = Wr.shape; Wq = torch.empty_like(Wr); g = q1.g; Idx = torch.empty(R, C, dtype=torch.int8, device=Wr.device)
    assert C % g == 0
    for c in range(0, C, g):
        sc, zp = q1.group_scale(Wr[:, c:c + g]); q, ix = q1.quant_cols(Wr[:, c:c + g], sc, zp); Wq[:, c:c + g] = q; Idx[:, c:c + g] = ix.to(torch.int8)
    q1.add_hist(Idx); return Wq

@torch.no_grad()
def gptq(Wr, Hr, q1, damp=0.01, blk=128):
    W = Wr.clone().float(); R, C = W.shape; H = Hr.clone().float()
    dead = torch.diag(H) == 0; H[dead, dead] = 1; W[:, dead] = 0
    H += damp * torch.diag(H).mean() * torch.eye(C, device=W.device)
    try:
        U = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(H)), upper=True)
    except Exception:
        H += 0.1 * torch.diag(H).mean() * torch.eye(C, device=W.device)
        U = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(H)), upper=True)
    blk = max(blk, q1.g); Q = torch.zeros_like(W); g = q1.g; Idx = torch.empty(R, C, dtype=torch.int8, device=W.device); assert C % g == 0
    for i1 in range(0, C, blk):
        i2 = min(i1 + blk, C); W1 = W[:, i1:i2].clone(); Q1 = torch.zeros_like(W1); E1 = torch.zeros_like(W1); Ub = U[i1:i2, i1:i2]
        for i in range(i2 - i1):
            if (i1 + i) % g == 0: sc, zp = q1.group_scale(W1[:, i:i + g])
            w = W1[:, i]; qq, ix = q1.quant_cols(w[:, None], sc, zp); q = qq[:, 0]; Q1[:, i] = q; Idx[:, i1 + i] = ix[:, 0].to(torch.int8)
            e = (w - q) / Ub[i, i]; W1[:, i:] -= e[:, None] @ Ub[i, i:][None, :]; E1[:, i] = e
        Q[:, i1:i2] = Q1; W[:, i2:] -= E1 @ U[i1:i2, i2:]
    q1.add_hist(Idx); return Q

# ------------------------------------------------------------------ eval
def wilson(k, n, z=1.96):
    if n == 0: return (0, 0)
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)

def load_items(path, task):
    return [json.loads(l) for l in open(path) if json.loads(l)["task"] == task]

@torch.no_grad()
def generate_labels(model, tok, items, labels, bs=48, max_new=6, mask_ids=None, retok=None):
    """real greedy generation; returns list of parsed label (or None). mask_ids: bool tensor [V] True=allowed output tokens (vocab trim sim)."""
    dev = next(model.parameters()).device; outs = {}
    prompts = [tok.apply_chat_template([{"role": "user", "content": it["user"]}], tokenize=False, add_generation_prompt=True) for it in items]
    ids = [tok(p, add_special_tokens=False)["input_ids"] for p in prompts]
    if retok is not None: ids = [retok(x) for x in ids]
    order = sorted(range(len(items)), key=lambda i: len(ids[i])); tok.padding_side = "left"
    pad = tok.pad_token_id if tok.pad_token_id is not None else 0
    for b in range(0, len(order), bs):
        idx = order[b:b + bs]; L = max(len(ids[i]) for i in idx)
        inp = torch.tensor([[pad] * (L - len(ids[i])) + ids[i] for i in idx], device=dev)
        att = torch.tensor([[0] * (L - len(ids[i])) + [1] * len(ids[i]) for i in idx], device=dev)
        kw = {}
        if mask_ids is not None:
            from transformers import LogitsProcessor, LogitsProcessorList
            m = mask_ids.to(dev)
            class Mk(LogitsProcessor):
                def __call__(self, input_ids, scores): return scores.masked_fill(~m[None, :scores.shape[1]], float("-inf"))
            kw["logits_processor"] = LogitsProcessorList([Mk()])
        g = model.generate(input_ids=inp, attention_mask=att, max_new_tokens=max_new, do_sample=False, pad_token_id=pad, **kw)
        for j, i in enumerate(idx): outs[i] = tok.decode(g[j, L:], skip_special_tokens=True)
    res = []
    for i in range(len(items)):
        t = outs[i].strip().lower(); lab = None
        for l in labels:
            if t.startswith(l.lower()): lab = l; break
        res.append(lab)
    return res

def score(items, preds, base_correct=None):
    corr = [p == it["answer"] for p, it in zip(preds, items)]; n = len(items); acc = sum(corr) / n
    out = dict(n=n, acc=acc, correct=corr)
    if base_correct is not None:
        bc = [i for i in range(n) if base_correct[i]]; keep = sum(corr[i] for i in bc)
        lo, hi = wilson(keep, len(bc)); out.update(retention=keep / len(bc), ret_lo=lo, ret_hi=hi, n_base_correct=len(bc))
        cells = collections.defaultdict(lambda: [0, 0])
        for i in bc:
            c = cells[(items[i]["src"], items[i]["kind"])]; c[1] += 1; c[0] += corr[i]
        r = {f"{k[0]}/{k[1]}": v[0] / v[1] for k, v in cells.items()}; out["min_cell"] = min(r.values()); out["min_cell_name"] = min(r, key=r.get)
    return out
