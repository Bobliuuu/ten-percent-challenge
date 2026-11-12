"""Variant B: GRADIENT-FREE DELETION feedback for L1 (Bonsai-8B 1-bit). Feedback only ZEROES existing per-128-group scales of the 1-bit down_proj of the last N layers
(unit = (layer, output row i, group g): weight[i, g*128:(g+1)*128] = sign * scale -> 0). No value is fitted, no gradient, no optimiser. The deletion set is a bitmap.
Saliency (closed form, first order in the residual of that layer, final norm treated as fixed):
  zeroing unit (i,g) changes the layer's output row i by  dy = - s_ig * a_ig(k),  a_ig(k) = sum_{c in group} sign[i,g,c] * k_gc  (k = down_proj input),
  hence the logit margin (label t vs competitor c) at an annotated site by  dm = - v_i * s_ig * a_ig(k),  v = gamma * (W_U[t]-W_U[c]) / rms(x_final).
gainA = mean over annotations of dm / gap (annotated texts);  gainC = same direction (label vs the model's own answer) on the UNLABELED calibration prompts;
dmg = calibration damage (squared output-row change weighted by ||gamma_i W_U[:, i]||^2, plus squared change of the own-answer margin at answer positions); wanda = s_ig * ||k_g|| (activation x weight)."""
import sys, os, json, zlib, numpy as np, torch
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE + "/.."); 
import bonsai_loader as BL, common, feedback as FB
G = 128
torch.backends.cuda.matmul.allow_tf32 = True

def _batches(cal, bs=16):
    order = sorted(range(len(cal)), key=lambda i: len(cal[i][0]))
    for s in range(0, len(order), bs):
        idx = order[s:s + bs]; L = max(len(cal[i][0]) for i in idx); ids = torch.zeros((len(idx), L), dtype=torch.long); am = torch.zeros((len(idx), L), dtype=torch.long)
        for b, i in enumerate(idx): q, ap = cal[i]; ids[b, :len(q)] = torch.tensor(q); am[b, :len(q)] = 1
        yield idx, ids, am

@torch.no_grad()
def calib_stats(m, tok, cal, layers, tR, tsub=6000, seed=0):
    """per layer li -> dict(dmg [D,G] (token part, answer-margin part), wan [D,G], gC [D,G]); all on GPU fp32."""
    dev = m.device; W = m.lm_head.weight.data; gamma = m.model.norm.weight.float(); eps = m.config.rms_norm_eps; LY = m.model.layers; D = m.config.hidden_size
    colnorm2 = (gamma * W.float().norm(dim=0)) ** 2                                   # [D] logit-space sensitivity of residual coordinate i
    # pass 1: final residual at answer positions, own next token a, runner-up b
    AP = []; Xs = []; st = {}
    h = m.model.norm.register_forward_pre_hook(lambda mod, inp: st.__setitem__("x", inp[0]))
    for idx, ids, am in _batches(cal):
        m.model(input_ids=ids.to(dev), attention_mask=am.to(dev))
        for b, i in enumerate(idx):
            q, ap = cal[i]
            for j, p in enumerate(ap): AP.append((i, p, q[p + 1], j == 0)); Xs.append(st["x"][b, p].float())
    h.remove(); X = torch.stack(Xs); rms = (X.pow(2).mean(-1, keepdim=True) + eps).sqrt(); xn = X * gamma / rms
    lg = (xn.to(W.dtype) @ W.T).float(); a = torch.tensor([r[2] for r in AP], device=dev); first = torch.tensor([r[3] for r in AP], device=dev)
    la = lg.gather(1, a[:, None])[:, 0]; lg2 = lg.clone(); lg2.scatter_(1, a[:, None], -1e9); b_ = lg2.argmax(1)
    v_ab = gamma[None] * (W[a].float() - W[b_].float()) / rms                           # [M,D]  d(margin own answer vs runner-up)/dy
    fi = first.nonzero()[:, 0]; v_R = gamma[None] * (W[tR].float()[None] - W[a[fi]].float()) / rms[fi]   # [P,D] d(margin Recipe vs own answer)/dy at first answer positions
    skip = (a[fi] == tR); v_R[skip] = 0
    g = torch.Generator().manual_seed(seed); out = {}
    for li in layers:
        wt = LY[li].mlp.down_proj.weight.data; F = wt.shape[1]; Gn = F // G; sg = torch.sign(wt.float()); sg[sg == 0] = 1; scl = wt.float().abs().reshape(D, Gn, G).amax(-1)
        Kt = []; Kap = []; st = {}; hh = LY[li].mlp.down_proj.register_forward_pre_hook(lambda mod, inp: st.__setitem__("k", inp[0]))
        class Stop(Exception): pass
        def stop(mod, inp): raise Stop()
        hs = LY[li].mlp.down_proj.register_forward_pre_hook(stop)   # runs after the capture hook (registered later)
        try:
            for idx, ids, am in _batches(cal):
                try: m.model(input_ids=ids.to(dev), attention_mask=am.to(dev))
                except Stop: pass
                k = st["k"]; Kt.append(k[am.bool().to(dev)])
                for b, i in enumerate(idx): Kap.append(k[b, cal[i][1]])
        finally: hh.remove(); hs.remove()
        Kt = torch.cat(Kt); Kap = torch.cat(Kap); sel = torch.randperm(len(Kt), generator=g)[:tsub].to(dev); Kt = Kt[sel]
        Ktg = Kt.reshape(len(Kt), Gn, G).permute(1, 0, 2).contiguous()                  # [G,T,128]
        Kag = Kap.reshape(len(Kap), Gn, G).permute(1, 0, 2).contiguous()                # [G,M,128]
        kn = Kt.float().pow(2).sum(-1).mean(); wan_g = Kt.float().reshape(len(Kt), Gn, G).pow(2).sum(-1).mean(0).sqrt()   # [Gn]
        dmg_t = torch.zeros(D, Gn, device=dev); dmg_a = torch.zeros(D, Gn, device=dev); gC = torch.zeros(D, Gn, device=dev)
        for r0 in range(0, D, 256):
            sb = sg[r0:r0 + 256].reshape(-1, Gn, G).permute(1, 2, 0).contiguous().half()   # [G,128,r]
            at = torch.bmm(Ktg, sb).float()                                              # [G,T,r]
            dmg_t[r0:r0 + 256] = ((at.pow(2).mean(1)) * (scl[r0:r0 + 256].T ** 2) * colnorm2[None, r0:r0 + 256]).T
            aa = torch.bmm(Kag, sb).float()                                              # [G,M,r]
            vv = v_ab[:, r0:r0 + 256]                                                    # [M,r]
            dmg_a[r0:r0 + 256] = (((aa * vv[None]).pow(2).mean(1)) * (scl[r0:r0 + 256].T ** 2)).T
            af = aa[:, fi]                                                               # [G,P,r]
            gC[r0:r0 + 256] = (-(af * v_R[:, r0:r0 + 256][None]).mean(1) * scl[r0:r0 + 256].T).T
            del at, aa, af
        out[li] = dict(dmg_t=dmg_t, dmg_a=dmg_a, gC=gC, wan=scl * wan_g[None])
        del Kt, Kap, Ktg, Kag
    return out

@torch.no_grad()
def ann_gain(m, tok, ann, layers, T=None):
    """per-annotation gain tensors {li: [n_ann, D, G]}: dm/gap summed over that annotation's mismatching answer sites. ann rows: dict(query,label)."""
    dev = m.device; W = m.lm_head.weight.data; gamma = m.model.norm.weight.float(); eps = m.config.rms_norm_eps; LY = m.model.layers; D = m.config.hidden_size; n = len(ann)
    per = {li: [] for li in layers}
    wts = {}
    for li in layers:
        wt = LY[li].mlp.down_proj.weight.data; sg = torch.sign(wt.float()); sg[sg == 0] = 1; wts[li] = (sg.reshape(D, -1, G), wt.float().abs().reshape(D, -1, G).amax(-1))
    for r in ann:
        sites = FB.make_sites(m, tok, [r], T)
        if not sites:
            for li in layers: per[li].append(None)
            continue
        feats = {li: [] for li in layers}; Xs = []; Ts = []
        for ids, pos, tgt in sites:
            st = {}; hs = [LY[li].mlp.down_proj.register_forward_pre_hook(lambda mod, inp, li=li: st.__setitem__(li, inp[0][0, pos].float())) for li in layers]
            hx = m.model.norm.register_forward_pre_hook(lambda mod, inp: st.__setitem__("x", inp[0][0, pos].float()))
            FB._fwd(m, torch.tensor([ids], device=dev)); [h.remove() for h in hs]; hx.remove()
            for li in layers: feats[li].append(st[li])
            Xs.append(st["x"]); Ts += tgt
        X = torch.cat(Xs); T_ = torch.tensor(Ts, device=dev); rms = (X.pow(2).mean(-1, keepdim=True) + eps).sqrt(); xn = X * gamma / rms
        lg = (xn.to(W.dtype) @ W.T).float(); zt = lg.gather(1, T_[:, None])[:, 0]; lg.scatter_(1, T_[:, None], -1e9); zc, ci = lg.max(1); gap = (zc - zt).clamp(min=1.0)
        v = gamma[None] * (W[T_].float() - W[ci].float()) / rms / gap[:, None]               # [S,D]
        for li in layers:
            K = torch.cat(feats[li]).half(); Kg = K.reshape(len(K), -1, G).permute(1, 0, 2).contiguous(); sg, scl = wts[li]; Gn = Kg.shape[0]; gain = torch.zeros(D, Gn, device=dev)
            for r0 in range(0, D, 512):
                sb = sg[r0:r0 + 512].permute(1, 2, 0).contiguous().half(); a = torch.bmm(Kg, sb).float()      # [G,S,r]
                gain[r0:r0 + 512] = (-(a * v[:, r0:r0 + 512][None]).sum(1) * scl[r0:r0 + 512].T).T
            per[li].append(gain)
    res = {}
    for li in layers:
        z = torch.zeros(D, per[li][[i for i, x in enumerate(per[li]) if x is not None][0]].shape[1], device=dev) if any(x is not None for x in per[li]) else None
        res[li] = torch.stack([x if x is not None else z for x in per[li]]) if z is not None else None
    return res

class Dele:
    """holds original down_proj weights of the candidate layers; apply(mask dict) zeroes units; restore()."""
    def __init__(self, m, layers):
        self.m = m; self.layers = layers; self.D = m.config.hidden_size; self.w0 = {li: m.model.layers[li].mlp.down_proj.weight.data.clone() for li in layers}
    @torch.no_grad()
    def apply(self, mask):          # mask {li: bool [D,G]} True = delete
        for li in self.layers:
            w = self.m.model.layers[li].mlp.down_proj.weight.data
            if li in mask: w.copy_((self.w0[li].view(self.D, -1, G) * (~mask[li])[:, :, None].half()).view_as(w))
            else: w.copy_(self.w0[li])
    @torch.no_grad()
    def restore(self):
        for li in self.layers: self.m.model.layers[li].mlp.down_proj.weight.data.copy_(self.w0[li])

def zterm(x, layers):
    r = torch.cat([x[li].flatten() for li in layers]).float().pow(2).mean().sqrt().clamp(min=1e-12); return {li: x[li] / r for li in layers}

def make_scores(cs, layers, rule, mu, nu):
    """returns fn(gainA_mean {li:[D,G]}) -> {li: score}. cs = calib_stats output. rule in gain/contrast/ratio/wanda/wandaratio"""
    zC = zterm({li: cs[li]["gC"].clamp(min=0) for li in layers}, layers); zD = zterm({li: (cs[li]["dmg_t"] + cs[li]["dmg_a"]).sqrt() for li in layers}, layers); zW = zterm({li: cs[li]["wan"] for li in layers}, layers)
    def fn(gA):
        zA = zterm(gA, layers)
        if rule == "gain": return zA
        if rule == "contrast": return {li: zA[li] - mu * zC[li] - nu * zD[li] for li in layers}
        if rule == "ratio": return {li: zA[li].clamp(min=0) / (zD[li] + 0.1) for li in layers}
        if rule == "wanda": return {li: zA[li] - nu * zW[li] for li in layers}
        if rule == "wandaratio": return {li: zA[li].clamp(min=0) / (zW[li] + 0.1) for li in layers}
        raise KeyError(rule)
    return fn

def topk_mask(score, layers, k):
    flat = torch.cat([score[li].flatten() for li in layers]); k = min(k, int((flat > 0).sum())); 
    if k <= 0: return {}, 0
    thr = flat.topk(k).values[-1]; mask = {li: score[li] >= thr for li in layers}; mask = {li: mk & (score[li] > 0) for li, mk in mask.items()}
    return mask, int(sum(mk.sum() for mk in mask.values()))

def pack_bitmap(mask, layers):
    """compressed bitmap: concatenated bits of all candidate layers (row-major [D,G]) -> zlib. returns bytes"""
    bits = torch.cat([(mask[li] if li in mask else torch.zeros_like(next(iter(mask.values())))).flatten().cpu() for li in layers]).numpy().astype(np.uint8)
    return zlib.compress(np.packbits(bits).tobytes(), 9)
def unpack_bitmap(b, layers, D, Gn):
    bits = np.unpackbits(np.frombuffer(zlib.decompress(b), np.uint8))[:len(layers) * D * Gn].reshape(len(layers), D, Gn).astype(bool); return {li: torch.from_numpy(bits[i]) for i, li in enumerate(layers)}

@torch.no_grad()
def first_tok(m, tok, prompts, bs=32):
    """argmax token at the first answer position for each prompt"""
    out = [None] * len(prompts); order = sorted(range(len(prompts)), key=lambda i: len(prompts[i])); tok.padding_side = "left"
    for s in range(0, len(order), bs):
        idx = order[s:s + bs]; enc = tok([common.chat(tok, prompts[i]) for i in idx], return_tensors="pt", padding=True, add_special_tokens=False, return_token_type_ids=False).to(m.device)
        lg = m(**enc, logits_to_keep=1).logits[:, -1].argmax(-1).tolist()
        for j, i in enumerate(idx): out[i] = lg[j]
    return out
