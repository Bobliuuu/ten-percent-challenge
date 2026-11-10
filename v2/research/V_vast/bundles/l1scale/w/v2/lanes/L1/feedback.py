"""ONE COMMAND THAT TAKES ANNOTATED ITEMS AS FEEDBACK.
  user mode:     python feedback.py --artifact DIR --task task.json --ann ann.jsonl      ann rows: {"text": ..., "label": <correct label word, may be a NEW word>}
  harness mode:  python feedback.py --artifact DIR --annotations ann.jsonl               ann rows: {id, query (full prompt), tools, gold:[{"name": label}]}
At most 25 rows. Feedback = a closed-form edit of EXISTING tensors, applied inside the model's own forward pass: columns of the existing MLP down_proj
matrices of the last LAYERS transformer layers (no head, no new tensor, no adapter, no lookup, annotations are not stored, nothing is in the prompt, no gradient/optimiser).
For every annotated answer position where the ORIGINAL model's argmax differs from the annotated token t (final residual x, best competitor c):
    residual target   dx = rms(x) * (gap + margin) * u / (u.u),   u = gamma * (W_U[t] - W_U[c])            (lifts the label logit over the competitor by `margin`)
    layer l (ascending) delivers 1/(layers left) of the CURRENT remaining dx through the COLS highest-scoring columns J of down_proj:
    dW[:, J] = R K^T (lam (C + awt C_ans + ridge I) + K K^T)^-1,   K = down_proj inputs (12288-d MLP activations) at the annotated sites, restricted to J,
    C = second moment of the same activations over all tokens of the shipped UNLABELED calibration prompts + the model's own greedy answers (protects ordinary behaviour),
    C_ans = the same second moment over answer/EOS positions only (protects the old labels).
Default carrier `scale`: instead of columns the same closed-form edit changes ONLY the existing per-128-group fp16 scales of the 1-bit down_proj (weight = sign * scale; no new tensor); DIR/patch.pt holds the replaced scales (could be written into scales.bin in place). `--carrier cols` stores fp16 replacement columns, `--carrier lr` rank-k factors; bonsai_loader.load overlays the patch on the existing down_proj tensors.
Always starts from the original weights (patch removed first). Zero rows => patch.pt removed => artifact byte-identical to the base artifact."""
import sys, os, json, argparse
import torch
sys.dont_write_bytecode = True   # never write into the artifact dir
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bonsai_loader as BL, common
DEF = dict(layers=6, cols=1024, lam=1000.0, awt=10.0, margin=16.0, ridge=0.05); LAM_SCALE = 100.0

class _Stop(Exception): pass

@torch.no_grad()
def _fwd(m, ids, mask=None):
    return m.model(input_ids=ids, attention_mask=mask).last_hidden_state

def _cap(layer_mod, store, key, pos=None, stop=False):
    def hook(mod, inp):
        k = inp[0]; store[key] = k if pos is None else k[0, pos]
        if stop: raise _Stop()
    return layer_mod.register_forward_pre_hook(hook)

@torch.no_grad()
def make_sites(m, tok, ann, T, aug=0):
    """annotated sequences + the answer positions where the ORIGINAL model's argmax differs from the annotated token. -> list of (ids, pos, tgt)"""
    dev = m.device; W = m.lm_head.weight; EOS = tok.convert_tokens_to_ids("<|im_end|>"); out = []
    lab = lambda r: r["label"] if "label" in r else r["gold"][0]["name"]
    for r in common.augment(ann, T, aug):
        p = tok(common.chat(tok, common.prompt_of(r, T)), add_special_tokens=False).input_ids; g = tok(lab(r), add_special_tokens=False).input_ids + [EOS]
        ids = p + g; pos = list(range(len(p) - 1, len(p) + len(g) - 1)); h = _fwd(m, torch.tensor([ids], device=dev))[0, pos]
        am = (h.to(W.dtype) @ W.T).argmax(-1).tolist(); keep = [j for j in range(len(g)) if am[j] != g[j]]
        if keep: out.append((ids, [pos[j] for j in keep], [g[j] for j in keep]))
    return out

@torch.no_grad()
def calib_seqs(m, tok, prompts, bs=16, mx=6):
    """calibration sequences = prompt + the model's own greedy answer (up to EOS). -> list of (ids, answer_positions)"""
    dev = m.device; EOS = tok.convert_tokens_to_ids("<|im_end|>"); P = [tok(common.chat(tok, q), add_special_tokens=False).input_ids for q in prompts]; out = [None] * len(P)
    order = sorted(range(len(P)), key=lambda i: len(P[i]))
    for s in range(0, len(order), bs):
        idx = order[s:s + bs]; L = max(len(P[i]) for i in idx); ids = torch.full((len(idx), L), tok.pad_token_id); am = torch.zeros((len(idx), L), dtype=torch.long)
        for r, i in enumerate(idx): ids[r, L - len(P[i]):] = torch.tensor(P[i]); am[r, L - len(P[i]):] = 1          # left pad for generate
        g = m.generate(input_ids=ids.to(dev), attention_mask=am.to(dev), max_new_tokens=mx, do_sample=False, pad_token_id=tok.pad_token_id)[:, L:].tolist()
        for r, i in enumerate(idx):
            o = g[r]; o = o[:o.index(EOS) + 1] if EOS in o else o
            out[i] = (P[i] + o, list(range(len(P[i]) - 1, len(P[i]) + len(o) - 1)))
    return out

@torch.no_grad()
def _cal_cov(m, cal, li, J, bs=16):
    """second moments of the layer-li down_proj inputs restricted to columns J: all tokens (C) and answer positions only (Ca), under the CURRENT model."""
    dev = m.device; r = len(J); C = torch.zeros((r, r), device=dev); Ca = torch.zeros((r, r), device=dev); n = na = 0; st = {}
    h = _cap(m.model.layers[li].mlp.down_proj, st, "k", stop=True)
    order = sorted(range(len(cal)), key=lambda i: len(cal[i][0]))
    try:
        for s in range(0, len(order), bs):
            idx = order[s:s + bs]; L = max(len(cal[i][0]) for i in idx); ids = torch.full((len(idx), L), 0, dtype=torch.long); am = torch.zeros((len(idx), L), dtype=torch.long); ma = torch.zeros((len(idx), L), dtype=torch.bool)
            for b, i in enumerate(idx): q, ap = cal[i]; ids[b, :len(q)] = torch.tensor(q); am[b, :len(q)] = 1; ma[b, ap] = True
            try: m.model(input_ids=ids.to(dev), attention_mask=am.to(dev))
            except _Stop: pass
            k = st["k"][..., J]; kk = k[am.bool().to(dev)].float(); C.addmm_(kk.T, kk); n += len(kk); kk = k[ma.to(dev)].float(); Ca.addmm_(kk.T, kk); na += len(kk)
    finally: h.remove()
    return C / max(n, 1), Ca / max(na, 1)

@torch.no_grad()
def compute_patch(m, tok, ann, T, calib_prompts, layers=DEF["layers"], cols=DEF["cols"], lam=None, awt=DEF["awt"], margin=DEF["margin"], ridge=DEF["ridge"], log=None, cal=None, aug=0, carrier="scale", sites=None):
    """Edits the loaded model IN PLACE (fp16 columns) and returns the patch {"layers": [..], "cols": [int16 [r]], "vals": [fp16 [hidden, r]]}; None if nothing needs teaching.
    The model must be the ORIGINAL (unpatched) one."""
    if lam is None: lam = LAM_SCALE if carrier == "scale" else DEF["lam"]      # per-carrier default strength (scale carrier: lam 100, as measured; others 1000)
    dev = m.device; W = m.lm_head.weight.data; LY = m.model.layers; NL = len(LY); gamma = m.model.norm.weight.float(); eps = m.config.rms_norm_eps; D = m.config.hidden_size
    sites = sites if sites is not None else make_sites(m, tok, ann, T, aug)
    if not sites: return None
    if carrier == "scale": return compute_patch_scale(m, tok, sites, calib_prompts, layers, lam, awt, margin, ridge, log, cal)
    cal = cal or calib_seqs(m, tok, calib_prompts); LS = list(range(NL - layers, NL)); out = {"layers": [], "cols": [], "vals": [], "orig": []}
    for step, li in enumerate(LS):
        rem = len(LS) - step; Ks, Xs, Ts = [], [], []
        for ids, pos, tgt in sites:
            st = {}; h1 = _cap(LY[li].mlp.down_proj, st, "k", pos); h2 = m.model.norm.register_forward_pre_hook(lambda mod, inp: st.__setitem__("x", inp[0][0, pos]))
            _fwd(m, torch.tensor([ids], device=dev)); h1.remove(); h2.remove(); Ks.append(st["k"].float()); Xs.append(st["x"].float()); Ts += tgt
        K = torch.cat(Ks).T.contiguous(); X = torch.cat(Xs); T_ = torch.tensor(Ts, device=dev)                  # K [12288, n], X [n, D]
        xn = X * torch.rsqrt(X.pow(2).mean(-1, keepdim=True) + eps) * gamma; lg = (xn.to(W.dtype) @ W.T).float()
        zt = lg.gather(1, T_[:, None])[:, 0]; lg.scatter_(1, T_[:, None], -1e9); zc, ci = lg.max(1); gap = (zc - zt + margin).clamp(min=0)
        u = gamma[None] * (W[T_].float() - W[ci].float()); rms = X.pow(2).mean(-1, keepdim=True).add(eps).sqrt()
        R = (rms * gap[:, None] * u / u.pow(2).sum(-1, keepdim=True).clamp(min=1e-9) / rem).T.contiguous()          # [D, n]
        J = ((K.abs() / K.norm(dim=0, keepdim=True).clamp(min=1e-6)).sum(1)).topk(cols).indices.sort().values
        C, Ca = _cal_cov(m, cal, li, J); Cs = C + awt * Ca; Am = lam * (Cs + ridge * Cs.diagonal().mean() * torch.eye(len(J), device=dev)); Kj = K[J]
        Z = torch.linalg.solve(Am, Kj); S = torch.linalg.inv(torch.eye(K.shape[1], device=dev) + Kj.T @ Z); dW = R @ S @ Z.T                      # [D, r]
        wt = LY[li].mlp.down_proj.weight; out["orig"].append(wt.data[:, J].clone())
        if carrier == "lr":      # same delta stored as rank-k fp16 factors (k <= number of sites): new = orig + A @ B
            U, Sv, Vh = torch.linalg.svd(dW, full_matrices=False); k = min(int((Sv > 1e-6 * Sv[0]).sum()), K.shape[1]); A_ = (U[:, :k] * Sv[:k]).half(); B_ = Vh[:k].half(); new = (wt.data[:, J].float() + A_.float() @ B_.float()).half()
            out.setdefault("A", []).append(A_.cpu()); out.setdefault("B", []).append(B_.cpu()); out.setdefault("kind", "lr")
        else: new = (wt.data[:, J].float() + dW).half(); out["vals"].append(new.cpu())
        wt.data[:, J] = new; out["layers"].append(li); out["cols"].append(J.short().cpu())
        if log: log(f"layer {li}: sites {K.shape[1]} mean gap {float(gap.mean()):.2f}")
    return out

@torch.no_grad()
def compute_patch_scale(m, tok, sites, calib_prompts, layers, lam, awt, margin, ridge, log=None, cal=None, rows_blk=64):
    """CARRIER 'scale': the edit changes ONLY the existing per-128-group scales of the 1-bit down_proj of the last `layers` layers (weights stay sign * scale; no new tensor, no new bytes in
    the pack: the patch.pt holds the replaced scale values, [4096, 96] fp16 per layer, and could equally be written back into scales.bin).
    Output row i of down_proj changes by dy_i = sum_g ds[i,g] * a_i[g](k),  a_i[g](k) = sum_{j in group g} sign[i,j] k_j.  Same closed form as compute_patch, per row, on the 96 group features."""
    dev = m.device; W = m.lm_head.weight.data; LY = m.model.layers; NL = len(LY); gamma = m.model.norm.weight.float(); eps = m.config.rms_norm_eps; D = m.config.hidden_size
    cal = cal or calib_seqs(m, tok, calib_prompts); LS = list(range(NL - layers, NL)); out = {"layers": [], "scales": [], "kind": "scale", "orig_w": []}
    for step, li in enumerate(LS):
        rem = len(LS) - step; Ks, Xs, Ts = [], [], []
        for ids, pos, tgt in sites:
            st = {}; h1 = _cap(LY[li].mlp.down_proj, st, "k", pos); h2 = m.model.norm.register_forward_pre_hook(lambda mod, inp: st.__setitem__("x", inp[0][0, pos]))
            _fwd(m, torch.tensor([ids], device=dev)); h1.remove(); h2.remove(); Ks.append(st["k"].float()); Xs.append(st["x"].float()); Ts += tgt
        Kt = torch.cat(Ks); X = torch.cat(Xs); T_ = torch.tensor(Ts, device=dev)                                    # Kt [n, F]
        xn = X * torch.rsqrt(X.pow(2).mean(-1, keepdim=True) + eps) * gamma; lg = (xn.to(W.dtype) @ W.T).float()
        zt = lg.gather(1, T_[:, None])[:, 0]; lg.scatter_(1, T_[:, None], -1e9); zc, ci = lg.max(1); gap = (zc - zt + margin).clamp(min=0)
        u = gamma[None] * (W[T_].float() - W[ci].float()); rms = X.pow(2).mean(-1, keepdim=True).add(eps).sqrt()
        R = (rms * gap[:, None] * u / u.pow(2).sum(-1, keepdim=True).clamp(min=1e-9) / rem)                          # [n, D]
        # calibration activations of this layer under the CURRENT model (all tokens + answer-position mask)
        F = Kt.shape[1]; G = F // 128; st = {}; h = _cap(LY[li].mlp.down_proj, st, "k", stop=True); KA = []; MA = []
        order = sorted(range(len(cal)), key=lambda i: len(cal[i][0]))
        try:
            for s0 in range(0, len(order), 16):
                idx = order[s0:s0 + 16]; L = max(len(cal[i][0]) for i in idx); ids = torch.zeros((len(idx), L), dtype=torch.long); am = torch.zeros((len(idx), L), dtype=torch.long); ma = torch.zeros((len(idx), L), dtype=torch.bool)
                for b, i in enumerate(idx): q, ap = cal[i]; ids[b, :len(q)] = torch.tensor(q); am[b, :len(q)] = 1; ma[b, ap] = True
                try: m.model(input_ids=ids.to(dev), attention_mask=am.to(dev))
                except _Stop: pass
                msk = am.bool().to(dev); KA.append(st["k"][msk]); MA.append(ma.to(dev)[msk])
        finally: h.remove()
        KA = torch.cat(KA); MA = torch.cat(MA); T0 = len(KA); wt = LY[li].mlp.down_proj.weight; w0 = wt.data.clone(); out["orig_w"].append(w0.cpu() if False else w0)
        sgn = torch.sign(w0.float()); sgn[sgn == 0] = 1; scl = w0.float().abs().reshape(D, G, 128).amax(-1)                 # [D, G] existing scales
        newscl = scl.clone(); Kg = Kt.reshape(len(Kt), G, 128); KAg = KA.reshape(T0, G, 128)
        for r0 in range(0, D, rows_blk):
            sb = sgn[r0:r0 + rows_blk].reshape(-1, G, 128)                                                         # [r, G, 128]
            An = torch.einsum("ngc,rgc->rng", Kg, sb)                                                              # [r, n, G]  features at the annotated sites
            Ac = torch.einsum("tgc,rgc->trg", KAg.float(), sb)                                                     # [T, r, G]
            C = torch.einsum("trg,trh->rgh", Ac, Ac) / T0; Ca = torch.einsum("trg,trh->rgh", Ac[MA], Ac[MA]) / max(int(MA.sum()), 1)
            Cs = C + awt * Ca; eye = torch.eye(G, device=dev)[None]; Am = lam * (Cs + ridge * Cs.diagonal(dim1=1, dim2=2).mean(-1)[:, None, None] * eye)
            Z = torch.linalg.solve(Am, An.transpose(1, 2))                                                         # [r, G, n]
            S_ = torch.linalg.inv(torch.eye(len(Kt), device=dev)[None] + An @ Z)                                    # [r, n, n]
            dS = torch.einsum("rgn,rnm,mr->rg", Z, S_, R[:, r0:r0 + rows_blk])                                      # [r, G]
            newscl[r0:r0 + rows_blk] += dS
        newscl = newscl.half(); wt.data.copy_((sgn.reshape(D, G, 128) * newscl.float()[:, :, None]).reshape(D, F).half())
        out["layers"].append(li); out["scales"].append(newscl.cpu())
        if log: log(f"layer {li}: sites {len(Kt)} mean gap {float(gap.mean()):.2f} rel scale change {float(((newscl.float()-scl).norm()/scl.norm())):.4f}")
        del KA, KAg, Ac
    return out

def merge_patch(old, new):
    """continual feedback: later patch (computed on the already-patched model) overrides the earlier one where both touch the same columns/layer."""
    if old is None: return new
    if new is None: return old
    assert old.get("kind", "cols") == new.get("kind", "cols")
    kind = new.get("kind", "cols"); res = {"layers": [], "cols": [], "vals": []} if kind == "cols" else ({"layers": [], "scales": [], "kind": "scale"} if kind == "scale" else None)
    assert res is not None, "merge only for the cols and scale carriers"
    ls = sorted(set(old["layers"]) | set(new["layers"]))
    for li in ls:
        o = old["layers"].index(li) if li in old["layers"] else None; n = new["layers"].index(li) if li in new["layers"] else None; res["layers"].append(li)
        if kind == "scale": res["scales"].append(new["scales"][n] if n is not None else old["scales"][o]); continue
        d = {}
        if o is not None:
            for c, j in enumerate(old["cols"][o].tolist()): d[j] = old["vals"][o][:, c]
        if n is not None:
            for c, j in enumerate(new["cols"][n].tolist()): d[j] = new["vals"][n][:, c]
        js = sorted(d); res["cols"].append(torch.tensor(js, dtype=torch.int16)); res["vals"].append(torch.stack([d[j] for j in js], 1))
    return res

def patch_bytes(P):
    if P is None: return 0
    return sum(t.numel() * t.element_size() for k in ("cols", "vals", "A", "B", "scales") for t in P.get(k, []))

def undo(m, P):
    """exact undo of compute_patch's in-place edit (experiments only)"""
    if P.get("kind") == "scale":
        for li, o in zip(P["layers"], P["orig_w"]): m.model.layers[li].mlp.down_proj.weight.data.copy_(o)
        return
    for li, c, o in zip(P["layers"], P["cols"], P["orig"]): m.model.layers[li].mlp.down_proj.weight.data[:, c.long().to(m.device)] = o

def apply_patch(m, P, save=None):
    """overlay patch columns on the existing down_proj tensors; if save is a dict, the replaced original columns are stored there (for exact restore in experiments)."""
    import bonsai_loader as _BL
    if P.get("kind") in ("lr", "scale"):
        _BL.apply_patch_dict(m, P); return
    for li, c, v in zip(P["layers"], P["cols"], P["vals"]):
        w = m.model.layers[li].mlp.down_proj.weight.data; c = c.long().to(w.device)
        if save is not None: save[li] = (c, w[:, c].clone())
        w[:, c] = v.to(w.device)

def restore(m, save):
    for li, (c, v) in save.items(): m.model.layers[li].mlp.down_proj.weight.data[:, c] = v
    save.clear()

def main():
    import argparse
    from transformers import AutoTokenizer
    a = argparse.ArgumentParser(); a.add_argument("--artifact", "--dir", dest="art", default=os.path.dirname(os.path.abspath(__file__))); a.add_argument("--task"); a.add_argument("--ann", "--annotations", dest="ann", required=True)
    a.add_argument("--calib"); a.add_argument("--append", action="store_true", help="continual feedback: keep the existing patch.pt and add this round on top of the already-patched model (default: start from the original weights)")
    a.add_argument("--carrier", default="scale", choices=["cols", "lr", "scale"], help="scale (default): only the existing per-128-group scales of the 1-bit down_proj change; cols: fp16 replacement columns; lr: rank-k factors"); a.add_argument("--aug", type=int, default=0)
    for k, v in DEF.items(): a.add_argument("--" + k, type=type(v), default=None if k == "lam" else v)   # lam default depends on the carrier
    a = a.parse_args(); pf = a.art + "/patch.pt"; T = common.task(a.task) if a.task else None
    ann = [json.loads(l) for l in open(a.ann) if l.strip()]; assert len(ann) <= 25, "at most 25 annotated items"
    old = torch.load(pf) if (a.append and os.path.exists(pf)) else None
    if os.path.exists(pf) and not a.append: os.remove(pf)      # default: always edit from the ORIGINAL weights
    if not ann: print(json.dumps({"patched_layers": [], "patch_bytes": 0})); return
    m, _ = BL.load(a.art, patch=pf if old is not None else None); tok = AutoTokenizer.from_pretrained(a.art)
    cp = a.calib or a.art + "/feedback_calib.jsonl"; cal = [common.prompt_of(json.loads(l), T) for l in open(cp) if l.strip()]
    P = compute_patch(m, tok, ann, T, cal, a.layers, a.cols, a.lam, a.awt, a.margin, a.ridge, aug=a.aug, carrier=a.carrier)
    if P is not None:
        P.pop("orig", None); P.pop("orig_w", None)
        if old is not None: P = merge_patch(old, P)
        torch.save(P, pf)
    print(json.dumps({"patched_layers": P["layers"] if P else [], "cols": a.cols, "patch_bytes": os.path.getsize(pf) if os.path.exists(pf) else 0}))

if __name__ == "__main__": main()
