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
The edited columns are stored as fp16 in DIR/patch.pt; common.load overlays them on the existing down_proj tensors.
Always starts from the original weights (patch removed first). Zero rows => patch.pt removed => artifact byte-identical to the base artifact."""
import sys, os, json, argparse
import torch
sys.dont_write_bytecode = True   # never write into the artifact dir
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common
DEF = dict(layers=6, cols=2048, lam=1000.0, awt=10.0, margin=16.0, ridge=0.05)

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
def make_sites(m, tok, ann, T):
    """annotated sequences + the answer positions where the ORIGINAL model's argmax differs from the annotated token. -> list of (ids, pos, tgt)"""
    dev = m.device; W = m.lm_head.weight; EOS = common.eos_id(tok); out = []
    lab = lambda r: r["label"] if "label" in r else r["gold"][0]["name"]
    for r in ann:
        p = tok(common.chat(tok, common.prompt_of(r, T)), add_special_tokens=False).input_ids; g = tok(lab(r), add_special_tokens=False).input_ids + [EOS]
        ids = p + g; pos = list(range(len(p) - 1, len(p) + len(g) - 1)); h = _fwd(m, torch.tensor([ids], device=dev))[0, pos]
        am = (h.to(W.dtype) @ W.T).argmax(-1).tolist(); keep = [j for j in range(len(g)) if am[j] != g[j]]
        if keep: out.append((ids, [pos[j] for j in keep], [g[j] for j in keep]))
    return out

@torch.no_grad()
def calib_seqs(m, tok, prompts, bs=16, mx=6):
    """calibration sequences = prompt + the model's own greedy answer (up to EOS). -> list of (ids, answer_positions)"""
    dev = m.device; EOS = common.eos_id(tok); P = [tok(common.chat(tok, q), add_special_tokens=False).input_ids for q in prompts]; out = [None] * len(P)
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
def compute_patch(m, tok, ann, T, calib_prompts, layers=DEF["layers"], cols=DEF["cols"], lam=DEF["lam"], awt=DEF["awt"], margin=DEF["margin"], ridge=DEF["ridge"], log=None, cal=None):
    """Edits the loaded model IN PLACE (fp16 columns) and returns the patch {"layers": [..], "cols": [int16 [r]], "vals": [fp16 [hidden, r]]}; None if nothing needs teaching.
    The model must be the ORIGINAL (unpatched) one."""
    dev = m.device; W = m.lm_head.weight.data; LY = m.model.layers; NL = len(LY); gamma = m.model.norm.weight.float(); eps = m.config.rms_norm_eps; D = m.config.hidden_size
    sites = make_sites(m, tok, ann, T)
    if not sites: return None
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
        wt = LY[li].mlp.down_proj.weight; out["orig"].append(wt.data[:, J].clone()); new = (wt.data[:, J].float() + dW).half(); wt.data[:, J] = new
        out["layers"].append(li); out["cols"].append(J.short().cpu()); out["vals"].append(new.cpu())
        if log: log(f"layer {li}: sites {K.shape[1]} mean gap {float(gap.mean()):.2f}")
    return out

def undo(m, P):
    """exact undo of compute_patch's in-place edit (experiments only)"""
    for li, c, o in zip(P["layers"], P["cols"], P["orig"]): m.model.layers[li].mlp.down_proj.weight.data[:, c.long().to(m.device)] = o

def apply_patch(m, P, save=None):
    """overlay patch columns on the existing down_proj tensors; if save is a dict, the replaced original columns are stored there (for exact restore in experiments)."""
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
    a.add_argument("--calib")
    for k, v in DEF.items(): a.add_argument("--" + k, type=type(v), default=v)
    a = a.parse_args(); pf = a.art + "/patch.pt"; T = common.task(a.task) if a.task else None
    ann = [json.loads(l) for l in open(a.ann) if l.strip()]; assert len(ann) <= 25, "at most 25 annotated items"
    if os.path.exists(pf): os.remove(pf)                       # always edit from the ORIGINAL weights
    if not ann: print(json.dumps({"patched_layers": [], "patch_bytes": 0})); return
    m = common.load(a.art, patch=False); tok = AutoTokenizer.from_pretrained(a.art); tok.pad_token = tok.pad_token or tok.eos_token
    cp = a.calib or a.art + "/feedback_calib.jsonl"; cal = [common.prompt_of(json.loads(l), T) for l in open(cp) if l.strip()]
    P = compute_patch(m, tok, ann, T, cal, a.layers, a.cols, a.lam, a.awt, a.margin, a.ridge)
    if P is not None: P.pop("orig", None); torch.save(P, pf)
    print(json.dumps({"patched_layers": P["layers"] if P else [], "cols": a.cols, "patch_bytes": os.path.getsize(pf) if P else 0}))

if __name__ == "__main__": main()
