"""L9b feedback core (shipped): closed-form diffuse edit of the shared-expert down_proj of the last MoE layers from <=25 annotated (request, correct label) pairs.
No optimiser, no new tensor, no prompt/memory: the edited columns REPLACE columns of an existing weight matrix; the carrier fb.bin stores those replacement columns (bf16) + indices.
Mechanism: teacher-force each annotation, where argmax != label take the final residual x and the closed-form target dx = rms(x)(gap+m) u/(u.u), u=(1+norm.w)(W_U[t]-W_U[c]);
layer l delivers 1/(layers left) of the CURRENT remaining dx: dW = R^T K (lam (C + awt C_ans + ridge) + K^T K)^-1 (keys K = sigmoid(shared_expert_gate) * down_proj input); then restricted to the r largest columns and re-solved."""
import os, json, torch
EOS = "<|im_end|>"

def apply_feedback(model, tok, R, ann, cal, nlayers=10, cols=512, lam=1000.0, awt=10.0, m=16.0, log=print):
    """ann: list of dict(user, answer). cal: list of dict(user, answer) (format covariance rows). Returns carrier {layer: (idx, cols_bf16[H, r])} and edits the model in place."""
    layers = model.model.layers; NL = len(layers); LAY = list(range(NL - nlayers, NL)); dev = R.dev0
    WU = model.lm_head.weight.float(); G = 1.0 + model.model.norm.weight.float(); eps = model.config.rms_norm_eps
    tgt = lambda lab: tok(lab + EOS, add_special_tokens=False).input_ids
    CAP = dict(on=False); hooks = []
    for l in LAY:
        mlp = layers[l].mlp
        hooks.append(mlp.shared_expert.down_proj.register_forward_pre_hook(lambda mod, a, l=l: CAP.__setitem__(("k", l), a[0].detach()) if CAP["on"] else None))
        hooks.append(mlp.shared_expert_gate.register_forward_hook(lambda mod, a, o, l=l: CAP.__setitem__(("g", l), torch.sigmoid(o.detach().float())) if CAP["on"] else None))
    hooks.append(model.model.norm.register_forward_pre_hook(lambda mod, a: CAP.__setitem__("fin", a[0].detach()) if CAP["on"] else None))
    def fwd(seqs):
        Lm = max(len(s) for s in seqs); ids = torch.full((len(seqs), Lm), tok.pad_token_id); am = torch.zeros((len(seqs), Lm), dtype=torch.long)
        for r, s in enumerate(seqs): ids[r, :len(s)] = torch.tensor(s); am[r, :len(s)] = 1
        CAP["on"] = True
        with torch.no_grad(): model.model(input_ids=ids.to(dev), attention_mask=am.to(dev), use_cache=False)
        CAP["on"] = False; return Lm
    keyat = lambda l, b, p, Lm: CAP[("k", l)][b * Lm + p].float() * CAP[("g", l)][b * Lm + p].float()
    R.encode(cal); DI = layers[LAY[0]].mlp.shared_expert.down_proj.weight.shape[1]
    Cs = {l: torch.zeros(DI, DI, device=dev) for l in LAY}; Ca = {l: torch.zeros(DI, DI, device=dev) for l in LAY}; nt = na = 0
    for s0 in range(0, len(cal), 16):
        bt = cal[s0:s0 + 16]; seqs = [r["ids"] + tgt(r["answer"]) for r in bt]; Lm = fwd(seqs)
        for l in LAY:
            for b, r in enumerate(bt):
                k = CAP[("k", l)][b * Lm:b * Lm + len(seqs[b])].float() * CAP[("g", l)][b * Lm:b * Lm + len(seqs[b])].float(); Cs[l] += k.T @ k; ka = k[len(r["ids"]) - 1:]; Ca[l] += ka.T @ ka
        nt += sum(len(s) for s in seqs); na += sum(len(tgt(r["answer"])) for r in bt)
    for l in LAY: Cs[l] /= nt; Ca[l] /= na
    R.encode(ann); items = [(r["ids"], r["answer"]) for r in ann]; seqs = [ids + tgt(lab) for ids, lab in items]
    def measure():
        Lm = fwd(seqs); x = CAP["fin"].float(); rows = []
        for b, (ids, lab) in enumerate(items):
            for j, tj in enumerate(tgt(lab)):
                p = len(ids) - 1 + j; xx = x[b, p]; lg = (G * xx / torch.sqrt((xx ** 2).mean() + eps)) @ WU.T; rows.append((b, p, tj, lg))
        return x, rows, Lm
    x0, rows, Lm = measure(); Sx = []
    for b, p, tj, lg in rows:
        c = int(torch.argmax(lg))
        if c != tj:
            xx = x0[b, p]; u = G * (WU[tj] - WU[c]); rms = torch.sqrt((xx ** 2).mean() + eps); Sx.append((b, p, xx + rms * (float(lg[c] - lg[tj]) + m) * u / (u @ u)))
    log("annotation sites needing correction:", len(Sx)); carrier = {}
    for i, l in enumerate(LAY):
        if not Sx: break
        x, _, Lm = measure(); K = torch.stack([keyat(l, b, p, Lm) for b, p, _ in Sx]); Rr = torch.stack([tg - x[b, p] for b, p, tg in Sx]) / (len(LAY) - i)
        B = lam * (Cs[l] + awt * Ca[l]); A = B + K.T @ K + 1e-4 * torch.eye(DI, device=dev) * B.diagonal().mean(); rhs = (Rr.T @ K).T; dW = torch.linalg.solve(A, rhs).T
        top = torch.arange(DI, device=dev)
        if cols < DI:
            top = dW.norm(dim=0).topk(cols).indices.sort().values; dW = torch.linalg.solve(A[top][:, top], rhs[top]).T
        W = layers[l].mlp.shared_expert.down_proj.weight; W.data[:, top] = (W.data[:, top].float() + dW.float()).to(W.dtype)
        carrier[l] = (top.cpu().to(torch.int16), W.data[:, top].cpu().clone())
    for h in hooks: h.remove()
    return carrier

def save_carrier(adir, carrier, meta=None):
    with open(os.path.join(adir, "fb.bin"), "wb") as f:
        index = {}
        for l, (idx, cols) in carrier.items():
            a = idx.numpy().tobytes(); b = cols.contiguous().view(torch.int16).numpy().tobytes(); index[str(l)] = [f.tell(), len(idx), cols.shape[0]]; f.write(a); f.write(b)
    json.dump(dict(index=index, meta=meta or {}), open(os.path.join(adir, "fb.json"), "w"))

def load_carrier(model, adir):
    p = os.path.join(adir, "fb.json")
    if not os.path.exists(p): return 0
    import numpy as np
    j = json.load(open(p)); raw = open(os.path.join(adir, "fb.bin"), "rb").read(); n = 0
    for l, (off, r, H) in j["index"].items():
        idx = torch.from_numpy(np.frombuffer(raw, dtype=np.int16, count=r, offset=off).astype(np.int64)); cols = torch.from_numpy(np.frombuffer(raw, dtype=np.int16, count=r * H, offset=off + 2 * r).copy()).view(torch.bfloat16).reshape(H, r)
        W = model.model.layers[int(l)].mlp.shared_expert.down_proj.weight; W.data[:, idx.to(W.device)] = cols.to(W.device); n += 1
    return n
