"""fbl3: closed-form diffuse feedback edit of EXISTING o_proj tensors (last layers) of the packed L3 model. Used by the shipped feedback.py and by the sweep jobs.
No optimiser, no new tensor, nothing in the prompt. Targets: teacher-force each annotated (prompt,label); at answer positions whose argmax != label take the final residual x and
dx = rms(x)(gap+m) u/(u.u), u = gamma (W_U[t]-W_U[c]); ascending over the edited layers, layer l delivers 1/(layers left) of the CURRENT remaining dx through o_proj.
Solvers (K = o_proj input at the annotated sites, R = residual target, C = key covariance of the shipped calibration prompts, Ca = same at answer positions):
  prot : dW = R^T K (lam (C + awt Ca) + K^T K + ridge)^-1                         (the original L3 solver)
  alpha: AlphaEdit-style null-space edit: keys are projected off the top-kp eigenspace of C, dW = R^T (Kp Kp^T + lamA I)^-1 Kp
Optional column restriction (cols=r): only the r input columns with most key energy at the annotated sites are rewritten (stored as a column patch)."""
import json, torch

class FB:
    def __init__(s, model, tok, task, cal, layers_all, dev="cuda", nlimit=None):
        s.model, s.tok, s.task, s.dev = model, tok, task, dev; s.L = model.model.layers; s.LAY = list(layers_all); s.EOS = "<|im_end|>"; s.cap = {}; s.hooks = []
        for l in s.LAY: s.hooks.append(s.L[l].self_attn.o_proj.register_forward_pre_hook(lambda m_, a, l=l: s.cap.__setitem__(l, a[0].detach())))
        s.hooks.append(model.model.norm.register_forward_pre_hook(lambda m_, a: s.cap.__setitem__("fin", a[0].detach())))
        s.WU = model.lm_head.weight.float(); s.G = model.model.norm.weight.float(); s.eps = model.config.rms_norm_eps
        s.backup = {}; s.eig = {}; s.cov(cal[:nlimit] if nlimit else cal)
    def chat(s, t): return s.tok(s.tok.apply_chat_template([{"role": "user", "content": s.task["prompt"].replace("{X}", t)}], tokenize=False, add_generation_prompt=True), add_special_tokens=False).input_ids
    def tg(s, lab): return s.tok(lab + s.EOS, add_special_tokens=False).input_ids
    @torch.no_grad()
    def fwd(s, seqs):
        n = max(len(x) for x in seqs); ids = torch.full((len(seqs), n), s.tok.pad_token_id); am = torch.zeros((len(seqs), n), dtype=torch.long)
        for r, x in enumerate(seqs): ids[r, :len(x)] = torch.tensor(x); am[r, :len(x)] = 1
        s.model.model(input_ids=ids.to(s.dev), attention_mask=am.to(s.dev), use_cache=False)
    @torch.no_grad()
    def cov(s, cal):
        s.Cs = {l: torch.zeros(4096, 4096, device=s.dev) for l in s.LAY}; s.Ca = {l: torch.zeros(4096, 4096, device=s.dev) for l in s.LAY}; nt = na = 0
        for s0 in range(0, len(cal), 16):
            bt = cal[s0:s0 + 16]; seqs = [s.chat(r["text"]) + s.tg(r["label"]) for r in bt]; s.fwd(seqs)
            for l in s.LAY:
                for b, r in enumerate(bt):
                    k = s.cap[l][b, :len(seqs[b])].float(); s.Cs[l] += k.T @ k; ka = k[len(s.chat(r["text"])) - 1:]; s.Ca[l] += ka.T @ ka
            nt += sum(len(x) for x in seqs); na += sum(len(s.tg(r["label"])) for r in bt)
        for l in s.LAY: s.Cs[l] /= nt; s.Ca[l] /= na
    def restore(s):
        for l, w in s.backup.items(): s.L[l].self_attn.o_proj.weight.data.copy_(w)
        s.backup = {}
    @torch.no_grad()
    def edit(s, rows, layers, solver="prot", lam=1000.0, awt=10.0, m=32.0, kp=256, lamA=1e-3, ridge=1e-4, cols=None, verbose=True):
        """rows: [{"text","label"}]. Edits the model IN PLACE (restore() undoes). Returns {state_dict_key: tensor} to ship (dense fp16 or column patch)."""
        s.restore(); layers = sorted(layers); items = [(s.chat(r["text"]), r["label"]) for r in rows]; seqs = [i + s.tg(lab) for i, lab in items]
        def measure():
            s.fwd(seqs); x = s.cap["fin"].float(); out = []
            for b, (ids, lab) in enumerate(items):
                for j, tj in enumerate(s.tg(lab)):
                    p = len(ids) - 1 + j; xx = x[b, p]; out.append((b, p, tj, (s.G * xx / torch.sqrt((xx ** 2).mean() + s.eps)) @ s.WU.T))
            return x, out
        x0, rws = measure(); S = []
        for b, p, tj, lg in rws:
            c = int(torch.argmax(lg))
            if c != tj: xx = x0[b, p]; u = s.G * (s.WU[tj] - s.WU[c]); rms = torch.sqrt((xx ** 2).mean() + s.eps); S.append((b, p, xx + rms * (float(lg[c] - lg[tj]) + m) * u / (u @ u)))
        if verbose: print("sites", len(S), "of", len(rws), flush=True)
        new = {}
        if not S: return new
        for i, l in enumerate(layers):
            x, _ = measure(); K = torch.stack([s.cap[l][b, p].float() for b, p, _ in S]); R = torch.stack([t - x[b, p] for b, p, t in S]) / (len(layers) - i)
            C = s.Cs[l]; Ca = s.Ca[l]; J = None
            if cols and cols < 4096:
                en = (K.abs() / K.norm(dim=1, keepdim=True).clamp(min=1e-9)).sum(0); J = en.topk(cols).indices; K = K[:, J]; C = C[J][:, J]; Ca = Ca[J][:, J]
            if solver == "prot":
                B = lam * (C + awt * Ca); A = B + K.T @ K + ridge * torch.eye(K.shape[1], device=s.dev) * B.diagonal().mean(); dW = torch.linalg.solve(A, (R.T @ K).T).T
            elif solver == "alpha":
                ev, U = torch.linalg.eigh(C); V = U[:, -min(kp, U.shape[1]):]; Kp = K - (K @ V) @ V.T; Gm = Kp @ Kp.T; lm_ = lamA * Gm.diagonal().mean().clamp(min=1e-9)
                dW = R.T @ torch.linalg.solve(Gm + lm_ * torch.eye(Gm.shape[0], device=s.dev), Kp)
            else: raise ValueError(solver)
            W = s.L[l].self_attn.o_proj.weight
            if l not in s.backup: s.backup[l] = W.data.clone()
            if J is None: W.data.add_(dW.to(W.dtype)); new[f"model.layers.{l}.self_attn.o_proj.weight"] = W.data.half().cpu()
            else:
                W.data[:, J] = (W.data[:, J].float() + dW).to(W.dtype); nm = f"model.layers.{l}.self_attn.o_proj.weight"
                new[nm + ".cols_idx"] = J.short().cpu(); new[nm + ".cols_w"] = W.data[:, J].half().cpu()
        return new

def edit_bytes(new): return sum(v.numel() * v.element_size() for v in new.values())
