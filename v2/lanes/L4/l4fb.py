"""L4 feedback: closed-form 'o_prot' edit (L11 recipe: protected key covariance, margin M, last layers) of the EXISTING o_proj tensors (the fp16 carrier) of the packed model.
No optimiser, no new tensor. teacher-forced annotated (prompt, label) -> final-residual target dx = rms (gap+M) u/(u.u), u = gamma (W_U[t]-W_U[c]); layer l delivers 1/(layers left) of the remaining dx:
dW = R^T K (lam (C + awt C_ans + ridge I) + K^T K)^-1. Shipped code."""
import torch
EOS = "<|im_end|>"
DEF = dict(layers=list(range(40, 48)), lam=1000.0, awt=10.0, ridge=0.05, m=32.0, margin_all=False, eos_m=None)

class FB:
    def __init__(s, lm, tok, cal, cfg=None):
        """cal: list of (prompt_ids, gold_label) calibration pairs of the OLD skills (protected)"""
        s.lm, s.tok, s.cfg = lm, tok, dict(DEF, **(cfg or {})); s.dev = "cuda"; s.L = lm.model.layers; s.LAY = s.cfg["layers"]
        s.cap = {}; s.hooks = []
        for l in s.LAY: s.hooks.append(s.L[l].self_attn.o_proj.register_forward_pre_hook(lambda m_, a, l=l: s.cap.__setitem__(l, a[0].detach())))
        s.hooks.append(lm.model.norm.register_forward_pre_hook(lambda m_, a: s.cap.__setitem__("fin", a[0].detach())))
        s.WU = lm.lm_head.weight.float(); s.G = lm.model.norm.weight.float(); s.eps = lm.config.rms_norm_eps; s.pad = tok.pad_token_id
        s.backup = {l: s.L[l].self_attn.o_proj.weight.data.clone() for l in s.LAY}; s.tg = {}
        Cs = {l: torch.zeros(4096, 4096, device=s.dev) for l in s.LAY}; Ca = {l: torch.zeros(4096, 4096, device=s.dev) for l in s.LAY}; nt = na = 0
        for s0 in range(0, len(cal), 16):
            bt = cal[s0:s0 + 16]; seqs = [p + s.t(a) for p, a in bt]; s.fwd(seqs)
            for l in s.LAY:
                for b, (p, a) in enumerate(bt):
                    k = s.cap[l][b, :len(seqs[b])].float(); Cs[l] += k.T @ k; ka = k[len(p) - 1:len(seqs[b])]; Ca[l] += ka.T @ ka
            nt += sum(len(x) for x in seqs); na += sum(len(s.t(a)) for _, a in bt)
        s.Cs = {l: Cs[l] / nt for l in s.LAY}; s.Ca = {l: Ca[l] / na for l in s.LAY}
    def t(s, lab):
        if lab not in s.tg: s.tg[lab] = s.tok(lab + EOS, add_special_tokens=False).input_ids
        return s.tg[lab]
    @torch.no_grad()
    def fwd(s, seqs):
        L = max(len(x) for x in seqs); ids = torch.full((len(seqs), L), s.pad); am = torch.zeros((len(seqs), L), dtype=torch.long)
        for r, x in enumerate(seqs): ids[r, :len(x)] = torch.tensor(x); am[r, :len(x)] = 1
        s.lm.model(input_ids=ids.to(s.dev), attention_mask=am.to(s.dev), use_cache=False)
    def restore(s):
        for l in s.LAY: s.L[l].self_attn.o_proj.weight.data.copy_(s.backup[l])
    def commit(s):   # accept the current weights as the new base (continual feedback)
        for l in s.LAY: s.backup[l] = s.L[l].self_attn.o_proj.weight.data.clone()
    @torch.no_grad()
    def edit(s, items):
        """items: list of (prompt_ids, label). returns number of sites."""
        c = s.cfg; M = c["m"]; seqs = [p + s.t(l) for p, l in items]
        def measure():
            s.fwd(seqs); x = s.cap["fin"].float(); rows = []
            for b, (p, lab) in enumerate(items):
                for j, tj in enumerate(s.t(lab)):
                    pos = len(p) - 1 + j; xx = x[b, pos]; lg = (s.G * xx / torch.sqrt((xx ** 2).mean() + s.eps)) @ s.WU.T; rows.append((b, pos, tj, lg, j == len(s.t(lab)) - 1))
            return x, rows
        x0, rows = measure(); S = []
        for b, pos, tj, lg, last in rows:
            o = lg.clone(); o[tj] = -1e9; cmp_ = int(torch.argmax(o)); gap = float(lg[cmp_] - lg[tj]); m_ = (c["eos_m"] if (last and c["eos_m"]) else M)
            if (gap > 0) or (c["margin_all"] and gap + m_ > 0):
                xx = x0[b, pos]; u = s.G * (s.WU[tj] - s.WU[cmp_]); rms = torch.sqrt((xx ** 2).mean() + s.eps); S.append((b, pos, xx + rms * (gap + m_) * u / (u @ u)))
        if not S: return 0
        n = len(s.LAY)
        for i, l in enumerate(s.LAY):
            x, _ = measure(); K = torch.stack([s.cap[l][b, p].float() for b, p, _ in S]); R = torch.stack([tg - x[b, p] for b, p, tg in S]) / (n - i)
            C = c["lam"] * (s.Cs[l] + c["awt"] * s.Ca[l]); A = C + c["ridge"] * C.diagonal().mean() * torch.eye(4096, device=s.dev) + K.T @ K
            dW = torch.linalg.solve(A, (R.T @ K).T).T; w = s.L[l].self_attn.o_proj.weight; w.data.add_(dW.to(w.dtype))
        return len(S)
    def close(s):
        for h in s.hooks: h.remove()
