"""Closed-form feedback edit of EXISTING o_proj tensors (L11 recipe), trimmed from v2/fb_ablation/fblib.py to what the shipped CLI needs:
protected solver (calibration key covariance, lam 1000, awt 10, ridge .05), margin 32, last 8 layers (56-63), only the top-`cols` input columns of each o_proj are rewritten.
No training, no gradient, no side memory. The edit is stored as the replaced columns (bf16) + their indices (patch.pt)."""
import collections, torch

EOS = "<|im_end|>"
METHOD = dict(layers=list(range(56, 64)), m=32.0, lam=1000.0, awt=10.0, ridge=0.05, cols=1536)

class FB:
    def __init__(s, tok, lm, prompt_fn, calib_texts, log=print, budget=20000):
        s.tok, s.lm, s.log = tok, lm, log; s.dev = lm.device; s.m = lm.model; s.L = s.m.layers; s.D = lm.config.hidden_size; s.EPS = lm.config.rms_norm_eps
        s.pad = tok.pad_token_id; s.eos_id = tok.convert_tokens_to_ids(EOS); s.budget = budget; s.prompt_fn = prompt_fn
        s.saved = {}; s.cache = {}; s._cap = None; s._cov = None; s._gi = None; s._hooks = []
        for li, lyr in enumerate(s.L): s._hooks.append(lyr.self_attn.o_proj.register_forward_pre_hook(s._mk(li)))
        s._hooks.append(s.m.norm.register_forward_pre_hook(s._hx))
        s.calP = [s.pid(t) for t in calib_texts]; s.calA = s.gen_ids(s.calP, 5)
        s.calseq = [p + a for p, a in zip(s.calP, s.calA)]; s.calpos = [list(range(len(p) - 1, len(p) + len(a) - 1)) for p, a in zip(s.calP, s.calA)]
    def pid(s, text): return s.tok(s.tok.apply_chat_template([{"role": "user", "content": s.prompt_fn(text)}], tokenize=False, add_generation_prompt=True), add_special_tokens=False).input_ids
    @torch.no_grad()
    def gen_ids(s, pids, mx, bs=48):
        out = [None] * len(pids); order = sorted(range(len(pids)), key=lambda i: len(pids[i]))
        for st in range(0, len(order), bs):
            idx = order[st:st + bs]; L = max(len(pids[i]) for i in idx); ids = torch.full((len(idx), L), s.pad); am = torch.zeros((len(idx), L), dtype=torch.long)
            for r, i in enumerate(idx): ids[r, L - len(pids[i]):] = torch.tensor(pids[i]); am[r, L - len(pids[i]):] = 1
            o = s.lm.generate(input_ids=ids.to(s.dev), attention_mask=am.to(s.dev), max_new_tokens=mx, do_sample=False, pad_token_id=s.pad)
            for r, i in enumerate(idx):
                t = o[r, L:].tolist()
                if s.eos_id in t: t = t[:t.index(s.eos_id) + 1]
                out[i] = t
        return out
    def gen_text(s, pids, mx=6): return [s.tok.decode(a, skip_special_tokens=True).strip() for a in s.gen_ids(pids, mx)]
    def close(s):
        for h in s._hooks: h.remove()
    # ---- hooks
    def _mk(s, li):
        def hook(mod, inp):
            if s._cov is not None:
                if li in s._cov["C"]:
                    k = inp[0]; kk = k[s._cov["mask"]].float(); s._cov["C"][li].addmm_(kk.T, kk); ka = k[s._cov["mask2"]].float(); s._cov["Ca"][li].addmm_(ka.T, ka)
                return
            if s._cap is not None and li in s._cap: r, c = s._gi; s._cap[li] = inp[0][r, c].float()
        return hook
    def _hx(s, mod, inp):
        if s._cap is not None and "x" in s._cap: r, c = s._gi; s._cap["x"] = inp[0][r, c].float()
    def _batches(s, seqs, budget):
        order = sorted(range(len(seqs)), key=lambda i: len(seqs[i])); i = 0
        while i < len(order):
            j = i
            while j < len(order) and (j - i + 1) * len(seqs[order[j]]) <= budget: j += 1
            j = max(j, i + 1); yield order[i:j]; i = j
    def _pad(s, seqs, idx):
        L = max(len(seqs[k]) for k in idx); ids = torch.full((len(idx), L), s.pad, dtype=torch.long); am = torch.zeros((len(idx), L), dtype=torch.long)
        for r, k in enumerate(idx): ids[r, :len(seqs[k])] = torch.tensor(seqs[k]); am[r, :len(seqs[k])] = 1
        return ids.to(s.dev), am.to(s.dev)
    @torch.no_grad()
    def run(s, seqs, pos, cap=()):
        n = len(seqs); out = {"h": [None] * n}
        for c in cap: out[c] = [None] * n
        for idx in s._batches(seqs, s.budget):
            ids, am = s._pad(seqs, idx); rows = []; cols = []; cnt = []
            for r, k in enumerate(idx): rows += [r] * len(pos[k]); cols += pos[k]; cnt.append(len(pos[k]))
            s._gi = (torch.tensor(rows, device=s.dev), torch.tensor(cols, device=s.dev)); s._cap = {c: None for c in cap}
            h = s.m(input_ids=ids, attention_mask=am).last_hidden_state; sp = torch.split(h[s._gi[0], s._gi[1]], cnt)
            for r, k in enumerate(idx): out["h"][k] = sp[r]
            for c in cap:
                spc = torch.split(s._cap[c], cnt)
                for r, k in enumerate(idx): out[c][k] = spc[r]
            s._cap = None; s._gi = None
        return out
    def logits(s, h): return (h.to(s.lm.lm_head.weight.dtype) @ s.lm.lm_head.weight.T).float()
    # ---- annotated sites: label tokens (word + EOS) the current model does not already emit
    @torch.no_grad()
    def sites_of(s, items):
        sq, ps, gs = [], [], []
        for text, lab in items:
            p = s.pid(text); g = s.tok(lab + EOS, add_special_tokens=False).input_ids; sq.append(p + g); ps.append(list(range(len(p) - 1, len(p) + len(g) - 1))); gs.append(g)
        o = s.run(sq, ps)["h"]; S = []
        for si, (h, g) in enumerate(zip(o, gs)):
            am = s.logits(h).argmax(-1).cpu().tolist()
            for j, t in enumerate(g):
                if am[j] != t: S.append((si, ps[si][j], t))
        return sq, S
    @torch.no_grad()
    def cov_many(s, layers):
        todo = [l for l in layers if l not in s.cache]
        if todo:
            dim = s.L[0].self_attn.o_proj.weight.shape[1]
            s._cov = dict(C={l: torch.zeros(dim, dim, device=s.dev) for l in todo}, Ca={l: torch.zeros(dim, dim, device=s.dev) for l in todo}); nt = na = 0
            for idx in s._batches(s.calseq, 16000):
                ids, am = s._pad(s.calseq, idx); m2 = torch.zeros_like(am, dtype=torch.bool)
                for r, k in enumerate(idx): m2[r, s.calpos[k]] = True
                s._cov["mask"] = am.bool(); s._cov["mask2"] = m2; nt += int(am.sum()); na += int(m2.sum()); s.m(input_ids=ids, attention_mask=am)
            for l in todo: s.cache[l] = dict(C=s._cov["C"][l] / nt, Ca=s._cov["Ca"][l] / na)
            s._cov = None
        return {l: s.cache[l] for l in layers}
    def site_cap(s, seqs, sites, caps):
        by = collections.defaultdict(list)
        for si, p, t in sites: by[si].append(p)
        order = sorted(by); o = s.run([seqs[i] for i in order], [by[i] for i in order], cap=caps)
        return {c: torch.cat(o[c]) for c in caps}
    def targets(s, x, sites, m):
        W = s.lm.lm_head.weight; G = s.m.norm.weight.float(); T = torch.tensor([t for _, _, t in sites], device=s.dev)
        h = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + s.EPS) * G; lg = s.logits(h); zt = lg.gather(1, T[:, None])[:, 0]; l2 = lg.clone(); l2.scatter_(1, T[:, None], -1e9); zc, ci = l2.max(1)
        gap = (zc - zt + m).clamp(min=0); u = G[None] * (W[T].float() - W[ci].float()); rms = x.pow(2).mean(-1, keepdim=True).add(s.EPS).sqrt()
        return rms * gap[:, None] * u / u.pow(2).sum(-1, keepdim=True).clamp(min=1e-9)
    def touch(s, li):
        if li not in s.saved: s.saved[li] = s.L[li].self_attn.o_proj.weight.data.clone()
    def restore(s):
        for li, w in s.saved.items(): s.L[li].self_attn.o_proj.weight.data.copy_(w)
        s.saved.clear()
    def edit(s, items, mt=METHOD):
        """items: [(request text, label word)] -> patch dict {layer: (cols int16 [c], new columns bf16 [D, c])}; the model is edited in place (restore() undoes it)."""
        seqs, sites = s.sites_of(items); patch = {}
        if not sites: return patch, 0
        layers = sorted(mt["layers"]); cvs = s.cov_many(layers)
        for step, li in enumerate(layers):
            cp = s.site_cap(seqs, sites, [li, "x"]); K = cp[li].T.contiguous(); R = (s.targets(cp["x"], sites, mt["m"]) / (len(layers) - step)).T.contiguous(); cv = cvs[li]
            sc = (K.abs() / K.norm(dim=0, keepdim=True).clamp(min=1e-9)).sum(1); J = sc.topk(mt["cols"]).indices.sort().values; Kj = K[J]
            Cs = (cv["C"] + mt["awt"] * cv["Ca"])[J][:, J]; A = mt["lam"] * (Cs + mt["ridge"] * Cs.diagonal().mean() * torch.eye(len(J), device=s.dev))
            Z = torch.linalg.solve(A, Kj); I_ = torch.eye(K.shape[1], device=s.dev); dWj = (R @ torch.linalg.solve(I_ + Kj.T @ Z, I_)) @ Z.T
            s.touch(li); w = s.L[li].self_attn.o_proj.weight; new = (w.data[:, J].float() + dWj).to(torch.bfloat16); w.data[:, J] = new
            patch[li] = (J.to(torch.int16).cpu(), new.cpu())
        return patch, len(sites)

def apply_patch(model, patch):
    for li, (J, cols) in patch.items(): model.model.layers[li].self_attn.o_proj.weight.data[:, J.long().to(model.device)] = cols.to(model.device)
