"""L11 feedback-ablation library: closed-form edits of EXISTING tensors of one dense LM (Qwen2.5-7B-Instruct), full model, real forward.
No optimiser; the only 'gradient' use is the Gauss-Newton Jacobian rows of the GN family (one linearised min-norm step, <=3 iterations).
Env: model + data + eval + edit methods. Used by run_abl.py (experiments) and answer.py / feedback.py (end-to-end CLI)."""
import json, os, time, hashlib, random, collections, math
import numpy as np, torch, torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

OLD = ["Alarm", "Weather", "Calendar", "Lists", "Transport"]
PROMPT = ("You are the router of a voice assistant. Decide which skill should handle the user's request.\n"
          "Answer with exactly one word from: Alarm, Weather, Calendar, Lists, Transport.\n\nRequest: {X}")
NEWS = {"Music": "Jukebox", "News": "Briefing"}
EOS = "<|im_end|>"

def rtn_scales(w, bits=4, g=128):
    R, C = w.shape; wf = w.float().view(R, C // g, g); qm = 2 ** (bits - 1) - 1
    return (wf.abs().amax(-1, keepdim=True) / qm).clamp(min=1e-8).half()
def rtn_apply(w, sc, bits=4, g=128):
    R, C = w.shape; qm = 2 ** (bits - 1) - 1; s = sc.float()
    return ((w.float().view(R, C // g, g) / s).round().clamp(-qm - 1, qm) * s).view(R, C)

class Env:
    def __init__(s, model, data_dir, q4=False, ncal=300, qbits=4, dev="cuda", log=print):
        s.log = log; s.dev = dev; s.T0 = time.time(); s.data_dir = data_dir
        s.tok = AutoTokenizer.from_pretrained(model); s.tok.padding_side = "right"
        s.lm = AutoModelForCausalLM.from_pretrained(model, torch_dtype=torch.bfloat16).to(dev).eval()
        for p in s.lm.parameters(): p.requires_grad_(False)
        s.m = s.lm.model; s.L = s.m.layers; s.NL = len(s.L); s.D = s.lm.config.hidden_size; s.EPS = s.lm.config.rms_norm_eps
        s.q4 = q4; s.scales = {}; s.qbits = qbits
        if q4:
            with torch.no_grad():
                for li, lyr in enumerate(s.L):
                    for nm, mod in [("q", lyr.self_attn.q_proj), ("k", lyr.self_attn.k_proj), ("v", lyr.self_attn.v_proj), ("o", lyr.self_attn.o_proj), ("gate", lyr.mlp.gate_proj), ("up", lyr.mlp.up_proj), ("down", lyr.mlp.down_proj)]:
                        sc = rtn_scales(mod.weight.data, qbits); s.scales[(nm, li)] = sc; mod.weight.data = rtn_apply(mod.weight.data, sc, qbits).to(torch.bfloat16)
        s.eos_id = s.tok.convert_tokens_to_ids(EOS); s.pad = s.tok.pad_token_id
        s.TG = {l: s.tok(l + EOS, add_special_tokens=False).input_ids for l in OLD + list(NEWS.values())}
        s.log("label tokens", {l: [s.tok.decode([i]) for i in v] for l, v in s.TG.items()})
        rows = lambda n: [json.loads(l) for l in open(f"{data_dir}/{n}.jsonl")]
        s.R = {n: rows(n) for n in ["neigh", "calib_old", "ann_Music", "teach_Music", "ann_News", "teach_News", "calnew_Music"] if os.path.exists(f"{data_dir}/{n}.jsonl")}
        for v in s.R.values():
            for r in v: r["pid"] = s.pid(r["text"])
        s.saved = {}; s._cap = None; s._cov = None; s._gi = None; s._install_hooks()
        s.cache = {}; s.prevK = {}
        s.calP = [r["pid"] for r in s.R["calib_old"][:ncal]]
        s.calA = s.gen_ids(s.calP, 5)
        s.calseq = [p + a for p, a in zip(s.calP, s.calA)]; s.calpos = [list(range(len(p) - 1, len(p) + len(a) - 1)) for p, a in zip(s.calP, s.calA)]
        s.log("cal answers", collections.Counter(s.tok.decode(a, skip_special_tokens=True).strip() for a in s.calA).most_common(6))
    # ---------------- tokens / generation
    def chat(s, u):
        try: return s.tok.apply_chat_template([{"role": "user", "content": u}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except TypeError: return s.tok.apply_chat_template([{"role": "user", "content": u}], tokenize=False, add_generation_prompt=True)
    def pid(s, text): return s.tok(s.chat(PROMPT.replace("{X}", text)), add_special_tokens=False).input_ids
    @torch.no_grad()
    def gen_ids(s, pids, mx, bs=48):
        out = [None] * len(pids); order = sorted(range(len(pids)), key=lambda i: len(pids[i]))
        s.tok.padding_side = "left"
        for st in range(0, len(order), bs):
            idx = order[st:st + bs]; L = max(len(pids[i]) for i in idx)
            ids = torch.full((len(idx), L), s.pad); am = torch.zeros((len(idx), L), dtype=torch.long)
            for r, i in enumerate(idx): ids[r, L - len(pids[i]):] = torch.tensor(pids[i]); am[r, L - len(pids[i]):] = 1
            o = s.lm.generate(input_ids=ids.to(s.dev), attention_mask=am.to(s.dev), max_new_tokens=mx, do_sample=False, pad_token_id=s.pad)
            for r, i in enumerate(idx):
                t = o[r, L:].tolist()
                if s.eos_id in t: t = t[:t.index(s.eos_id) + 1]
                out[i] = t
        s.tok.padding_side = "right"; return out
    def gen_text(s, pids, mx=6): return [s.tok.decode(a, skip_special_tokens=True).strip() for a in s.gen_ids(pids, mx)]
    # ---------------- hooks
    def _install_hooks(s):
        def mk(kind, li):
            def hook(mod, inp):
                if s._cov is not None:
                    if (kind, li) in s._cov["C"]:
                        k = inp[0]; kk = k[s._cov["mask"]].float(); s._cov["C"][(kind, li)].addmm_(kk.T, kk); s._cov["sum"][(kind, li)] += kk.sum(0)
                        ka = k[s._cov["mask2"]].float(); s._cov["Ca"][(kind, li)].addmm_(ka.T, ka); s._cov["suma"][(kind, li)] += ka.sum(0)
                    return
                if s._cap is not None and (kind, li) in s._cap:
                    r, c = s._gi; s._cap[(kind, li)] = inp[0][r, c].float()
            return hook
        for li, lyr in enumerate(s.L):
            lyr.mlp.down_proj.register_forward_pre_hook(mk("down", li)); lyr.self_attn.o_proj.register_forward_pre_hook(mk("o", li)); lyr.mlp.register_forward_pre_hook(mk("mlp_in", li))
        def hx(mod, inp):
            if s._cap is not None and "x" in s._cap: r, c = s._gi; s._cap["x"] = inp[0][r, c].float()
        s.m.norm.register_forward_pre_hook(hx)
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
    def run(s, seqs, pos, cap=(), budget=40000):
        """returns dict: 'h' list of normed hidden [np,D] at pos, 'x' pre-norm final residual (if in cap), (kind,li) keys -> list of [np,dim]"""
        n = len(seqs); out = {"h": [None] * n}
        for c in cap: out[c] = [None] * n
        for idx in s._batches(seqs, budget):
            ids, am = s._pad(seqs, idx); rows = []; cols = []; cnt = []
            for r, k in enumerate(idx): rows += [r] * len(pos[k]); cols += pos[k]; cnt.append(len(pos[k]))
            s._gi = (torch.tensor(rows, device=s.dev), torch.tensor(cols, device=s.dev)); s._cap = {c: None for c in cap}
            h = s.m(input_ids=ids, attention_mask=am).last_hidden_state; hh = h[s._gi[0], s._gi[1]]
            sp = torch.split(hh, cnt)
            for r, k in enumerate(idx): out["h"][k] = sp[r]
            for c in cap:
                spc = torch.split(s._cap[c], cnt)
                for r, k in enumerate(idx): out[c][k] = spc[r]
            s._cap = None; s._gi = None
        return out
    def logits(s, h): return (h.to(s.lm.lm_head.weight.dtype) @ s.lm.lm_head.weight.T).float()
    # ---------------- evaluation (teacher-forced greedy == real greedy generation up to numerics; validated separately)
    def seq_of(s, r, lab):
        if lab not in s.TG: s.TG[lab] = s.tok(lab + EOS, add_special_tokens=False).input_ids
        g = s.TG[lab]; return r["pid"] + g, list(range(len(r["pid"]) - 1, len(r["pid"]) + len(g) - 1)), g
    @torch.no_grad()
    def acc(s, rows, labfn, idxs=None):
        idxs = list(range(len(rows))) if idxs is None else idxs
        if not idxs: return np.zeros(0, dtype=np.int8)
        sq, ps, gs = zip(*[s.seq_of(rows[i], labfn(rows[i])) for i in idxs]); o = s.run(list(sq), list(ps))["h"]; res = np.zeros(len(idxs), dtype=np.int8)
        for n, (h, g) in enumerate(zip(o, gs)): res[n] = int((s.logits(h).argmax(-1).cpu() == torch.tensor(g)).all())
        return res
    # ---------------- sites for annotated items: list of (seq_idx, pos, token)
    @torch.no_grad()
    def sites_of(s, items):
        """items: list of (row, label). returns seqs, per-seq all label positions, sites [(si,pos,t)] where argmax!=t (current weights)"""
        sq, ps, gs = zip(*[s.seq_of(r, l) for r, l in items]); o = s.run(list(sq), list(ps))["h"]; S = []
        for si, (h, g) in enumerate(zip(o, gs)):
            am = s.logits(h).argmax(-1).cpu().tolist()
            for j, t in enumerate(g):
                if am[j] != t: S.append((si, ps[si][j], t))
        return list(sq), S
    # ---------------- calibration statistics (key covariance of module inputs over calibration prompts + own greedy answers)
    @torch.no_grad()
    def cov(s, kind, li):
        key = ("cov", kind, li)
        if key in s.cache: return s.cache[key]
        need = [(kind, l) for l in ([li] if not isinstance(li, (list, tuple)) else li)]
        return s.cov_many(kind, [li])[("cov", kind, li)]
    @torch.no_grad()
    def cov_many(s, kind, layers):
        todo = [l for l in layers if ("cov", kind, l) not in s.cache]
        if todo:
            dim = s.L[0].mlp.down_proj.weight.shape[1] if kind == "down" else s.D
            s._cov = dict(C={(kind, l): torch.zeros(dim, dim, device=s.dev) for l in todo}, Ca={(kind, l): torch.zeros(dim, dim, device=s.dev) for l in todo},
                          sum={(kind, l): torch.zeros(dim, device=s.dev) for l in todo}, suma={(kind, l): torch.zeros(dim, device=s.dev) for l in todo}); nt = na = 0
            for idx in s._batches(s.calseq, 16000):
                ids, am = s._pad(s.calseq, idx); m2 = torch.zeros_like(am, dtype=torch.bool)
                for r, k in enumerate(idx): m2[r, s.calpos[k]] = True
                s._cov["mask"] = am.bool(); s._cov["mask2"] = m2; nt += int(am.sum()); na += int(m2.sum()); s.m(input_ids=ids, attention_mask=am)
            for l in todo:
                k = (kind, l); s.cache[("cov", kind, l)] = dict(C=s._cov["C"][k] / nt, Ca=s._cov["Ca"][k] / na, mu=s._cov["sum"][k] / nt, mua=s._cov["suma"][k] / na)
            s._cov = None
        return {("cov", kind, l): s.cache[("cov", kind, l)] for l in layers}
    def clear_cache(s):
        s.cache = {k: v for k, v in s.cache.items() if k == "calh"}; torch.cuda.empty_cache()
    # ---------------- weight bookkeeping (everything edited is stored back in bf16 = what a shipped bf16 body would hold)
    def _named(s, n):
        if not hasattr(s, "_pd"): s._pd = dict(s.lm.named_parameters())
        return s._pd[n]
    def tname(s, kind, li): return {"down": f"model.layers.{li}.mlp.down_proj.weight", "o": f"model.layers.{li}.self_attn.o_proj.weight", "up": f"model.layers.{li}.mlp.up_proj.weight",
                                    "gate": f"model.layers.{li}.mlp.gate_proj.weight", "norm_post": f"model.layers.{li}.post_attention_layernorm.weight", "norm_in": f"model.layers.{li}.input_layernorm.weight",
                                    "vbias": f"model.layers.{li}.self_attn.v_proj.bias"}[kind]
    def touch(s, n):
        if n not in s.saved: s.saved[n] = s._named(n).data.clone()
    def restore(s):
        for n, w in s.saved.items(): s._named(n).data.copy_(w)
        s.saved.clear(); s.prevK = {}
    def write(s, kind, li, dW, carrier="bf16"):
        n = s.tname(kind, li); s.touch(n); w = s._named(n); new = w.data.float() + dW
        if carrier == "requant4": new = rtn_apply(new, s.scales[(kind, li)], s.qbits)
        w.data.copy_(new.to(torch.bfloat16))
    # ---------------- targets: closed-form final-residual change lifting the annotated token over the best competitor by margin m
    def logits_x(s, x):
        h = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + s.EPS) * s.m.norm.weight.float(); return s.logits(h)
    def targets(s, x, sites, m):
        W = s.lm.lm_head.weight; G = s.m.norm.weight.float(); T = torch.tensor([t for _, _, t in sites], device=s.dev)
        lg = s.logits_x(x); zt = lg.gather(1, T[:, None])[:, 0]; l2 = lg.clone(); l2.scatter_(1, T[:, None], -1e9); zc, ci = l2.max(1); gap = (zc - zt + m).clamp(min=0)
        u = G[None] * (W[T].float() - W[ci].float()); rms = x.pow(2).mean(-1, keepdim=True).add(s.EPS).sqrt()
        return rms * gap[:, None] * u / u.pow(2).sum(-1, keepdim=True).clamp(min=1e-9), gap
    def site_cap(s, seqs, sites, caps):
        bysi = collections.defaultdict(list)
        for si, p, t in sites: bysi[si].append(p)
        order = sorted(bysi); o = s.run([seqs[i] for i in order], [bysi[i] for i in order], cap=caps, budget=60000)
        return {c: torch.cat(o[c]) for c in caps}      # row order == sites order (sites sorted by (si,pos))
    # ---------------- solvers: K [I,n] keys, R [D,n] residual targets -> dW [D,I]
    def _chol(s, key, build):
        if key not in s.cache:
            A = build(); jit = 0.0
            for _ in range(8):
                Lc, info = torch.linalg.cholesky_ex(A + jit * A.diagonal().mean() * torch.eye(A.shape[0], device=s.dev))
                if int(info) == 0: break
                jit = max(jit * 10, 1e-5)
            s.cache[key] = Lc
        return s.cache[key]
    def solve_prot(s, kind, li, K, R, mt, cv):
        lam = mt.get("lam", 1000.0); awt = mt.get("awt", 10.0); ridge = mt.get("ridge", 0.05)
        def build():
            C = cv["C"] + awt * cv["Ca"]; return lam * (C + ridge * C.diagonal().mean() * torch.eye(C.shape[0], device=s.dev))
        Lc = s._chol(("chol", kind, li, lam, awt, ridge), build); Z = torch.cholesky_solve(K, Lc); I_ = torch.eye(K.shape[1], device=s.dev)
        return (R @ torch.linalg.solve(I_ + K.T @ Z, I_)) @ Z.T
    def solve_ultra(s, kind, li, K, R, mt, cv):
        lam = mt.get("lam", 1.0); A = lam * (cv["C"].diagonal() + mt.get("ridge", 0.05) * cv["C"].diagonal().mean())    # diagonal, memory-free preconditioner, no covariance matrix
        Z = K / A[:, None]; I_ = torch.eye(K.shape[1], device=s.dev); return (R @ torch.linalg.solve(I_ + K.T @ Z, I_)) @ Z.T
    def solve_alpha(s, kind, li, K, R, mt, cv):
        kp = mt.get("kp", 1024); key = ("alphaV", kind, li, kp)
        if key not in s.cache:
            U, S_, V = torch.svd_lowrank(cv["C"], q=kp + 32, niter=3); s.cache[key] = U[:, :kp].contiguous()
        V = s.cache[key]; Kp = K - V @ (V.T @ K)       # null-space projection of the keys (protected subspace = top eigenspace of the calibration key covariance)
        G = Kp.T @ Kp; lam = mt.get("lamA", 1e-3) * G.diagonal().mean().clamp(min=1e-9); return R @ torch.linalg.solve(G + lam * torch.eye(K.shape[1], device=s.dev), Kp.T)
    def solve_contrast(s, kind, li, K, R, mt, cv):
        lam = mt.get("lam", 1000.0); awt = mt.get("awt", 10.0); ridge = mt.get("ridge", 0.05)
        def build():
            C = cv["C"] + awt * cv["Ca"]; return lam * (C + ridge * C.diagonal().mean() * torch.eye(C.shape[0], device=s.dev))
        Lc = s._chol(("chol", kind, li, lam, awt, ridge), build); kb = K.mean(1); v = torch.cholesky_solve((kb - cv["mua"])[:, None], Lc)[:, 0]    # whitened contrast direction (annotated keys vs calibration answer keys)
        sdir = R.mean(1); return sdir[:, None] * (v / (v @ kb).clamp(min=1e-9))[None]      # rank-1, calibrated so the mean annotated key gets the mean target
    def solve_steer2(s, kind, li, K, R, mt, cv):
        r = mt.get("r", 64); sd = (cv["C"].diagonal() - cv["mu"] ** 2).clamp(min=1e-12).sqrt(); z = (K.mean(1) - cv["mua"]) / sd
        J = z.abs().topk(r).indices; a = torch.zeros_like(z); a[J] = z[J]; kb = K.mean(1); sdir = R.mean(1)
        return sdir[:, None] * (a / (a @ kb).clamp(min=1e-9))[None]    # rank-1 over r selected neurons/dims (Steer2Edit-style: steering direction distributed over weight columns by contrast score)
    # ---------------- module carriers whose OUTPUT adds straight into the residual: down_proj, o_proj
    def edit_module(s, mt, seqs, sites, carrier="bf16"):
        kind = mt["carrier"]; layers = sorted(mt["layers"]); solver = mt.get("solver", "prot"); m = mt.get("m", 16.0); mu = mt.get("cont", 1.0)
        cvs = s.cov_many(kind, layers) if solver != "none" else {}
        for step, li in enumerate(layers):
            cp = s.site_cap(seqs, sites, [(kind, li), "x"]); K = cp[(kind, li)].T.contiguous(); X = cp["x"]
            dx, gap = s.targets(X, sites, m); R = (dx / (len(layers) - step)).T.contiguous(); K0 = K; cv = cvs[("cov", kind, li)]
            if (kind, li) in s.prevK and mt.get("keep_prev", True):    # continual: earlier feedback's keys are protected (their effect must not move)
                Kp = s.prevK[(kind, li)]; K = torch.cat([K, mu * Kp], 1); R = torch.cat([R, torch.zeros(R.shape[0], Kp.shape[1], device=s.dev)], 1)
            dW = getattr(s, "solve_" + solver)(kind, li, K, R, mt, cv)
            s.write(kind, li, dW, carrier); s.prevK[(kind, li)] = torch.cat([s.prevK.get((kind, li), K0[:, :0]), K0], 1)
    # ---------------- MLP up / gate carriers (linearised through silu*up and a min-norm pseudo-inverse of down_proj)
    def edit_mlp_in(s, mt, seqs, sites, carrier="bf16"):
        kind = mt["carrier"]; layers = sorted(mt["layers"]); m = mt.get("m", 16.0); lam = mt.get("lam", 1000.0); ridge = mt.get("ridge", 0.05); awt = mt.get("awt", 10.0)
        cvs = s.cov_many("mlp_in", layers)
        for step, li in enumerate(layers):
            lyr = s.L[li]; cp = s.site_cap(seqs, sites, [("mlp_in", li), "x"]); Kin = cp[("mlp_in", li)]; X = cp["x"]; dx, gap = s.targets(X, sites, m); R = dx / (len(layers) - step)   # [n,D]
            Wd = lyr.mlp.down_proj.weight.float(); Gd = Wd @ Wd.T; Gd = Gd + 1e-3 * Gd.diagonal().mean() * torch.eye(Gd.shape[0], device=s.dev)
            dint = (Wd.T @ torch.linalg.solve(Gd, R.T)).T      # [n,I] min-norm intermediate change producing R through down_proj
            xb = Kin.to(torch.bfloat16); zg = (xb @ lyr.mlp.gate_proj.weight.T).float(); zu = (xb @ lyr.mlp.up_proj.weight.T).float(); sg = torch.sigmoid(zg); g = zg * sg
            if kind == "up": den = g
            else: den = zu * sg * (1 + zg * (1 - sg))
            T = dint * den / (den ** 2 + 0.01 * (den ** 2).mean(0, keepdim=True) + 1e-12)       # [n,I] damped inverse
            cv = cvs[("cov", "mlp_in", li)]; K = Kin.T.contiguous(); Tt = T.T.contiguous()
            Lc = s._chol(("chol", "mlp_in", li, lam, awt, ridge), lambda: lam * ((cv["C"] + awt * cv["Ca"]) + ridge * (cv["C"] + awt * cv["Ca"]).diagonal().mean() * torch.eye(s.D, device=s.dev)))
            Z = torch.cholesky_solve(K, Lc); I_ = torch.eye(K.shape[1], device=s.dev); dW = (Tt @ torch.linalg.solve(I_ + K.T @ Z, I_)) @ Z.T
            s.write(kind, li, dW, carrier)
    # ---------------- lm_head rows (UNSHIPPABLE reference, = fb_legal mnH)
    def edit_lm_row(s, mt, seqs, sites):
        mg = mt.get("m", 8.0); W = s.lm.lm_head.weight; cp = s.site_cap(seqs, sites, ["x"]); X = cp["x"]
        Hs = X * torch.rsqrt(X.pow(2).mean(-1, keepdim=True) + s.EPS) * s.m.norm.weight.float()
        if "calh" not in s.cache:
            hs = s.run(s.calseq, s.calpos)["h"]; s.cache["calh"] = torch.cat(hs).float()
        CH = s.cache["calh"]; M = CH.T @ CH / len(CH); M = M + 0.05 * M.diagonal().mean() * torch.eye(s.D, device=s.dev); by = collections.defaultdict(list)
        for i, (_, _, t) in enumerate(sites): by[t].append(i)
        n = "lm_head.weight"; s.touch(n)
        for t, ii in by.items():
            H = Hs[ii]; wt = W.data[t].float(); bl = (H.to(W.dtype) @ W.T).float(); z = H @ wt; bl[:, t] = -1e9; g = (bl.max(1).values - z + mg).clamp(min=0)
            MiH = torch.linalg.solve(M, H.T); G = H @ MiH; lam = 1e-3 * G.diagonal().mean(); W.data[t] = (wt + MiH @ torch.linalg.solve(G + lam * torch.eye(len(H), device=s.dev), g)).to(W.dtype)
    # ---------------- Gauss-Newton family: one linearised min-norm step per iteration on a small parameter set (norm gains, v biases, embedding rows)
    def _gn_margins(s, seqs, sites, params, embed_ids=None, chunk=6):
        """returns margin vector (z_t - z_c)  [n] and Jacobian rows [n, P] wrt params (list of tensors) or wrt embedding rows"""
        bysi = collections.defaultdict(list)
        for k, (si, p, t) in enumerate(sites): bysi[si].append((k, p, t))
        order = sorted(bysi); Jrows = [None] * len(sites); mar = torch.zeros(len(sites), device=s.dev)
        W = s.lm.lm_head.weight
        for c0 in range(0, len(order), chunk):
            idx = order[c0:c0 + chunk]; ids, am = s._pad(seqs, idx)
            with torch.enable_grad():
                if embed_ids is not None:
                    emb = s.m.embed_tokens(ids).detach().requires_grad_(True); h = s.m(inputs_embeds=emb, attention_mask=am).last_hidden_state
                else: h = s.m(input_ids=ids, attention_mask=am).last_hidden_state
                for r, si in enumerate(idx):
                    for k, p, t in bysi[si]:
                        z = (h[r, p].to(W.dtype) @ W.T).float(); zt = z[t]; z2 = z.clone().detach(); z2[t] = -1e9; c = int(z2.argmax()); mg = zt - z[c]; mar[k] = float(mg)
                        gr = torch.autograd.grad(mg, [emb] if embed_ids is not None else params, retain_graph=True)
                        if embed_ids is not None:
                            g_ = gr[0][r]; sel = ids[r]; Jrows[k] = torch.stack([g_[(sel == v) & (am[r].bool())].sum(0) for v in embed_ids]).float().reshape(-1)
                        else: Jrows[k] = torch.cat([x.float().reshape(-1) for x in gr])
                del h
        return mar, torch.stack(Jrows)
    def _fisher(s, params, nitem=48):
        key = ("fisher", tuple(id(p) for p in params))
        if key not in s.cache:
            F_ = torch.zeros(sum(p.numel() for p in params), device=s.dev)
            for i in range(nitem):
                seq = s.calseq[i]; ps = s.calpos[i]; ids = torch.tensor([seq], device=s.dev)
                with torch.enable_grad():
                    h = s.m(input_ids=ids).last_hidden_state[0, ps]; lg = (h.to(s.lm.lm_head.weight.dtype) @ s.lm.lm_head.weight.T).float(); tg = torch.tensor(seq[ps[0] + 1:ps[-1] + 2], device=s.dev)
                    loss = F.cross_entropy(lg, tg); gr = torch.autograd.grad(loss, params)
                F_ += torch.cat([g.float().reshape(-1) for g in gr]) ** 2
            s.cache[key] = F_ / nitem
        return s.cache[key]
    def edit_gn(s, mt, seqs, sites):
        kind = mt["carrier"]; m = mt.get("m", 8.0); iters = mt.get("iters", 3); lam = mt.get("lam", 1e-3); awt = mt.get("awt", 10.0)
        if kind == "embed":
            tmpl = set(s.pid("")); toks = sorted({t for sq in seqs for t in sq} - tmpl); names = ["model.embed_tokens.weight"]; params = None
            cnt = collections.Counter(t for sq in s.calP for t in set(sq)); Md = torch.tensor([1.0 + awt * cnt[t] / len(s.calP) for t in toks], device=s.dev)[:, None].expand(-1, s.D).reshape(-1)
        else:
            lays = sorted(mt["layers"]) if kind != "norm_final" else [0]; names = ["model.norm.weight"] if kind == "norm_final" else [s.tname(kind, li) for li in lays]; params = [s._named(n) for n in names]
            for p in params: p.requires_grad_(True)
            Md = 1.0 + awt * s._fisher(params) / (s._fisher(params).mean() + 1e-30)
        for n in names: s.touch(n)
        try:
            for it in range(iters):
                mar, J = s._gn_margins(seqs, sites, params, embed_ids=toks if kind == "embed" else None)
                gap = (m - mar).clamp(min=0) / (iters - it)
                if float(gap.max()) <= 0: break
                Mi = 1.0 / Md; JM = J * Mi[None]; G = JM @ J.T; ll = lam * G.diagonal().mean().clamp(min=1e-12); d = (JM.T @ torch.linalg.solve(G + ll * torch.eye(len(G), device=s.dev), gap))
                with torch.no_grad():
                    if kind == "embed":
                        E = s._named(names[0]); dd = d.reshape(len(toks), s.D)
                        for r, v in enumerate(toks): E.data[v] = (E.data[v].float() + dd[r]).to(torch.bfloat16)
                    else:
                        o = 0
                        for p in params: n_ = p.numel(); p.data.copy_((p.data.float() + d[o:o + n_].reshape(p.shape)).to(p.dtype)); o += n_
        finally:
            if params is not None:
                for p in params: p.requires_grad_(False)
    # ---------------- dispatcher
    def apply(s, mt, items):
        t0 = time.time(); c = mt["carrier"]; carrier = mt.get("store", "bf16")
        if c == "none": return dict(nsites=0, secs=0.0)
        seqs, sites = s.sites_of(items)
        if sites:
            if c in ("down", "o"): s.edit_module(mt, seqs, sites, carrier)
            elif c in ("up", "gate"): s.edit_mlp_in(mt, seqs, sites, carrier)
            elif c == "lm_row": s.edit_lm_row(mt, seqs, sites)
            elif c == "multi":
                for sub in mt["parts"]:
                    sq2, st2 = s.sites_of(items)
                    if st2: s.apply_part(sub, sq2, st2)
            else: s.edit_gn(mt, seqs, sites)
        torch.cuda.synchronize(); return dict(nsites=len(sites), secs=time.time() - t0)
    def apply_part(s, sub, seqs, sites):
        c = sub["carrier"]
        if c in ("down", "o"): s.edit_module(sub, seqs, sites, sub.get("store", "bf16"))
        elif c in ("up", "gate"): s.edit_mlp_in(sub, seqs, sites, sub.get("store", "bf16"))
        else: s.edit_gn(sub, seqs, sites)
    # ---------------- carrier size: what the edit adds on top of the unedited tensors (stored-dtype diff)
    def delta_stats(s):
        dense = sparse = 0; lowrank = 0; tensors = 0; changed_frac = []
        for n, w0 in s.saved.items():
            w1 = s._named(n).data; d = (w1.float() - w0.float()); tensors += 1
            if d.dim() == 2 and "embed" not in n and "lm_head" not in n:
                dense += d.numel() * 2; e = (d ** 2).sum(0); es, _ = e.sort(descending=True); cs = es.cumsum(0); k = int((cs < 0.99 * cs[-1]).sum()) + 1 if float(cs[-1]) > 0 else 0
                sparse += k * (d.shape[0] * 2 + 4); changed_frac.append(float((d != 0).float().mean()))
            elif d.dim() == 2:   # row-edit carrier (embedding / lm_head): rows touched
                rws = int((d.abs().sum(1) > 0).sum()); dense += rws * d.shape[1] * 2 + rws * 4; sparse += rws * d.shape[1] * 2 + rws * 4
            else: dense += d.numel() * 2; sparse += d.numel() * 2
        return dict(bytes_dense=dense, bytes_sparse99=sparse, tensors=tensors, nz_frac=float(np.mean(changed_frac)) if changed_frac else 0.0)
