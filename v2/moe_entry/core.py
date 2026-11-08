import os, json, time, math, glob, collections, shutil
import packfmt
import torch, torch.nn as nn, torch.nn.functional as F
MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507"; FP16_BYTES = 61_064_245_248
NL, NE, TOPK, HID, FF = 48, 128, 8, 2048, 768
T0 = time.time(); log = lambda *a: print(f"[{time.time()-T0:6.0f}s]", *a, flush=True)

def wilson(k, n, z=1.96):
    if n == 0: return (0.0, 0.0)
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); r = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - r) / d, (c + r) / d)

def rtn_codes(w, bits, g, mse=True):
    """RTN returning (dq, q uint8 [..,in], scale fp16 [..,in/g], zero uint8 [..,in/g])"""
    sh = w.shape; wg = w.float().reshape(*sh[:-1], sh[-1] // g, g); mx = wg.amax(-1, keepdim=True); mn = wg.amin(-1, keepdim=True); qm = 2 ** bits - 1
    def q(mn_, mx_):
        sc = ((mx_ - mn_) / qm).clamp(min=1e-8).half().float(); z = (-mn_ / sc).round().clamp(0, qm); qq = (wg / sc + z).round().clamp(0, qm); return qq, sc, z, (qq - z) * sc
    best = q(mn, mx); berr = ((best[3] - wg) ** 2).sum(-1, keepdim=True)
    if mse:
        for r in (0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65):
            c = q(mn * r, mx * r); err = ((c[3] - wg) ** 2).sum(-1, keepdim=True); m = err < berr
            best = tuple(torch.where(m, a, b) for a, b in zip(c, best)); berr = torch.where(m, err, berr)
    qq, sc, z, dq = best
    return dq.reshape(sh), qq.reshape(sh).to(torch.uint8), sc.squeeze(-1).half(), z.squeeze(-1).to(torch.uint8)

def rtn(w, bits, g, mse=False):
    """asymmetric per-group RTN along last dim; fp16 scale, integer zero-point. returns dequantised float tensor"""
    sh = w.shape; wg = w.float().reshape(*sh[:-1], sh[-1] // g, g); mx = wg.amax(-1, keepdim=True); mn = wg.amin(-1, keepdim=True); qm = 2 ** bits - 1
    def q(mn_, mx_):
        sc = ((mx_ - mn_) / qm).clamp(min=1e-8).half().float(); z = (-mn_ / sc).round().clamp(0, qm)
        return ((wg / sc + z).round().clamp(0, qm) - z) * sc
    if not mse: return q(mn, mx).reshape(sh)
    best = None; berr = None
    for r in (1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65):
        dq = q(mn * r, mx * r); err = ((dq - wg) ** 2).sum(-1, keepdim=True)
        if best is None: best, berr = dq, err
        else:
            m = err < berr; best = torch.where(m, dq, best); berr = torch.where(m, err, berr)
    return best.reshape(sh)


def had(n):
    H = torch.ones(1, 1)
    while H.shape[0] < n: H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    return H / math.sqrt(n)
def rot_matrix(dim, blk, seed, dev):
    g = torch.Generator().manual_seed(seed); sg = (torch.randint(0, 2, (dim,), generator=g) * 2 - 1).float()
    B = had(blk); Q = torch.zeros(dim, dim)
    for s0 in range(0, dim, blk): Q[s0:s0 + blk, s0:s0 + blk] = B
    return (sg[:, None] * Q).to(dev)   # Q = D * blockdiag(H); orthogonal
@torch.no_grad()
def gptq(W, H, bits, g, damp=0.01, codes=False):
    """W [o,i] float, H [i,i] float (sum x x^T). asym per-group (g along i) with fp16 scale + int zero, groups fixed when reached. returns dequantised W"""
    W = W.float().clone(); H = H.float().clone(); i_ = W.shape[1]; qm = 2 ** bits - 1
    dead = torch.diag(H) == 0; H[dead, dead] = 1; W[:, dead] = 0
    H += damp * torch.mean(torch.diag(H)) * torch.eye(i_, device=W.device)
    try: Hi = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(H)), upper=True)
    except Exception: return rtn_codes(W, bits, g, True) if codes else rtn(W, bits, g, True)
    Q = torch.zeros_like(W); QI = torch.zeros(W.shape, dtype=torch.uint8, device=W.device); SC = []; ZZ = []
    for b0 in range(0, i_, g):
        W1 = W[:, b0:b0 + g].clone(); Hi1 = Hi[b0:b0 + g, b0:b0 + g]; E = torch.zeros_like(W1)
        mn = W1.amin(1, keepdim=True); mx = W1.amax(1, keepdim=True); sc = ((mx - mn) / qm).clamp(min=1e-8).half().float(); z = (-mn / sc).round().clamp(0, qm); SC.append(sc.half()); ZZ.append(z.to(torch.uint8))
        for j in range(W1.shape[1]):
            w = W1[:, j:j + 1]; qi = (w / sc + z).round().clamp(0, qm); q = (qi - z) * sc; Q[:, b0 + j:b0 + j + 1] = q; QI[:, b0 + j:b0 + j + 1] = qi.to(torch.uint8)
            err = (w - q) / Hi1[j, j]; W1[:, j:] -= err @ Hi1[j:j + 1, j:]; E[:, j:j + 1] = err
        W[:, b0 + g:] -= E @ Hi[b0:b0 + g, b0 + g:]
    if codes: return Q, QI, torch.cat(SC, 1), torch.cat(ZZ, 1)
    return Q

def bpw(bits, g): return 16.0 if not bits else bits + (16 + bits) / g


def moe_fwd(self, hs):
    B, T, D = hs.shape; x = hs.view(-1, D); lg = self.gate(x)
    p = F.softmax(lg.float(), -1); w, ii = torch.topk(p, TOPK, -1); w = (w / w.sum(-1, keepdim=True)).to(x.dtype)
    out = torch.zeros_like(x); rec = getattr(self, "rec", None); hs_ = getattr(self, "hset", None); valid = getattr(self, "valid", None)
    em = F.one_hot(ii, NE).permute(2, 1, 0); hit = (em.sum((1, 2)) > 0).nonzero().flatten().tolist()
    for e in hit:
        k_, t_ = torch.where(em[e]); xe = x[t_]; ex = self.experts[e]
        a = ex.act_fn(ex.gate_proj(xe)) * ex.up_proj(xe); y = ex.down_proj(a); we = w[t_, k_]
        out.index_add_(0, t_, y * we[:, None])
        if rec is not None:
            v = valid[t_] if valid is not None else torch.ones(len(t_), device=x.device)
            rec[0][e] += (we.float() * v).sum(); rec[1][e] += v.sum(); rec[2][e] += (we.float() * y.float().norm(dim=-1) * v).sum()
        if hs_ is not None and e in hs_:
            v = valid[t_].bool() if valid is not None else slice(None); h = hs_[e]
            xv = xe[v].float(); av = a[v].float(); h[0] += xv.T @ xv; h[1] += av.T @ av; h[2] += xv.shape[0]
    return out.view(B, T, D)

class GateW(nn.Module):
    def __init__(s, gate): super().__init__(); s.g = gate; s.mask = None; s.rec = None; s.valid = None
    def forward(s, x):
        lg = s.g(x)
        if s.rec is not None:
            p = F.softmax(lg.float(), -1); w, i = torch.topk(p, TOPK, -1); w = w / w.sum(-1, keepdim=True)
            if s.valid is not None: w = w * s.valid[:, None]; cnt = s.valid
            else: cnt = torch.ones(w.shape[0], device=w.device)
            s.rec[0].index_add_(0, i.reshape(-1), w.reshape(-1)); s.rec[1].index_add_(0, i.reshape(-1), (cnt[:, None].expand_as(w)).reshape(-1))
        if s.mask is not None: lg = lg + s.mask
        return lg

def main(cfgs, vol):
    global DS; DS = cfgs[0].get("ds", "news")
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from huggingface_hub import snapshot_download
    snap = snapshot_download(MODEL, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model"])
    tokb = sum(os.path.getsize(os.path.join(snap, f)) for f in ["tokenizer.json", "vocab.json", "merges.txt", "tokenizer_config.json", "config.json", "generation_config.json"] if os.path.exists(os.path.join(snap, f)))
    tok = AutoTokenizer.from_pretrained(MODEL); tok.padding_side = "left"
    lm = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, device_map={"": 0}).eval()
    for p in lm.parameters(): p.requires_grad_(False)
    dev = "cuda"; os.makedirs("/r/res", exist_ok=True)
    blocks = [l.mlp for l in lm.model.layers]; gates = []
    import types
    for b in blocks: gw = GateW(b.gate); b.gate = gw; gates.append(gw); b.forward = types.MethodType(moe_fwd, b)
    # CPU copies of original weights
    log("copy orig to cpu")
    O = []
    for l, lay in enumerate(lm.model.layers):
        ex = lay.mlp.experts
        O.append(dict(gate=torch.stack([e.gate_proj.weight.data.cpu() for e in ex]), up=torch.stack([e.up_proj.weight.data.cpu() for e in ex]), down=torch.stack([e.down_proj.weight.data.cpu() for e in ex]),
                      attn={n: getattr(lay.self_attn, n).weight.data.cpu().clone() for n in ("q_proj", "k_proj", "v_proj", "o_proj")}))
    OE = lm.model.embed_tokens.weight.data.cpu().clone(); OH = lm.lm_head.weight.data.cpu().clone()
    log("orig copied")
    chat = lambda u: tok.apply_chat_template([{"role": "user", "content": u}], tokenize=False, add_generation_prompt=True)
    def load(n, k=None):
        pth = f"/root/splits/{n.lower()}.jsonl" if DS == "news" else f"/root/data/{'voice' if DS.startswith('voice') else DS}_{n.lower()}.jsonl"
        rows = [json.loads(l) for l in open(pth)]
        if DS == "voices":
            for r in rows: r["user"] = "Route the request to one skill: clock, calendar, lists, email, smart_home, music, weather, transport, food, qa. Answer with the skill name only.\nRequest: " + r["text"]
        for r in rows: r["ids"] = tok(chat(r["user"]), add_special_tokens=False).input_ids
        return rows
    DATA = {}
    def getset(n):
        if n not in DATA: DATA[n] = load(n)
        return DATA[n]
    def batches(rows, budget):
        order = sorted(range(len(rows)), key=lambda i: len(rows[i]["ids"])); i = 0
        while i < len(order):
            j = i
            while j < len(order) and (j - i + 1) * len(rows[order[j]]["ids"]) <= budget and (j - i) < 128: j += 1
            j = max(j, i + 1); yield order[i:j]; i = j
    def pad(seqs):
        L = max(len(s) for s in seqs); ids = torch.full((len(seqs), L), tok.pad_token_id); am = torch.zeros((len(seqs), L), dtype=torch.long)
        for r, s in enumerate(seqs): ids[r, L - len(s):] = torch.tensor(s); am[r, L - len(s):] = 1
        return ids.to(dev), am.to(dev)
    @torch.no_grad()
    def gen_eval(rows, budget=40000, max_new=5):
        outs = [None] * len(rows)
        for idx in batches(rows, budget):
            ids, am = pad([rows[k]["ids"] for k in idx]); L = ids.shape[1]
            g = lm.generate(input_ids=ids, attention_mask=am, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.pad_token_id)
            txt = tok.batch_decode(g[:, L:], skip_special_tokens=True)
            for k, t in zip(idx, txt): outs[k] = t.strip()
        return outs
    def prefix_len(rows):
        a = rows[0]["ids"]; p = len(a)
        for r in rows[1:200]:
            b = r["ids"]; k = 0
            while k < min(p, len(b)) and a[k] == b[k]: k += 1
            p = k
        return p
    def vmask(am, ids_rows, skip, P, first_full):
        """valid flat mask; if skip: positions of the shared prompt prefix are excluded (counted only for the first row of the first batch)"""
        v = am.clone().float()
        if skip:
            L = am.shape[1]
            for b, seq in enumerate(ids_rows):
                if first_full and b == 0: continue
                st = L - len(seq); v[b, st:st + P] = 0
        return v.flatten()
    @torch.no_grad()
    def routing_mass(rows, budget=40000, skip=False):
        P = prefix_len(rows) if skip else 0; first = [True]
        rec = [(torch.zeros(NE, device=dev), torch.zeros(NE, device=dev), torch.zeros(NE, device=dev)) for _ in range(NL)]
        for l, b in enumerate(blocks): b.rec = rec[l]
        for idx in batches(rows, budget):
            ids, am = pad([rows[k]["ids"] for k in idx]); v = vmask(am, [rows[k]["ids"] for k in idx], skip, P, first[0]); first[0] = False
            for b in blocks: b.valid = v
            lm.model(input_ids=ids, attention_mask=am)
        for b in blocks: b.rec = None; b.valid = None
        return tuple(torch.stack([r[j] for r in rec]).cpu() for j in range(3))
    @torch.no_grad()
    def collect_hessians(rows, layers, kept, budget=30000, skip=False):
        P = prefix_len(rows) if skip else 0; first = [True]
        for l in layers: blocks[l].hset = {e: [torch.zeros(HID, HID, device=dev), torch.zeros(FF, FF, device=dev), 0] for e in kept[l]}
        for idx in batches(rows, budget):
            ids, am = pad([rows[k]["ids"] for k in idx]); v = vmask(am, [rows[k]["ids"] for k in idx], skip, P, first[0]); first[0] = False
            for l in layers: blocks[l].valid = v
            lm.model(input_ids=ids, attention_mask=am, use_cache=False, logits_to_keep=1) if False else lm.model.forward(input_ids=ids, attention_mask=am, use_cache=False)
        out = {l: blocks[l].hset for l in layers}
        for l in layers: blocks[l].hset = None; blocks[l].valid = None
        return out
    # ---- apply config
    PK = [None]
    def keep_lists(cfg, mass0):
        mass = {"mass": mass0, "reap": reap / cnt.clamp(min=1), "massreap": (mass0 / mass0.sum(1, keepdim=True)) * (reap / cnt.clamp(min=1) / (reap / cnt.clamp(min=1)).mean(1, keepdim=True))}[cfg.get("score", "mass")]
        if cfg.get("kept_counts"): return [sorted(torch.topk(mass[l], cfg["kept_counts"][l]).indices.tolist()) for l in range(NL)]
        if cfg.get("alpha") is not None:
            mf = get_mass(False)[0]; ms = get_mass(True)[0]; pp = (mf - ms).clamp(min=0); a = cfg["alpha"]
            mass0 = a * pp / pp.sum(1, keepdim=True) + (1 - a) * ms / ms.sum(1, keepdim=True); mass = mass0
        k = cfg.get("keep", NE)
        if cfg.get("kept_file"): return [sorted(x) for x in json.load(open(cfg["kept_file"]))]
        if isinstance(k, float) and k <= 1.0: k = int(round(k * NE))
        if k >= NE: return [list(range(NE)) for _ in range(NL)]
        if cfg.get("mode", "layer") == "layer":
            return [sorted(torch.topk(mass[l], k).indices.tolist()) for l in range(NL)]
        frac = mass / mass.sum(1, keepdim=True); tot = int(k * NL); kmin = cfg.get("kmin", 16)
        keep = [set(torch.topk(frac[l], kmin).indices.tolist()) for l in range(NL)]; rem = tot - kmin * NL
        cand = sorted(((frac[l, e].item(), l, e) for l in range(NL) for e in range(NE) if e not in keep[l]), reverse=True)[:rem]
        for _, l, e in cand: keep[l].add(e)
        return [sorted(x) for x in keep]
    def bytes_of(cfg, kept):
        eb, eg = cfg.get("eb"), cfg.get("eg", 128); ab, gb = cfg.get("attn_b"), cfg.get("attn_g", 128)
        nk = sum(len(k) for k in kept); epar = 3 * HID * FF
        d = {}
        d["experts"] = nk * epar * bpw(eb, eg) / 8 if cfg.get("quant") != "vq4" else nk * (epar * 2.0 / 8 + 4) + 2048
        d["router"] = nk * HID * 2
        d["keep_index"] = nk if nk < NL * NE else 0
        ap = HID * 4096 + 2 * HID * 512 + 4096 * HID
        d["attn"] = NL * ap * bpw(ab, gb) / 8
        d["embed"] = 151936 * HID * bpw(cfg.get("emb_b"), 128) / 8
        d["head"] = 151936 * HID * bpw(cfg.get("head_b"), 128) / 8
        d["norms"] = (NL * (2 * HID + 256) + HID) * 2
        d["tokenizer_config"] = tokb + cfg.get("extra_bytes", 0)
        tot = sum(d.values()); return dict(parts={k: int(v) for k, v in d.items()}, total=int(tot), ratio=tot / FP16_BYTES)
    @torch.no_grad()
    def apply(cfg, kept):
        if cfg.get("pack"):
            ad = f"/r/res/art_{DS}_{cfg['name']}"; os.makedirs(ad, exist_ok=True); PK[0] = packfmt.Writer(ad + "/weights.bin")
            for l, lay in enumerate(lm.model.layers):
                for nm_, t_ in (("input_layernorm", lay.input_layernorm.weight), ("post_attention_layernorm", lay.post_attention_layernorm.weight), ("q_norm", lay.self_attn.q_norm.weight), ("k_norm", lay.self_attn.k_norm.weight), ("router", gates[l].g.weight.data[kept[l]])):
                    PK[0].add_fp16(f"L{l}.{nm_}", t_)
            PK[0].add_fp16("final_norm", lm.model.norm.weight)
        else: PK[0] = None
        mse = cfg.get("quant", "rtn") == "rtnmse"; eb, eg = cfg.get("eb"), cfg.get("eg", 128)
        for l, lay in enumerate(lm.model.layers):
            ki = kept[l]; ex = lay.mlp.experts
            for nm, mod in (("gate", "gate_proj"), ("up", "up_proj"), ("down", "down_proj")):
                w = O[l][nm][ki].to(dev)
                if eb and cfg.get("quant") not in ("gptq", "vq4"):
                    if PK[0] is not None:
                        w, q_, s_, z_ = rtn_codes(w, eb, eg, mse)
                        for j, e in enumerate(ki): PK[0].add_q(f"L{l}.E{e}.{nm}", q_[j], s_[j], z_[j], eb, eg)
                    else: w = rtn(w, eb, eg, mse)
                for j, e in enumerate(ki): getattr(ex[e], mod).weight.data.copy_(w[j].to(torch.bfloat16))
                del w
            m = torch.zeros(NE, device=dev); m[[e for e in range(NE) if e not in set(ki)]] = float("-inf"); gates[l].mask = m if len(ki) < NE else None
            for n_, w0 in O[l]["attn"].items():
                w = w0.to(dev)
                if cfg.get("attn_b"):
                    if PK[0] is not None: w, q_, s_, z_ = rtn_codes(w, cfg["attn_b"], cfg.get("attn_g", 128), mse); PK[0].add_q(f"L{l}.{n_}", q_, s_, z_, cfg["attn_b"], cfg.get("attn_g", 128))
                    else: w = rtn(w, cfg["attn_b"], cfg.get("attn_g", 128), mse)
                getattr(lay.self_attn, n_).weight.data.copy_(w.to(torch.bfloat16))
        for tgt, src, b, nm_ in ((lm.model.embed_tokens, OE, cfg.get("emb_b"), "embed"), (lm.lm_head, OH, cfg.get("head_b"), "head")):
            parts = []
            for s in range(0, src.shape[0], 20000):
                w = src[s:s + 20000].to(dev)
                if b:
                    w, q_, s_, z_ = rtn_codes(w, b, 128, False); parts.append((q_.cpu(), s_.cpu(), z_.cpu()))
                tgt.weight.data[s:s + 20000].copy_(w.to(torch.bfloat16))
            if PK[0] is not None and b: PK[0].add_q(nm_, torch.cat([p[0] for p in parts]), torch.cat([p[1] for p in parts]), torch.cat([p[2] for p in parts]), b, 128)
    @torch.no_grad()
    def gptq_all(cfg, kept):
        calrows = getset("cal")[:cfg.get("nh", 1000)]; eb, eg = cfg["eb"], cfg.get("eg", 128); blk = cfg.get("hblk", 256); fb = 0
        for g0 in range(0, NL, cfg.get("lgroup", 6)):
            layers = list(range(g0, min(NL, g0 + cfg.get("lgroup", 6)))); Hs = collect_hessians(calrows, layers, kept, skip=cfg.get("skip", False)); log("hessians", layers[0], layers[-1])
            for l in layers:
                Qi = rot_matrix(HID, min(blk, HID), 1000 + l, dev) if cfg.get("had", True) else torch.eye(HID, device=dev); Qd = rot_matrix(FF, min(blk, FF), 2000 + l, dev) if cfg.get("had", True) else torch.eye(FF, device=dev)
                ex = lm.model.layers[l].mlp.experts
                for e in kept[l]:
                    Hin, Hdn, n = Hs[l][e]
                    gw = O[l]["gate"][e].to(dev).float(); uw = O[l]["up"][e].to(dev).float(); dw = O[l]["down"][e].to(dev).float()
                    if n < cfg.get("nmin", 32):
                        fb += 1; qg, a1, a2, a3 = rtn_codes(gw, eb, eg, True); qu, b1, b2, b3 = rtn_codes(uw, eb, eg, True); qd, c1, c2, c3 = rtn_codes(dw, eb, eg, True)
                        if PK[0] is not None: PK[0].add_q(f"L{l}.E{e}.gate", a1, a2, a3, eb, eg); PK[0].add_q(f"L{l}.E{e}.up", b1, b2, b3, eb, eg); PK[0].add_q(f"L{l}.E{e}.down", c1, c2, c3, eb, eg)
                    else:
                        Hi_ = Qi.T @ Hin @ Qi; Hd_ = Qd.T @ Hdn @ Qd; gu = torch.cat([gw, uw], 0) @ Qi; q, qq_, ss_, zz_ = gptq(gu, Hi_, eb, eg, cfg.get("damp", 0.01), codes=True); q = q @ Qi.T; qg, qu = q[:FF], q[FF:]
                        qd, qd_q, qd_s, qd_z = gptq(dw @ Qd, Hd_, eb, eg, cfg.get("damp", 0.01), codes=True); qd = qd @ Qd.T
                        if PK[0] is not None:
                            PK[0].add_q(f"L{l}.E{e}.gate", qq_[:FF], ss_[:FF], zz_[:FF], eb, eg); PK[0].add_q(f"L{l}.E{e}.up", qq_[FF:], ss_[FF:], zz_[FF:], eb, eg); PK[0].add_q(f"L{l}.E{e}.down", qd_q, qd_s, qd_z, eb, eg)
                    ex[e].gate_proj.weight.data.copy_(qg.to(torch.bfloat16)); ex[e].up_proj.weight.data.copy_(qu.to(torch.bfloat16)); ex[e].down_proj.weight.data.copy_(qd.to(torch.bfloat16))
                Hs[l] = None
            del Hs; torch.cuda.empty_cache()
        log("gptq done, rtn fallbacks", fb)
    @torch.no_grad()
    def vq_all(cfg, kept):
        import qlib, qlib_b
        class VQh(qlib_b.VQ):   # fp16-exact per-expert sigma and fp16 codebook (so the packed file reproduces the simulation bit for bit)
            def prepare(s, W):
                super().prepare(W); s.sig = s.sig.half().float(); s.cbs = [c.half().float() for c in s.cbs]; s.cns = [0.5 * (c ** 2).sum(1) for c in s.cbs]
        calrows = getset("cal")[:cfg.get("nh", 1000)]; G = cfg.get("lgroup", 4); damp = cfg.get("damp", 0.01); cbw = [None]
        for g0 in range(0, NL, G):
            layers = list(range(g0, min(NL, g0 + G))); Hs = collect_hessians(calrows, layers, kept, skip=cfg.get("skip", False)); log("hessians", layers[0], layers[-1])
            for l in layers:
                ki = kept[l]; E = len(ki); ex = lm.model.layers[l].mlp.experts
                Hin = torch.stack([Hs[l][e][0] for e in ki]); Hdn = torch.stack([Hs[l][e][1] for e in ki]); Hs[l] = None
                Wgu = torch.cat([O[l]["gate"][ki], O[l]["up"][ki]], 1).to(dev).float(); Wdn = O[l]["down"][ki].to(dev).float()
                for nm, W, H, seed in (("gu", Wgu, Hin, 1000 + l), ("dn", Wdn, Hdn, 2000 + l)):
                    Wr, Hr, back = qlib_b.rotb(W, H, seed, "had"); rd = VQh(4, 256, "gauss"); Qm = qlib_b.engine_b(Wr, Hr, rd, damp); dq = back(Qm)
                    if PK[0] is not None:
                        cbw[0] = rd.cbs[0].half(); X = (Qm / rd.sig).reshape(-1, 4); idx = qlib.nn_argmax(X, rd.cbs[0], rd.cns[0]).reshape(E, -1).to(torch.uint8).cpu(); sg = rd.sig.flatten().half().cpu()
                        for j, e in enumerate(ki): PK[0].add_vq(f"L{l}.E{e}.{nm}", idx[j], sg[j], list(W.shape[1:]), seed)
                    for j, e in enumerate(ki):
                        if nm == "gu": ex[e].gate_proj.weight.data.copy_(dq[j, :FF].to(torch.bfloat16)); ex[e].up_proj.weight.data.copy_(dq[j, FF:].to(torch.bfloat16))
                        else: ex[e].down_proj.weight.data.copy_(dq[j].to(torch.bfloat16))
                del Hin, Hdn, Wgu, Wdn
            del Hs; torch.cuda.empty_cache()
        if PK[0] is not None: PK[0].add_fp16("vq_cb", cbw[0])
        log("vq done")
    # ---- routing mass (cached)
    ncal = max([c.get("ncal", 2000) for c in cfgs]); mp = f"/r/res/mass_{DS}_{ncal}.pt"
    MS = {}
    def get_mass(skip):
        if skip in MS: return MS[skip]
        p_ = mp.replace(".pt", "_skip.pt") if skip else mp
        if os.path.exists(p_): MS[skip] = torch.load(p_)
        else:
            cal = getset("cal")[:ncal]; MS[skip] = routing_mass(cal, skip=skip); torch.save(MS[skip], p_); vol.commit(); log("mass done", skip, len(cal))
        return MS[skip]
    mass, cnt, reap = get_mass(False)
    for cfg in cfgs:
        t1 = time.time(); name = cfg["name"]; log("CFG", name, cfg)
        mass, cnt, reap = get_mass(bool(cfg.get('skip', False)))
        kept = keep_lists(cfg, mass); by = bytes_of(cfg, kept)
        if cfg.get("bytes_only"): print("RESULT_JSON", json.dumps(dict(name=name, cfg=cfg, bytes=by, kept_per_layer=[len(k) for k in kept]))); continue
        apply(cfg, kept)
        if cfg.get('quant') == 'gptq': gptq_all(cfg, kept)
        if cfg.get('quant') == 'vq4': vq_all(cfg, kept)
        tot = 0
        if PK[0] is not None:
            ad = f"/r/res/art_{DS}_{name}"; PK[0].idx["_meta"] = dict(kept=kept, had=cfg.get("had", True) if cfg.get("quant") == "gptq" else False, vq=(cfg.get("quant") == "vq4"), hblk=cfg.get("hblk", 256), eb=cfg.get("eb"), eg=cfg.get("eg", 128)); PK[0].close(ad + "/index.json")
            for f_ in ["config.json", "generation_config.json", "tokenizer.json", "vocab.json", "merges.txt", "tokenizer_config.json"]:
                if os.path.exists(os.path.join(snap, f_)): shutil.copy(os.path.join(snap, f_), ad + "/" + f_)
            tot = sum(os.path.getsize(os.path.join(ad, f_)) for f_ in os.listdir(ad)); log("ARTIFACT bytes", tot, tot / FP16_BYTES)
        res = dict(name=name, cfg=cfg, bytes=by, artifact=(dict(bytes=tot, ratio=tot / FP16_BYTES) if cfg.get('pack') else None), kept_per_layer=[len(k) for k in kept], sets={})
        for sn in cfg.get("sets", ["SEL"]):
            rows = getset(sn)
            if cfg.get("sub"): rows = rows[::cfg["sub"]]
            outs = gen_eval(rows); res_s = dict(outs=outs, ids=[r["id"] for r in rows]); correct = [o == r["answer"] for o, r in zip(outs, rows)]
            bp = f"/r/res/base_{DS}_{sn}_{cfg.get('sub',1)}.json"
            if name == "base": json.dump(dict(outs=outs, ids=res_s["ids"]), open(bp, "w")); vol.commit()
            sm = dict(n=len(rows), acc=sum(correct) / len(rows))
            if os.path.exists(bp):
                b = json.load(open(bp)); assert b["ids"] == res_s["ids"]; bc = [o == r["answer"] for o, r in zip(b["outs"], rows)]
                nb = sum(bc); kk = sum(1 for x, y in zip(correct, bc) if x and y); sm.update(base_acc=nb / len(rows), n_bc=nb, kept_bc=kk, ret=kk / max(nb, 1), ret_lo=wilson(kk, nb)[0])
                cells = collections.defaultdict(lambda: [0, 0])
                for x, y, r in zip(correct, bc, rows):
                    if y: c = cells[r["src"] + "|" + r["kind"]]; c[0] += 1; c[1] += int(x)
                sm["cells"] = {k: dict(n=v[0], ret=v[1] / v[0], lo=wilson(v[1], v[0])[0]) for k, v in sorted(cells.items())}
                srcs = collections.defaultdict(lambda: [0, 0])
                for x, y, r in zip(correct, bc, rows):
                    if y: c = srcs[r["src"]]; c[0] += 1; c[1] += int(x)
                sm["src_ret"] = {k: round(v[1] / v[0], 4) for k, v in srcs.items()}
                sm["min_cell"] = min(v["ret"] for v in sm["cells"].values())
                bad = collections.Counter(o for x, y, o in zip(correct, bc, outs) if y and not x); sm["flip_outs"] = dict(bad.most_common(8))
            res["sets"][sn] = sm; json.dump(res_s, open(f"/r/res/outs_{DS}_{name}_{sn}.json", "w"))
        res["secs"] = round(time.time() - t1); json.dump(res, open(f"/r/res/{DS}_{name}.json", "w")); vol.commit()
        print("RESULT_JSON", json.dumps(res), flush=True)
        if cfg.get("fb"):
            import fb; fb.feedback(cfg, dict(lm=lm, tok=tok, chat=chat, gen_eval=gen_eval, getset=getset, log=log))
