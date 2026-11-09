"""L4 compression + real packing (compression side; NOT shipped). Same recipe as v2/rival3/core.py (top-K experts by routing mass, GPTQ+block-Hadamard 2-bit g128 experts, RTN 6b attn, 5b embed, 4b head),
but every quantised tensor is also emitted as integer codes + fp16 scales + integer zeros and written to the packed file (l4fmt). last CARRIER layers' o_proj stay raw bf16 (feedback carrier)."""
import os, sys, json, time, math, hashlib, types, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, "/root")
import torch, torch.nn as nn
import l4fmt as F4
import core
from core import NL, NE, HID, FF, log, GateW, moe_fwd, FP16_BYTES
MODEL = core.MODEL

def rtn_c(w, bits, g, mse=False):
    sh = w.shape; wg = w.float().reshape(*sh[:-1], sh[-1] // g, g); mx = wg.amax(-1, keepdim=True); mn = wg.amin(-1, keepdim=True); qm = 2 ** bits - 1
    def q(mn_, mx_):
        sc = ((mx_ - mn_) / qm).clamp(min=1e-8).half().float(); z = (-mn_ / sc).round().clamp(0, qm); c = (wg / sc + z).round().clamp(0, qm)
        return c, sc, z, (c - z) * sc
    if not mse: c, sc, z, dq = q(mn, mx)
    else:
        best = None
        for r in (1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65):
            c, sc, z, dq = q(mn * r, mx * r); err = ((dq - wg) ** 2).sum(-1, keepdim=True)
            if best is None: best, berr = (c, sc, z, dq), err
            else:
                m = err < berr; best = tuple(torch.where(m, a, b) for a, b in zip((c, sc, z, dq), best)); berr = torch.where(m, err, berr)
        c, sc, z, dq = best
    return dq.reshape(sh), c.reshape(sh).to(torch.uint8), sc.squeeze(-1), z.squeeze(-1).to(torch.uint8)

@torch.no_grad()
def gptq_c(W, H, bits, g, damp=0.01):
    W = W.float().clone(); H = H.float().clone(); i_ = W.shape[1]; qm = 2 ** bits - 1
    dead = torch.diag(H) == 0; H[dead, dead] = 1; W[:, dead] = 0
    H += damp * torch.mean(torch.diag(H)) * torch.eye(i_, device=W.device)
    try: Hi = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(H)), upper=True)
    except Exception: return rtn_c(W, bits, g, True)
    Q = torch.zeros_like(W); C = torch.zeros(W.shape, dtype=torch.uint8, device=W.device); SC = []; Z = []
    for b0 in range(0, i_, g):
        W1 = W[:, b0:b0 + g].clone(); Hi1 = Hi[b0:b0 + g, b0:b0 + g]; E = torch.zeros_like(W1)
        mn = W1.amin(1, keepdim=True); mx = W1.amax(1, keepdim=True); sc = ((mx - mn) / qm).clamp(min=1e-8).half().float(); z = (-mn / sc).round().clamp(0, qm)
        SC.append(sc); Z.append(z)
        for j in range(W1.shape[1]):
            w = W1[:, j:j + 1]; c = (w / sc + z).round().clamp(0, qm); q = (c - z) * sc; Q[:, b0 + j:b0 + j + 1] = q; C[:, b0 + j:b0 + j + 1] = c.to(torch.uint8)
            err = (w - q) / Hi1[j, j]; W1[:, j:] -= err @ Hi1[j:j + 1, j:]; E[:, j:j + 1] = err
        W[:, b0 + g:] -= E @ Hi[b0:b0 + g, b0 + g:]
    return Q, C, torch.cat(SC, 1), torch.cat(Z, 1).to(torch.uint8)

def size_formula(kept, cfg, tokbytes, carrier):
    """exact packed size of the arrays (excl. header/trailer ~ tens of KB)"""
    eb, g = cfg["eb"], 128; pb = lambda n, b: (n * b + 7) // 8; nk = sum(len(k) for k in kept)
    exp = 0
    for k in kept:
        n = len(k)
        for rows, cols in ((FF, HID), (FF, HID), (HID, FF)):
            exp += pb(n * rows * cols, eb) + n * rows * (cols // g) * 2 + pb(n * rows * (cols // g), eb)
    qs = lambda rows, cols, b: pb(rows * cols, b) + rows * (cols // g) * 2 + pb(rows * (cols // g), b)
    ab = cfg["attn_b"]; attn = 0
    for l in range(NL):
        for rows, cols, o in ((4096, HID, 0), (512, HID, 0), (512, HID, 0), (HID, 4096, 1)):
            attn += 2 * rows * cols if (o and l >= NL - carrier) else qs(rows, cols, ab)
    emb = qs(151936, HID, cfg["emb_b"]); head = qs(151936, HID, cfg["head_b"])
    other = nk * HID * 2 + NL * (2 * HID + 256) * 2 + HID * 2 + NL * 16 + nk + tokbytes
    return exp + attn + emb + head + other

def pack_main(task, cfgs, vol, outdir):
    cfg = cfgs[0]; DS = cfg["ds"]; core.DS = DS
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from huggingface_hub import snapshot_download
    snap = snapshot_download(MODEL, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model"])
    files = ["tokenizer.json", "vocab.json", "merges.txt", "tokenizer_config.json", "config.json", "generation_config.json"]
    tokraw = {f: open(os.path.join(snap, f), "rb").read() for f in files if os.path.exists(os.path.join(snap, f))}
    import zlib; tokb = sum(len(zlib.compress(b, 9)) for b in tokraw.values())
    tok = AutoTokenizer.from_pretrained(MODEL); tok.padding_side = "left"
    lm = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, device_map={"": 0}).eval()
    for p in lm.parameters(): p.requires_grad_(False)
    dev = "cuda"; os.makedirs(outdir, exist_ok=True); CAR = cfg.get("carrier", 8)
    blocks = [l.mlp for l in lm.model.layers]
    for b in blocks: gw = GateW(b.gate); b.gate = gw; b.forward = types.MethodType(moe_fwd, b)
    O = []
    for l, lay in enumerate(lm.model.layers):
        ex = lay.mlp.experts
        O.append(dict(gate=torch.stack([e.gate_proj.weight.data.cpu() for e in ex]), up=torch.stack([e.up_proj.weight.data.cpu() for e in ex]), down=torch.stack([e.down_proj.weight.data.cpu() for e in ex]),
                      attn={n: getattr(lay.self_attn, n).weight.data.cpu().clone() for n in ("q_proj", "k_proj", "v_proj", "o_proj")}))
    OE = lm.model.embed_tokens.weight.data.cpu().clone(); OH = lm.lm_head.weight.data.cpu().clone(); log("orig copied")
    chat = lambda u: tok.apply_chat_template([{"role": "user", "content": u}], tokenize=False, add_generation_prompt=True)
    rows_ = [json.loads(l) for l in open(f"/root/data/{DS}_cal.jsonl")]
    for r in rows_: r["ids"] = tok(chat(r["user"]), add_special_tokens=False).input_ids
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
    def collect_hessians(rows, layers, kept, budget=30000):
        for l in layers: blocks[l].hset = {e: [torch.zeros(HID, HID, device=dev), torch.zeros(FF, FF, device=dev), 0] for e in kept[l]}
        for idx in batches(rows, budget):
            ids, am = pad([rows[k]["ids"] for k in idx]); v = am.flatten().float()
            for l in layers: blocks[l].valid = v
            lm.model.forward(input_ids=ids, attention_mask=am, use_cache=False)
        out = {l: blocks[l].hset for l in layers}
        for l in layers: blocks[l].hset = None; blocks[l].valid = None
        return out
    mp = f"/r/res/mass_{DS}_{cfg.get('ncal', 2000)}.pt"; assert os.path.exists(mp), "routing mass cache missing " + mp
    mass, cnt, reap = torch.load(mp)
    kept = [sorted(torch.topk(mass[l], cfg["keep"]).indices.tolist()) for l in range(NL)]
    # trim the least-mass experts (from early layers) until the exact packed size fits the target
    target = int(cfg["target_ratio"] * FP16_BYTES) - cfg.get("code_bytes", 120000) - 200000; tl = 0
    while size_formula(kept, cfg, tokb, CAR) > target:
        l = tl % NL; tl += 1
        order = sorted(kept[l], key=lambda e: float(mass[l, e])); kept[l] = sorted(set(kept[l]) - {order[0]})
    log("kept per layer", [len(k) for k in kept], "formula bytes", size_formula(kept, cfg, tokb, CAR))
    wr = F4.Writer(f"{outdir}/body.l4p"); eb, eg = cfg["eb"], 128; hashes = {}
    hs = lambda n, t: hashes.__setitem__(n, hashlib.sha256(t.contiguous().view(torch.int16).cpu().numpy().tobytes()).hexdigest())
    wr.blob("files.list", json.dumps(list(tokraw)).encode())
    for f, b in tokraw.items(): wr.blob("file." + f, b)
    wr.blob("kept", json.dumps(kept).encode())
    @torch.no_grad()
    def setw(p, w, name):
        p.data.copy_(w.to(torch.bfloat16)); hs(name, p.data)
    with torch.no_grad():
        for tgt, src, b, nm in ((lm.model.embed_tokens, OE, cfg["emb_b"], "embed"), (lm.lm_head, OH, cfg["head_b"], "head")):
            Cs, Ss, Zs = [], [], []
            for s in range(0, src.shape[0], 20000):
                dq, c, sc, z = rtn_c(src[s:s + 20000].to(dev), b, 128, False); tgt.weight.data[s:s + 20000].copy_(dq.to(torch.bfloat16)); Cs.append(c); Ss.append(sc); Zs.append(z)
            wr.quant(nm, torch.cat(Cs), torch.cat(Ss), torch.cat(Zs), b); hs(nm, tgt.weight.data); del Cs, Ss, Zs
        wr.raw("norm", lm.model.norm.weight.data); hs("norm", lm.model.norm.weight.data)
        for l, lay in enumerate(lm.model.layers):
            ki = kept[l]
            for nm, mod in (("q", "q_proj"), ("k", "k_proj"), ("v", "v_proj"), ("o", "o_proj")):
                w = O[l]["attn"][mod].to(dev); p = getattr(lay.self_attn, mod).weight
                if nm == "o" and l >= NL - CAR: wr.raw(f"L{l}.o.raw", w.to(torch.bfloat16)); setw(p, w, f"L{l}.o")
                else:
                    dq, c, sc, z = rtn_c(w, cfg["attn_b"], 128, False); wr.quant(f"L{l}.{nm}", c, sc, z, cfg["attn_b"]); setw(p, dq, f"L{l}.{nm}")
            for nm, p in (("ln1", lay.input_layernorm.weight), ("ln2", lay.post_attention_layernorm.weight), ("qn", lay.self_attn.q_norm.weight), ("kn", lay.self_attn.k_norm.weight)):
                wr.raw(f"L{l}.{nm}", p.data); hs(f"L{l}.{nm}", p.data)
            g = lay.mlp.gate.g.weight.data; wr.raw(f"L{l}.router", g[ki]); mk = torch.zeros(NE, device=dev); mk[[e for e in range(NE) if e not in set(ki)]] = float("-inf"); lay.mlp.gate.mask = mk
            gz = g.clone(); gz[[e for e in range(NE) if e not in set(ki)]] = 0; lay.mlp.gate.g.weight.data.copy_(gz); hs(f"L{l}.router", gz)
    # ---- GPTQ experts, group of layers at a time (hessians need the current partially quantised model, as in core.gptq_all)
    LG = cfg.get("lgroup", 6); nmin = cfg.get("nmin", 32); fb = 0
    with torch.no_grad():
        for g0 in range(0, NL, LG):
            layers = list(range(g0, min(NL, g0 + LG))); Hs = collect_hessians(rows_[:cfg.get("nh", 1000)], layers, kept); log("hessians", layers[0], layers[-1])
            for l in layers:
                Qi = F4.rot_matrix(HID, 256, 1000 + l, dev); Qd = F4.rot_matrix(FF, 256, 2000 + l, dev); ex = lm.model.layers[l].mlp.experts
                acc = {k: [] for k in ("gc", "gs", "gz", "uc", "us", "uz", "dc", "ds", "dz")}; rot = []
                for e in kept[l]:
                    Hin, Hdn, n = Hs[l][e]; gw = O[l]["gate"][e].to(dev).float(); uw = O[l]["up"][e].to(dev).float(); dw = O[l]["down"][e].to(dev).float()
                    if n < nmin:
                        fb += 1; rot.append(0)
                        (qg, cg, sg, zg), (qu, cu, su, zu), (qd, cd, sd, zd) = rtn_c(gw, eb, eg, True), rtn_c(uw, eb, eg, True), rtn_c(dw, eb, eg, True)
                    else:
                        rot.append(1); Hi_ = Qi.T @ Hin @ Qi; Hd_ = Qd.T @ Hdn @ Qd
                        q, c, sc, z = gptq_c(torch.cat([gw, uw], 0) @ Qi, Hi_, eb, eg, cfg.get("damp", 0.01)); q = q @ Qi.T
                        qg, qu = q[:FF], q[FF:]; cg, cu, sg, su, zg, zu = c[:FF], c[FF:], sc[:FF], sc[FF:], z[:FF], z[FF:]
                        qd, cd, sd, zd = gptq_c(dw @ Qd, Hd_, eb, eg, cfg.get("damp", 0.01)); qd = qd @ Qd.T
                    for k_, v_ in zip(acc, (cg, sg, zg, cu, su, zu, cd, sd, zd)): acc[k_].append(v_)
                    ex[e].gate_proj.weight.data.copy_(qg.to(torch.bfloat16)); ex[e].up_proj.weight.data.copy_(qu.to(torch.bfloat16)); ex[e].down_proj.weight.data.copy_(qd.to(torch.bfloat16))
                    hs(f"L{l}.e{e}.g", ex[e].gate_proj.weight.data); hs(f"L{l}.e{e}.u", ex[e].up_proj.weight.data); hs(f"L{l}.e{e}.d", ex[e].down_proj.weight.data)
                wr.u8(f"L{l}.rot", torch.tensor(rot))
                for nm, (a, b_, c_) in (("gate", ("gc", "gs", "gz")), ("up", ("uc", "us", "uz")), ("down", ("dc", "ds", "dz"))):
                    wr.quant(f"L{l}.{nm}", torch.stack(acc[a]), torch.stack(acc[b_]), torch.stack(acc[c_]), eb)
                Hs[l] = None
            del Hs; torch.cuda.empty_cache()
    log("gptq done, rtn fallbacks", fb); wr.close()
    sz = os.path.getsize(f"{outdir}/body.l4p"); json.dump(hashes, open(f"{outdir}/hashes.json", "w"))
    h = hashlib.sha256(); f = open(f"{outdir}/body.l4p", "rb")
    for b in iter(lambda: f.read(1 << 24), b""): h.update(b)
    info = dict(task=task, cfg=cfg, bytes=sz, ratio=sz / FP16_BYTES, sha256=h.hexdigest(), kept_per_layer=[len(k) for k in kept], rtn_fallback_experts=fb, tok_zlib_bytes=tokb, n_hashes=len(hashes))
    json.dump(info, open(f"{outdir}/pack_info.json", "w")); vol.commit(); print("PACK_RESULT_JSON", json.dumps(info), flush=True); return info
