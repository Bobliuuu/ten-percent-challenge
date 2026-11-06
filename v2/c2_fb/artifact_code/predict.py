"""MDC-6 artifact loader. load(artifact_dir, memory_path=None) -> f(rows) -> predicted label indices (per-task logit bias applied).
rows: {task,text,names,inst}. Decodes ANS streams -> dequantizes (Hadamard-rotated ECSQ) -> rebuilds Qwen2 body in bf16 on GPU."""
import os, io, json, lzma, math, tarfile, tempfile, base64, numpy as np
LET = "ABCDEFGHIJKLMN"
def fmt(r):
    s = r["inst"] + "\n\nText: " + r["text"] + "\n\nOptions:\n"
    for l, c in zip(LET, r["names"]): s += f"{l}. {c}\n"
    return s + "Answer:"
def _build(art, dev="cuda"):
    import torch, constriction
    from transformers import AutoConfig, AutoModel, AutoTokenizer
    M = json.loads(lzma.decompress(open(f"{art}/index.json.xz", "rb").read()))
    td = tempfile.mkdtemp(); tarfile.open(f"{art}/tok.tar.xz").extractall(td); tok = AutoTokenizer.from_pretrained(td)
    d = json.load(open(f"{art}/config.json")); cfg = AutoConfig.for_model(**d)
    model = AutoModel.from_config(cfg, torch_dtype=torch.bfloat16).to(dev).eval()
    sd = model.state_dict()
    # misc
    blob = lzma.decompress(open(f"{art}/misc.bin.xz", "rb").read())
    def getm(k):
        shp, o = M["misc"][k]; n = int(np.prod(shp)); a = np.frombuffer(blob, dtype=np.int16, count=n, offset=o).reshape(shp)
        return torch.from_numpy(a.copy()).view(torch.bfloat16)
    for k in M["misc"]:
        if k == "__Wl": continue
        kk = k[6:] if k.startswith("model.") else k
        if kk in sd: sd[kk].copy_(getm(k).to(dev))
    Wl = getm("__Wl").to(dev)
    # embedding
    V, H = cfg.vocab_size, M["H"]; nk = M["n_keep"]
    keep = torch.from_numpy(np.cumsum(np.array(M["keep"], dtype=np.int64)))
    raw = np.fromfile(f"{art}/emb.bin", dtype=np.uint8)
    pk4 = raw[:nk * H // 2].reshape(nk, H // 2); es = raw[nk * H // 2:].view(np.float16).reshape(nk, H // 64)
    codes = np.zeros((nk, H), np.uint8); codes[:, 0::2] = pk4 & 15; codes[:, 1::2] = pk4 >> 4
    cw = torch.from_numpy(codes).to(dev).float(); sc = torch.from_numpy(es.copy()).to(dev).float().repeat_interleave(64, 1)
    E = ((cw - 7.5) * sc).to(torch.bfloat16)
    sd["embed_tokens.weight"].zero_(); sd["embed_tokens.weight"][keep.to(dev)] = E
    # body
    HB = 128
    Hm = torch.ones(1, 1)
    while Hm.shape[0] < HB: Hm = torch.cat([torch.cat([Hm, Hm], 1), torch.cat([Hm, -Hm], 1)], 0)
    Hb = (Hm / math.sqrt(HB)).to(dev)
    body = np.fromfile(f"{art}/body.bin", dtype=np.uint32); signs = np.fromfile(f"{art}/signs.bin", dtype=np.uint8)
    for e in M["mods"]:
        R, C, g, K = e["R"], e["C"], e["g"], e["K"]
        def dec(cnt, off, ln, n):
            p = np.array(cnt, dtype=np.float64); m = constriction.stream.model.Categorical(p / p.sum(), perfect=False)
            return constriction.stream.stack.AnsCoder(body[off // 4: off // 4 + ln].copy()).decode(m, n)
        wi = torch.from_numpy(dec(e["cw"], e["ow"], e["lw"], R * C).astype(np.uint8)).to(dev).view(R, C)
        si = torch.from_numpy(dec(e["cs"], e["os"], e["ls"], R * (C // g)).astype(np.uint8)).to(dev).view(R, C // g)
        s = torch.exp2((si.float() - 200) / 16).repeat_interleave(g, 1)
        Wr = (wi.float() - K) * s
        sg = torch.from_numpy(np.unpackbits(signs[e["sgoff"]:e["sgoff"] + (C + 7) // 8])[:C].astype(np.float32)).to(dev) * 2 - 1
        W = (Wr.view(R, C // HB, HB) @ Hb).reshape(R, C) * sg
        o, n = e["name"].split(".", 1)
        sd[f"layers.{M['orig_idx'].index(int(o))}.{n}.weight"].copy_(W.to(torch.bfloat16))
    # layer index remap when layers were dropped
    keepset = set(keep.tolist()); kl = sorted(keepset); kstr = None
    vinv = tok.convert_ids_to_tokens(list(range(len(tok))))
    kstr = {vinv[i]: i for i in kl if i < len(vinv)}; maxl = max(len(k) for k in kstr)
    def remap(ids):
        out = []
        for i in ids:
            if i in keepset: out.append(i); continue
            t_ = vinv[i] if i < len(vinv) else None
            if t_ is None: out.append(i); continue
            p = 0
            while p < len(t_):
                for q in range(min(len(t_), p + maxl), p, -1):
                    if t_[p:q] in kstr: out.append(kstr[t_[p:q]]); p = q; break
                else: out.append(i); p += 1
        return out
    bias = {k: np.array(v) for k, v in M["bias"].items()}
    def spans(r):
        pre = r["inst"] + "\n\nText: " + r["text"] + "\n\nOptions:\n"; st = []; p = len(pre)
        for l, c in zip(LET, r["names"]): st.append(p); p += len(f"{l}. {c}\n")
        return st + [p]
    @torch.no_grad()
    def run(rows, bs=16):
        """-> (letter logits [N,14] raw, FQ [N,H] last-token hidden at 3/4 depth, FO [N,5,H] per-option mean-pooled hidden at 2/3 depth)"""
        enc, ostart = [], []
        for r in rows:
            s = fmt(r); eo = tok(s, return_offsets_mapping=True, add_special_tokens=False)
            ids, cs = [], []
            for t_, (a, _) in zip(eo["input_ids"], eo["offset_mapping"]):
                m = remap([t_]); ids += m; cs += [a] * len(m)
            enc.append(ids[-1024:]); ostart.append((np.array(cs[-1024:]), spans(r)))
        order = np.argsort([-len(e) for e in enc]); out = np.zeros((len(rows), 14), np.float32)
        FQ = np.zeros((len(rows), H), np.float16); FO = np.zeros((len(rows), 5, H), np.float16)
        for b in range(0, len(rows), bs):
            ix = order[b:b + bs]; mx = max(len(enc[i]) for i in ix)
            ids = torch.zeros(len(ix), mx, dtype=torch.long); att = torch.zeros(len(ix), mx, dtype=torch.long)
            for j, i in enumerate(ix): ids[j, :len(enc[i])] = torch.tensor(enc[i]); att[j, :len(enc[i])] = 1
            o = model(input_ids=ids.to(dev), attention_mask=att.to(dev), output_hidden_states=True); hs = o.last_hidden_state; HS = o.hidden_states; NL = len(HS) - 1
            a = att.to(dev); last = a.sum(1) - 1; ar = torch.arange(len(ix))
            lg = (hs[ar, last] @ Wl.T).float().cpu().numpy()
            fq = HS[(3 * NL) // 4][ar, last].float().cpu().numpy(); hm = HS[(2 * NL) // 3].float()
            for j, i in enumerate(ix):
                lg[j, len(rows[i]["names"]):] = -1e9; out[i] = lg[j]; FQ[i] = fq[j]
                cs, st = ostart[i]
                for q in range(len(rows[i]["names"])):
                    msk = np.nonzero((cs >= st[q]) & (cs < st[q + 1]))[0]
                    if len(msk): FO[i, q] = hm[j, torch.from_numpy(msk).to(dev)].mean(0).cpu().numpy()
        return out, FQ, FO
    return dict(run=run, bias=bias)

# ---------------------------------------------------------------------------------------------------------------
# Feedback head for C2 (tool use). Shipped in the artifact: head.npz (global, fixed). Outside: memory JSON (closed-form statistics).
# Features of the DECODED model: per-option mean-pooled hidden at 2/3 depth (contextualised by the query, since options follow it).
# No retrieval: annotations are folded into (a) per-tool running prototype means, (b) per-tool (true,pred) tallies, (c) sums of
# none-margin for none-gold / tool-gold items. Nothing per item is kept; the prompt tokens are never touched.
# ---------------------------------------------------------------------------------------------------------------
def _lsm(L):
    L = L - L.max(-1, keepdims=True); return L - np.log(np.exp(L).sum(-1, keepdims=True))
def _enc(a): return base64.b64encode(np.asarray(a, np.float32).tobytes()).decode()
def _dec(s): return np.frombuffer(base64.b64decode(s), np.float32).copy()
NONE = "\x00none"
def tool_name(opt): return opt.split(":")[0].strip()

class Head:
    def __init__(s, d):
        z = np.load(os.path.join(d, "head.npz"))
        s.meta = json.loads(bytes(z["meta"]).decode()); s.mu0 = z["mu0"].astype(np.float32); s.P = z["P"].astype(np.float32)
        s.Ci = z["Ci"].astype(np.float32); s.m = z["m"].astype(np.float32)   # m: [4, r] = mpos, mneg, mposE, mnegE
        s.pr = s.meta["p"]
    def proj(s, FO): return (FO.astype(np.float32) - s.mu0) @ s.P.T          # [N,5,r]
    def gscore(s, U):
        """global option-relevance LDA score [N,5]"""
        g = np.zeros(U.shape[:2], np.float32)
        for lo, hi, a, b in ((0, 4, 0, 1), (4, 5, 2, 3)):
            d = s.Ci @ (s.m[a] - s.m[b]); c = 0.5 * (s.m[a] + s.m[b])
            g[:, lo:hi] = (U[:, lo:hi] - c) @ d
        return g
    def tscore(s, u, mu, neg=None):
        neg = s.m[1] if neg is None else neg
        d = s.Ci @ (mu - neg); return float((u - 0.5 * (mu + neg)) @ d)

class Mem:
    """{"v":1,"proto":{tool:{"n":count,"mu":b64 fp32[r]}},"tp":{tool:[true,pred]}}   (tool "\x00none" = the 'none fits' option)"""
    def __init__(s, path=None):
        m = json.load(open(path)) if path else {}
        s.proto = m.get("proto", {}); s.tp = m.get("tp", {})
    def dump(s, path): json.dump({"v": 1, "proto": s.proto, "tp": s.tp}, open(path, "w"), separators=(",", ":"))
    def empty(s): return not s.proto and not s.tp

def _scores(H, M, lg, U, g, names, use_mem=True):
    """lg [5] raw logits, U [5,r], g [5] global score -> option scores [5] (base log-softmax + gated head + memory terms)"""
    p = H.pr if H else {}
    s = _lsm(lg[:5].astype(np.float64))
    if H: s = s + p["lam0"] * g
    if not (H and use_mem) or M.empty(): return s
    if NONE in M.proto: s[4] += p["lam_n"] * H.tscore(U[4], _dec(M.proto[NONE]["mu"]), H.m[3])
    if NONE in M.tp: a, b = M.tp[NONE]; s[4] += p["eta_n"] * np.log((a + p["ba"]) / (b + p["ba"]))
    for o in range(4):
        t = tool_name(names[o]); e = M.proto.get(t)
        if e: s[o] += p["lam_t"] * H.tscore(U[o], _dec(e["mu"]))
        if t in M.tp:
            a, b = M.tp[t]; s[o] += p["eta"] * np.log((a + p["ba"]) / (b + p["ba"]))
    return s

def decide(H, M, rows, LG, FO):
    U = H.proj(FO) if H else None; G = H.gscore(U) if H else None
    return [int(_scores(H, M, LG[i], U[i] if H else None, G[i] if H else None, r["names"]).argmax()) for i, r in enumerate(rows)]

def ingest_feats(H, M, rows, LG, FO):
    """fold annotated rows (row["answer"] = correct option index, 4 = none fits) into M. In-place."""
    U = H.proj(FO); G = H.gscore(U); p = H.pr
    for i, r in enumerate(rows):
        pred = int(_scores(H, M, LG[i], U[i], G[i], r["names"]).argmax()); y = int(r["answer"])
        t = NONE if y == 4 else tool_name(r["names"][y]); e = M.proto.get(t)
        n, mu = (e["n"], _dec(e["mu"])) if e else (p["n0t"], H.m[2] if y == 4 else H.m[0])
        mu = (n * mu + p["w"] * U[i, y]) / (n + p["w"]); M.proto[t] = {"n": float(n + p["w"]), "mu": _enc(mu)}
        if y != 4: M.tp.setdefault(t, [0, 0])[0] += 1
        M.tp.setdefault(NONE if pred == 4 else tool_name(r["names"][pred]), [0, 0])[1] += 1
        if y == 4: M.tp.setdefault(NONE, [0, 0])[0] += 1
    return M

def ingest(art, annotations, memory_in, memory_out, B=None):
    """annotations: rows {inst,text,names,answer}; answer = index of the correct option (4 = 'None of these tools can do this')."""
    B = B or _build(art); H = Head(art); M = Mem(memory_in)
    LG, _, FO = B["run"](annotations); ingest_feats(H, M, annotations, LG, FO); M.dump(memory_out); return os.path.getsize(memory_out)

def load(art, memory=None, use_bias=False, dev="cuda"):
    """f(rows) -> predicted option indices. rows: {task,text,names,inst}. With memory=None and the head gated OFF this is exactly the base argmax."""
    B = _build(art, dev)
    H = Head(art) if os.path.exists(os.path.join(art, "head.npz")) and os.environ.get("TENPCT_NOHEAD") != "1" else None
    M = Mem(memory) if memory else Mem()
    def f(rows):
        LG, _, FO = B["run"](rows)
        if use_bias:
            for i, r in enumerate(rows): b = B["bias"][r["task"]]; LG[i, :len(b)] += b
        return decide(H, M, rows, LG, FO)
    f.logits = lambda rows: B["run"](rows)[0]; f.build = B
    return f
