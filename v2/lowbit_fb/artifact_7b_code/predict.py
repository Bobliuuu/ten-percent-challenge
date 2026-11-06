"""MDC-6 artifact loader + feedback-capable calibrated head (PCH; head.npz + memory JSON, see v2/lowbit_fb/RESULTS.md). Base loader by v2/altpush. load(artifact_dir, memory_path=None) -> f(rows) -> predicted label indices (per-task logit bias applied).
rows: {task,text,names,inst}. Decodes ANS streams -> dequantizes (Hadamard-rotated ECSQ) -> rebuilds Qwen2 body in bf16 on GPU."""
import os, io, json, lzma, math, tarfile, tempfile, numpy as np, torch
LET = "ABCDEFGHIJKLMN"
def fmt(r):
    s = r["inst"] + "\n\nText: " + r["text"] + "\n\nOptions:\n"
    for l, c in zip(LET, r["names"]): s += f"{l}. {c}\n"
    return s + "Answer:"
def _build(art, dev="cuda"):
    import constriction
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
    @torch.no_grad()
    def run(rows, bs=16):
        """-> (letter logits [N,14] raw, features [N,2H])"""
        enc = [remap(tok.encode(fmt(r)))[-1024:] for r in rows]
        order = np.argsort([-len(e) for e in enc]); out = np.zeros((len(rows), 14), np.float32); FT = None
        for b in range(0, len(rows), bs):
            ix = order[b:b + bs]; mx = max(len(enc[i]) for i in ix)
            ids = torch.zeros(len(ix), mx, dtype=torch.long); att = torch.zeros(len(ix), mx, dtype=torch.long)
            for j, i in enumerate(ix): ids[j, :len(enc[i])] = torch.tensor(enc[i]); att[j, :len(enc[i])] = 1
            o = model(input_ids=ids.to(dev), attention_mask=att.to(dev), output_hidden_states=True); hs = o.last_hidden_state; HS = o.hidden_states; NL = len(HS) - 1
            a = att.to(dev); last = a.sum(1) - 1; ar = torch.arange(len(ix))
            lg = (hs[ar, last] @ Wl.T).float().cpu().numpy()
            mm = (HS[NL // 2].float() * a.unsqueeze(-1)).sum(1) / a.sum(1, keepdim=True)
            ft = torch.cat([HS[(3 * NL) // 4][ar, last].float(), mm], 1).cpu().numpy()
            if FT is None: FT = np.zeros((len(rows), ft.shape[1]), np.float32)
            for j, i in enumerate(ix): lg[j, len(rows[i]["names"]):] = -1e9; out[i] = lg[j]; FT[i] = ft[j]
        return out, FT
    return dict(run=run, bias=bias)

# ---------------------------------------------------------------------------------------------------------------
# Feedback-capable calibrated head (PCH). All state lives in head.npz (fixed, in the artifact) + memory JSON (outside).
# Features = concat(last-token hidden at layer 3/4 depth, mean-pooled hidden at layer 1/2 depth) of the DECODED model.
# ---------------------------------------------------------------------------------------------------------------
import base64

def _lsm(L):
    L = L - L.max(); return L - np.log(np.exp(L).sum())

class Head:
    def __init__(s, d):
        z = np.load(os.path.join(d, "head.npz"))
        s.meta = json.loads(bytes(z["meta"]).decode()); s.mu0 = z["mu0"].astype(np.float32); s.P = z["P"].astype(np.float32)
        s.Ci = z["Ci"].astype(np.float32); s.keys = s.meta["keys"]; s.cmu = z["cmu"].astype(np.float32); s.cn = z["cn"].astype(np.float64)
        s.idx = {k: i for i, k in enumerate(s.keys)}; s.T = s.meta["tasks"]; s.c0, s.c1 = s.meta.get("cols", [0, s.P.shape[1]])
    def proj(s, f): return (f[s.c0:s.c1] - s.mu0) @ s.P.T
    def score(s, z, mus): return z @ (s.Ci @ mus.T) - 0.5 * np.einsum("cr,rs,cs->c", mus, s.Ci, mus)

def _key(task, name): return task + "\x1f" + name

def _enc(a): return base64.b64encode(np.asarray(a, np.float32).tobytes()).decode()
def _dec(s): return np.frombuffer(base64.b64decode(s), np.float32).copy()

class Mem:
    """memory = {"v":1,"cls":{key:{"n":count,"mu":b64 fp32[r]}},"tp":{task:{name:[true_count,pred_count]}}}  (class sufficient statistics only)"""
    def __init__(s, path=None):
        m = json.load(open(path)) if path else {}
        s.cls = m.get("cls", {}); s.tp = m.get("tp", {})
    def dump(s, path):
        json.dump({"v": 1, "cls": s.cls, "tp": s.tp}, open(path, "w"), separators=(",", ":"))
    def touched(s, task): return any(k.startswith(task + "\x1f") for k in s.cls) or task in s.tp

def _state(H, M, task):
    keys = [k for k in H.keys if k.startswith(task + "\x1f")]
    mus = {k: H.cmu[H.idx[k]] for k in keys}; ns = {k: H.cn[H.idx[k]] for k in keys}
    for k, v in M.cls.items():
        if k.startswith(task + "\x1f"): mus[k] = _dec(v["mu"]); ns[k] = v["n"]
    return mus, ns

def _decide(H, M, row, lg, f):
    """final letter index for an MDC row: base letter logits lg, feature f, head H, memory M."""
    names = row["names"]; K = len(names); task = row.get("task"); T = H.T.get(task) if H else None
    base = _lsm(lg[:K].astype(np.float64))
    if T is None: return int(base.argmax())
    mem_t = M.touched(task); lam = T["lam_fb"] if mem_t else T["lam0"]
    if lam == 0 and not mem_t: return int(base.argmax())
    mus, ns = _state(H, M, task); z = H.proj(f)
    old = {_key(task, n): i for i, n in enumerate(names) if _key(task, n) in H.idx}
    newk = [k for k in mus if k not in H.idx]
    allk = list(old) + newk
    S = H.score(z, np.stack([mus[k] for k in allk])); sc = dict(zip(allk, S))
    s = base.copy()
    for k, i in old.items(): s[i] += lam * sc[k]
    if newk:   # new-label gate: base+head decision unless the class geometry says "new" -> answer with the Other option
        if max(allk, key=lambda k: sc[k]) in newk:
            oth = [i for i, n in enumerate(names) if _key(task, n) not in H.idx]
            if oth: return int(oth[-1])
        return int(base.argmax()) if T.get("gate_new", True) else int(s.argmax())
    b = np.zeros(K)
    for i, n in enumerate(names):
        if task in M.tp and _key(task, n) in H.idx:
            t, p = M.tp[task].get(n, [0, 0]); b[i] = T["eta"] * np.log((t + T["ba"]) / (p + T["ba"]))
    return int((s + b).argmax())

def _base(B, rows):
    lg, FT = B["run"](rows)
    for i, r in enumerate(rows):
        b = B["bias"][r["task"]]; lg[i, :len(b)] += b          # shipped per-task logit bias (part of the base artifact)
    return lg, FT

def ingest(artifact_dir, annotations, memory_in, memory_out, B=None):
    """annotations: list of rows {task,inst,text,names,label}; label = correct class NAME (a name not among the head's classes
    marks a NEW label; `names` must then contain an 'Other' fallback option). Closed form: per-class running mean + count and
    (true,pred) tallies. No gradients, no stored items, no retrieval, no prompt edits. Writes memory_out; returns its bytes."""
    B = B or _build(artifact_dir); H = Head(artifact_dir); M = Mem(memory_in)
    LG, FT = _base(B, annotations)
    for a, lg, f in zip(annotations, LG, FT):
        task = a["task"]; T = H.T[task]; z = H.proj(f); lab = a["label"]; names = a["names"]
        pred = names[_decide(H, M, a, lg, f)]          # shipped system's answer BEFORE this update
        mus, ns = _state(H, M, task); k = _key(task, lab)
        if k not in mus:   # new class: prior = mean of the task's old prototypes with weight n0
            mus[k] = np.mean(list(mus.values()), 0); ns[k] = T["n0"]
        w = T["w"]; mu = (ns[k] * mus[k] + w * z) / (ns[k] + w)
        M.cls[k] = {"n": float(ns[k] + w), "mu": _enc(mu)}
        if _key(task, lab) in H.idx and _key(task, pred) in H.idx:
            tp = M.tp.setdefault(task, {}); tp.setdefault(lab, [0, 0])[0] += 1; tp.setdefault(pred, [0, 0])[1] += 1
    M.dump(memory_out); return os.path.getsize(memory_out)

def load(art, memory=None, use_bias=True, dev="cuda"):
    B = _build(art, dev)
    H = Head(art) if os.path.exists(os.path.join(art, "head.npz")) and os.environ.get("TENPCT_NOHEAD") != "1" else None
    M = Mem(memory) if memory else Mem()
    def f(rows):
        lg, FT = _base(B, rows) if use_bias else B["run"](rows)
        return [_decide(H, M, r, lg[i], FT[i]) for i, r in enumerate(rows)]
    f.logits = lambda rows: B["run"](rows)[0]
    f.base = lambda rows: _base(B, rows)
    f.build = B
    return f
