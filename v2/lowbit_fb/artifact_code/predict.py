"""Self-contained predictor for the packed-ternary BitNet-b1.58-2B-4T artifact (see v2/lowbit/ARTIFACT.md).
Contract: load(artifact_dir, memory_path|None) -> f(rows) -> list of letters.
Rows: MMLU {id,subject,question,choices} or MDC-6 {id,task,inst,text,names}. Rebuilds the model ONLY from the files in
artifact_dir (entropy-coded ternary weights, 4..8-bit embeddings, norms, tokenizer, config); no network, no HF hub, no base weights.
Optional memory = tiny JSON of class statistics (see feedback section): running class means + counts + label-bias tallies; NO items, NO answers.
Calibrated head (head.npz, counted in size): frozen features of the decoded model -> closed-form LDA scores, added to the letter logits."""
import json, os, lzma, struct, tempfile, numpy as np

def _probs_from_q(q):
    q = q.astype(np.int64); q2 = 65536 - q[..., 0] - q[..., 1]
    return (np.stack([q[..., 0], q[..., 1], q2], -1) / 65536.0).astype(np.float32)

def _decode_matrix(mode, q, words, R, C):
    import constriction
    pr = _probs_from_q(q)
    per = np.broadcast_to(pr[0], (R * C, 3)) if mode == "g" else (np.repeat(pr, C, axis=0) if mode == "r" else np.tile(pr, (R, 1)))
    s = constriction.stream.stack.AnsCoder(words).decode(constriction.stream.model.Categorical(perfect=False), np.ascontiguousarray(per))
    return s.reshape(R, C).astype(np.int8) - 1

def _pack_hf(t):
    R, C = t.shape; pr = R // 4; u = (t + 1).astype(np.uint8); p = np.zeros((pr, C), np.uint8)
    for i in range(4): p |= (u[i * pr:(i + 1) * pr] << (2 * i))
    return p

def _fmt_mmlu(r):
    return ("The following are multiple choice questions (with answers) about " + r["subject"].replace("_", " ") + ".\n\n" + r["question"] + "\n"
            + "".join(f"{l}. {c}\n" for l, c in zip("ABCD", r["choices"])) + "Answer:")

def _fmt_mdc(r):
    return r["inst"] + "\n\nText: " + r["text"] + "\n\nOptions:\n" + "".join(f"{l}. {c}\n" for l, c in zip("ABCDEFGHIJKLMN", r["names"])) + "Answer:"

def _build(d):
    import torch
    from safetensors.torch import save_file
    from transformers import AutoTokenizer, AutoModelForCausalLM
    rd = lambda n: open(os.path.join(d, n), "rb").read()
    mj = json.loads(lzma.decompress(rd("meta.xz")))
    tb = np.frombuffer(rd("ternary.bin"), dtype=np.uint8); sd = {}
    for m in mj["meta"]:
        q = np.frombuffer(tb[m["qoff"]:m["woff"]].tobytes(), dtype=np.uint16).reshape(-1, 2)
        w = np.frombuffer(tb[m["woff"]:m["woff"] + 4 * m["nw"]].tobytes(), dtype=np.uint32)
        sd[m["k"]] = torch.from_numpy(_pack_hf(_decode_matrix(m["mode"], q, w, m["R"], m["C"])))
    sb = rd("small.bin")
    for s in mj["smeta"]:
        dt = getattr(torch, s["dtype"].split(".")[-1]); n = int(np.prod(s["shape"])); isz = torch.tensor([], dtype=dt).element_size()
        sd[s["k"]] = torch.frombuffer(bytearray(sb[s["off"]:s["off"] + n * isz]), dtype=dt).reshape(s["shape"]).clone()
    eb, G, nk, H = mj["eb"], mj["G"], mj["nkeep"], mj["H"]
    bits = np.unpackbits(np.frombuffer(rd("emb.bin"), dtype=np.uint8))[:nk * H * eb].reshape(-1, eb)
    qv = np.zeros(bits.shape[0], np.uint8)
    for j in range(eb): qv = (qv << 1) | bits[:, j]
    qv = torch.from_numpy(qv.reshape(nk, H // G, G)).float()
    sc = torch.from_numpy(np.frombuffer(rd("emb_scale.bin"), dtype=np.float16).reshape(nk, H // G, 1).copy()).float()
    mn = torch.from_numpy(np.frombuffer(rd("emb_min.bin"), dtype=np.float16).reshape(nk, H // G, 1).copy()).float()
    sd[mj["emb_key"]] = (qv * sc + mn).reshape(nk, H).to(torch.bfloat16)
    if mj["head_key"]: sd[mj["head_key"]] = torch.zeros(nk, H, dtype=torch.bfloat16)
    keep = np.frombuffer(rd("keep_ids.bin"), dtype=np.uint32 if mj["keep_dtype"] == "u32" else np.uint16).tolist()
    letter_ids = mj["letter_ids"]
    letters = torch.from_numpy(np.frombuffer(rd("head_letters.bin"), dtype=np.int16).reshape(len(letter_ids), H).copy()).view(torch.bfloat16).cuda()
    D = tempfile.mkdtemp(prefix="art_")
    raw = lzma.decompress(rd("tokenizer.xz")); p = 0
    while p < len(raw):
        n = struct.unpack("<I", raw[p:p + 4])[0]; p += 4; nm = raw[p:p + n].decode(); p += n
        n = struct.unpack("<I", raw[p:p + 4])[0]; p += 4; open(f"{D}/{nm}", "wb").write(raw[p:p + n]); p += n
    cj = json.loads(lzma.decompress(rd("config.xz"))); cj["vocab_size"] = nk; json.dump(cj, open(f"{D}/config.json", "w"))
    save_file(sd, f"{D}/model.safetensors", metadata={"format": "pt"})
    tok = AutoTokenizer.from_pretrained(D, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(D, torch_dtype=torch.bfloat16, local_files_only=True).cuda().eval()
    Vfull = mj["Vfull"]; vocab_inv = tok.convert_ids_to_tokens(list(range(Vfull)))
    keepmask = np.zeros(Vfull, bool); keepmask[keep] = True
    old2new = {o: n for n, o in enumerate(keep)}; keepstr = {vocab_inv[i]: i for i in keep}; maxl = max(len(k) for k in keepstr)
    unk = old2new.get(tok.unk_token_id, old2new[keep[0]])
    def remap(ids):
        o = []
        for i in ids:
            if keepmask[i]: o.append(old2new[i]); continue
            t_ = vocab_inv[i]; p = 0
            while p < len(t_):
                for q in range(min(len(t_), p + maxl), p, -1):
                    if t_[p:q] in keepstr: o.append(old2new[keepstr[t_[p:q]]]); p = q; break
                else: o.append(unk); p += 1
        return o
    return dict(model=model, remap=remap, letters=letters, tok=tok, torch=torch)

# ---------------------------------------------------------------------------------------------------------------
# Feedback-capable calibrated head (PCH). All state lives in head.npz (fixed, in the artifact) + memory JSON (outside).
# Features = concat(last-token hidden at layer 3/4 depth, mean-pooled hidden at layer 1/2 depth) of the DECODED model.
# ---------------------------------------------------------------------------------------------------------------
import base64

def _lsm(L):
    L = L - L.max(); return L - np.log(np.exp(L).sum())

def _fwd(B, r):
    """-> (letter logits float32 [K], feature float32 [d])"""
    torch = B["torch"]; model = B["model"]; mdc = "names" in r; K = len(r["names"]) if mdc else 4
    x = torch.tensor([B["remap"](B["tok"].encode((_fmt_mdc if mdc else _fmt_mmlu)(r)))[-1024:]]).cuda()
    o = model.model(input_ids=x, output_hidden_states=True); hs = o.hidden_states; NL = len(hs) - 1
    h = o.last_hidden_state[0, -1]
    lg = (h @ B["letters"][:K].T).float().cpu().numpy()
    f = torch.cat([hs[(3 * NL) // 4][0, -1].float(), hs[NL // 2][0].float().mean(0)]).cpu().numpy()
    return lg, f

class Head:
    def __init__(s, d):
        z = np.load(os.path.join(d, "head.npz"))
        s.meta = json.loads(bytes(z["meta"]).decode()); s.mu0 = z["mu0"].astype(np.float32); s.P = z["P"].astype(np.float32)
        s.Ci = z["Ci"].astype(np.float32); s.keys = s.meta["keys"]; s.cmu = z["cmu"].astype(np.float32); s.cn = z["cn"].astype(np.float64)
        s.idx = {k: i for i, k in enumerate(s.keys)}; s.T = s.meta["tasks"]
    def proj(s, f): return (f - s.mu0) @ s.P.T
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

def ingest(artifact_dir, annotations, memory_in, memory_out, B=None):
    """annotations: list of rows {task,inst,text,names,label}; label = correct class NAME (a name not among the head's classes
    marks a NEW label; `names` must then contain an 'Other' fallback option). Closed form: per-class running mean + count and
    (true,pred) tallies. No gradients, no stored items, no retrieval, no prompt edits. Writes memory_out; returns its bytes."""
    B = B or _build(artifact_dir); H = Head(artifact_dir); M = Mem(memory_in)
    for a in annotations:
        lg, f = _fwd(B, a); task = a["task"]; T = H.T[task]; z = H.proj(f); lab = a["label"]; names = a["names"]
        pred = names[_decide(H, M, a, lg, f)]          # shipped system's answer BEFORE this update
        mus, ns = _state(H, M, task); k = _key(task, lab)
        if k not in mus:   # new class: prior = mean of the task's old prototypes with weight n0
            mus[k] = np.mean(list(mus.values()), 0); ns[k] = T["n0"]
        w = T["w"]; mu = (ns[k] * mus[k] + w * z) / (ns[k] + w)
        M.cls[k] = {"n": float(ns[k] + w), "mu": _enc(mu)}
        if _key(task, lab) in H.idx and _key(task, pred) in H.idx:
            tp = M.tp.setdefault(task, {}); tp.setdefault(lab, [0, 0])[0] += 1; tp.setdefault(pred, [0, 0])[1] += 1
    M.dump(memory_out); return os.path.getsize(memory_out)

def load(d, memory=None):
    B = _build(d); torch = B["torch"]
    H = Head(d) if os.path.exists(os.path.join(d, "head.npz")) and os.environ.get("TENPCT_NOHEAD") != "1" else None
    M = Mem(memory) if memory else Mem()
    def f(rows):
        out = []
        with torch.no_grad():
            for r in rows:
                lg, ft = _fwd(B, r)
                out.append("ABCDEFGHIJKLMN"[_decide(H, M, r, lg, ft)] if "names" in r else "ABCD"[int(lg[:4].argmax())])
        return out
    return f
