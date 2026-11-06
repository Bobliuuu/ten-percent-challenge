"""MDC-6 artifact loader. load(artifact_dir, memory_path=None) -> f(rows) -> predicted label indices (per-task logit bias applied).
rows: {task,text,names,inst}. Decodes ANS streams -> dequantizes (Hadamard-rotated ECSQ) -> rebuilds Qwen2 body in bf16 on GPU."""
import os, io, json, lzma, math, tarfile, tempfile, numpy as np, torch
LET = "ABCDEFGHIJKLMN"
def fmt(r):
    s = r["inst"] + "\n\nText: " + r["text"] + "\n\nOptions:\n"
    for l, c in zip(LET, r["names"]): s += f"{l}. {c}\n"
    return s + "Answer:"
def load(art, memory=None, use_bias=True, dev="cuda"):
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
    def logits(rows, bs=16):
        enc = [remap(tok.encode(fmt(r)))[-1024:] for r in rows]
        order = np.argsort([-len(e) for e in enc]); out = np.zeros((len(rows), 14), np.float32)
        for b in range(0, len(rows), bs):
            ix = order[b:b + bs]; mx = max(len(enc[i]) for i in ix)
            ids = torch.zeros(len(ix), mx, dtype=torch.long); att = torch.zeros(len(ix), mx, dtype=torch.long)
            for j, i in enumerate(ix): ids[j, :len(enc[i])] = torch.tensor(enc[i]); att[j, :len(enc[i])] = 1
            hs = model(input_ids=ids.to(dev), attention_mask=att.to(dev)).last_hidden_state
            last = att.sum(1).to(dev) - 1
            lg = (hs[torch.arange(len(ix)), last] @ Wl.T).float().cpu().numpy()
            for j, i in enumerate(ix): lg[j, len(rows[i]["names"]):] = -1e9; out[i] = lg[j]
        return out
    def f(rows):
        lg = logits(rows)
        if use_bias:
            for i, r in enumerate(rows):
                b = bias[r["task"]]; lg[i, :len(b)] += b
        return lg.argmax(1).tolist()
    f.logits = logits
    return f
