"""Self-contained predictor for the packed-ternary BitNet-b1.58-2B-4T artifact (see v2/lowbit/ARTIFACT.md).
Contract: load(artifact_dir, memory_path|None) -> f(rows) -> list of letters.
Rows: MMLU {id,subject,question,choices} or MDC-6 {id,task,inst,text,names}. Rebuilds the model ONLY from the files in
artifact_dir (entropy-coded ternary weights, 4..8-bit embeddings, norms, tokenizer, config); no network, no HF hub, no base weights.
Optional memory = JSON {id: letter} overrides (feedback store, counted in size)."""
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

def load(d, memory=None):
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
    mem = json.load(open(memory)) if memory else {}
    def f(rows):
        out = []
        with torch.no_grad():
            for r in rows:
                if r["id"] in mem: out.append(mem[r["id"]]); continue
                mdc = "names" in r; K = len(r["names"]) if mdc else 4
                x = torch.tensor([remap(tok.encode((_fmt_mdc if mdc else _fmt_mmlu)(r)))[-1024:]]).cuda()
                h = model.model(input_ids=x).last_hidden_state[0, -1]
                out.append("ABCDEFGHIJKLMN"[int((h @ letters[:K].T).float().argmax())])
        return out
    return f
