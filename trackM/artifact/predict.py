"""Self-contained SST-2 BERT artifact (10% challenge). Needs only python + numpy + torch (no transformers, no base ckpt, no network).
Contract: load(artifact_dir, memory_path|None) -> f(list[str]) -> list[int].
Pieces: model.qpk (bit-packed GPTQ/RTN weights + zlib vocab + fp16 small tensors), WordPiece tokenizer (below), BERT forward (below),
lexical feedback memory (optional file; tiny sparse token->int8 weights added to the logit difference)."""
import json, os, struct, unicodedata, zlib, tempfile
try: tempfile.gettempdir()
except Exception:   # sandboxed run with no writable default tmp dir (torch import needs one): use the writable parent of the artifact copy
    tempfile.tempdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch

MAXLEN = 64
SPECIAL = (0, 100, 101, 102, 103)   # [PAD] [UNK] [CLS] [SEP] [MASK] ids are not memory features

# ---------------- model.qpk reader
def _unpack(buf, b, n):
    bits = np.unpackbits(np.frombuffer(buf, dtype=np.uint8))[:n * b].reshape(n, b)
    return (bits << np.arange(b)).sum(1).astype(np.float32)

def _dq(data, d):
    O, I = d["shape"]; b, g = d["bits"], d["g"]; o1, o2, o3 = d["o"]; gg = I if g <= 0 else g
    q = _unpack(data[o1:o1 + (O * I * b + 7) // 8], b, O * I).reshape(O, I // gg, gg)
    ng = O * (I // gg)
    s = np.frombuffer(data[o2:o2 + 2 * ng], dtype=np.float16).astype(np.float32).reshape(O, I // gg, 1)
    z = np.frombuffer(data[o3:o3 + 2 * ng], dtype=np.float16).astype(np.float32).reshape(O, I // gg, 1)
    return torch.from_numpy(((q - z) * s).reshape(O, I).copy())

def read_model(path):
    raw = open(path, "rb").read(); n = struct.unpack("<I", raw[:4])[0]
    meta = json.loads(raw[4:4 + n]); data = memoryview(raw)[4 + n:]
    V = meta["cfg"]["vocab_size"]; W = {}
    for k, d in meta["q"].items():
        if "word_embeddings" not in k: W[k] = _dq(data, d)
    mask = np.unpackbits(np.frombuffer(data[meta["mask_o"]:meta["mask_o"] + (V + 7) // 8], dtype=np.uint8))[:V].astype(bool)
    E = torch.zeros(V, meta["cfg"]["hidden_size"])
    E[torch.from_numpy(mask)] = _dq(data, meta["q"]["bert.embeddings.word_embeddings.weight#a"])
    E[torch.from_numpy(~mask)] = _dq(data, meta["q"]["bert.embeddings.word_embeddings.weight#b"])
    W["bert.embeddings.word_embeddings.weight"] = E
    for k, d in meta["fp16"].items():
        cnt = int(np.prod(d["shape"]))
        W[k] = torch.from_numpy(np.frombuffer(data[d["o"]:d["o"] + 2 * cnt], dtype=np.float16).astype(np.float32).reshape(d["shape"]).copy())
    vocab = zlib.decompress(bytes(data[meta["vocab_o"]:meta["vocab_o"] + meta["vocab_len"]])).decode().split("\n")
    return meta["cfg"], W, vocab

# ---------------- WordPiece tokenizer (BERT uncased)
def _is_punct(c):
    o = ord(c)
    if 33 <= o <= 47 or 58 <= o <= 64 or 91 <= o <= 96 or 123 <= o <= 126: return True
    return unicodedata.category(c).startswith("P")

def _is_cjk(o):
    return (0x4E00 <= o <= 0x9FFF or 0x3400 <= o <= 0x4DBF or 0x20000 <= o <= 0x2A6DF or 0x2A700 <= o <= 0x2B73F or
            0x2B740 <= o <= 0x2B81F or 0x2B820 <= o <= 0x2CEAF or 0xF900 <= o <= 0xFAFF or 0x2F800 <= o <= 0x2FA1F)

class Tok:
    def __init__(s, vocab):
        s.v = {t: i for i, t in enumerate(vocab)}; s.unk = s.v["[UNK]"]; s.cls = s.v["[CLS]"]; s.sep = s.v["[SEP]"]
    def basic(s, text):
        out = []
        for c in text:
            o = ord(c)
            if o == 0 or o == 0xFFFD or (unicodedata.category(c) in ("Cc", "Cf") and c not in "\t\n\r"): continue
            if c in "\t\n\r" or unicodedata.category(c) == "Zs": out.append(" ")
            elif _is_cjk(o): out.append(" " + c + " ")
            else: out.append(c)
        words = []
        for w in "".join(out).split():
            w = unicodedata.normalize("NFD", w.lower())
            w = "".join(c for c in w if unicodedata.category(c) != "Mn")
            cur = ""
            for c in w:
                if _is_punct(c):
                    if cur: words.append(cur); cur = ""
                    words.append(c)
                else: cur += c
            if cur: words.append(cur)
        return words
    def wp(s, w):
        if len(w) > 100: return [s.unk]
        ids, st = [], 0
        while st < len(w):
            en, f = len(w), None
            while st < en:
                p = w[st:en]; p = ("##" + p) if st else p
                if p in s.v: f = s.v[p]; break
                en -= 1
            if f is None: return [s.unk]
            ids.append(f); st = en
        return ids
    def encode(s, text):
        ids = []
        for w in s.basic(text): ids += s.wp(w)
        return [s.cls] + ids[:MAXLEN - 2] + [s.sep]

# ---------------- BERT forward (inference only)
class Bert:
    def __init__(s, cfg, W):
        s.W, s.H, s.nh, s.L, s.eps = W, cfg["hidden_size"], cfg["num_attention_heads"], cfg["num_hidden_layers"], cfg["layer_norm_eps"]
    def ln(s, x, p): return torch.nn.functional.layer_norm(x, (s.H,), s.W[p + ".weight"], s.W[p + ".bias"], s.eps)
    def lin(s, x, p): return torch.nn.functional.linear(x, s.W[p + ".weight"], s.W[p + ".bias"])
    @torch.no_grad()
    def __call__(s, ids, am):
        W, (B, T) = s.W, ids.shape; dh = s.H // s.nh
        x = W["bert.embeddings.word_embeddings.weight"][ids] + W["bert.embeddings.position_embeddings.weight"][:T][None] \
            + W["bert.embeddings.token_type_embeddings.weight"][0]
        x = s.ln(x, "bert.embeddings.LayerNorm")
        bias = (1.0 - am[:, None, None, :].float()) * -1e9
        for l in range(s.L):
            p = f"bert.encoder.layer.{l}."
            sp = lambda t: t.view(B, T, s.nh, dh).transpose(1, 2)
            q, k, v = (sp(s.lin(x, p + f"attention.self.{n}")) for n in ("query", "key", "value"))
            a = torch.softmax(q @ k.transpose(-1, -2) / dh ** 0.5 + bias, -1) @ v
            x = s.ln(s.lin(a.transpose(1, 2).reshape(B, T, s.H), p + "attention.output.dense") + x, p + "attention.output.LayerNorm")
            h = torch.nn.functional.gelu(s.lin(x, p + "intermediate.dense"))
            x = s.ln(s.lin(h, p + "output.dense") + x, p + "output.LayerNorm")
        pooled = torch.tanh(s.lin(x[:, 0], "bert.pooler.dense"))
        return s.lin(pooled, "classifier")

# ---------------- memory file (sparse lexical residual; format shared with feedback/ingest tooling)
def read_memory(path):
    """-> dict(w={tokid: float}, anchors=[(ids,label,base_diff)])"""
    if not path or not os.path.exists(path) or os.path.getsize(path) == 0: return dict(w={}, anchors=[])
    raw = zlib.decompress(open(path, "rb").read()); nw, na = struct.unpack("<II", raw[:8]); o = 8
    ids = np.frombuffer(raw, "<u2", nw, o); o += 2 * nw; vals = np.frombuffer(raw, "i1", nw, o); o += nw
    lens = np.frombuffer(raw, "u1", na, o); o += na
    lab = np.frombuffer(raw, "u1", na, o); o += na
    bd = np.frombuffer(raw, "<f2", na, o); o += 2 * na
    flat = np.frombuffer(raw, "<u2", int(lens.sum()), o)
    anc, c = [], 0
    for i in range(na):
        anc.append((flat[c:c + int(lens[i])].tolist(), int(lab[i]), float(bd[i]))); c += int(lens[i])
    return dict(w={int(i): float(v) / 8.0 for i, v in zip(ids, vals)}, anchors=anc)

def write_memory(path, w, anchors):
    ks = sorted(w); ids = np.array(ks, "<u2"); vals = np.array([int(round(w[k] * 8)) for k in ks], "i1")
    lens = np.array([len(a[0]) for a in anchors], "u1"); lab = np.array([a[1] for a in anchors], "u1")
    bd = np.array([a[2] for a in anchors], "<f2"); flat = np.array([i for a in anchors for i in a[0]], "<u2")
    raw = struct.pack("<II", len(ks), len(anchors)) + ids.tobytes() + vals.tobytes() + lens.tobytes() + lab.tobytes() + bd.tobytes() + flat.tobytes()
    open(path, "wb").write(zlib.compress(raw, 9))

# ---------------- public API
class Model:
    def __init__(s, d, memory=None, threads=None):
        if threads: torch.set_num_threads(threads)
        cfg, W, vocab = read_model(os.path.join(d, "model.qpk"))
        s.tok, s.net = Tok(vocab), Bert(cfg, W)
        s.mem = read_memory(memory)
    def toks(s, texts):   # memory features: set of non-special wordpiece ids
        return [sorted(set(s.tok.encode(t)) - set(SPECIAL)) for t in texts]
    def logits(s, texts, bs=64):
        enc = [s.tok.encode(t) for t in texts]; order = sorted(range(len(enc)), key=lambda i: len(enc[i])); out = [None] * len(enc)
        for i in range(0, len(order), bs):
            idx = order[i:i + bs]; T = max(len(enc[j]) for j in idx)
            ids = torch.zeros(len(idx), T, dtype=torch.long); am = torch.zeros(len(idx), T, dtype=torch.long)
            for r, j in enumerate(idx): ids[r, :len(enc[j])] = torch.tensor(enc[j]); am[r, :len(enc[j])] = 1
            lg = s.net(ids, am)
            for r, j in enumerate(idx): out[j] = lg[r]
        return torch.stack(out).numpy() if out else np.zeros((0, 2), np.float32)
    def predict_logits(s, texts):
        lg = s.logits(texts)
        if s.mem["w"]:
            w = s.mem["w"]; z = np.array([sum(w.get(i, 0.0) for i in t) / np.sqrt(max(len(t), 1)) for t in s.toks(texts)], np.float32)
            lg = lg + np.stack([-z, z], 1) / 2
        return lg

def load(artifact_dir, memory_path=None):
    m = Model(artifact_dir, memory_path)
    return lambda texts: [int(x) for x in m.predict_logits(list(texts)).argmax(-1)]
