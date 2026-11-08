"""near-miss lane e (native low-bit starts other than Bonsai). HF greedy probe + byte accounting for packed-ternary releases.
usage: ne_probe.py MODEL TAG [--limit N] [--cands auto|csv "N:bits:g,..."] [--sizes-only]
START MODEL = the released low-bit model itself (fp16 reference = 2 bytes x its total param count).
Lossy levers (train-free, no gradients): vocab-prefix truncation (closed BPE), RTN group quantisation of embedding / lm_head rows. Ternary body is lossless (entropy-coded bytes counted).
Outputs results/TAG__CAND.json (t1/cls rows in probe_box schema) and results/TAG__sizes.json."""
import os, sys, json, re, time, math, lzma, random, argparse
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__)); D = os.path.join(HERE, "..", "data"); OUT = os.path.join(HERE, "..", "results"); os.makedirs(OUT, exist_ok=True)
os.environ.setdefault("HF_HOME", "/workspace/hf")
from huggingface_hub import snapshot_download
sys.path.insert(0, HERE)
ap = argparse.ArgumentParser(); ap.add_argument("model"); ap.add_argument("tag"); ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--cands", default="auto"); ap.add_argument("--sizes-only", action="store_true"); ap.add_argument("--t1max", type=int, default=300); ap.add_argument("--budget", type=float, default=0.099)
ap.add_argument("--fp16-params", type=float, default=0, help="override total params for the fp16 reference")
A = ap.parse_args()
from safetensors import safe_open
src = snapshot_download(A.model, allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt", "*.jinja"])

# ---------------- accounting ----------------
def unpack4(p):
    from transformers.integrations.bitnet import unpack_weights
    return unpack_weights(p, torch.int8)
def ternary_cost_bits(w):
    """w int8 [R,C] in {-1,0,1}: exact ideal-coder bits for best of global/row/col table (16 bits per table entry stored) -- same mode set as v2/lowbit/build_artifact.py"""
    s = (w.to(torch.int64) + 1); R, C = s.shape
    def H(cnt):  # cnt [...,3]
        c = cnt.double(); p = c / c.sum(-1, keepdim=True).clamp_min(1); return -(c * torch.where(p > 0, torch.log2(p.clamp_min(1e-30)), torch.zeros_like(p))).sum()
    oh = lambda k: (s == k)
    g = torch.stack([oh(k).sum() for k in range(3)])[None]; r = torch.stack([oh(k).sum(1) for k in range(3)], 1); c = torch.stack([oh(k).sum(0) for k in range(3)], 1)
    return min(float(H(g)) + 3 * 16, float(H(r)) + R * 2 * 16, float(H(c)) + C * 2 * 16)
def account():
    sizes = {"ternary_params": 0, "ternary_bits_ideal": 0.0, "ternary_bytes_5pb": 0, "small_bytes": 0, "emb": None, "head": None, "other_params": 0}
    fl = [f for f in os.listdir(src) if f.endswith(".safetensors")]
    for fn in fl:
        f = safe_open(os.path.join(src, fn), "pt")
        for k in f.keys():
            t = f.get_tensor(k)
            if k.endswith("embed_tokens.weight"): sizes["emb"] = list(t.shape); continue
            if k.endswith("lm_head.weight"): sizes["head"] = list(t.shape); continue
            if t.dtype == torch.uint8 and t.ndim == 2:
                w = unpack4(t.cuda()); sizes["ternary_params"] += w.numel(); sizes["ternary_bits_ideal"] += ternary_cost_bits(w); sizes["ternary_bytes_5pb"] += (w.numel() + 4) // 5
            else:
                sizes["small_bytes"] += t.numel() * 2; sizes["other_params"] += t.numel()
    sizes["total_params"] = sizes["ternary_params"] + sizes["other_params"] + (sizes["emb"][0] * sizes["emb"][1] if sizes["emb"] else 0) + (sizes["head"][0] * sizes["head"][1] if sizes["head"] else 0)
    sizes["ternary_bytes_ideal"] = sizes["ternary_bits_ideal"] / 8
    return sizes
S = account(); S["fp16_ref_bytes"] = 2 * (A.fp16_params or S["total_params"]); S["model"] = A.model
print("SIZES", json.dumps(S), flush=True)

# ---------------- tokenizer trim ----------------
tok_path = os.path.join(src, "tokenizer.json")
TJ = json.load(open(tok_path)); TC = json.load(open(os.path.join(src, "tokenizer_config.json")))
chat_t = TC.get("chat_template") or ""
if not chat_t and os.path.exists(os.path.join(src, "chat_template.jinja")): chat_t = open(os.path.join(src, "chat_template.jinja")).read()
if isinstance(chat_t, list): chat_t = json.dumps(chat_t)
FULLV = max(max(TJ["model"]["vocab"].values()), max((a["id"] for a in TJ["added_tokens"]), default=0)) + 1
def special_ids():
    names = set(re.findall(r"<[^<>\s]{1,40}>", chat_t)) | {"<tool_call>", "</tool_call>", "<think>", "</think>"}
    for k in ("bos_token", "eos_token", "pad_token", "unk_token"):
        v = TC.get(k); v = v.get("content") if isinstance(v, dict) else v
        if v: names.add(v)
    ids = set()
    for a in TJ["added_tokens"]:
        if a["content"] in names or a.get("special"):
            if a["content"] in names: ids.add(a["id"])
    return sorted(ids)
SPECIAL = special_ids()
if os.environ.get("TASKVOCAB", "1") == "1":   # task-vocab trim: keep every token of the FIXED task prompt templates + label words (not of any eval item)
    _tk = __import__("tokenizers").Tokenizer.from_file(tok_path)
    _tpl = [open(D + "/tox_prompt.txt").read().replace("{t}", ""), open(D + "/inj_prompt.txt").read().replace("{t}", ""), "Alarm, Weather, Music, Calendar, Lists, News, Transport.\n\nRequest:",
            "safe toxic hate sexual violence injection benign Alarm Weather Music Calendar Lists News Transport", "<tool_call>\n{\"name\": \"\", \"arguments\": {}}\n</tool_call>", TC.get("chat_template") if isinstance(TC.get("chat_template"), str) else "",
            "You are a function calling assistant. You may call one or more of these functions to answer the user's request.\n<tools>\n</tools>\n"]
    EXTRA = sorted({i for s in _tpl for i in _tk.encode(s, add_special_tokens=False).ids}); SPECIAL = sorted(set(SPECIAL) | set(EXTRA))
    print("TASKVOCAB extra ids", len(EXTRA), flush=True)
print("SPECIAL_IDS", len(SPECIAL), "FULLV", FULLV, flush=True)
def trim_tok(N):
    """prefix truncation: keep ids < N plus template/special tokens. returns (tokenizers.Tokenizer, old->new map, keep list, shipped xz bytes)"""
    from tokenizers import Tokenizer
    if N >= FULLV: return Tokenizer.from_file(tok_path), None, list(range(FULLV)), len(lzma.compress(open(tok_path, "rb").read(), preset=9 | lzma.PRESET_EXTREME)) + 3000
    tj = json.loads(json.dumps(TJ)); m = tj["model"]
    keep = sorted(set(range(N)) | set(i for i in SPECIAL if i < FULLV)); mp = {o: n for n, o in enumerate(keep)}
    added_ids = {a["id"] for a in tj["added_tokens"]}
    nv = {t: mp[i] for t, i in m["vocab"].items() if i in mp}
    for a in tj["added_tokens"]:
        if a["id"] in mp and a["content"] not in nv: pass
    tj["added_tokens"] = [dict(a, id=mp[a["id"]]) for a in tj["added_tokens"] if a["id"] in mp]
    for a in tj["added_tokens"]:
        if a["content"] in m["vocab"] or True: nv[a["content"]] = a["id"]
    m["vocab"] = nv
    def ok(mg):
        a, b = (mg if isinstance(mg, list) else mg.split(" ")); return a in nv and b in nv and (a + b) in nv
    m["merges"] = [mg for mg in m["merges"] if ok(mg)]
    def remap(o):
        if isinstance(o, dict):
            return {k: ([mp.get(x, x) for x in v] if k == "ids" and isinstance(v, list) else (mp.get(v, v) if k == "id" and isinstance(v, int) else remap(v))) for k, v in o.items()}
        if isinstance(o, list): return [remap(x) for x in o]
        return o
    for kk in ("post_processor", "decoder", "normalizer", "pre_tokenizer"):
        if tj.get(kk): tj[kk] = remap(tj[kk])
    blob = json.dumps(tj, ensure_ascii=False, separators=(",", ":")).encode()
    return Tokenizer.from_str(blob.decode()), mp, keep, len(lzma.compress(blob, preset=9 | lzma.PRESET_EXTREME)) + 3000   # +3000: tokenizer_config/special map/chat template (xz)

# ---------------- embedding quantisation ----------------
def rtn(W, bits, g):
    """W float [R,C] -> (dequant float, payload bytes) ; asym min/max per group of g along C; fp16 scale+min; bit-planes packed exactly."""
    R, C = W.shape; assert C % g == 0
    x = W.float().reshape(R, C // g, g); lo = x.amin(-1, keepdim=True).half().float(); hi = x.amax(-1, keepdim=True)
    sc = ((hi - lo) / (2 ** bits - 1)).half().float().clamp_min(1e-8); q = torch.round((x - lo) / sc).clamp(0, 2 ** bits - 1)
    deq = (q * sc + lo).reshape(R, C)
    nbytes = (R * C * bits + 7) // 8 + R * (C // g) * 4
    return deq, nbytes
def quant_bytes(rows, dim, bits, g): return (rows * dim * bits + 7) // 8 + rows * (dim // g) * 4 if bits < 16 else rows * dim * 2

# ---------------- data / prompts ----------------
def load_items():
    t1 = [json.loads(l) for l in open(D + "/probe_items.jsonl")]
    TOXP = open(D + "/tox_prompt.txt").read(); INJP = open(D + "/inj_prompt.txt").read(); cls = []
    for f in ("s1_dev.jsonl", "s3_dev.jsonl"): cls += [("tox", json.loads(l)) for l in open(D + "/tox_" + f)]
    for f in ("dev_S1.jsonl", "dev_S3.jsonl"): cls += [("inj", json.loads(l)) for l in open(D + "/inj_" + f)]
    random.Random(0).shuffle(cls); cls = cls[:1400]
    if A.limit: t1 = t1[::max(1, len(t1) // A.limit)][:A.limit]; cls = cls[::max(1, len(cls) // A.limit)][:A.limit]
    return t1, cls, TOXP, INJP
from transformers import AutoTokenizer, AutoModelForCausalLM
htok = AutoTokenizer.from_pretrained(src)
TOOLFB = ("You are a function calling assistant. You may call one or more of these functions to answer the user's request.\n<tools>\n{tools}\n</tools>\n"
          "To call a function, reply with a JSON object inside <tool_call></tool_call> tags: <tool_call>\n{{\"name\": <function-name>, \"arguments\": <args-json-object>}}\n</tool_call>\n"
          "If no function applies, answer in plain text without a tool call.")
def render(msgs, tools=None):
    if tools:
        try:
            s = htok.apply_chat_template(msgs, tools=[{"type": "function", "function": t} for t in tools], add_generation_prompt=True, tokenize=False)
            if tools[0]["name"] in s: return s
        except Exception: pass
        sysm = TOOLFB.format(tools="\n".join(json.dumps({"type": "function", "function": t}) for t in tools))
        try: return htok.apply_chat_template([{"role": "system", "content": sysm}] + msgs, add_generation_prompt=True, tokenize=False)
        except Exception: return htok.apply_chat_template([{"role": "user", "content": sysm + "\n\n" + msgs[0]["content"]}], add_generation_prompt=True, tokenize=False)
    return htok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
t1_items, cls_items, TOXP, INJP = load_items()
P1 = [render([{"role": "user", "content": it["query"]}], it["tools"]) for it in t1_items]
P2 = [render([{"role": "user", "content": (TOXP if k == "tox" else INJP).replace("{t}", r["text"])}]) for k, r in cls_items]
SK = [json.loads(l) for l in open(D + "/skill7_dev.jsonl")]
if A.limit: SK = SK[::max(1, len(SK) // A.limit)][:A.limit]
P3 = [render([{"role": "user", "content": r["user"]}]) for r in SK]
print("PROMPT_SAMPLE", repr(P1[0][:300]), "...", repr(P2[0][-200:]), flush=True)
EOS_STR = set()
for k in ("eos_token",):
    v = TC.get(k); v = v.get("content") if isinstance(v, dict) else v
    if v: EOS_STR.add(v)
gc = {}
try: gc = json.load(open(src + "/generation_config.json"))
except Exception: pass
eos_old = gc.get("eos_token_id", htok.eos_token_id); eos_old = eos_old if isinstance(eos_old, list) else [eos_old]
for nm in ("<|eot_id|>", "<|im_end|>", "<|end_of_text|>"):
    i = htok.convert_tokens_to_ids(nm)
    if isinstance(i, int) and i >= 0 and i != htok.unk_token_id: eos_old.append(i)
eos_old = sorted(set(i for i in eos_old if i is not None))

# ---------------- model + generation ----------------
def load_base():
    return AutoModelForCausalLM.from_pretrained(src, torch_dtype=torch.bfloat16, device_map="cuda")
@torch.no_grad()
def gen_batch(model, ids_list, eos_new, pad, max_new, tok_budget=40000, max_bs=64):
    order = sorted(range(len(ids_list)), key=lambda i: -len(ids_list[i])); res = [None] * len(ids_list); i = 0
    while i < len(order):
        L = len(ids_list[order[i]]); bs = max(1, min(max_bs, tok_budget // (L + max_new))); idx = order[i:i + bs]; i += bs
        ml = max(len(ids_list[j]) for j in idx)
        inp = torch.full((len(idx), ml), pad, dtype=torch.long); am = torch.zeros((len(idx), ml), dtype=torch.long)
        for r, j in enumerate(idx): x = ids_list[j]; inp[r, ml - len(x):] = torch.tensor(x); am[r, ml - len(x):] = 1
        out = model.generate(input_ids=inp.cuda(), attention_mask=am.cuda(), max_new_tokens=max_new, do_sample=False, eos_token_id=eos_new, pad_token_id=pad)
        for r, j in enumerate(idx):
            g = out[r, ml:].tolist(); cut = len(g)
            for q, t in enumerate(g):
                if t in eos_new: cut = q; break
            res[j] = g[:cut]
    return res
def run_probe(model, tk, mp, tag):
    enc = (lambda s: tk.encode(s, add_special_tokens=False).ids)
    eos_new = sorted(set((mp[e] if mp else e) for e in eos_old if (mp is None or e in mp)))
    pad = eos_new[0]; t0 = time.time()
    i1 = [enc(p) for p in P1]; i2 = [enc(p) for p in P2]
    o1 = gen_batch(model, i1, eos_new, pad, A.t1max); o2 = gen_batch(model, i2, eos_new, pad, 16, max_bs=96); i3 = [enc(p) for p in P3]; o3 = gen_batch(model, i3, eos_new, pad, 8, max_bs=96)
    dec = lambda g: tk.decode(g, skip_special_tokens=False)
    R = {"model": A.model, "cand": tag, "ntok_t1": sum(map(len, i1)), "ntok_cls": sum(map(len, i2)),
         "t1": [{"id": it["id"], "kind": it["kind"], "source": it["source"], "gold": it["gold"], "out": dec(o)} for it, o in zip(t1_items, o1)],
         "cls": [{"task": k, "id": r["id"], "gold": r["gold"], "kind": r.get("kind"), "source": r.get("source"), "out": dec(o)} for (k, r), o in zip(cls_items, o2)], "skill7": [{"id": r["id"], "gold": r["answer"], "kind": r["kind"], "source": r["src"], "out": dec(o)} for r, o in zip(SK, o3)], "secs": time.time() - t0}
    json.dump(R, open(f"{OUT}/{A.tag}__{tag}.json", "w")); print("PROBE_DONE", A.tag, tag, round(R["secs"]), "ntok", R["ntok_t1"], R["ntok_cls"], flush=True)

def vocab_dims(model):
    emb = model.get_input_embeddings().weight; head = model.get_output_embeddings().weight
    return emb.shape[0], emb.shape[1], (emb.data_ptr() == head.data_ptr())
def plan_cands(V, dim, tied):
    other = S["ternary_bytes_ideal"] + S["small_bytes"] + 8192 + 40000
    leftover0 = A.budget * S["fp16_ref_bytes"] - other
    mult = 1 if tied else 2; C = []
    for bits, g in ((8, 64), (4, 64), (3, 64), (2, 64)):
        N = V
        for _ in range(3):
            tokb = trim_tok(N)[3] if N < V else len(lzma.compress(open(tok_path, "rb").read(), preset=9)) + 3000
            left = leftover0 - tokb - N / 8
            Nf = int(left // (mult * quant_bytes(1, dim, bits, g)))
            N = max(2048, min(V, Nf + 0)) if Nf < V else V
        C.append((N, bits, g, Nf >= 2048))
    # trim-only bf16 embed
    left = leftover0 - 60000; Nf = int(left // (mult * dim * 2))
    if 2048 <= Nf < V: C.insert(0, (Nf, 16, 64, True))
    seen = set(); out = []
    for c in C:
        if c[:3] not in seen: seen.add(c[:3]); out.append(c)
    return out
def build_cand(base, N, bits, g):
    tk, mp, keep, tokb = trim_tok(N)
    emb = base.get_input_embeddings().weight.data; V, dim = emb.shape; head = base.get_output_embeddings().weight.data; tied = base.get_input_embeddings().weight.data_ptr() == base.get_output_embeddings().weight.data_ptr()
    kk = torch.tensor(keep, device=emb.device); res = {"keep": len(keep)}
    def prep(W):
        Wk = W[kk].float()
        if bits < 16: Wk, nb = rtn(Wk, bits, g)
        else: nb = Wk.shape[0] * Wk.shape[1] * 2
        return Wk.to(torch.bfloat16), nb
    We, nbe = prep(emb); Wh, nbh = (We, 0) if tied else prep(head)
    return tk, mp, keep, We, Wh, tied, nbe + nbh + (len(keep) + 7) // 8, tokb
@torch.no_grad()
def apply_cand(model, We, Wh, tied):
    n, d = We.shape; e = torch.nn.Embedding(n, d).to("cuda", torch.bfloat16); e.weight.data.copy_(We); model.set_input_embeddings(e)
    if tied:
        model.get_output_embeddings().weight = e.weight if hasattr(model.get_output_embeddings(), "weight") else None
        if model.get_output_embeddings().weight.shape[0] != n: raise RuntimeError("tie fail")
    else:
        h = torch.nn.Linear(d, n, bias=False).to("cuda", torch.bfloat16); h.weight.data.copy_(Wh); model.set_output_embeddings(h)
    model.config.vocab_size = n
if A.sizes_only:
    json.dump(S, open(f"{OUT}/{A.tag}__sizes.json", "w"), indent=1); sys.exit()
model = load_base(); V, dim, tied = vocab_dims(model); print("EMB", V, dim, "tied", tied, flush=True)
S["vocab"] = V; S["dim"] = dim; S["tied"] = tied
if A.cands == "auto": cands = plan_cands(V, dim, tied)
else: cands = [tuple(int(x) for x in c.split(":")) + (True,) for c in A.cands.split(",")]
S["cands"] = cands; json.dump(S, open(f"{OUT}/{A.tag}__sizes.json", "w"), indent=1); print("CANDS", cands, flush=True)
tk0, _, _, _ = trim_tok(FULLV)
run_probe(model, tk0, None, "base")
orig = (model.get_input_embeddings(), model.get_output_embeddings())
orig_e = model.get_input_embeddings().weight.data.clone(); orig_h = model.get_output_embeddings().weight.data.clone()
summary = []
for (N, bits, g, fit) in cands:
    # restore full-size embedding / head, build candidate, run
    model.set_input_embeddings(torch.nn.Embedding.from_pretrained(orig_e, freeze=True));
    if tied: model.get_output_embeddings().weight = model.get_input_embeddings().weight
    else: model.set_output_embeddings(torch.nn.Linear(dim, V, bias=False).to("cuda", torch.bfloat16)); model.get_output_embeddings().weight.data.copy_(orig_h)
    model.config.vocab_size = V
    tk, mp, keep, We, Wh, tied_, embbytes, tokb = build_cand(model, N, bits, g)
    total = S["ternary_bytes_ideal"] + S["small_bytes"] + embbytes + tokb + 8192
    row = {"N": N, "bits": bits, "g": g, "keep": len(keep), "emb_bytes": embbytes, "tok_bytes": tokb, "total_bytes": total, "pct_of_fp16": 100 * total / S["fp16_ref_bytes"],
           "pct_if_5_per_byte": 100 * (total - S["ternary_bytes_ideal"] + S["ternary_bytes_5pb"]) / S["fp16_ref_bytes"], "fit": fit}
    print("CAND", json.dumps(row), flush=True)
    apply_cand(model, We, Wh, tied_)
    tag = f"N{N}_b{bits}_g{g}"; run_probe(model, tk, mp, tag); summary.append(dict(row, tag=tag)); json.dump(summary, open(f"{OUT}/{A.tag}__cands.json", "w"), indent=1)
    del We, Wh; torch.cuda.empty_cache()
print("RESULT_JSON", json.dumps({"tag": A.tag, "sizes": {k: v for k, v in S.items() if k != "cands"}, "cands": summary}), flush=True)
