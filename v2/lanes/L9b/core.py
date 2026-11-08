"""L9b core: data, chat template, batched greedy generation, scoring, pack + load of the real packed artifact (Qwen3.5-35B-A3B, SKILL7)."""
import json, os, math, collections, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qpack
MID = "Qwen/Qwen3.5-35B-A3B"

def wilson(k, n, z=1.96):
    if n == 0: return 0.0
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); r = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)); return (c - r) / d
def norm(s): return s.strip().lower().split("\n")[0].strip(" .,:;\"'`*") if s.strip() else ""
def rows_of(path): return [json.loads(l) for l in open(path)]

class Runner:
    def __init__(s, model, tok):
        s.model, s.tok = model, tok; s.dev0 = next(model.parameters()).device
    def chat(s, u):
        kws = dict(tokenize=False, add_generation_prompt=True)
        try: return s.tok.apply_chat_template([{"role": "user", "content": u}], enable_thinking=False, **kws)
        except Exception: return s.tok.apply_chat_template([{"role": "user", "content": u}], **kws)
    def encode(s, rows):
        for r in rows:
            if "ids" not in r: r["ids"] = s.tok(s.chat(r["user"]), add_special_tokens=False).input_ids
        return rows
    def batches(s, rows, budget):
        order = sorted(range(len(rows)), key=lambda i: len(rows[i]["ids"])); i = 0
        while i < len(order):
            j = i
            while j < len(order) and (j - i + 1) * len(rows[order[j]]["ids"]) <= budget and (j - i) < 64: j += 1
            j = max(j, i + 1); yield order[i:j]; i = j
    def pad(s, seqs):
        L = max(len(x) for x in seqs); ids = torch.full((len(seqs), L), s.tok.pad_token_id); am = torch.zeros((len(seqs), L), dtype=torch.long)
        for r, x in enumerate(seqs): ids[r, L - len(x):] = torch.tensor(x); am[r, L - len(x):] = 1
        return ids.to(s.dev0), am.to(s.dev0)
    @torch.no_grad()
    def gen(s, rows, max_new=8, budget=30000):
        s.encode(rows); outs = [None] * len(rows)
        for idx in s.batches(rows, budget):
            ids, am = s.pad([rows[k]["ids"] for k in idx]); L = ids.shape[1]
            g = s.model.generate(input_ids=ids, attention_mask=am, max_new_tokens=max_new, do_sample=False, pad_token_id=s.tok.pad_token_id)
            for k, t in zip(idx, s.tok.batch_decode(g[:, L:], skip_special_tokens=True)): outs[k] = t
        return outs

def correct(outs, rows): return [norm(o) == r["answer"].lower() for o, r in zip(outs, rows)]

def score(cor, base_cor, rows):
    """retention = P(correct | base correct); per (src,kind) cells with >=8 base-correct."""
    bc = [i for i, b in enumerate(base_cor) if b]; k = sum(cor[i] for i in bc)
    cells = collections.defaultdict(lambda: [0, 0]); kinds = collections.defaultdict(lambda: [0, 0])
    for i in bc:
        c = cells[rows[i]["src"] + "|" + rows[i]["kind"]]; c[0] += 1; c[1] += int(cor[i]); c = kinds[rows[i]["kind"]]; c[0] += 1; c[1] += int(cor[i])
    wc = [(v[1] / v[0], c, v[0]) for c, v in cells.items() if v[0] >= 8]; wk = [(v[1] / v[0], c, v[0]) for c, v in kinds.items() if v[0] >= 8]
    return dict(n=len(rows), base_acc=sum(base_cor) / len(rows), comp_acc=sum(cor) / len(rows), n_bc=len(bc), ret=k / max(1, len(bc)), ret_lo=wilson(k, len(bc)),
                worst_cell=min(wc) if wc else None, worst_kind=min(wk) if wk else None, cells={c: [v[1], v[0]] for c, v in sorted(cells.items())})

# ------------------------------------------------------------------ routing mass
def install_router_hooks(model):
    """returns (blocks, mass list, STAT). wraps every router.forward to accumulate routing weight per expert (valid tokens only)."""
    blocks = []
    for name, mod in model.named_modules():
        ch = dict(mod.named_children())
        if "experts" in ch and "gate" in ch: blocks.append((name, mod, ch["gate"], ch["experts"]))
    E = blocks[0][3].gate_up_proj.shape[0]; STAT = dict(on=False, valid=None)
    mass = [torch.zeros(E, device=r.weight.device) for _, _, r, _ in blocks]
    def mk(li, r):
        orig = r.forward
        def fwd(*a, **k):
            out = orig(*a, **k)
            if STAT["on"]:
                w, ix = out[1], out[2]; v = STAT["valid"].to(w.device)
                mass[li].index_add_(0, ix.reshape(-1).to(mass[li].device), (w.float() * v[:, None]).reshape(-1).to(mass[li].device))
            return out
        r.forward = fwd
    for li, (_, _, r, _) in enumerate(blocks): mk(li, r)
    return blocks, mass, STAT

@torch.no_grad()
def mass_chunks(R, rows, blocks, mass, STAT, nchunk=12):
    R.encode(rows); out = []
    for c in range(nchunk):
        sub = rows[c::nchunk]
        for m in mass: m.zero_()
        STAT["on"] = True
        for idx in R.batches(sub, 20000):
            ids, am = R.pad([sub[k]["ids"] for k in idx]); STAT["valid"] = am.flatten().float().cpu()
            R.model(input_ids=ids, attention_mask=am, use_cache=False, logits_to_keep=1)
        STAT["on"] = False; out.append(torch.stack([m.float().cpu() for m in mass]))
    return torch.stack(out)   # [nchunk, L, E]

# ------------------------------------------------------------------ pack
def pack_model(model, tok, kept, outdir, eb=3, attn_b=6, emb_b=4, head_b=4, g=128, extra=None):
    """kept: LongTensor [L, k] sorted expert ids per MoE layer. Writes manifest.json + body.bin + tokenizer/config. Returns total bytes."""
    import re
    W = qpack.Writer(outdir); names = {}
    for n, p in model.named_parameters():
        d = p.data
        m = re.match(r"model\.layers\.(\d+)\.mlp\.(experts\.(gate_up_proj|down_proj)|gate\.weight)$", n)
        if m:
            li = int(m.group(1)); ix = kept[li].to(d.device)
            if m.group(2) == "gate.weight": W.tensor(n, d.index_select(0, ix), 0, 0)
            else: W.tensor(n, d.index_select(0, ix), eb, g, chunk_rows=2048)
        elif "embed_tokens" in n: W.tensor(n, d, emb_b, g)
        elif n.startswith("lm_head"): W.tensor(n, d, head_b, g)
        elif d.dim() == 2 and d.numel() >= 65536 and d.shape[-1] % g == 0: W.tensor(n, d, attn_b, g)
        else: W.tensor(n, d, 0, 0)
    cfg = model.config.to_dict(); cfg["num_experts"] = int(kept.shape[1])
    tkd = {}
    W.close(dict(model_cls=type(model).__name__, config_cls=type(model.config).__name__, config=cfg, kept=kept.tolist(), cfg=dict(eb=eb, attn_b=attn_b, emb_b=emb_b, head_b=head_b, g=g), extra=extra or {}))
    tok.save_pretrained(outdir)
    for f in ("core.py", "qpack.py", "answer.py", "feedback.py"):
        src = os.path.join(os.path.dirname(os.path.abspath(__file__)), f)
        if os.path.exists(src): open(os.path.join(outdir, f), "w").write(open(src).read())
    return sum(os.path.getsize(os.path.join(dp, f)) for dp, _, fs in os.walk(outdir) for f in fs)

def load_packed(adir, dev="cuda:0"):
    import transformers
    from transformers import AutoTokenizer
    m = json.load(open(os.path.join(adir, "manifest.json"))); body = np.memmap(os.path.join(adir, "body.bin"), dtype=np.uint8, mode="r")
    config = getattr(transformers, m["config_cls"]).from_dict(m["config"])
    old = torch.get_default_dtype(); torch.set_default_dtype(torch.bfloat16)
    with torch.device(dev): model = getattr(transformers, m["model_cls"])(config)
    torch.set_default_dtype(old); model = model.to(torch.bfloat16).eval()
    P = dict(model.named_parameters()); missing = [n for n in P if n not in m["tensors"]]; assert not missing, missing[:5]
    with torch.no_grad():
        for n, e in m["tensors"].items(): P[n].data.copy_(qpack.read_tensor(body, e, dev))
    for p in model.parameters(): p.requires_grad_(False)
    tok = AutoTokenizer.from_pretrained(adir); tok.padding_side = "left"
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    return model, tok, m
