#!/usr/bin/env python3
"""run_arms: L13 byte-trick ablation on a dense HF model. For each arm: fake-quantise the linear layers (+ optional emb/head/vocab trim), real greedy generation on the task dev items,
retention of base-correct items with Wilson bounds, exact bit accounting. Results appended to results/<out>.jsonl (one JSON per arm) -> push after every run.
python run_arms.py --model Qwen/Qwen2.5-14B-Instruct --arms arms_scale.json --out q14_scale [--device cuda] [--nsamp 64]"""
import argparse, json, os, sys, time, math, gc, collections
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from btcore import *

HERE = os.path.dirname(os.path.abspath(__file__))
ITEMS = os.path.join(HERE, "..", "task_search", "EASY", "items_dev.jsonl")
CALIB = os.path.join(HERE, "..", "task_search", "EASY", "calib.jsonl")
if not os.path.exists(ITEMS): ITEMS = os.path.join(HERE, "items_dev.jsonl"); CALIB = os.path.join(HERE, "calib.jsonl")
LIN = [("self_attn", "q_proj"), ("self_attn", "k_proj"), ("self_attn", "v_proj"), ("self_attn", "o_proj"), ("mlp", "gate_proj"), ("mlp", "up_proj"), ("mlp", "down_proj")]
GROUP_OF = {"q_proj": "qkv", "k_proj": "qkv", "v_proj": "qkv", "o_proj": "o", "gate_proj": "gu", "up_proj": "gu", "down_proj": "down"}
GID = {"qkv": 1, "o": 2, "gu": 3, "down": 4}
def log(*a): print(time.strftime("%H:%M:%S"), *a, flush=True)

def mod(layer, p, n): return getattr(getattr(layer, p), n)

class Catch(torch.nn.Module):
    def __init__(s, m): super().__init__(); s.m = m; s.args = None; s.kw = None
    def forward(s, *a, **kw): s.args = a; s.kw = kw; raise ValueError("caught")

R3 = os.path.join(HERE, "..", "rival3", "data")
def load_task(task, limit=0):
    """returns items, labels, calib path. SKILL7 = EASY-A (task_search/EASY dev items); BANK8 = EASY-B (rival3 bank+clin SEL sets, first 400 each by stable hash; never the exam)."""
    import hashlib
    if task == "BANK8":
        items = []
        for src in ("bank", "clin"):
            rows = [json.loads(l) for l in open(os.path.join(R3, src, "sel.jsonl"))]
            rows.sort(key=lambda r: hashlib.md5(r["id"].encode()).hexdigest())
            for r in rows[:400]: r["answer"] = r["kind"]; r["task"] = "BANK8"; items.append(r)
        labels = sorted({i["answer"] for i in items}); return (items[:limit] if limit else items), labels, os.path.join(R3, "bank", "cal.jsonl")
    items = load_items(ITEMS, task); labels = json.load(open(os.path.join(HERE, "..", "task_search", "EASY", "PLAN.json")))[task]["labels"]
    return (items[:limit] if limit else items), labels, CALIB

def build_calib(tok, n_seq, T, calib_path=None):
    texts = []
    for l in open(calib_path or CALIB):
        r = json.loads(l); r.setdefault("answer", r.get("kind", "")); texts.append(tok.apply_chat_template([{"role": "user", "content": r["user"]}, {"role": "assistant", "content": r.get("answer", r.get("kind", ""))}], tokenize=False))
    ids = []
    for t in texts: ids += tok(t, add_special_tokens=False)["input_ids"]
    need = n_seq * T
    while len(ids) < need: ids = ids + ids
    return torch.tensor(ids[:need]).reshape(n_seq, T)

@torch.no_grad()
def capture(model, calib, dev, bs):
    layers = model.model.layers; c = Catch(layers[0]); layers[0] = c; xs = []; kw = None
    for b in range(0, calib.shape[0], bs):
        try: model(calib[b:b + bs].to(dev), use_cache=False)
        except ValueError: pass
        xs.append(c.args[0].clone() if c.args else c.kw["hidden_states"].clone()); kw = c.kw
    layers[0] = c.m; kw = {k: v for k, v in kw.items() if k != "hidden_states"}; return xs, kw

def run_layer(layer, x, kw):
    o = layer(x, **kw); return o[0] if isinstance(o, tuple) else o

@torch.no_grad()
def collect_H(layer, xs, kw, seeds, dev):
    """H in the rotated basis for the 4 input groups. returns dict gid->H (fp32)"""
    Hs = {}; hooks = []; cnt = {"n": 0}
    def mk(name, mod_):
        def hook(m, inp):
            x = inp[0].reshape(-1, inp[0].shape[-1]).float(); Q = blockQ(x.shape[1], seeds[name], dev); xr = x @ Q.T
            Hs[name] = Hs.get(name, 0) + xr.T @ xr
            if name == "qkv": cnt["n"] += x.shape[0]
        return hook
    hooks = [mod(layer, "self_attn", "q_proj").register_forward_pre_hook(mk("qkv", None)), mod(layer, "self_attn", "o_proj").register_forward_pre_hook(mk("o", None)),
             mod(layer, "mlp", "gate_proj").register_forward_pre_hook(mk("gu", None)), mod(layer, "mlp", "down_proj").register_forward_pre_hook(mk("down", None))]
    outs = [run_layer(layer, x, kw) for x in xs]
    for h in hooks: h.remove()
    return {k: v * (2.0 / cnt["n"]) for k, v in Hs.items()}, outs

def level_for(cfg, l, name):
    lv = cfg.get("levels")
    if lv and f"{l}.{name}" in lv: return lv[f"{l}.{name}"]
    return cfg["L"]

def mkq(cfg, L): return Q1(L, cfg.get("g", 128), cfg.get("share", 1), cfg.get("coder", "fp16"), cfg.get("asym", False), cfg.get("grid", "uniform"), theta=cfg.get("theta", 0.0))

@torch.no_grad()
def quantize_linears(model, orig, cfg, calib, dev, bs=8, sens=None):
    """sequential GPTQ/RTN over all layers. orig: list of dict name->cpu weight. returns accounting dict. sens: if set, ONLY compute sensitivity tables (no weights changed)."""
    layers = model.model.layers; nl = len(layers); method = cfg.get("method", "gptq")
    need_H = method == "gptq" or sens is not None
    xs, kw = capture(model, calib, dev, bs) if need_H else (None, None)
    acct = collections.defaultdict(float); acct["scale_by"] = collections.defaultdict(float); per = {}
    t0 = time.time()
    for l in range(nl):
        layer = layers[l]; seeds = {g: 1000 * l + GID[g] for g in GID}
        if need_H: Hs, outs = collect_H(layer, xs, kw, seeds, dev)
        for p, n in LIN:
            m = mod(layer, p, n); gname = GROUP_OF[n]; W = orig[l][f"{p}.{n}"].to(dev).float(); C = W.shape[1]
            Q = blockQ(C, seeds[gname], dev); Wr = W @ Q.T
            if sens is not None:
                Hr = Hs[gname]; row = {}
                for L in sens["Ls"]:
                    q1 = mkq(cfg, L); Wq = rtn(Wr, q1); D = Wq - Wr; tr = float(((D @ Hr) * D).sum()); b = q1.bits()
                    row[L] = dict(trace=tr, idx=b["idx"], scale=b["scale"], n=W.numel())
                sens["tab"][f"{l}.{p}.{n}"] = row; continue
            L = level_for(cfg, l, f"{p}.{n}"); q1 = mkq(cfg, L)
            Wq = gptq(Wr, Hs[gname], q1, damp=cfg.get("damp", 0.01)) if method == "gptq" else rtn(Wr, q1)
            m.weight.data = (Wq @ Q).to(m.weight.dtype)
            b = q1.bits(); acct["n"] += W.numel(); acct["idx"] += b["idx"]; acct["nscale"] += b["nscale"]; acct["zp"] += b.get("zp", 0.0)
            for k, v in b["scale"].items(): acct["scale_by"][k] += v
            del Wq, Wr, W
        if need_H:
            if sens is None or True: xs = outs if sens is not None else [run_layer(layer, x, kw) for x in xs]
            del Hs
        if l % 8 == 0: log("layer", l, "of", nl, round(time.time() - t0), "s")
        torch.cuda.empty_cache() if dev != "cpu" else None
    return acct

def total_pct(acct, nparams_total, other_bits, coding):
    """total bytes of the quantised artifact as % of fp16 base: linear idx + scales(coding) + zp + `other_bits` (everything not in the linears: emb, head, norms, biases, tok, code)."""
    sb = acct["scale_by"].get(coding, acct["scale_by"]["fp16"])
    bits = acct["idx"] + sb + acct["zp"] + other_bits
    return 100 * bits / (16 * nparams_total)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True); ap.add_argument("--arms", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--task", default="SKILL7"); ap.add_argument("--device", default="cuda"); ap.add_argument("--nsamp", type=int, default=64); ap.add_argument("--T", type=int, default=512)
    ap.add_argument("--limit", type=int, default=0); ap.add_argument("--bs", type=int, default=8); ap.add_argument("--gbs", type=int, default=48); ap.add_argument("--only", default=""); ap.add_argument("--max_new", type=int, default=12)
    a = ap.parse_args(); dev = a.device
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16 if dev != "cpu" else torch.float32, attn_implementation="sdpa").to(dev).eval()
    items, labels, calib_path = load_task(a.task, a.limit)
    nparams = sum(p.numel() for p in model.parameters())
    emb_n = model.get_input_embeddings().weight.numel(); tied = model.config.tie_word_embeddings
    lin_n = sum(mod(model.model.layers[0], p, n).weight.numel() for p, n in LIN) * len(model.model.layers)
    other_n = nparams - lin_n - emb_n - (0 if tied else model.get_output_embeddings().weight.numel())
    log("params", nparams, "linear", lin_n, "emb", emb_n, "tied", tied, "other", other_n)
    orig = [{f"{p}.{n}": mod(model.model.layers[l], p, n).weight.data.detach().cpu().clone() for p, n in LIN} for l in range(len(model.model.layers))]
    emb0 = model.get_input_embeddings().weight.data.cpu().clone(); head0 = None if tied else model.get_output_embeddings().weight.data.cpu().clone()
    calib = build_calib(tok, a.nsamp, a.T, calib_path); outp = os.path.join(HERE, "results", a.out + ".jsonl"); os.makedirs(os.path.dirname(outp), exist_ok=True)
    done = set()
    if os.path.exists(outp): done = {json.loads(l)["arm"] for l in open(outp)}
    arms = json.load(open(a.arms)); arms = [x for x in arms if not a.only or x["arm"] in a.only.split(",")]
    base = None; bpath = os.path.join(HERE, "results", a.out + "_base_preds.json")
    if os.path.exists(bpath): base = json.load(open(bpath))
    else:
        base = generate_labels(model, tok, items, labels, bs=a.gbs, max_new=a.max_new); json.dump(base, open(bpath, "w"))
    bs0 = score(items, base); base_correct = bs0["correct"]; log("BASE acc", bs0["acc"], "n", len(items))
    def restore():
        for l in range(len(model.model.layers)):
            for p, n in LIN: mod(model.model.layers[l], p, n).weight.data = orig[l][f"{p}.{n}"].to(dev)
        model.get_input_embeddings().weight.data = emb0.to(dev)
        if not tied: model.get_output_embeddings().weight.data = head0.to(dev)
    for arm in arms:
        if arm["arm"] in done: continue
        cfg = arm; t0 = time.time(); restore(); res = dict(arm=arm["arm"], cfg=cfg, model=a.model, task=a.task, base_acc=bs0["acc"])
        res["nparams"] = nparams; res["lin_n"] = lin_n
        try:
            other_bits = 16.0 * other_n; mask = None; retok = None
            if cfg.get("alloc"):   # bit allocation from a sensitivity table (computed once, cached)
                sp = os.path.join(HERE, "results", a.out + "_sens.json")
                if os.path.exists(sp): sens = json.load(open(sp))
                else:
                    sens = dict(Ls=cfg.get("Ls", [3, 4, 5, 6, 8, 12]), tab={}); quantize_linears(model, orig, dict(cfg, method="rtn"), calib, dev, a.bs, sens=sens); json.dump(sens, open(sp, "w")); restore()
                    sens["kl"] = kl_layer_sens(model, orig, dev, calib, cfg); json.dump(sens, open(sp, "w")); restore(); sens = json.load(open(sp))
                cfg["levels"] = allocate(sens, cfg)
                res["levels_hist"] = dict(collections.Counter(cfg["levels"].values()))
            if cfg.get("L"):
                acct = quantize_linears(model, orig, cfg, calib, dev, a.bs); res["acct"] = {k: (dict(v) if isinstance(v, dict) else v) for k, v in acct.items()}
                res["lin_idx_bpw"] = acct["idx"] / acct["n"]; res["lin_scale_bpw"] = {k: v / acct["n"] for k, v in acct["scale_by"].items()}
            else: acct = None
            # embedding / head
            eb = 16.0 * emb_n; hb = 0.0 if tied else 16.0 * model.get_output_embeddings().weight.numel()
            if cfg.get("emb"):
                e = cfg["emb"]; eb = quant_matrix(model.get_input_embeddings(), emb0, e, dev, 77777)
                if tied: model.get_output_embeddings().weight = model.get_input_embeddings().weight
            if cfg.get("head") and not tied: hb = quant_matrix(model.get_output_embeddings(), head0, cfg["head"], dev, 88888)
            if cfg.get("vocab_keep"):
                mask, retok, vinfo = vocab_trim(tok, model, cfg["vocab_keep"], items); res["vocab"] = vinfo
                kept = int(mask.sum()); V = emb_n // model.config.hidden_size
                eb = eb * kept / V; hb = hb * kept / V if not tied else 0.0
            res["emb_bits"] = eb; res["head_bits"] = hb; res["other_bits"] = other_bits
            preds = generate_labels(model, tok, items, labels, bs=a.gbs, mask_ids=mask, retok=retok, max_new=a.max_new)
            sc = score(items, preds, base_correct); sc.pop("correct"); res["score"] = sc
            if acct:
                tot_other = other_bits + eb + hb
                res["pct_by_coding"] = {k: total_pct(acct, nparams, tot_other, k) for k in acct["scale_by"]}
                res["pct_fp16_scales"] = total_pct(acct, nparams, tot_other, "fp16")
            else: res["pct_total"] = 100 * (other_bits + eb + hb + 16.0 * lin_n) / (16 * nparams)
        except Exception as ex:
            import traceback; res["error"] = repr(ex); traceback.print_exc()
        res["secs"] = time.time() - t0; open(outp, "a").write(json.dumps(res) + "\n"); log("ARM", arm["arm"], json.dumps(res.get("score"))[:300], round(res["secs"]), "s")
        gc.collect(); torch.cuda.empty_cache() if dev != "cpu" else None
    log("ALL_DONE")

# ------------------------------------------------------------------ embedding / head
@torch.no_grad()
def quant_matrix(m, orig_cpu, e, dev, seed):
    """e: {kind: rtn|vq|lowrank, L, g, coder / K,d / rank}. returns total bits for the matrix."""
    W = orig_cpu.to(dev).float(); V, C = W.shape; kind = e.get("kind", "rtn")
    if kind == "fp8":
        m.weight.data = W.to(torch.float8_e4m3fn).to(m.weight.dtype); return 8.0 * W.numel()
    if kind == "rtn":
        Q = blockQ(C, seed, dev); Wr = W @ Q.T; q1 = Q1(e["L"], e.get("g", 128), 1, e.get("coder", "fp16")); out = torch.empty_like(Wr)
        for r in range(0, V, 16384): out[r:r + 16384] = rtn(Wr[r:r + 16384], q1) if False else _rtn_chunk(Wr[r:r + 16384], q1)
        q1.hist = q1.hist  # accumulated
        b = q1.bits(); m.weight.data = (out @ Q).to(m.weight.dtype)
        sb = b["scale"].get(e.get("acct", "fp16"), b["scale"]["fp16"]) if isinstance(b["scale"], dict) else 0
        return b["idx"] + sb
    if kind == "lowrank":
        r = e["rank"]; U, S, Vh = torch.svd_lowrank(W, q=r + 16, niter=3); A = U[:, :r] * S[:r]; B = Vh.T[:r] if Vh.shape[0] != r else Vh
        B = Vh[:, :r].T; m.weight.data = (A @ B).to(m.weight.dtype); return 16.0 * (A.numel() + B.numel()) if not e.get("lr_bits") else e["lr_bits"] * (A.numel() + B.numel())
    if kind == "vq":
        d, K = e["d"], e["K"]; X = W.reshape(-1, d); n = X.shape[0]; g = torch.Generator(device=dev).manual_seed(0)
        sub = X[torch.randint(0, n, (min(n, 400000),), device=dev, generator=g)]; cb = sub[torch.randperm(sub.shape[0], device=dev)[:K]].clone()
        for _ in range(12):
            a_ = torch.cdist(sub, cb).argmin(1); cb = torch.stack([sub[a_ == k].mean(0) if (a_ == k).any() else cb[k] for k in range(K)])
        idx = torch.empty(n, dtype=torch.long, device=dev)
        for i in range(0, n, 2_000_000): idx[i:i + 2_000_000] = torch.cdist(X[i:i + 2_000_000], cb).argmin(1)
        m.weight.data = cb[idx].reshape(V, C).to(m.weight.dtype); h = torch.bincount(idx, minlength=K).cpu()
        return hist_bits(h) * 1.005 + 16.0 * cb.numel()
    raise ValueError(kind)

@torch.no_grad()
def _rtn_chunk(Wr, q1):
    R, C = Wr.shape; out = torch.empty_like(Wr); g = q1.g; Idx = torch.empty(R, C, dtype=torch.int8, device=Wr.device)
    for c in range(0, C, g):
        sc, zp = q1.group_scale(Wr[:, c:c + g]); q, ix = q1.quant_cols(Wr[:, c:c + g], sc, zp); out[:, c:c + g] = q; Idx[:, c:c + g] = ix.to(torch.int8)
    q1.add_hist(Idx); return out

# ------------------------------------------------------------------ vocab trim (simulated: mask output, re-tokenise input into kept pieces)
def vocab_trim(tok, model, keep_cfg, items):
    V = model.get_input_embeddings().weight.shape[0]; N = keep_cfg["N"]; nspecial_from = len(tok) - len(tok.get_added_vocab())
    allowed = torch.zeros(V, dtype=torch.bool); allowed[:min(N, V)] = True; allowed[nspecial_from:] = True
    vocab = tok.get_vocab(); inv = {i: t for t, i in vocab.items()}
    kept_tokens = {inv[i]: i for i in range(min(N, len(inv))) if i in inv}
    stat = dict(N=N, kept=int(allowed.sum()), replaced=0, total=0)
    labs = set()
    for it in items: labs.add(it["answer"])
    stat["label_tokens_kept"] = all(all(i < N for i in tok(l, add_special_tokens=False)["input_ids"]) for l in labs)
    maxlen = max(len(t) for t in kept_tokens)
    def retok(ids):
        out = []
        for i in ids:
            stat["total"] += 1
            if i < N or i >= nspecial_from: out.append(i); continue
            stat["replaced"] += 1; s = inv[i]; j = 0
            while j < len(s):   # longest kept prefix
                for k in range(min(len(s) - j, maxlen), 0, -1):
                    if s[j:j + k] in kept_tokens: out.append(kept_tokens[s[j:j + k]]); j += k; break
                else: raise RuntimeError("no byte fallback for " + repr(s[j]))
        return out
    return allowed, retok, stat

# ------------------------------------------------------------------ bit allocation
def kl_layer_sens(model, orig, dev, calib, cfg):
    """per-layer KL(base||quantised-one-layer) at RTN L=3 on 8 calib sequences (final logits)."""
    layers = model.model.layers; x = calib[:4].to(dev); kls = []
    with torch.no_grad():
        ref = torch.log_softmax(model(x, use_cache=False).logits.float(), -1)
        for l in range(len(layers)):
            saved = {}
            for p, n in LIN:
                m = mod(layers[l], p, n); saved[(p, n)] = m.weight.data; W = m.weight.data.float(); C = W.shape[1]; Q = blockQ(C, 5, dev)
                m.weight.data = (rtn(W @ Q.T, Q1(3, cfg.get("g", 128))) @ Q).to(m.weight.dtype)
            lq = torch.log_softmax(model(x, use_cache=False).logits.float(), -1)
            kls.append(float((ref.exp() * (ref - lq)).sum(-1).mean()))
            for (p, n), w in saved.items(): mod(layers[l], p, n).weight.data = w
    return kls

def allocate(sens, cfg):
    """Lagrangian allocation of per-tensor levels at equal total bits as uniform cfg['L']. cfg['alloc'] in {'trace','kl'}."""
    tab = sens["tab"]; Ls = [str(x) for x in sens["Ls"]]; coder = cfg.get("acct", "fp16"); Lref = str(cfg["L"]); mode = cfg["alloc"]
    kl = sens.get("kl")
    def bits(row, L):
        r = row[L]; sc = r["scale"].get(coder, r["scale"]["fp16"]); return r["idx"] + sc
    items = []
    for k, row in tab.items():
        l = int(k.split(".")[0]); w = 1.0
        if mode == "kl": w = kl[l] / max(sum(tab[f"{l}.{p}.{n}"][Ls[0]]["trace"] for p, n in LIN), 1e-30)
        items.append((k, row, w))
    tgt = sum(bits(r, Lref) for _, r, _ in items)
    def pick(lam):
        tot = 0; ch = {}
        for k, row, w in items:
            best = min(Ls, key=lambda L: w * row[L]["trace"] + lam * bits(row, L)); ch[k] = int(best); tot += bits(row, best)
        return tot, ch
    lo, hi = 1e-30, 1e30
    for _ in range(200):
        mid = math.sqrt(lo * hi); tot, ch = pick(mid)
        if tot > tgt: lo = mid
        else: hi = mid
    tot, ch = pick(hi); return ch

if __name__ == "__main__": main()
