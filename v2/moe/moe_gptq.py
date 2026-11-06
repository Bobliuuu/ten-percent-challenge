"""Sequential GPTQ for tiered-precision pruned Qwen3-30B-A3B. Run on a single 96GB GPU box (not Modal).
env: HF_HOME, MOE_DATA, MOE_OUT (needs stats.pt from moe_sweep.py), MOE_MAXMEM
usage: python moe_gptq.py '<json list of configs>'  config={name,tiers:[[frac,bits],...],g,attn_bits,emb_bits,damp}
"""
import json, os, sys, time, numpy as np, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from safetensors import safe_open
from huggingface_hub import snapshot_download
MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507"
DATA = os.environ.get("MOE_DATA", "."); OUT = os.environ.get("MOE_OUT", "out")
def fmt(r):
    s = r["subject"].replace("_", " ")
    q = f"The following are multiple choice questions (with answers) about {s}.\n\n{r['question']}\n"
    for l, c in zip("ABCD", r["choices"]): q += f"{l}. {c}\n"
    return q + "Answer:"
tok = AutoTokenizer.from_pretrained(MODEL); tok.padding_side = "left"
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="auto", max_memory={0: os.environ.get("MOE_MAXMEM", "88GiB")}); model.eval()
cfg = model.config; NE, TK, NL = cfg.num_experts, cfg.num_experts_per_tok, cfg.num_hidden_layers
letters = [tok.encode(" " + l)[0] for l in "ABCD"]
lm_rows = model.lm_head.weight[letters].detach().clone().float()
dev = [json.loads(l) for l in open(DATA + "/dev.jsonl")]; calib = [json.loads(l) for l in open(DATA + "/calib.jsonl")]
stats = torch.load(os.path.join(OUT, "stats.pt"))
sd = snapshot_download(MODEL); widx = json.load(open(os.path.join(sd, "model.safetensors.index.json")))["weight_map"]; _fh = {}
def getw(n):
    f = widx[n]
    if f not in _fh: _fh[f] = safe_open(os.path.join(sd, f), "pt", device="cpu")
    return _fh[f].get_tensor(n)
layers = model.model.layers; blocks = [l.mlp for l in layers]
S = dict(valid=None, H=None, Hli=-1); keepm = [None] * NL
def mk_fwd(li):
    def fwd(self, x):
        B, T, D = x.shape; xf = x.reshape(-1, D)
        logits = self.gate(xf).float()
        if keepm[li] is not None: logits = logits.masked_fill(~keepm[li].to(logits.device), float("-inf"))
        w = torch.softmax(logits, -1); tw, ti = torch.topk(w, TK, -1); tw = tw / tw.sum(-1, keepdim=True)
        out = torch.zeros_like(xf)
        flat = ti.reshape(-1); order = torch.argsort(flat); cnts = torch.bincount(flat, minlength=NE).tolist()
        tokidx = order // TK; ws = tw.reshape(-1)[order]; off = 0
        col = S["H"] is not None and S["Hli"] == li
        for e, c in enumerate(cnts):
            if c == 0: continue
            ids = tokidx[off:off + c]; g = ws[off:off + c]; off += c
            ex = self.experts[e]; xin = xf[ids]
            if col:
                gg = ex.gate_proj(xin); uu = ex.up_proj(xin); inter = ex.act_fn(gg) * uu; h = ex.down_proj(inter)
                v = S["valid"].to(x.device)[ids]
                if v.any():
                    Xe = xin[v].float(); Ie = inter[v].float()
                    S["H"]["gu"][e] += Xe.T @ Xe; S["H"]["d"][e] += Ie.T @ Ie; S["H"]["n"][e] += int(v.sum())
            else: h = ex(xin)
            out.index_add_(0, ids, h * g[:, None].to(h.dtype))
        return out.view(B, T, D), logits
    return fwd
for li, b in enumerate(blocks): b.forward = mk_fwd(li).__get__(b)
def batches(texts, maxtok=12000, maxlen=1024):
    enc = [tok.encode(t)[-maxlen:] for t in texts]; idx = sorted(range(len(enc)), key=lambda i: len(enc[i])); i = 0
    while i < len(idx):
        j = i
        while j < len(idx) and (j - i + 1) * len(enc[idx[j]]) <= maxtok and j - i < 64: j += 1
        j = max(j, i + 1); sel = idx[i:j]; L = max(len(enc[k]) for k in sel)
        ids = torch.full((len(sel), L), tok.pad_token_id); am = torch.zeros((len(sel), L), dtype=torch.long)
        for r, k in enumerate(sel):
            ids[r, L - len(enc[k]):] = torch.tensor(enc[k]); am[r, L - len(enc[k]):] = 1
        yield sel, ids, am; i = j
@torch.no_grad()
def evaluate(rows):
    preds = [None] * len(rows)
    for sel, ids, am in batches([fmt(r) for r in rows]):
        hs = model.model(input_ids=ids.cuda(), attention_mask=am.cuda()).last_hidden_state[:, -1].float()
        lg = hs @ lm_rows.to(hs.device).T
        for r, k in enumerate(sel): preds[k] = int(lg[r].argmax())
    return float(np.mean([p == r["answer"] for p, r in zip(preds, rows)])), preds

def gparams(Wg, bits):
    """Wg [N,g] -> scale, zp [N,1] with MSE clip search."""
    mn = Wg.min(-1, keepdim=True).values; mx = Wg.max(-1, keepdim=True).values; qmax = 2 ** bits - 1; best = None
    for r in [1.0, .95, .9, .85, .8, .75, .7, .65, .6]:
        c = (mx + mn) / 2; hw = (mx - mn) / 2 * r; lo = c - hw; hi = c + hw
        sc = ((hi - lo) / qmax).clamp_min(1e-8).half().float(); zp = torch.round(-lo / sc).clamp(0, qmax)
        dq = (torch.clamp(torch.round(Wg / sc) + zp, 0, qmax) - zp) * sc; err = ((dq - Wg) ** 2).sum(-1, keepdim=True)
        if best is None: best = [err, sc, zp]
        else:
            m = err < best[0]; best = [torch.where(m, err, best[0]), torch.where(m, sc, best[1]), torch.where(m, zp, best[2])]
    return best[1], best[2]
@torch.no_grad()
def gptq(W, H, bits, g, damp=0.01):
    """W [E,o,i] float32, H [E,i,i]. returns dq [E,o,i], ints uint8."""
    E, o, i = W.shape; W = W.clone(); H = H.clone(); qmax = 2 ** bits - 1
    H = torch.nan_to_num(H, nan=0.0, posinf=0.0, neginf=0.0); W = torch.nan_to_num(W)
    d = torch.diagonal(H, dim1=1, dim2=2); dead = d == 0
    idx = torch.arange(i, device=W.device)
    H[:, idx, idx] = torch.where(dead, torch.ones_like(d), d); W.masked_fill_(dead[:, None, :], 0)
    md = torch.diagonal(H, dim1=1, dim2=2).mean(-1)
    H[:, idx, idx] += damp * md[:, None]
    try:
        L = torch.linalg.cholesky(H); Hinv = torch.cholesky_inverse(L); U = torch.linalg.cholesky(Hinv, upper=True)
    except Exception as ex:
        print("cholesky fail, boost damp", ex, flush=True)
        H[:, idx, idx] += 0.5 * md[:, None]
        try: L = torch.linalg.cholesky(H); Hinv = torch.cholesky_inverse(L); U = torch.linalg.cholesky(Hinv, upper=True)
        except Exception: 
            print("cholesky fail again -> RTN fallback", flush=True); return rtn(W, bits, g)
    Q = torch.zeros_like(W); QI = torch.zeros(W.shape, dtype=torch.uint8, device=W.device)
    for i1 in range(0, i, g):
        i2 = min(i1 + g, i); W1 = W[:, :, i1:i2].clone(); Err = torch.zeros_like(W1); U1 = U[:, i1:i2, i1:i2]
        sc, zp = gparams(W1.reshape(-1, i2 - i1), bits); sc = sc.view(E, o, 1); zp = zp.view(E, o, 1)
        for j in range(i2 - i1):
            w = W1[:, :, j:j + 1]; q = torch.clamp(torch.round(w / sc) + zp, 0, qmax); dq = (q - zp) * sc
            e = (w - dq) / U1[:, j, j][:, None, None]
            W1[:, :, j:] -= e @ U1[:, j:j + 1, j:]
            Err[:, :, j:j + 1] = e; Q[:, :, i1 + j:i1 + j + 1] = dq; QI[:, :, i1 + j:i1 + j + 1] = q.to(torch.uint8)
        if i2 < i: W[:, :, i2:] -= Err @ U[:, i1:i2, i2:]
    return Q, QI
def rtn(W, bits, g):
    shp = W.shape; Wg = W.float().reshape(-1, g); sc, zp = gparams(Wg, bits); qmax = 2 ** bits - 1
    q = torch.clamp(torch.round(Wg / sc) + zp, 0, qmax); return ((q - zp) * sc).reshape(shp), q.reshape(shp).to(torch.uint8)
def hent(hist):
    p = hist / hist.sum(); return float(-(p[p > 0] * torch.log2(p[p > 0])).sum())

class Stop(Exception): pass
def run_config(c):
    t0 = time.time(); name = c["name"]; tiers = c["tiers"]; g = c.get("g", 128); ab = c.get("attn_bits", 4); eb = c.get("emb_bits", 6); damp = c.get("damp", 0.01)
    acc_bits = dict(raw=0.0, ent=0.0)
    def add(n, bits, hist, grp):
        acc_bits["raw"] += n * bits + n / grp * (16 + bits); acc_bits["ent"] += hent(hist) * n + n / grp * (16 + bits)
    # embedding
    W = getw("model.embed_tokens.weight").cuda(); dq, q = rtn(W, eb, 128)
    model.model.embed_tokens.weight.data.copy_(dq.to(torch.bfloat16)); add(W.numel(), eb, torch.bincount(q.reshape(-1).long(), minlength=2 ** eb).double().cpu(), 128); del W, dq, q
    # capture layer-0 inputs
    store = []
    def pre(mod, args, kwargs): 
        kw = {k: v for k, v in kwargs.items() if k != "past_key_values"}; kw["use_cache"] = False; store.append([args[0].detach(), kw]); raise Stop()
    h = layers[0].register_forward_pre_hook(pre, with_kwargs=True)
    valids = []
    for sel, ids, am in batches([fmt(r) + " " + "ABCD"[r["answer"]] for r in calib], maxtok=12000, maxlen=512):
        try:
            with torch.no_grad(): model.model(input_ids=ids.cuda(), attention_mask=am.cuda(), use_cache=False)
        except Stop: pass
        valids.append(am.reshape(-1).bool())
    h.remove(); print("captured", len(store), "batches", time.time() - t0, flush=True)
    sc_ = stats["reap"] / stats["cnt"].clamp_min(1)
    rank_bits = []
    for li in range(NL):
        order = torch.argsort(sc_[li].float(), descending=True); tb = torch.zeros(NE, dtype=torch.long); pos = 0; 
        for frac, bits in tiers:
            n = int(round(frac * NE)); tb[order[pos:pos + n]] = bits; pos += n
        rank_bits.append(tb)
    n_other = 0
    for li in range(NL):
        layer = layers[li]; a = layer.self_attn; b = blocks[li]; tb = rank_bits[li]; keepm[li] = tb > 0
        # restore originals
        with torch.no_grad():
            for m in ("q_proj", "k_proj", "v_proj", "o_proj"): getattr(a, m).weight.data.copy_(getw(f"model.layers.{li}.self_attn.{m}.weight").to(getattr(a, m).weight.device))
            for e in torch.nonzero(tb > 0).flatten().tolist():
                for m in ("gate_proj", "up_proj", "down_proj"):
                    getattr(b.experts[e], m).weight.data.copy_(getw(f"model.layers.{li}.mlp.experts.{e}.{m}.weight").to(getattr(b.experts[e], m).weight.device))
        Hs = dict(gu=torch.zeros(NE, 2048, 2048, device="cuda"), d=torch.zeros(NE, 768, 768, device="cuda"), n=[0] * NE)
        Ha = dict(qkv=torch.zeros(2048, 2048, device="cuda"), o=torch.zeros(4096, 4096, device="cuda"))
        cur = {"v": None}
        def hq(mod, inp): 
            x = inp[0].reshape(-1, inp[0].shape[-1])[cur["v"]].float(); Ha["qkv"] += x.T @ x
        def ho(mod, inp):
            x = inp[0].reshape(-1, inp[0].shape[-1])[cur["v"]].float(); Ha["o"] += x.T @ x
        h1 = a.q_proj.register_forward_pre_hook(hq); h2 = a.o_proj.register_forward_pre_hook(ho)
        S["H"] = Hs; S["Hli"] = li
        with torch.no_grad():
            for (x, kw), v in zip(store, valids):
                S["valid"] = v; cur["v"] = v.to("cuda"); layer(x, **kw)
        S["H"] = None; h1.remove(); h2.remove()
        ntok = sum(int(v.sum()) for v in valids)
        with torch.no_grad():
            # attention GPTQ
            for names, H, key in ((("q_proj", "k_proj", "v_proj"), Ha["qkv"] / ntok, "qkv"), (("o_proj",), Ha["o"] / ntok, "o")):
                for m in names:
                    p = getattr(a, m).weight; W = p.data.float()[None]
                    dq, q = gptq(W, H[None], ab, 128, damp); p.data.copy_(dq[0].to(torch.bfloat16))
                    add(W.numel(), ab, torch.bincount(q.reshape(-1).long(), minlength=2 ** ab).double().cpu(), 128)
            # experts by tier
            for bits in sorted(set(tb.tolist()) - {0}):
                ki = torch.nonzero(tb == bits).flatten().tolist(); kt = torch.tensor(ki, device="cuda")
                nn_ = torch.tensor([max(Hs["n"][e], 1) for e in ki], device="cuda", dtype=torch.float32)[:, None, None]
                for m, key in (("gate_proj", "gu"), ("up_proj", "gu"), ("down_proj", "d")):
                    W = torch.stack([getattr(b.experts[e], m).weight.data for e in ki]).float(); H = Hs[key][kt] / nn_
                    dq, q = gptq(W, H, bits, g, damp)
                    for j, e in enumerate(ki): getattr(b.experts[e], m).weight.data.copy_(dq[j].to(torch.bfloat16))
                    add(W.numel(), bits, torch.bincount(q.reshape(-1).long(), minlength=2 ** bits).double().cpu(), g)
                    del W, H, dq, q
            n_other += int((tb > 0).sum()) * 2048 * 2 + 2 * 2048 * 2 + 2 * 128 * 2
            # next inputs
            for s in store:
                o = layer(s[0], **s[1]); s[0] = torch.nan_to_num(o[0] if isinstance(o, tuple) else o, nan=0.0, posinf=6e4, neginf=-6e4)
        del Hs, Ha; torch.cuda.empty_cache()
        print(f"[{name}] layer {li} done {time.time()-t0:.0f}s", flush=True)
    del store
    n_other += 2048 * 2 + 4 * 2048 * 2
    fp16 = 30532122624 * 2; raw = acc_bits["raw"] / 8 + n_other; ent = acc_bits["ent"] / 8 + n_other
    for k in range(NL): pass
    acc, preds = evaluate(dev)
    return dict(name=name, tiers=tiers, g=g, attn_bits=ab, emb_bits=eb, damp=damp, dev_acc=acc, bytes_raw=raw, bytes_ent=ent, ratio_raw=raw / fp16, ratio_ent=ent / fp16, secs=time.time() - t0), preds
if __name__ == "__main__":
    cfgs = json.loads(sys.argv[1]); resf = os.path.join(OUT, "results_gptq.json")
    res = json.load(open(resf)) if os.path.exists(resf) else {}
    base_preds = None
    for c in cfgs:
        r, preds = run_config(c)
        print("RESULT_JSON " + json.dumps(r), flush=True); res[r["name"]] = r; json.dump(res, open(resf, "w"), indent=1)
        # reset keepm for safety
        for li in range(NL): keepm[li] = None
