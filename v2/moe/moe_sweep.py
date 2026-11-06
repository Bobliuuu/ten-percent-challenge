"""MoE prune+quantize sweep (Qwen3-30B-A3B-Instruct-2507). Simulated (fake-quant) with exact byte accounting.
Launch: modal run --detach v2/moe/moe_sweep.py
"""
import modal, json, os, math, time
app = modal.App("moe-10pct")
vol = modal.Volume.from_name("moe-10pct-vol", create_if_missing=True)
HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(HERE, "..", "base")
img = (modal.Image.debian_slim(python_version="3.11")
       .pip_install("torch==2.5.1", "transformers==4.57.1", "accelerate", "huggingface_hub", "hf_transfer", "numpy")
       .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HOME": "/vol/hf"})
       .add_local_file(os.path.join(BASE, "dev.jsonl"), "/data/dev.jsonl")
       .add_local_file(os.path.join(BASE, "calib.jsonl"), "/data/calib.jsonl"))
MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507"

def fmt(r):
    s = r["subject"].replace("_", " ")
    q = f"The following are multiple choice questions (with answers) about {s}.\n\n{r['question']}\n"
    for l, c in zip("ABCD", r["choices"]): q += f"{l}. {c}\n"
    return q + "Answer:"

GRID = [  # (name, keep_frac, bits, score)
    ("f0.75_b2", .75, 2, "reap"), ("f0.5_b2", .5, 2, "reap"),
    ("f0.5_b3", .5, 3, "reap"), ("f0.375_b3", .375, 3, "reap"), ("f0.375_b4", .375, 4, "reap"),
    ("f0.25_b4", .25, 4, "reap"), ("f0.75_b3", .75, 3, "reap"), ("f1.0_b3", 1.0, 3, "reap"),
    ("f0.5_b3_freq", .5, 3, "freq"), ("f0.375_b4_freq", .375, 4, "freq"),
    ("f0.5_b4", .5, 4, "reap"), ("f0.25_b3", .25, 3, "reap"), ("f0.5_b3_g64", .5, 3, "reap"),
]

@app.function(image=img, gpu=["H100:2","A100-80GB:2"], cpu=16, memory=300000, timeout=24*3600, volumes={"/vol": vol})
def run(grid_json: str = ""):
    import torch, numpy as np, torch.nn.functional as F
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import transformers.models.qwen3_moe.modeling_qwen3_moe as Q
    grid = json.loads(grid_json) if grid_json else GRID
    tok = AutoTokenizer.from_pretrained(MODEL); tok.padding_side = "left"
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, device_map="auto",
                                                 max_memory={0: "62GiB", 1: "62GiB"}); model.eval()
    print("loaded", time.time() - t0, flush=True)
    cfg = model.config
    NE, TK, NL = cfg.num_experts, cfg.num_experts_per_tok, cfg.num_hidden_layers
    letters = [tok.encode(" " + l)[0] for l in "ABCD"]
    lm_rows = model.lm_head.weight[letters].detach().clone().float()  # 4 rows kept (fp16 in size acct)
    dev = [json.loads(l) for l in open("/data/dev.jsonl")]
    calib = [json.loads(l) for l in open("/data/calib.jsonl")]
    blocks = [l.mlp for l in model.model.layers]
    S = dict(collect=False, valid=None)
    stats = {"cnt": torch.zeros(NL, NE, dtype=torch.float64), "reap": torch.zeros(NL, NE, dtype=torch.float64),
             "gate": torch.zeros(NL, NE, dtype=torch.float64)}
    keepm = [None] * NL
    def mk_fwd(li):
        def fwd(self, x):
            B, T, D = x.shape; xf = x.reshape(-1, D)
            logits = self.gate(xf).float()
            if keepm[li] is not None: logits = logits.masked_fill(~keepm[li].to(logits.device), float("-inf"))
            w = torch.softmax(logits, -1); tw, ti = torch.topk(w, TK, -1); tw = tw / tw.sum(-1, keepdim=True)
            out = torch.zeros_like(xf)
            flat = ti.reshape(-1); order = torch.argsort(flat); cnts = torch.bincount(flat, minlength=NE).tolist()
            tokidx = order // TK; ws = tw.reshape(-1)[order]; off = 0
            valid = S["valid"].to(x.device) if S["collect"] else None
            for e, c in enumerate(cnts):
                if c == 0: continue
                ids = tokidx[off:off + c]; g = ws[off:off + c]; off += c
                h = self.experts[e](xf[ids])
                out.index_add_(0, ids, h * g[:, None].to(h.dtype))
                if S["collect"]:
                    v = valid[ids].double(); nrm = h.float().norm(dim=-1).double()
                    stats["cnt"][li, e] += v.sum().item(); stats["reap"][li, e] += (v * g.double() * nrm).sum().item()
                    stats["gate"][li, e] += (v * g.double()).sum().item()
            return out.view(B, T, D), logits
        return fwd
    for li, b in enumerate(blocks): b.forward = mk_fwd(li).__get__(b)
    # decoder layer expects tuple? handle both
    import inspect
    # patch decoder layer mlp usage: transformers 4.57 decoder does `hidden_states = self.mlp(hidden_states)`, may unpack tuple
    src = inspect.getsource(Q.Qwen3MoeDecoderLayer.forward)
    tuple_ok = "router_logits" in src or "= self.mlp(" in src and "," in src.split("= self.mlp(")[0].split("\n")[-1]
    if not tuple_ok:
        for li, b in enumerate(blocks):
            f0 = b.forward
            b.forward = (lambda f: (lambda x: f(x)[0]))(f0)
    print("decoder src mlp line:", [l for l in src.split("\n") if "self.mlp" in l], "tuple_ok", tuple_ok, flush=True)

    def batches(texts, maxtok=24000, maxlen=1024):
        enc = [tok.encode(t)[-maxlen:] for t in texts]
        idx = sorted(range(len(enc)), key=lambda i: len(enc[i]))
        i = 0
        while i < len(idx):
            j = i; 
            while j < len(idx) and (j - i + 1) * len(enc[idx[j]]) <= maxtok and j - i < 64: j += 1
            j = max(j, i + 1); sel = idx[i:j]; L = max(len(enc[k]) for k in sel)
            ids = torch.full((len(sel), L), tok.pad_token_id); am = torch.zeros((len(sel), L), dtype=torch.long)
            for r, k in enumerate(sel):
                ids[r, L - len(enc[k]):] = torch.tensor(enc[k]); am[r, L - len(enc[k]):] = 1
            yield sel, ids, am; i = j

    @torch.no_grad()
    def evaluate(rows):
        preds = [None] * len(rows); texts = [fmt(r) for r in rows]
        for sel, ids, am in batches(texts):
            hs = model.model(input_ids=ids.to("cuda:0"), attention_mask=am.to("cuda:0")).last_hidden_state[:, -1].float()
            lg = hs @ lm_rows.to(hs.device).T
            for r, k in enumerate(sel): preds[k] = int(lg[r].argmax())
        acc = float(np.mean([p == r["answer"] for p, r in zip(preds, rows)]))
        return acc, preds

    out_all = {}
    def emit(name, d):
        d["name"] = name; d["t"] = time.time() - t0; out_all[name] = d
        print("RESULT_JSON " + json.dumps(d), flush=True)
        json.dump(out_all, open("/vol/results.json", "w"), indent=1); vol.commit()

    # base
    base_acc, base_preds = evaluate(dev)
    emit("base_bf16", dict(dev_acc=base_acc, n=len(dev)))
    # calibration stats
    ctexts = [fmt(r) + " " + "ABCD"[r["answer"]] for r in calib]
    S["collect"] = True
    for sel, ids, am in batches(ctexts, maxtok=24000, maxlen=768):
        S["valid"] = am.reshape(-1).bool()
        with torch.no_grad(): model.model(input_ids=ids.to("cuda:0"), attention_mask=am.to("cuda:0"))
    S["collect"] = False
    torch.save(stats, "/vol/stats.pt"); vol.commit()
    print("calib done", time.time() - t0, "min cnt/layer0", stats["cnt"][0].min().item(), flush=True)
    # originals on CPU
    orig = {}
    for li, b in enumerate(blocks):
        orig[li] = {m: torch.stack([getattr(ex, m).weight.detach().cpu() for ex in b.experts]) for m in ("gate_proj", "up_proj", "down_proj")}
    attn_orig = {}
    for li, l in enumerate(model.model.layers):
        for m in ("q_proj", "k_proj", "v_proj", "o_proj"): attn_orig[(li, m)] = getattr(l.self_attn, m).weight.detach().cpu().clone()
    emb_orig = model.model.embed_tokens.weight.detach().cpu().clone()

    def quant(W, bits, g, clip=True):
        """W [..., in] float; groupwise asym along last dim. returns dequant, int tensor (uint8)."""
        shp = W.shape; Wg = W.float().reshape(-1, g)
        mn = Wg.min(-1, keepdim=True).values; mx = Wg.max(-1, keepdim=True).values
        qmax = 2 ** bits - 1
        best = None
        for r in ([1.0, .95, .9, .85, .8, .75, .7, .65] if clip else [1.0]):
            c = (mx + mn) / 2; hw = (mx - mn) / 2 * r
            lo, hi = c - hw, c + hw
            sc = ((hi - lo) / qmax).clamp_min(1e-8).half().float()
            zp = torch.round(-lo / sc).clamp(0, qmax)
            q = torch.clamp(torch.round(Wg / sc) + zp, 0, qmax)
            dq = (q - zp) * sc
            err = ((dq - Wg) ** 2).sum(-1, keepdim=True)
            if best is None: best = [err, dq, q]
            else:
                m = err < best[0]; best[0] = torch.where(m, err, best[0]); best[1] = torch.where(m, dq, best[1]); best[2] = torch.where(m, q, best[2])
        return best[1].reshape(shp), best[2].reshape(shp).to(torch.uint8)
    def ent_bits(q, bits):
        h = torch.bincount(q.reshape(-1).long(), minlength=2 ** bits).double(); p = h / h.sum()
        return float(-(p[p > 0] * torch.log2(p[p > 0])).sum()) * q.numel()

    # attn+emb at 4 bit g128 (constant across grid); returns bits
    def quant_fixed(bits=4, g=128):
        raw = ent = 0.0
        for li, l in enumerate(model.model.layers):
            for m in ("q_proj", "k_proj", "v_proj", "o_proj"):
                W = attn_orig[(li, m)].cuda(); dq, q = quant(W, bits, g)
                getattr(l.self_attn, m).weight.data.copy_(dq.to(torch.bfloat16).to(getattr(l.self_attn, m).weight.device))
                n = W.numel(); raw += n * bits + n / g * (16 + bits); ent += ent_bits(q, bits) + n / g * (16 + bits)
        W = emb_orig.cuda(); dq, q = quant(W, bits, g)
        model.model.embed_tokens.weight.data.copy_(dq.to(torch.bfloat16).to(model.model.embed_tokens.weight.device))
        n = W.numel(); raw += n * bits + n / g * (16 + bits); ent += ent_bits(q, bits) + n / g * (16 + bits)
        return raw, ent
    fixed_raw, fixed_ent = quant_fixed()
    total_params = sum(p.numel() for p in model.parameters())
    fp16_bytes = total_params * 2
    print("params", total_params, "fp16 bytes", fp16_bytes, "fixed bits", fixed_raw, fixed_ent, flush=True)
    other_bytes = lambda keepn: (NL * (keepn * cfg.hidden_size * 2 + 2 * cfg.hidden_size * 2 + 2 * cfg.head_dim * 2) + cfg.hidden_size * 2 + 4 * cfg.hidden_size * 2)
    acc, p = evaluate(dev)
    emit("attn4_emb4_only", dict(dev_acc=acc, agree=float(np.mean([a == b for a, b in zip(p, base_preds)])),
         fixed_bytes_raw=fixed_raw / 8, fixed_bytes_ent=fixed_ent / 8))

    for name, f, bits, score, *rest in grid:
        g = 64 if name.endswith("g64") else 128
        t1 = time.time()
        keepn = max(1, int(round(f * NE)))
        sc = (stats["reap"] / stats["cnt"].clamp_min(1)) if score == "reap" else stats["cnt"]
        raw = ent = 0.0
        for li, b in enumerate(blocks):
            top = torch.topk(sc[li].float(), keepn).indices
            km = torch.zeros(NE, dtype=torch.bool); km[top] = True; keepm[li] = km
            ki = top.sort().values.tolist()
            for m in ("gate_proj", "up_proj", "down_proj"):
                W = orig[li][m][ki].cuda(); dq, q = quant(W, bits, g)
                for j, e in enumerate(ki):
                    p_ = getattr(b.experts[e], m).weight; p_.data.copy_(dq[j].to(torch.bfloat16).to(p_.device))
                n = W.numel(); raw += n * bits + n / g * (16 + bits); ent += ent_bits(q, bits) + n / g * (16 + bits)
        ob = sum(other_bytes(keepn) for _ in [0])
        tot_raw = (fixed_raw + raw) / 8 + ob; tot_ent = (fixed_ent + ent) / 8 + ob
        acc, p = evaluate(dev)
        emit(name, dict(keep_frac=f, keep_n=keepn, bits=bits, group=g, score=score, dev_acc=acc,
             agree=float(np.mean([a == b for a, b in zip(p, base_preds)])), base_acc=base_acc,
             bytes_raw=tot_raw, bytes_ent=tot_ent, fp16_bytes=fp16_bytes,
             ratio_raw=tot_raw / fp16_bytes, ratio_ent=tot_ent / fp16_bytes, secs=time.time() - t1))
    # restore not needed (each config re-quantizes kept experts from orig; pruned experts masked)

@app.local_entrypoint()
def main(grid: str = ""):
    run.remote(grid)
