"""Task-specific expert pruning + RTN on Qwen3-30B-A3B-Instruct-2507 (dev data only).
Simulated (fake-quant) eval; exact byte accounting (bit-packed codes + fp16 scale + packed zero per group + bf16 norms/router).
Metric: greedy FIRST token of the chat-templated answer == label's first token (full-vocab argmax, so format failures count wrong).
Launch: modal run --detach v2/moe_task/moe_task.py --cfg v2/moe_task/cfg_e2.json
"""
import modal, json, os, math, time, random, collections
app = modal.App("moe-task-10pct")
vol = modal.Volume.from_name("moe-10pct-vol", create_if_missing=True)
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
C2 = os.path.join(HERE, "..", "commercial", "C2")
img = (modal.Image.debian_slim(python_version="3.11")
       .pip_install("torch==2.5.1", "transformers==4.57.1", "accelerate", "huggingface_hub", "hf_transfer", "numpy", "datasets")
       .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HOME": "/vol/hf"})
       .add_local_dir(DATA, "/data").add_local_dir(C2, "/c2"))
MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507"
FP16 = 61_064_245_248
BUDGET = int(0.099 * FP16)
LET = "ABCDEFGHIJKLMN"

def c2_user(r):
    s = r["inst"] + "\n\nText: " + r["text"] + "\n\nOptions:\n"
    for l, c in zip(LET, r["names"]): s += f"{l}. {c}\n"
    return s + "Answer:"

def load_tasks():
    rd = lambda p: [json.loads(l) for l in open(p)]
    rng = random.Random(0)
    T = {}
    # INJ dev: stratified subset of kinds (dev only)
    dev = rd("/data/dev_inj.jsonl"); by = collections.defaultdict(list)
    for r in dev: by[r["kind"]].append(r)
    quota = {"S1_benign": 400, "S1_injection": 400, "S3_benign": 200, "S3_injection": 200, "S2_benign": 130, "S2_injection": 130}
    sel = []
    for k, q in quota.items():
        v = by[k][:]; rng.shuffle(v); sel += v[:q]
    T["INJ"] = dict(dev=[dict(id=r["id"], user=r["query"], label=r["gold"].capitalize(), kind=r["gold"], src=r["source"]) for r in sel],
                    calib=[r["user"] for r in rd("/data/calib_inj.jsonl")])
    nd = rd("/data/items_dev.jsonl")
    T["NEWS"] = dict(dev=[dict(id=r["id"], user=r["user"], label=r["answer"], kind=r["kind"], src=r["src"]) for r in nd],
                     calib=[r["user"] for r in rd("/data/calib.jsonl")])
    cd = rd("/c2/dev.jsonl")[:1000]
    T["C2"] = dict(dev=[dict(id=r["id"], user=c2_user(r), label=LET[int(r["answer"])], kind=LET[int(r["answer"])], src=r.get("origin", "")) for r in cd],
                   calib=[c2_user(r) for r in rd("/c2/calib.jsonl")[:700]])
    return T

@app.function(image=img, gpu="A100-80GB", cpu=16, memory=300000, timeout=int(3.5 * 3600), volumes={"/vol": vol})
def run(cfg_json: str):
    import torch, numpy as np
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import transformers.models.qwen3_moe.modeling_qwen3_moe as Q
    cfgs = json.loads(cfg_json); OUT = "/vol/task"; os.makedirs(OUT, exist_ok=True)
    tag = cfgs.get("tag", "run")
    tok = AutoTokenizer.from_pretrained(MODEL)
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, device_map="auto", max_memory={0: "77GiB"}); model.eval()
    cfg = model.config; NE, TK, NL, H, I = cfg.num_experts, cfg.num_experts_per_tok, cfg.num_hidden_layers, cfg.hidden_size, cfg.moe_intermediate_size
    print("loaded", time.time() - t0, flush=True)
    T = load_tasks()
    # label first tokens
    ftok = lambda s: tok.encode(s)[0]
    labs = {"INJ": ["Injection", "Benign"], "NEWS": sorted({r["label"] for r in T["NEWS"]["dev"]}), "C2": list(LET[:5])}
    for t, ls in labs.items():
        ids = [ftok(l) for l in ls]; assert len(set(ids)) == len(ids), (t, ls, ids)
    tmpl = lambda u: tok.apply_chat_template([{"role": "user", "content": u}], add_generation_prompt=True, tokenize=False)

    blocks = [l.mlp for l in model.model.layers]
    S = dict(collect=False, valid=None)
    st = lambda: {k: torch.zeros(NL, NE, dtype=torch.float64) for k in ("mass", "reap", "cnt")}
    stats = st(); keepm = [None] * NL
    def mk_fwd(li):
        def fwd(self, x):
            B, Tn, D = x.shape; xf = x.reshape(-1, D)
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
                    stats["cnt"][li, e] += v.sum().item(); stats["mass"][li, e] += (v * g.double()).sum().item()
                    stats["reap"][li, e] += (v * g.double() * nrm).sum().item()
            return out.view(B, Tn, D), logits
        return fwd
    import inspect
    for li, b in enumerate(blocks): b.forward = mk_fwd(li).__get__(b)
    src = inspect.getsource(Q.Qwen3MoeDecoderLayer.forward)
    tuple_ok = "router_logits" in src or "= self.mlp(" in src and "," in src.split("= self.mlp(")[0].split("\n")[-1]
    if not tuple_ok:
        for b in blocks: b.forward = (lambda f: (lambda x: f(x)[0]))(b.forward)
    print("tuple_ok", tuple_ok, flush=True)

    def batches(enc, maxtok=14000):
        idx = sorted(range(len(enc)), key=lambda i: len(enc[i])); i = 0
        while i < len(idx):
            j = i
            while j < len(idx) and (j - i + 1) * len(enc[idx[j]]) <= maxtok and j - i < 64: j += 1
            j = max(j, i + 1); sel = idx[i:j]; L = max(len(enc[k]) for k in sel)
            ids = torch.full((len(sel), L), tok.pad_token_id); am = torch.zeros((len(sel), L), dtype=torch.long)
            for r, k in enumerate(sel):
                ids[r, L - len(enc[k]):] = torch.tensor(enc[k]); am[r, L - len(enc[k]):] = 1
            yield sel, ids, am; i = j
    MAXLEN = 1800
    def encode(users):
        e = [tok.encode(tmpl(u)) for u in users]
        nt = sum(len(x) > MAXLEN for x in e); e = [x if len(x) <= MAXLEN else x[:200] + x[-(MAXLEN - 200):] for x in e]
        return e, nt

    @torch.no_grad()
    def evaluate(task):
        rows = T[task]["dev"]; enc = ENC[task]; preds = [None] * len(rows)
        W = model.lm_head.weight
        for sel, ids, am in batches(enc):
            hs = model.model(input_ids=ids.to("cuda:0"), attention_mask=am.to("cuda:0")).last_hidden_state[:, -1]
            lg = hs @ W.to(hs.device).T
            top = lg.argmax(-1).tolist()
            for r, k in enumerate(sel): preds[k] = top[r]
        return preds
    ENC = {}
    for t in T:
        ENC[t], nt = encode([r["user"] for r in T[t]["dev"]])
        print(t, "dev n", len(ENC[t]), "max len", max(map(len, ENC[t])), "truncated", nt, flush=True)

    def score(task, preds):
        rows = T[task]["dev"]; gold = [ftok(r["label"]) for r in rows]
        return [p == g for p, g in zip(preds, gold)]
    def summarize(task, ok, base_ok=None):
        rows = T[task]["dev"]; d = dict(acc=float(np.mean(ok)), n=len(ok))
        for key in ("kind", "src"):
            grp = collections.defaultdict(list)
            for r, o, i in zip(rows, ok, range(len(ok))): grp[r[key]].append(i)
            d["acc_" + key] = {k: float(np.mean([ok[i] for i in v])) for k, v in grp.items()}
            if base_ok is not None:
                d["faith_" + key] = {k: float(sum(ok[i] and base_ok[i] for i in v) / max(1, sum(base_ok[i] for i in v))) for k, v in grp.items()}
        if base_ok is not None:
            d["faith"] = float(sum(a and b for a, b in zip(ok, base_ok)) / max(1, sum(base_ok)))
            d["n_base_correct"] = int(sum(base_ok))
        return d
    results = {}
    def emit(name, d):
        d["name"] = name; d["t"] = time.time() - t0; results[name] = d
        print("RESULT_JSON " + json.dumps(d), flush=True)
        json.dump(results, open(f"{OUT}/{tag}_results.json", "w"), indent=1); vol.commit()

    # ---- E1 base
    BASE = {}
    for t in cfgs.get("tasks", ["INJ", "NEWS", "C2"]):
        p = evaluate(t); ok = score(t, p); BASE[t] = ok
        tot = collections.Counter(p); fail = float(np.mean([pp not in [ftok(l) for l in labs[t]] for pp in p]))
        d = summarize(t, ok); d["format_fail"] = fail
        emit(f"base_{t}", d)

    # ---- originals on CPU
    ORIG = [[[getattr(b.experts[e], m).weight.data.cpu().clone() for m in ("gate_proj", "up_proj", "down_proj")] for e in range(NE)] for b in blocks]
    ATT = {(li, m): getattr(l.self_attn, m).weight.data.cpu().clone() for li, l in enumerate(model.model.layers) for m in ("q_proj", "k_proj", "v_proj", "o_proj")}
    EMB = model.model.embed_tokens.weight.data.cpu().clone(); HEAD = model.lm_head.weight.data.cpu().clone()
    print("orig copied", time.time() - t0, flush=True)
    GRID = (1.0, 0.95, 0.9, 0.85, 0.8, 0.75)
    def quant(W, bits, g):
        shp = W.shape; Wg = W.float().reshape(-1, g); qmax = 2 ** bits - 1
        mn = Wg.min(-1, keepdim=True).values; mx = Wg.max(-1, keepdim=True).values; mid = (mn + mx) / 2; half = (mx - mn) / 2
        best = None
        for s in GRID:
            lo = mid - half * s; hi = mid + half * s
            sc = ((hi - lo) / qmax).clamp_min(1e-8).half().float()
            zp = torch.round(-lo / sc).clamp(0, qmax)
            q = torch.clamp(torch.round(Wg / sc) + zp, 0, qmax); dq = (q - zp) * sc
            err = ((dq - Wg) ** 2).sum(-1, keepdim=True)
            if best is None: best = [err, dq]
            else:
                m = err < best[0]; best[0] = torch.where(m, err, best[0]); best[1] = torch.where(m, dq, best[1])
        return best[1].reshape(shp)
    def qbytes(n, bits, g):  # codes + fp16 scale + packed zero-point per group
        return n * bits / 8 + n / g * (2 + bits / 8)

    def fixed_bytes(c):
        qd, kvd = cfg.num_attention_heads * cfg.head_dim, cfg.num_key_value_heads * cfg.head_dim
        attn_n = NL * (H * qd * 2 + H * kvd * 2); V = cfg.vocab_size
        b = qbytes(attn_n, c["ab"], c["g"]) + (qbytes(V * H, c["mb"], c["g"]) if c["mb"] < 16 else 2 * V * H) \
            + (qbytes(V * H, c["hb"], c["g"]) if c["hb"] < 16 else 2 * V * H)
        b += NL * (2 * H + 2 * cfg.head_dim) * 2 + H * 2          # norms (bf16)
        return b
    def total_bytes(c, K):
        return fixed_bytes(c) + K * (qbytes(3 * H * I, c["eb"], c["g"]) + H * 2)  # + router row (bf16) per kept expert
    def solve_K(c):
        K = 0
        while K < NL * NE and total_bytes(c, K + 1) <= BUDGET: K += 1
        return K

    def collect(texts):
        global_stats = st()
        for k in global_stats: stats[k].zero_()
        enc, _ = encode(texts); S["collect"] = True
        for sel, ids, am in batches(enc):
            S["valid"] = am.reshape(-1).bool()
            with torch.no_grad(): model.model(input_ids=ids.to("cuda:0"), attention_mask=am.to("cuda:0"))
        S["collect"] = False
        return {k: v.clone() for k, v in stats.items()}
    wiki = None
    def generic(n=400):
        from datasets import load_dataset
        d = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
        out = [x["text"].strip() for x in d if len(x["text"].strip()) > 80 and not x["text"].strip().startswith("=")]
        return out[:n]
    wiki = generic()
    STAT = {}
    def get_stats(task, variant):
        key = (task, variant)
        if key in STAT: return STAT[key]
        texts = list(T[task]["calib"])
        if variant == "full":
            others = [u for o in T if o != task for u in T[o]["calib"][:150]]
            texts = texts + others + wiki
        s = collect(texts); STAT[key] = s
        torch.save(s, f"{OUT}/{tag}_stats_{task}_{variant}.pt"); vol.commit()
        print("stats", key, len(texts), "min mass layer0", s["cnt"][0].min().item(), flush=True)
        return s

    def select(sc, K, mode, minpl=16):
        frac = sc / sc.sum(1, keepdim=True)
        if mode == "uniform":
            k = K // NL; return [sorted(torch.topk(frac[l], k).indices.tolist()) for l in range(NL)]
        keep = [set(torch.topk(frac[l], minpl).indices.tolist()) for l in range(NL)]
        n = sum(len(k) for k in keep)
        for i in torch.argsort(frac.flatten(), descending=True).tolist():
            if n >= K: break
            l, e = divmod(i, NE)
            if e not in keep[l]: keep[l].add(e); n += 1
        return [sorted(k) for k in keep]

    def apply(c, keep):
        g = c["g"]
        for li, b in enumerate(blocks):
            km = torch.zeros(NE, dtype=torch.bool); km[keep[li]] = True; keepm[li] = km
            ks = keep[li]
            for mi, m in enumerate(("gate_proj", "up_proj", "down_proj")):
                for c0 in range(0, len(ks), 16):
                    kc = ks[c0:c0 + 16]
                    W = torch.stack([ORIG[li][e][mi] for e in kc]).cuda()
                    dq = quant(W, c["eb"], g)
                    for j, e in enumerate(kc):
                        p = getattr(b.experts[e], m).weight; p.data.copy_(dq[j].to(torch.bfloat16).to(p.device))
        for li, l in enumerate(model.model.layers):
            for m in ("q_proj", "k_proj", "v_proj", "o_proj"):
                p = getattr(l.self_attn, m).weight; W = ATT[(li, m)].cuda()
                p.data.copy_(quant(W, c["ab"], g).to(torch.bfloat16).to(p.device) if c["ab"] < 16 else W.to(p.device))
        for p, W0, bits in ((model.model.embed_tokens.weight, EMB, c["mb"]), (model.lm_head.weight, HEAD, c["hb"])):
            if bits < 16:
                for r0 in range(0, W0.shape[0], 32768):
                    p.data[r0:r0 + 32768] = quant(W0[r0:r0 + 32768].cuda(), bits, g).to(torch.bfloat16).to(p.device)
            else: p.data.copy_(W0.to(p.device))
    def restore():  # un-prune/un-quant everything (originals)
        for li, b in enumerate(blocks):
            keepm[li] = None
            for e in range(NE):
                for mi, m in enumerate(("gate_proj", "up_proj", "down_proj")):
                    p = getattr(b.experts[e], m).weight; p.data.copy_(ORIG[li][e][mi].to(p.device))

    for c0 in cfgs["configs"]:
        c = dict(task="INJ", score="mass", calib="full", sel="var", eb=3, ab=6, mb=5, hb=4, g=128, K=None); c.update(c0)
        t1 = time.time(); t = c["task"]
        s = get_stats(t, c["calib"]); sc = s["mass"] if c["score"] == "mass" else s["reap"] / s["cnt"].clamp_min(1)
        K = c["K"] or solve_K(c); K = min(K, NL * NE)
        keep = select(sc, K, c["sel"])
        nb = total_bytes(c, sum(len(k) for k in keep))
        restore(); apply(c, keep)
        ok = score(t, evaluate(t)); d = summarize(t, ok, BASE[t])
        d.update(cfg=c, K=sum(len(k) for k in keep), per_layer=[len(k) for k in keep], bytes=nb, ratio=nb / FP16, secs=time.time() - t1)
        emit(c["name"], d)
        json.dump({"keep": keep}, open(f"{OUT}/{tag}_{c['name']}_keep.json", "w")); vol.commit()
    print("DONE", time.time() - t0, flush=True)

@app.local_entrypoint()
def main(cfg: str):
    run.remote(open(cfg).read())
