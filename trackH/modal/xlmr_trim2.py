# trackH: XLM-R sentiment, clean techniques only: row deletion (vocab trim) + RTN + ideal entropy coding. No Hessian/GPTQ, no training.
# Run: cd trackH/modal && modal run xlmr_trim.py   (Ks run in parallel on L4; calib built once on CPU)
# Data: dev/locked from mteb/amazon_polarity TEST by md5(text)%10 (0-1 LOCKED: dropped at load, never used; 2-5 dev). Calib = TRAIN splits only.
import modal, json
app = modal.App("trackh-xlmr")
img = modal.Image.debian_slim().pip_install("torch", "transformers", "datasets", "numpy", "sentencepiece", "huggingface_hub")
MID = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
TOK = "FacebookAI/xlm-roberta-base"
KS = [20000, 30000]

@app.function(image=img, cpu=8, memory=32768, timeout=3600)
def calib(n_amz=120000):
    import numpy as np
    from datasets import load_dataset
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TOK)
    texts, src = [], {}
    def add(name, it):
        t = [x for x in it if x]; src[name] = len(t); texts.extend(t)
    for name, f in [
        ("amazon_polarity_train", lambda: load_dataset("mteb/amazon_polarity", split=f"train[:{n_amz}]")["text"]),
        ("imdb_train", lambda: load_dataset("stanfordnlp/imdb", split="train")["text"]),
        ("yelp_train", lambda: load_dataset("fancyzhx/yelp_polarity", split="train[:40000]")["text"]),
        ("tweet_eval_sent_train", lambda: load_dataset("cardiffnlp/tweet_eval", "sentiment", split="train")["text"]),
        ("tweet_eval_emoji_train", lambda: load_dataset("cardiffnlp/tweet_eval", "emoji", split="train")["text"]),
        ("multilingual_amazon", lambda: load_dataset("mteb/amazon_reviews_multi", split="train[:30000]")["text"]),
    ]:
        try: add(name, f())
        except Exception as e: print("skip", name, repr(e)[:120])
    print("calib sources", src, "total", len(texts), flush=True)
    cnt = np.zeros(250002, dtype=np.int64)
    for i in range(0, len(texts), 2000):
        enc = tok(texts[i:i+2000], truncation=True, max_length=256)["input_ids"]
        cnt += np.bincount(np.concatenate([np.array(x) for x in enc]), minlength=250002)[:250002]
    return cnt.tolist(), src

@app.function(image=img, gpu="L4", timeout=3600, max_containers=8)
def evalk(K, cnt):
    import copy, hashlib, math, numpy as np, torch, os
    from collections import Counter
    from datasets import load_dataset
    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    ML = 128
    tok = AutoTokenizer.from_pretrained(TOK)
    spm_bytes = os.path.getsize(hf_hub_download(TOK, "sentencepiece.bpe.model"))
    ds = load_dataset("mteb/amazon_polarity", split="test")
    rows = [(t, l) for t, l in zip(ds["text"], ds["label"])]
    rows = [r for r in rows if int(hashlib.md5(r[0].encode()).hexdigest(), 16) % 10 in (2, 3, 4, 5)]  # locked 0,1 dropped
    rows = rows[:6000]
    dev_ids = [tok(t, truncation=True, max_length=ML)["input_ids"] for t, _ in rows]
    y = np.array([l for _, l in rows])
    model = AutoModelForSequenceClassification.from_pretrained(MID).eval().cuda()
    V = model.get_input_embeddings().weight.shape[0]
    cnt = np.array(cnt)
    @torch.no_grad()
    def logits(m, ids):
        order = np.argsort([len(x) for x in ids]); out = []
        for i in range(0, len(ids), 128):
            b = [ids[j] for j in order[i:i+128]]; L = max(map(len, b))
            x = torch.full((len(b), L), 1, dtype=torch.long); a = torch.zeros_like(x)
            for k, s in enumerate(b): x[k, :len(s)] = torch.tensor(s); a[k, :len(s)] = 1
            out.append(m(input_ids=x.cuda(), attention_mask=a.cuda()).logits.float().cpu())
        lg = torch.cat(out); r = torch.empty_like(lg); r[torch.tensor(order)] = lg; return r
    bl = logits(model, dev_ids); base = bl[:, [0, 2]].argmax(1).numpy()
    margin = (bl[:, 2] - bl[:, 0]).abs().numpy(); hard = margin <= np.quantile(margin, 0.25)
    base_acc = float((base == y).mean())
    nparam = sum(p.numel() for p in model.parameters()); base_total = nparam * 4 + spm_bytes
    # vocab selection
    order = np.argsort(-cnt, kind="stable")
    others = [int(i) for i in order[:K] if i >= 4 and cnt[i] > 0]
    keep = [0, 1, 2, 3] + sorted(set(others) | {250001})  # mask token kept; specials stay at idx 0..3 (pad idx 1 => position ids intact)
    kset = set(keep); remap = {t: i for i, t in enumerate(keep)}
    pieces = tok.convert_ids_to_tokens(list(range(V)))
    kp = {pieces[t]: remap[t] for t in keep if t >= 4}; maxlen = max(len(p) for p in kp)
    memo = {}
    def split(t):
        if t in memo: return memo[t]
        p, o, i = pieces[t], [], 0
        while i < len(p):
            for j in range(min(len(p), i + maxlen), i, -1):
                if p[i:j] in kp: o.append(kp[p[i:j]]); i = j; break
            else: o.append(3); i += 1
        memo[t] = o; return o
    def mapids(ids, fb):
        out = []
        for t in ids:
            if t in remap: out.append(remap[t])
            elif fb == "split": out += split(t)
            else: out.append(3)
        return out
    gaps = np.diff(keep[4:]); gc = np.array(list(Counter(gaps.tolist()).values())) / len(gaps)
    id_bytes = float(-(gc * np.log2(gc)).sum()) * len(gaps) / 8
    def rtn(w, bits, g, clip=False, sym=False):
        w2 = w.reshape(-1, g); n = 2**bits - 1
        mn0 = w2.min(1, keepdim=True)[0]; mx0 = w2.max(1, keepdim=True)[0]
        am0 = w2.abs().max(1, keepdim=True)[0]
        best = None
        for r in ([1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6] if clip else [1.0]):
            if sym:
                h = (n // 2); sc = (am0 * r / h).clamp_min(1e-8).half().float(); qq = (w2 / sc).round().clamp(-h, h); dq = qq * sc; q = qq + h; side = 2
            else:
                c = (mx0 + mn0) / 2; hw = (mx0 - mn0) / 2 * r; mn = (c - hw).half().float(); sc = ((2 * hw) / n).clamp_min(1e-8).half().float()
                q = ((w2 - mn) / sc).round().clamp(0, n); dq = q * sc + mn; side = 4
            err = ((dq - w2) ** 2).sum(1, keepdim=True)
            if best is None: best = [err, dq, q]
            else:
                m_ = err < best[0]; best = [torch.where(m_, err, best[0]), torch.where(m_, dq, best[1]), torch.where(m_, q, best[2])]
        return best[1].reshape(w.shape), best[2].long().flatten(), w2.shape[0] * side
    def ent(q, bits):
        c = torch.bincount(q, minlength=2**bits).double(); p = c[c > 0] / c.sum(); return float(-(p * p.log2()).sum()) * q.numel() / 8
    def build(eb, ebg, wb, wg, clip=False, sym=False):
        m = copy.deepcopy(model); E = m.get_input_embeddings().weight.data[keep].clone(); b = id_bytes
        if eb < 16:
            E, q, sb = rtn(E, eb, ebg, clip, sym); b += ent(q, eb) + sb
        else: b += E.numel() * 2
        emb = torch.nn.Embedding(E.shape[0], E.shape[1], padding_idx=1); emb.weight.data = E.contiguous(); m.set_input_embeddings(emb)
        for n, p in m.named_parameters():
            if p is emb.weight: continue
            if p.ndim == 2 and "position_embeddings" not in n and "token_type" not in n and wb < 16:
                Wq, q, sb = rtn(p.data, wb, wg, clip, sym); p.data = Wq; b += ent(q, wb) + sb
            else: p.data = p.data.half().float(); b += p.numel() * 2  # biases/LN/pos emb stored fp16
        return m, b
    res = []
    def run(name, eb, ebg, wb, wg, fb, clip=False, sym=False):
        m, wbytes = build(eb, ebg, wb, wg, clip, sym)
        pr = logits(m, [mapids(x, fb) for x in dev_ids])[:, [0, 2]].argmax(1).numpy()
        r = dict(K=K, kept=len(keep), cfg=name, fb=fb, weights_pct=100 * wbytes / (nparam * 4), total_pct=100 * (wbytes + spm_bytes) / base_total,
                 agree=float((pr == base).mean()), agree_hard=float((pr[hard] == base[hard]).mean()), acc=float((pr == y).mean()), base_acc=base_acc, n=len(y))
        print(json.dumps(r), flush=True); res.append(r); del m; torch.cuda.empty_cache()
    run("fp16 all", 16, 0, 16, 0, "split")
    run("e4g64 enc-fp16", 4, 64, 16, 0, "split"); run("e4g64clip enc-fp16", 4, 64, 16, 0, "split", True)
    run("e-fp16 w6g64", 16, 0, 6, 64, "split"); run("e-fp16 w8g64", 16, 0, 8, 64, "split")
    for eb in (4,):
        for wb, wg in [(8, 128), (7, 64), (7, 128), (6, 64), (6, 128), (5, 64), (5, 128)]:
            run(f"e{eb}g64clip w{wb}g{wg}clip", eb, 64, wb, wg, "split", True)
    for wb, wg in [(8, 64), (7, 64), (7, 128), (6, 64), (6, 128), (5, 64)]:
        run(f"e4g64clip w{wb}g{wg}sym+clip", 4, 64, wb, wg, "split", True, True)
    return res

@app.local_entrypoint()
def main():
    cnt, src = calib.remote()
    out = [r for rs in evalk.map(KS, kwargs={}, order_outputs=True) for r in rs] if False else []
    for rs in evalk.starmap([(K, cnt) for K in KS]): out += rs
    json.dump({"calib": src, "results": out}, open("results_xlmr2.json", "w"), indent=1)
    print("K kept cfg fb weights% total% agree agree_hard acc(base)")
    for r in out: print(r["K"], r["kept"], r["cfg"], r["fb"], f'{r["weights_pct"]:.2f} {r["total_pct"]:.2f} {r["agree"]:.4f} {r["agree_hard"]:.4f} {r["acc"]:.4f}({r["base_acc"]:.4f})')
