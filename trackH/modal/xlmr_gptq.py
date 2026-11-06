# trackH round 3: GPTQ (Hessian from unlabeled calib sentences, no training) on XLM-R sentiment, vocab-trim + emb RTN4 + bias correction.
# Run: modal run xlmr_gptq.py   -> results_gptq.json ; dev/locked from amazon_polarity TEST md5%10 (0,1 locked, dropped); calib/Hessian/bias texts from TRAIN sets only.
import modal, json
app = modal.App("trackh-gptq")
img = modal.Image.debian_slim().pip_install("torch", "transformers", "datasets", "numpy", "sentencepiece", "huggingface_hub", "constriction")
MID = "cardiffnlp/twitter-xlm-roberta-base-sentiment"; TOK = "FacebookAI/xlm-roberta-base"

@app.function(image=img, cpu=8, memory=32768, timeout=3600)
def calib(n_amz=120000, nh=1200, nb=2400):
    import numpy as np, random
    from datasets import load_dataset
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TOK); texts = []
    for f in [lambda: load_dataset("mteb/amazon_polarity", split=f"train[:{n_amz}]")["text"], lambda: load_dataset("stanfordnlp/imdb", split="train")["text"],
              lambda: load_dataset("fancyzhx/yelp_polarity", split="train[:40000]")["text"], lambda: load_dataset("cardiffnlp/tweet_eval", "sentiment", split="train")["text"],
              lambda: load_dataset("cardiffnlp/tweet_eval", "emoji", split="train")["text"]]:
        texts += [x for x in f() if x]
    cnt = np.zeros(250002, dtype=np.int64)
    for i in range(0, len(texts), 2000):
        enc = tok(texts[i:i+2000], truncation=True, max_length=256)["input_ids"]
        cnt += np.bincount(np.concatenate([np.array(x) for x in enc]), minlength=250002)[:250002]
    random.Random(0).shuffle(texts)
    return cnt.tolist(), texts[:nh], texts[nh:nh+nb]

@app.function(image=img, gpu=["L4","A10G","L40S","A100"], timeout=7200, max_containers=12, memory=32768)
def job(cfg, cnt, htexts, btexts):
    import copy, hashlib, lzma, os, numpy as np, torch, time
    from collections import Counter
    from datasets import load_dataset
    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import constriction
    t0 = time.time(); K, bits, g, clip = cfg["K"], cfg["bits"], cfg["g"], cfg.get("clip", False); eb, eg = 4, 64
    tok = AutoTokenizer.from_pretrained(TOK); dv = "cuda"
    spm_bytes = os.path.getsize(hf_hub_download(TOK, "sentencepiece.bpe.model"))
    ds = load_dataset("mteb/amazon_polarity", split="test")
    rows = [(t, l) for t, l in zip(ds["text"], ds["label"]) if int(hashlib.md5(t.encode()).hexdigest(), 16) % 10 in (2, 3, 4, 5)][:6000]
    enc = lambda ts: [tok(t, truncation=True, max_length=128)["input_ids"] for t in ts]
    dev_ids, y = enc([r[0] for r in rows]), np.array([r[1] for r in rows]); h_ids, b_ids = enc(htexts), enc(btexts)
    model = AutoModelForSequenceClassification.from_pretrained(MID).eval().to(dv)
    nparam = sum(p.numel() for p in model.parameters()); base_total = nparam * 4 + spm_bytes
    cnt = np.array(cnt); order = np.argsort(-cnt, kind="stable")
    keep = [0, 1, 2, 3] + sorted(set(int(i) for i in order[:K] if i >= 4 and cnt[i] > 0) | {250001}); remap = {t: i for i, t in enumerate(keep)}
    mp = lambda ids: [[remap.get(t, 3) for t in x] for x in ids]
    gaps = np.diff(keep[4:]); gc = np.array(list(Counter(gaps.tolist()).values())) / len(gaps); id_bytes = float(-(gc * np.log2(gc)).sum()) * len(gaps) / 8
    def batches(ids, bs=64):
        order = np.argsort([len(x) for x in ids]); 
        for i in range(0, len(ids), bs):
            idx = order[i:i+bs]; b = [ids[j] for j in idx]; L = max(map(len, b))
            x = torch.full((len(b), L), 1, dtype=torch.long); a = torch.zeros_like(x)
            for k, s in enumerate(b): x[k, :len(s)] = torch.tensor(s); a[k, :len(s)] = 1
            yield idx, x.to(dv), a.to(dv)
    @torch.no_grad()
    def logits(m, ids):
        out = torch.zeros(len(ids), 3)
        for idx, x, a in batches(ids, 128): out[torch.tensor(idx)] = m(input_ids=x, attention_mask=a).logits.float().cpu()
        return out
    bl = logits(model, dev_ids); base = (bl[:, 2] > bl[:, 0]).numpy(); marg = (bl[:, 2] - bl[:, 0]).abs().numpy(); hard = marg <= np.quantile(marg, 0.25)
    bb = logits(model, b_ids); bbase = (bb[:, 2] > bb[:, 0]).numpy()
    # ---- quant primitives (asym, fp16 scale, integer zero)
    def qparams(w, b_):
        n = 2**b_ - 1; best = None
        for r in ([1.0, 0.95, 0.9, 0.85, 0.8, 0.75] if clip else [1.0]):
            lo, hi = w.min(1, keepdim=True)[0], w.max(1, keepdim=True)[0]; c = (lo + hi) / 2; hw = (hi - lo) / 2 * r; lo, hi = c - hw, c + hw
            s = ((hi - lo).clamp(min=1e-8) / n).half().float().clamp(min=1e-8); z = (-lo / s).round().clamp(0, n)
            e = (((((w / s + z).round().clamp(0, n)) - z) * s - w) ** 2).sum(1, keepdim=True)
            if best is None: best = [e, s, z]
            else:
                m_ = e < best[0]; best = [torch.where(m_, e, best[0]), torch.where(m_, s, best[1]), torch.where(m_, z, best[2])]
        return best[1], best[2]
    def qd(w, s, z, b_):
        n = 2**b_ - 1; q = (w / s + z).round().clamp(0, n); return q, (q - z) * s
    def rtn(W, b_, gg):
        R, C = W.shape; Wg = W.reshape(-1, gg); s, z = qparams(Wg, b_); q, d = qd(Wg, s, z, b_); return d.reshape(R, C), q.reshape(R, C), s, z
    def gptq(W, H, b_, gg, damp=0.01, blk=128):
        W = W.clone().float(); R, C = W.shape; H = H.clone(); dead = torch.diag(H) == 0; H[dead, dead] = 1; W[:, dead] = 0
        perm = torch.argsort(torch.diag(H), descending=True); W = W[:, perm]; H = H[perm][:, perm]; invperm = torch.argsort(perm)
        H += damp * torch.mean(torch.diag(H)) * torch.eye(C, device=dv)
        L = torch.linalg.cholesky(H); Hinv = torch.linalg.cholesky(torch.cholesky_inverse(L), upper=True)
        Q = torch.zeros_like(W); codes = torch.zeros(R, C, device=dv); S = []; Z = []
        for i1 in range(0, C, blk):
            i2 = min(i1 + blk, C); W1 = W[:, i1:i2].clone(); Q1 = torch.zeros_like(W1); Err = torch.zeros_like(W1); Hi = Hinv[i1:i2, i1:i2]
            for i in range(i2 - i1):
                c = i1 + i
                if c % gg == 0:
                    s, z = qparams(W1[:, i:i+gg], b_); S.append(s); Z.append(z)
                w = W1[:, i:i+1]; q, d = qd(w, s, z, b_); Q1[:, i:i+1] = d; codes[:, c:c+1] = q
                err = (w - d) / Hi[i, i]; W1[:, i:] -= err @ Hi[i:i+1, i:]; Err[:, i:i+1] = err
            Q[:, i1:i2] = Q1; W[:, i2:] -= Err @ Hinv[i1:i2, i2:]
        return Q[:, invperm], codes, torch.cat(S, 1), torch.cat(Z, 1)
    # ---- real coder
    stats = dict(code=0, ent=0, side=0, ncodes=0)
    def pack(codes, b_, s, z, extra=0):
        c = codes.flatten().long(); cts = torch.bincount(c, minlength=2**b_).double(); p = cts / cts.sum(); pp = p[p > 0]
        stats["ent"] += float(-(pp * pp.log2()).sum()) * c.numel() / 8
        sym = c.cpu().numpy().astype(np.int32); probs = (cts / cts.sum()).float().cpu().numpy()
        probs = np.maximum(probs, 1e-9); probs /= probs.sum()
        ac = constriction.stream.stack.AnsCoder(); ac.encode_reverse(sym, constriction.stream.model.Categorical(probs.astype(np.float32), perfect=False))
        stats["code"] += len(ac.get_compressed()) * 4 + (2**b_) * 2
        stats["side"] += s.numel() * 2 + len(lzma.compress(z.flatten().to(torch.uint8).cpu().numpy().tobytes())) + extra
    m = copy.deepcopy(model)
    E = m.get_input_embeddings().weight.data[keep].clone(); Ed, ecodes, es, ez = rtn(E, eb, eg); pack(ecodes, eb, es, ez)
    emb = torch.nn.Embedding(E.shape[0], E.shape[1], padding_idx=1); emb.weight.data = Ed.contiguous(); m.set_input_embeddings(emb)
    hm = mp(h_ids); enc_layers = m.roberta.encoder.layer
    for i, l in enumerate(enc_layers):
        for st in [[("q", l.attention.self.query), ("k", l.attention.self.key), ("v", l.attention.self.value)], [("o", l.attention.output.dense)], [("f1", l.intermediate.dense)], [("f2", l.output.dense)]]:
            Hs = {n: torch.zeros(mod.in_features, mod.in_features, device=dv) for n, mod in st}; hooks = []; cur = {}
            def mk(n):
                def h(mod, inp, out):
                    x = inp[0].detach().float().reshape(-1, inp[0].shape[-1]); x = x[cur["mask"]]; Hs[n].add_(x.T @ x)
                return h
            for n, mod in st: hooks.append(mod.register_forward_hook(mk(n)))
            with torch.no_grad():
                for idx, x, a in batches(hm, 32): cur["mask"] = a.reshape(-1).bool(); m(input_ids=x, attention_mask=a)
            for hk in hooks: hk.remove()
            for n, mod in st:
                W = mod.weight.data; d, codes, s, z = gptq(W, Hs[n], bits, g); mod.weight.data = d; pack(codes, bits, s, z, extra=W.shape[1] * 2)
            del Hs
    for mod in [m.classifier.dense, m.classifier.out_proj]:
        if mod.weight.shape[1] % g == 0 and mod.weight.numel() % g == 0:
            d, codes, s, z = rtn(mod.weight.data, bits, g); mod.weight.data = d; pack(codes, bits, s, z)
    qn = {id(p) for mod in [m.classifier.dense, m.classifier.out_proj] for p in [mod.weight]} | {id(emb.weight)}
    qn |= {id(mod.weight) for l in enc_layers for mod in [l.attention.self.query, l.attention.self.key, l.attention.self.value, l.attention.output.dense, l.intermediate.dense, l.output.dense]}
    rest = 0
    for n, p in m.named_parameters():
        if id(p) in qn: continue
        rest += p.numel() * 2; p.data = p.data.half().float()
    wbytes_real = stats["code"] + stats["side"] + rest + id_bytes; wbytes_ent = stats["ent"] + stats["side"] + rest + id_bytes
    # ---- eval + bias correction (bias chosen on unlabeled bias-calib texts vs base preds)
    lq_b = logits(m, mp(b_ids)); dq_b = (lq_b[:, 2] - lq_b[:, 0]).numpy()
    grid = np.arange(-1.5, 1.51, 0.05); ag = [((dq_b + bsv > 0) == bbase).mean() for bsv in grid]; bias = float(grid[int(np.argmax(ag))])
    lq = logits(m, mp(dev_ids)); dq = (lq[:, 2] - lq[:, 0]).numpy()
    r = dict(cfg=cfg, kept=len(keep), real_weights_pct=100 * wbytes_real / (nparam * 4), real_total_pct=100 * (wbytes_real + spm_bytes) / base_total,
             ent_total_pct=100 * (wbytes_ent + spm_bytes) / base_total, real_bytes=wbytes_real + spm_bytes, bias=bias, sec=time.time() - t0, base_acc=float((base == y).mean()))
    for tag, bs_ in (("nobias", 0.0), ("bias", bias)):
        pr = (dq + bs_) > 0; r[f"agree_{tag}"] = float((pr == base).mean()); r[f"hard_{tag}"] = float((pr[hard] == base[hard]).mean()); r[f"acc_{tag}"] = float((pr == y).mean())
    print(json.dumps(r), flush=True); return r

@app.local_entrypoint()
def main():
    cnt, h, b = calib.remote()
    cfgs = [dict(K=30000, bits=bt, g=g, clip=c) for bt, g, c in [(4, 64, False), (4, 64, True), (5, 64, False), (5, 128, False), (5, 64, True), (6, 64, False), (6, 128, False), (6, 64, True)]]
    cfgs += [dict(K=50000, bits=bt, g=g, clip=c) for bt, g, c in [(5, 64, True), (6, 64, False), (6, 64, True)]]
    out = list(job.starmap([(c, cnt, h, b) for c in cfgs]))
    json.dump(out, open("results_gptq.json", "w"), indent=1)
    print("K bits g clip | real_total% ent_total% | agree nobias/bias | hard nobias/bias | acc(base)")
    for r in sorted(out, key=lambda r: r["real_total_pct"]):
        c = r["cfg"]; print(c["K"], c["bits"], c["g"], c.get("clip"), f'| {r["real_total_pct"]:.2f} {r["ent_total_pct"]:.2f} | {r["agree_nobias"]:.4f}/{r["agree_bias"]:.4f} | {r["hard_nobias"]:.4f}/{r["hard_bias"]:.4f} | {r["acc_bias"]:.4f}({r["base_acc"]:.4f}) bias={r["bias"]:.2f}')
