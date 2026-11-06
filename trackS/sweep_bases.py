# trackS: which base (embedding-heavy multilingual classifier) + English-ish task gets to ~10% bytes with >=99.8% agreement?
# Per base: REAL trimmed tokenizer (edited sentencepiece proto / trimmed WordPiece vocab.txt), top-K vocab by calibration counts,
# RTN at several (emb bits, encoder bits), real rANS coder (constriction) byte counts, agreement vs the fp32 base on a hash-fixed dev split.
# Splits: md5(text)%10 : 0,1 = LOCKED (dropped at load, never touched); 2-5 = dev (all choices); 6-9 unused. Calibration = TRAIN splits only.
# Run: cd trackS && modal run sweep_bases.py [--only minilm_nli,...]
import modal, json, sys
app = modal.App("trackS-bases")
img = modal.Image.debian_slim().pip_install("torch", "transformers", "datasets", "numpy", "sentencepiece", "protobuf", "huggingface_hub", "constriction")

SPECS = {
 "minilm_nli": dict(mid="MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli", fam="spm", layout="xlmr", spm="sentencepiece.bpe.model", task="nli", Ks=[8000, 15000, 25000, 40000, 60000]),
 "mdeberta_nli": dict(mid="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli", fam="spm", layout="deberta", spm="spm.model", task="nli", Ks=[8000, 15000, 25000, 40000, 60000]),
 "xlmrL_xnli": dict(mid="joeddav/xlm-roberta-large-xnli", fam="spm", layout="xlmr", spm="sentencepiece.bpe.model", task="nli", Ks=[15000, 25000, 40000]),
 "distilmbert_sent": dict(mid="lxyuan/distilbert-base-multilingual-cased-sentiments-student", fam="wp", task="sent3", Ks=[8000, 15000, 25000, 40000, 60000]),
 "nlptown_stars": dict(mid="nlptown/bert-base-multilingual-uncased-sentiment", fam="wp", task="stars", Ks=[8000, 15000, 25000, 40000, 60000]),
 "papluca_langid": dict(mid="papluca/xlm-roberta-base-language-detection", fam="spm", layout="xlmr", spm="sentencepiece.bpe.model", task="langid", Ks=[20000, 40000, 70000, 100000]),
 "textdetox_tox": dict(mid="textdetox/xlmr-large-toxicity-classifier", fam="spm", layout="xlmr", spm="sentencepiece.bpe.model", task="tox", Ks=[8000, 15000, 25000, 40000, 60000]),
}
# (emb bits, emb group, enc bits, enc group)
CFGS = [(16, 0, 16, 0), (8, 128, 8, 128), (6, 64, 8, 128), (6, 64, 6, 128), (4, 64, 6, 128), (5, 64, 5, 64), (4, 64, 5, 64), (4, 64, 4, 64)]
N_DEV, N_STRESS, ML = 8000, 3000, 192

@app.function(image=img, gpu="L4", cpu=8, memory=49152, timeout=7200, max_containers=10)
def run_model(key):
    import copy, hashlib, lzma, os, shutil, time
    import numpy as np, torch
    from datasets import load_dataset
    from huggingface_hub import hf_hub_download, snapshot_download
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    S = SPECS[key]; mid = S["mid"]; task = S["task"]; t0 = time.time()
    def log(*a): print(f"[{key} {time.time()-t0:.0f}s]", *a, flush=True)
    md5m = lambda s: int(hashlib.md5(s.encode()).hexdigest(), 16)

    model0 = AutoModelForSequenceClassification.from_pretrained(mid).eval()
    cfg = model0.config; l2i = {str(v).lower(): int(k) for k, v in cfg.id2label.items()}
    nparam = sum(p.numel() for p in model0.parameters())
    emb_n = model0.get_input_embeddings().weight.numel()
    V = model0.get_input_embeddings().weight.shape[0]
    log("params", nparam, "emb", emb_n, "share", emb_n / nparam, "labels", cfg.id2label)

    # ---------------- data ----------------
    def L(name, *a, **k):
        try: return load_dataset(name, *a, **k)
        except Exception as e: log("SKIP", name, a, repr(e)[:150]); return None
    def col(name, cfg_, split, c, n=None):
        d = L(name, cfg_, split=split) if cfg_ else L(name, split=split)
        if d is None: return []
        d = d.select(range(min(n, len(d)))) if n else d
        return d[c]
    calib = []
    def addc(xs, tag):
        xs = [x for x in xs if isinstance(x, str) and len(x) > 3]; calib.extend(xs); log("calib", tag, len(xs))
    rows, stress = [], []   # (a, b, y, allowed)
    def general():
        addc(col("Salesforce/wikitext", "wikitext-103-raw-v1", "train", "text", 300000), "wikitext103")
        addc(col("fancyzhx/ag_news", None, "train", "text", 120000), "agnews")
    if task == "nli":
        general()
        m = L("nyu-mll/multi_nli", split="train[:200000]")
        if m: addc(m["premise"] + m["hypothesis"], "mnli_train")
        s = L("stanfordnlp/snli", split="train[:150000]")
        if s: addc(s["premise"] + s["hypothesis"], "snli_train")
        addc(col("Yelp/yelp_review_full", None, "train", "text", 30000), "yelp_train")
        mp = [l2i.get("entailment", 0), l2i.get("neutral", 1), l2i.get("contradiction", 2)]
        for sp in ["validation_matched", "validation_mismatched"]:
            d = L("nyu-mll/multi_nli", split=sp)
            if d: rows += [(p, h, mp[l], None) for p, h, l in zip(d["premise"], d["hypothesis"], d["label"])]
        d = L("stanfordnlp/snli", split="test")
        if d: rows += [(p, h, mp[l], None) for p, h, l in zip(d["premise"], d["hypothesis"], d["label"]) if l in (0, 1, 2)]
        for r in ["test_r1", "test_r2", "test_r3"]:   # STRESS: adversarial NLI, never in calib
            d = L("facebook/anli", split=r)
            if d: stress += [(p, h, mp[l], None) for p, h, l in zip(d["premise"], d["hypothesis"], d["label"])]
    elif task == "sent3":
        general()
        addc(col("mteb/amazon_polarity", None, "train", "text", 100000), "amz_train")
        addc(col("fancyzhx/yelp_polarity", None, "train", "text", 60000), "yelp_train")
        addc(col("cardiffnlp/tweet_eval", "sentiment", "train", "text"), "tweet_train")
        addc(col("cardiffnlp/tweet_eval", "emoji", "train", "text", 50000), "tweet_emoji")
        n, p_, u = l2i["negative"], l2i["positive"], l2i["neutral"]
        d = L("cardiffnlp/tweet_eval", "sentiment", split="test")
        if d: rows += [(t, None, [n, u, p_][l], None) for t, l in zip(d["text"], d["label"])]
        for nm in ["mteb/amazon_polarity", "fancyzhx/yelp_polarity"]:
            d = L(nm, split="test")
            if d:
                d = d.select(range(min(15000, len(d)))); rows += [(t, None, [n, p_][l], (n, p_)) for t, l in zip(d["text"], d["label"])]
        d = L("stanfordnlp/imdb", split="test")   # STRESS: long movie reviews, not in calib
        if d: stress += [(t, None, [n, p_][l], (n, p_)) for t, l in zip(d["text"], d["label"])]
    elif task == "stars":
        general()
        addc(col("Yelp/yelp_review_full", None, "train", "text", 150000), "yelp_train")
        addc(col("mteb/amazon_polarity", None, "train", "text", 60000), "amz_train")
        d = L("Yelp/yelp_review_full", split="test")
        if d: rows += [(t, None, l, None) for t, l in zip(d["text"], d["label"])]
        d = L("stanfordnlp/imdb", split="test")   # STRESS
        if d: stress += [(t, None, 0, None) for t in d["text"]]
    elif task == "tox":
        general()
        d = L("google/civil_comments", split="train[:250000]")
        if d: addc(d["text"], "civil_train")
        addc(col("cardiffnlp/tweet_eval", "sentiment", "train", "text"), "tweet_train")
        d = L("google/civil_comments", split="test")
        if d: rows += [(t, None, int(v >= 0.5), None) for t, v in zip(d["text"], d["toxicity"])]
        d = L("cardiffnlp/tweet_eval", "offensive", split="test")   # STRESS
        if d: stress += [(t, None, l, None) for t, l in zip(d["text"], d["label"])]
    elif task == "langid":
        d = L("papluca/language-identification", split="train")
        if d: addc(d["text"], "papluca_train")
        d = L("papluca/language-identification", split="test")
        if d: rows += [(t, None, l2i[lab.lower()], None) for t, lab in zip(d["text"], d["labels"])]
        for lg in ["ar", "bg", "de", "el", "en", "es", "fr", "hi", "ru", "sw", "th", "tr", "ur", "vi", "zh"]:   # STRESS: XNLI premises
            if lg in l2i:
                d = L("facebook/xnli", lg, split="test[:400]")
                if d: stress += [(t, None, l2i[lg], None) for t in d["premise"]]
    key_of = lambda r: (r[0] + "\x1f" + (r[1] or ""))
    def split_dev(rs, n):
        rs = [r for r in rs if md5m(key_of(r)) % 10 in (2, 3, 4, 5)]      # 0,1 LOCKED dropped, 6-9 unused
        rs.sort(key=lambda r: md5m("s" + key_of(r))); return rs[:n]
    n_before = len(rows); dev = split_dev(rows, N_DEV); stressd = split_dev(stress, N_STRESS)
    log("dev", len(dev), "of", n_before, "stress", len(stressd), "calib texts", len(calib))
    y = np.array([r[2] for r in dev])

    # ---------------- tokenizers ----------------
    work = f"/tmp/{key}"; os.makedirs(work, exist_ok=True)
    if S["fam"] == "spm":
        import sentencepiece as spm
        from sentencepiece import sentencepiece_model_pb2 as pb
        try: spm_path = hf_hub_download(mid, S["spm"])
        except Exception: spm_path = hf_hub_download("FacebookAI/xlm-roberta-base", "sentencepiece.bpe.model")
        lay = S["layout"]
        mp0 = pb.ModelProto(); mp0.ParseFromString(open(spm_path, "rb").read()); NP = len(mp0.pieces)
        if lay == "xlmr": CLS, PADID, SEP, UNK, OFF = 0, 1, 2, 3, 1
        else: CLS, PADID, SEP, UNK, OFF = 1, 0, 2, 3, 0
        def m_id(sp_id):
            if lay == "xlmr": return 3 if sp_id == 0 else sp_id + 1
            return sp_id
        class Tk:
            def __init__(self, path): self.sp = spm.SentencePieceProcessor(model_file=path)
            def enc_many(self, texts):
                return [[m_id(i) for i in ids] for ids in self.sp.encode([t[:3000] for t in texts])]
            def build(self, a, b):
                if b is None: return [CLS] + a + [SEP]
                if lay == "xlmr": return [CLS] + a + [SEP, SEP] + b + [SEP]
                return [CLS] + a + [SEP] + b + [SEP]
            def encode_rows(self, rs):
                A = self.enc_many([r[0] for r in rs]); bi = [i for i, r in enumerate(rs) if r[1] is not None]
                B = [None] * len(rs)
                for i, ids in zip(bi, self.enc_many([rs[i][1] for i in bi])): B[i] = ids
                out = []
                for a, b in zip(A, B):
                    lim = ML - (4 if b is not None else 2); a = list(a); b = list(b) if b is not None else None
                    while len(a) + (len(b) if b is not None else 0) > lim:
                        if b is not None and len(b) > len(a): b.pop()
                        else: a.pop()
                    out.append(self.build(a, b))
                return out
        tk0 = Tk(spm_path); base_tok_bytes = os.path.getsize(spm_path); base_unk_id = UNK
        try:
            hf = AutoTokenizer.from_pretrained(mid); sm = dev[:300]
            ref = [hf(r[0], r[1], truncation=True, max_length=ML)["input_ids"] if r[1] else hf(r[0], truncation=True, max_length=ML)["input_ids"] for r in sm]
            mine = tk0.encode_rows(sm); log("tokenizer parity vs HF: mismatches", sum(a != b for a, b in zip(ref, mine)), "/", len(sm))
        except Exception as e: log("parity check skipped", repr(e)[:100])
        cnt = np.zeros(V, dtype=np.int64)
        for i in range(0, len(calib), 20000):
            flat = np.fromiter((t for ids in tk0.enc_many(calib[i:i + 20000]) for t in ids), dtype=np.int64)
            cnt += np.bincount(flat, minlength=V)[:V]
        log("counted", int(cnt.sum()), "tokens; base tok bytes", base_tok_bytes)
        mask_old = V - 1 if lay == "xlmr" else 250101
        ascii_keep = {i for i, p in enumerate(mp0.pieces) if p.type == 1 and len(p.piece.replace("▁", "")) == 1 and all(ord(c) < 128 for c in p.piece)}
        def make_trim(K):
            order = np.argsort(-cnt, kind="stable"); sel = set()
            for t in order:
                if len(sel) >= K or cnt[t] == 0: break
                sp_id = int(t) - OFF
                if lay == "xlmr" and int(t) < 4: continue
                if sp_id < 0 or sp_id >= NP or mp0.pieces[sp_id].type != 1: continue
                sel.add(sp_id)
            sel |= ascii_keep
            keep_sp = [i for i, p in enumerate(mp0.pieces) if p.type != 1 or i in sel]
            mp = pb.ModelProto(); mp.CopyFrom(mp0); del mp.pieces[:]
            for i in keep_sp: mp.pieces.add().CopyFrom(mp0.pieces[i])
            mp.trainer_spec.vocab_size = len(keep_sp)
            path = f"{work}/trim_{K}.model"; data = mp.SerializeToString(); open(path, "wb").write(data)
            if lay == "xlmr": rows_old = [0, 1, 2, 3] + [m_id(i) for i in keep_sp if i >= 3] + [mask_old]
            else: rows_old = [m_id(i) for i in keep_sp] + [mask_old]
            return Tk(path), rows_old, len(data), len(lzma.compress(data, preset=9 | lzma.PRESET_EXTREME))
        pad_new = lambda tk: PADID
    else:  # WordPiece
        d0 = snapshot_download(mid, allow_patterns=["*.txt", "*.json"], ignore_patterns=["*onnx*", "model*.json", "*index.json"])
        tok0 = AutoTokenizer.from_pretrained(mid)
        vocab = [l.rstrip("\n") for l in open(f"{d0}/vocab.txt", encoding="utf-8")]
        base_tok_bytes = os.path.getsize(f"{d0}/vocab.txt"); base_unk_id = tok0.unk_token_id
        class Tk:
            def __init__(self, tok): self.t = tok
            def encode_rows(self, rs): return self.t([r[0][:3000] for r in rs], truncation=True, max_length=ML)["input_ids"]
        tk0 = Tk(tok0); PADID = tok0.pad_token_id
        cnt = np.zeros(V, dtype=np.int64)
        for i in range(0, len(calib), 20000):
            enc = tok0([t[:3000] for t in calib[i:i + 20000]], truncation=True, max_length=256)["input_ids"]
            cnt += np.bincount(np.fromiter((t for ids in enc for t in ids), dtype=np.int64), minlength=V)[:V]
        log("counted", int(cnt.sum()))
        special = set(tok0.all_special_ids)
        ascii_keep = {i for i, w in enumerate(vocab) if (len(w) == 1 or (w.startswith("##") and len(w) == 3)) and all(ord(c) < 128 for c in w)}
        def make_trim(K):
            order = np.argsort(-cnt, kind="stable"); sel = set(special) | ascii_keep; base_n = len(sel)
            for t in order:
                if len(sel) >= K + base_n or cnt[t] == 0: break
                sel.add(int(t))
            keep = sorted(sel); nd = f"{work}/wp_{K}"; shutil.rmtree(nd, ignore_errors=True); os.makedirs(nd)
            for f in os.listdir(d0):
                if f.endswith(".json") and f != "tokenizer.json" and "config" in f or f == "special_tokens_map.json": shutil.copy(f"{d0}/{f}", nd)
            data = "\n".join(vocab[i] for i in keep) + "\n"; open(f"{nd}/vocab.txt", "w", encoding="utf-8").write(data)
            tkw = AutoTokenizer.from_pretrained(nd); b = data.encode()
            return Tk(tkw), keep, len(b), len(lzma.compress(b, preset=9 | lzma.PRESET_EXTREME))
        pad_new = lambda tk: tk.t.pad_token_id
    # ---------------- model eval ----------------
    dev_ids0 = tk0.encode_rows(dev); st_ids0 = tk0.encode_rows(stressd)
    model0 = model0.cuda()
    @torch.no_grad()
    def logits(m, ids, pad):
        order = np.argsort([len(x) for x in ids]); out = []; bs = 64
        for i in range(0, len(ids), bs):
            b = [ids[j] for j in order[i:i + bs]]; Lm = max(map(len, b))
            x = torch.full((len(b), Lm), pad, dtype=torch.long); a = torch.zeros_like(x)
            for k, s in enumerate(b): x[k, :len(s)] = torch.tensor(s); a[k, :len(s)] = 1
            out.append(m(input_ids=x.cuda(), attention_mask=a.cuda()).logits.float().cpu())
        lg = torch.cat(out); r = torch.empty_like(lg); r[torch.tensor(order)] = lg; return r
    def decide(lg, rs):
        p = lg.argmax(1).numpy().copy()
        for i, r in enumerate(rs):
            if r[3] is not None: p[i] = r[3][int(lg[i, list(r[3])].argmax())]
        return p
    bl = logits(model0, dev_ids0, PADID); base = decide(bl, dev)
    top2 = bl.topk(2, 1).values; margin = (top2[:, 0] - top2[:, 1]).numpy(); hard = margin <= np.quantile(margin, 0.25)
    has_st = len(stressd) > 0
    sbase = decide(logits(model0, st_ids0, PADID), stressd) if has_st else None
    base_acc = float((base == y).mean()); log("BASE dev acc", base_acc, "n", len(y), "stress n", len(stressd))
    base_total = nparam * 4 + base_tok_bytes
    base_state = {k: v.detach().cpu().clone() for k, v in model0.state_dict().items()}
    emb_name = [n for n, p in model0.named_parameters() if p is model0.get_input_embeddings().weight][0]

    # ---------------- quant + real coder ----------------
    def rtn(w, bits, g):
        w2 = w.reshape(-1, g) if w.numel() % g == 0 else w.reshape(1, -1)
        mn = w2.min(1, keepdim=True)[0]; mx = w2.max(1, keepdim=True)[0]
        n = 2 ** bits - 1; sc = ((mx - mn) / n).clamp_min(1e-8); q = ((w2 - mn) / sc).round().clamp(0, n)
        mn16, sc16 = mn.half().float(), sc.half().float()
        return (q * sc16 + mn16).reshape(w.shape), q.to(torch.uint8).flatten(), w2.shape[0] * 4
    import constriction
    verified = [False]
    def rans_bytes(q, bits):
        qn = q.cpu().numpy().astype(np.int32); c = np.bincount(qn, minlength=2 ** bits).astype(np.float64)
        c[c == 0] = 1e-3; pr = c / c.sum()
        mdl = constriction.stream.model.Categorical(pr, perfect=False)
        cd = constriction.stream.stack.AnsCoder(); cd.encode_reverse(qn, mdl); comp = cd.get_compressed()
        if not verified[0]:
            dd = constriction.stream.stack.AnsCoder(comp); assert (dd.decode(mdl, len(qn)) == qn).all(); verified[0] = True; log("rANS roundtrip OK")
        return comp.nbytes + (2 ** bits) * 2
    def build(rows_old, eb, ebg, wb, wg):
        m = copy.deepcopy(model0).cpu()
        E = base_state[emb_name][torch.tensor(rows_old)].clone().cuda().float(); b_pack = b_ans = 0.0
        if eb < 16:
            Eq, q, sb = rtn(E, eb, ebg); b_pack += q.numel() * eb / 8 + sb; b_ans += rans_bytes(q, eb) + sb; E = Eq
        else: b_pack += E.numel() * 2; b_ans += E.numel() * 2; E = E.half().float()
        emb = torch.nn.Embedding(E.shape[0], E.shape[1], padding_idx=PADID); emb.weight.data = E.cpu().contiguous(); m.set_input_embeddings(emb)
        ew = emb.weight
        for n, p in m.named_parameters():
            if p is ew: continue
            if p.ndim == 2 and "position_embeddings" not in n and "token_type" not in n and "rel_embeddings" not in n and wb < 16:
                Wq, q, sb = rtn(p.data.cuda().float(), wb, wg); p.data = Wq.cpu(); b_pack += q.numel() * wb / 8 + sb; b_ans += rans_bytes(q, wb) + sb
            else: p.data = p.data.half().float(); b_pack += p.numel() * 2; b_ans += p.numel() * 2
        return m.cuda(), b_pack, b_ans
    results = []
    for K in S["Ks"]:
        tkK, rows_old, tb_raw, tb_xz = make_trim(K)
        dev_ids = tkK.encode_rows(dev); st_ids = tkK.encode_rows(stressd); padK = pad_new(tkK)
        unk_new = 3 if S["fam"] == "spm" else tkK.t.unk_token_id
        unk_rate = float(np.mean([sum(t == unk_new for t in x) / max(1, len(x)) for x in dev_ids]))
        unk0 = float(np.mean([sum(t == base_unk_id for t in x) / max(1, len(x)) for x in dev_ids0]))
        len_ratio = float(np.mean([len(a) / max(1, len(b)) for a, b in zip(dev_ids, dev_ids0)]))
        log(f"K={K} rows={len(rows_old)} tok_raw={tb_raw} tok_xz={tb_xz} unk_rate={unk_rate:.5f} (base {unk0:.5f}) seqlen_ratio={len_ratio:.3f}")
        for (eb, ebg, wb, wg) in CFGS:
            m, bp, ba = build(rows_old, eb, ebg, wb, wg)
            pr = decide(logits(m, dev_ids, padK), dev); prs = decide(logits(m, st_ids, padK), stressd) if has_st else None
            r = dict(model=key, K=K, kept=len(rows_old), eb=eb, wb=wb, tok_raw=tb_raw, tok_xz=tb_xz, nparam=nparam, emb_share=emb_n / nparam,
                     pct_packed=100 * (bp + tb_raw) / base_total, pct_coded=100 * (ba + tb_xz) / base_total, pct_w_only=100 * ba / (nparam * 4),
                     agree=float((pr == base).mean()), agree_hard=float((pr[hard] == base[hard]).mean()), disagree_n=int((pr != base).sum()),
                     stress_agree=float((prs == sbase).mean()) if has_st else None, acc=float((pr == y).mean()), base_acc=base_acc, n=len(y),
                     unk_rate=unk_rate, unk_base=unk0, seqlen_ratio=len_ratio)
            print("RES " + json.dumps(r), flush=True); results.append(r); del m; torch.cuda.empty_cache()
    return results

@app.local_entrypoint()
def main(only: str = ""):
    keys = [k for k in SPECS if not only or k in only.split(",")]
    out = []
    for rs in run_model.map(keys, return_exceptions=True):
        if isinstance(rs, Exception): print("FAILED", repr(rs)[:800])
        else: out += rs
    json.dump(out, open("results_bases.json", "w"), indent=1)
    print("model K eb wb pct_coded pct_packed agree stress hard acc(base)")
    for r in out: print(r["model"], r["K"], r["eb"], r["wb"], f'{r["pct_coded"]:.2f} {r["pct_packed"]:.2f} {r["agree"]:.4f} {r["stress_agree"]} {r["agree_hard"]:.4f} {r["acc"]:.4f}({r["base_acc"]:.4f})')
