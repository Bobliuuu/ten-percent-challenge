# trackH: vocab trim + RTN + entropy bytes for a big-embedding sentiment classifier.
# usage: python run.py MODEL_ID [maxlen]. Labels: positive/negative only (neutral mapped by argmax over the 2 polar logits).
# Data: mteb/amazon_polarity TEST parquet (never SST-2). md5(text)%10: 0-1 LOCKED (never loaded for any computation), 2-5 DEV, 6-9 CALIB.
import sys, hashlib, glob, math, json, numpy as np, pandas as pd, torch
from collections import Counter
from transformers import AutoTokenizer, AutoModelForSequenceClassification
mid = sys.argv[1]; ML = int(sys.argv[2]) if len(sys.argv) > 2 else 128
dev_ = 'mps'
f = glob.glob(glob.os.path.expanduser('~/.cache/huggingface/hub/datasets--mteb--amazon_polarity/snapshots/*/data/test-*.parquet'))[0]
df = pd.read_parquet(f); pass
df['h'] = df.text.map(lambda s: int(hashlib.md5(s.encode()).hexdigest(), 16) % 10)
df = df[df.h >= 2]  # locked rows (h<2) dropped immediately, never touched
dev = df[df.h <= 5].head(2000); cal = df[df.h >= 6].head(4000)
tok = AutoTokenizer.from_pretrained('FacebookAI/xlm-roberta-base' if 'xlm-roberta' in mid else mid)
model = AutoModelForSequenceClassification.from_pretrained(mid).eval()
lab = {v.lower(): k for k, v in model.config.id2label.items()}
pos, neg = [lab[k] for k in lab if k.startswith('pos') or k == 'label_2'][0], [lab[k] for k in lab if k.startswith('neg') or k == 'label_0'][0]
print('labels', model.config.id2label, pos, neg)
def enc(texts): return [tok(t, truncation=True, max_length=ML)['input_ids'] for t in texts]
dev_ids, cal_ids = enc(dev.text.tolist()), enc(cal.text.tolist())
ydev = dev.label.values  # 1=positive
@torch.no_grad()
def predict(m, ids, remap=None):
    m = m.to(dev_); out = []
    order = np.argsort([len(x) for x in ids])
    for i in range(0, len(ids), 64):
        b = [ids[j] for j in order[i:i+64]]; L = max(map(len, b))
        if remap is not None: b = [[remap.get(t, remap['unk']) for t in x] for x in b]
        x = torch.full((len(b), L), model.config.pad_token_id, dtype=torch.long); a = torch.zeros_like(x)
        for k, s in enumerate(b): x[k, :len(s)] = torch.tensor(s); a[k, :len(s)] = 1
        lg = m(input_ids=x.to(dev_), attention_mask=a.to(dev_)).logits.float().cpu()
        out.append(lg[:, [neg, pos]].argmax(1))
    p = torch.cat(out).numpy(); r = np.empty_like(p); r[order] = p; m.cpu(); return r
base = predict(model, dev_ids)
print('BASE dev acc (amazon_polarity, never-trained-on?)', (base == ydev).mean())
# vocab trim
cnt = Counter(t for x in cal_ids for t in x)
special = set(tok.all_special_ids)
V = model.get_input_embeddings().weight.shape[0]
def rtn(w, bits, g):  # symmetric-ish asym per group along last dim; returns dequant, ints, scale count
    sh = w.shape; w2 = w.reshape(-1, g) if g else w.reshape(1, -1)
    mn, mx = w2.min(1, keepdim=True)[0], w2.max(1, keepdim=True)[0]
    n = 2**bits - 1; sc = ((mx - mn) / n).clamp_min(1e-8); q = ((w2 - mn) / sc).round().clamp(0, n)
    return (q * sc + mn).reshape(sh), q.to(torch.int64).flatten(), w2.shape[0]
def ent_bytes(q, bits):
    c = torch.bincount(q, minlength=2**bits).double(); p = c[c > 0] / c.sum()
    return float(-(p * p.log2()).sum()) * q.numel() / 8
base_bytes = sum(p.numel() for p in model.parameters()) * 4
print('base fp32 bytes', base_bytes, 'emb share', model.get_input_embeddings().weight.numel() * 4 / base_bytes)
res = []
def run(name, topk, eb, ebg, wb, wg):
    import copy; m = copy.deepcopy(model)
    keep = sorted(special | {t for t, _ in cnt.most_common(topk)} if topk else range(V)); remap = None
    E = m.get_input_embeddings().weight.data
    byt = 0; byt_ent = 0
    if topk:
        remap = {t: i for i, t in enumerate(keep)}; remap['unk'] = remap[tok.unk_token_id]
        E = E[keep].clone()
        d = np.diff(keep); idxb = len(keep) * 1.5  # delta ids ~1.5B each (conservative-ish; entropy of gaps below)
        gc = Counter(d.tolist()); pg = np.array(list(gc.values())) / len(d); idxb = float(-(pg * np.log2(pg)).sum()) * len(d) / 8
        byt += idxb; byt_ent += idxb
    nE = E.numel()
    if eb < 16:
        Eq, q, ns = rtn(E.float(), eb, ebg); sb = ns * 4  # fp16 scale+min
        byt += nE * eb / 8 + sb; byt_ent += ent_bytes(q, eb) + sb; E = Eq
    else: byt += nE * 4; byt_ent += nE * 4
    # replace embedding: new small Embedding
    emb = torch.nn.Embedding(E.shape[0], E.shape[1]); emb.weight.data = E.contiguous()
    m.set_input_embeddings(emb); m.config.vocab_size = E.shape[0]
    if hasattr(m, 'roberta') and hasattr(m, 'lm_head'): pass
    emb_ids = {id(emb.weight)}
    for n, p in m.named_parameters():
        if id(p) in emb_ids: continue
        if p.ndim == 2 and wb < 16 and 'embeddings' not in n:
            Wq, q, ns = rtn(p.data.float(), wb, wg); p.data = Wq.reshape(p.shape); sb = ns * 4
            byt += p.numel() * wb / 8 + sb; byt_ent += ent_bytes(q, wb) + sb
        else: byt += p.numel() * 4; byt_ent += p.numel() * 4  # fp32 (biases, LN, position emb)
    pr = predict(m, dev_ids, remap)
    r = dict(name=name, topk=topk, kept=len(keep), bytes_pct=100 * byt / base_bytes, ent_bytes_pct=100 * byt_ent / base_bytes,
             agree=float((pr == base).mean()), acc=float((pr == ydev).mean()))
    print(json.dumps(r), flush=True); res.append(r)
run('trim only fp32', 30000, 16, 0, 16, 0)
run('trim5k fp32', 5000, 16, 0, 16, 0)
for k in (30000, 12000, 5000):
    run(f'trim{k} emb4g64 w4g64', k, 4, 64, 4, 64)
run('trim5k emb4g64 w8g128', 5000, 4, 64, 8, 128)
run('trim5k emb4g64 w6g64', 5000, 4, 64, 6, 64)
run('trim12k emb4g64 w6g64', 12000, 4, 64, 6, 64)
run('trim5k emb3g64 w4g64', 5000, 3, 64, 4, 64)
json.dump(res, open(f'results_{mid.replace("/","_")}.json', 'w'), indent=1)
