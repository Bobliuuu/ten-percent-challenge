"""Deletion carrier (variant A): feedback = a binary mask over the EXISTING per-128-group fp16 scales of the 1-bit Bonsai MLP matrices of the last layers.
Zeroing a group scale deletes the whole group (128 weights). No value is changed, nothing new is stored but the bitmap of deleted groups.
Learned by gradient on <=25 annotations (CE on label tokens + EOS) + KL retention on the shipped UNLABELED calibration prompts + sparsity penalty,
hard forward mask / straight-through backward (inspired by Deletion Tuning, arXiv 2611.10866)."""
import json, random, zlib, struct, torch, torch.nn as nn, torch.nn.functional as F
G = 128
MODS = {"down": "down_proj", "gate": "gate_proj", "up": "up_proj"}

class MaskedLinearFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w0, m):
        D, Fi = w0.shape; w = (w0.view(D, Fi // G, G) * m.to(w0.dtype)[:, :, None]).view(D, Fi)
        ctx.save_for_backward(x, w, w0); return x @ w.T
    @staticmethod
    def backward(ctx, go):
        x, w, w0 = ctx.saved_tensors; D, Fi = w0.shape
        gx = go @ w; gW = go.reshape(-1, D).T @ x.reshape(-1, Fi)
        gm = (gW.float() * w0.float()).view(D, Fi // G, G).sum(-1)
        return gx, None, gm

class DelLinear(nn.Module):
    """wraps an existing nn.Linear. modes: la (training, hard mask from logits la<0, budget-capped), w_eff (baked eval), none (original)."""
    def __init__(self, lin):
        super().__init__(); self.lin = lin; D, Fi = lin.weight.shape; self.la = None; self.w_eff = None; self.shape = (D, Fi // G)
    def forward(self, x):
        if self.w_eff is not None: return F.linear(x, self.w_eff)
        if self.la is not None: return MaskedLinearFn.apply(x, self.lin.weight, self.cur_mask)
        return self.lin(x)

def attach(m, layers, mods):
    out = {}
    for li in layers:
        for mk in mods:
            parent = m.model.layers[li].mlp; name = MODS[mk]; cur = getattr(parent, name)
            if not isinstance(cur, DelLinear): cur = DelLinear(cur); setattr(parent, name, cur)
            out[(li, mk)] = cur
    return out

def n_groups(dl): return dl.shape[0] * dl.shape[1]

def hard_masks(mods, budget_frac=None, budget=None):
    """deleted = la<0, capped to the `budget` lowest logits globally. returns dict key->bool deleted [D,Gn]"""
    tot = sum(n_groups(d) for d in mods.values()); k = budget if budget is not None else int(budget_frac * tot)
    allla = torch.cat([d.la.detach().flatten() for d in mods.values()]); dele = allla < 0
    if int(dele.sum()) > k:
        th = torch.topk(allla, k, largest=False).values.max(); dele = allla <= th; dele &= allla < 0
        if int(dele.sum()) > k:   # ties
            idx = torch.nonzero(dele).flatten(); sel = idx[torch.argsort(allla[idx])[:k]]; dele = torch.zeros_like(dele); dele[sel] = True
    out = {}; o = 0
    for key, d in mods.items(): n = n_groups(d); out[key] = dele[o:o + n].view(d.shape); o += n
    return out

def set_train_masks(mods, budget_frac):
    dm = hard_masks(mods, budget_frac)
    for key, d in mods.items(): d.cur_mask = (~dm[key]).float() + (d.la - d.la.detach())
    return dm

def bake(mods, dm):
    for key, d in mods.items():
        w0 = d.lin.weight.data; D, Fi = w0.shape; d.w_eff = (w0.view(D, Fi // G, G) * (~dm[key]).to(w0.dtype)[:, :, None]).view(D, Fi); d.la = None
def unbake(mods):
    for d in mods.values(): d.w_eff = None; d.la = None

def ids_of(dm):
    """sorted flat indices of deleted groups per (layer,module)"""
    return {f"{k[0]}.{k[1]}": torch.nonzero(v.flatten()).flatten().cpu().tolist() for k, v in dm.items() if v.any()}

def _varint(n):
    b = bytearray()
    while True:
        x = n & 0x7f; n >>= 7
        if n: b.append(x | 0x80)
        else: b.append(x); return bytes(b)
def encode(ids, shapes):
    """compressed bitmap file: JSON-free binary. per key: name, count, zlib(delta-varint list) vs zlib(packbits bitmap), smaller one. returns bytes"""
    import numpy as np
    out = bytearray(b"DEL1")
    for name in sorted(ids):
        L = ids[name]; ng = shapes[name]
        dv = bytearray(); prev = -1
        for i in L: dv += _varint(i - prev - 1); prev = i
        a = zlib.compress(bytes(dv), 9); bm = np.zeros(ng, dtype=np.uint8); bm[L] = 1; b = zlib.compress(np.packbits(bm).tobytes(), 9)
        kind, pay = (0, a) if len(a) <= len(b) else (1, b); nm = name.encode()
        out += struct.pack("<B", len(nm)) + nm + struct.pack("<IIBI", ng, len(L), kind, len(pay)) + pay
    return bytes(out)
def decode(buf):
    import numpy as np
    assert buf[:4] == b"DEL1"; p = 4; res = {}
    while p < len(buf):
        n = buf[p]; p += 1; name = buf[p:p + n].decode(); p += n; ng, cnt, kind, ln = struct.unpack("<IIBI", buf[p:p + 13]); p += 13; pay = zlib.decompress(buf[p:p + ln]); p += ln
        if kind == 0:
            L = []; prev = -1; i = 0
            while i < len(pay):
                v = 0; s = 0
                while True:
                    c = pay[i]; i += 1; v |= (c & 0x7f) << s; s += 7
                    if not c & 0x80: break
                prev = prev + v + 1; L.append(prev)
        else:
            bm = np.unpackbits(np.frombuffer(pay, dtype=np.uint8))[:ng]; L = np.nonzero(bm)[0].tolist()
        assert len(L) == cnt; res[name] = L
    return res

@torch.no_grad()
def first_logp(m, tok, prompts, chat, bs=16):
    """log-probs of the first answer token for each full prompt (right padded, last real position)."""
    dev = m.device; out = []
    for s in range(0, len(prompts), bs):
        P = [tok(chat(tok, q), add_special_tokens=False).input_ids for q in prompts[s:s + bs]]; L = max(map(len, P))
        ids = torch.full((len(P), L), tok.pad_token_id); am = torch.zeros((len(P), L), dtype=torch.long)
        for i, p in enumerate(P): ids[i, :len(p)] = torch.tensor(p); am[i, :len(p)] = 1
        h = m.model(input_ids=ids.to(dev), attention_mask=am.to(dev)).last_hidden_state; last = am.sum(1) - 1
        hl = h[torch.arange(len(P)), last.to(dev)]; out.append(F.log_softmax(m.lm_head(hl).float(), -1).half())
    return torch.cat(out)

def train_mask(m, tok, chat, ann_prompts, label, calib_prompts, ref_logp, mods, cfg, log=None, seed=0):
    """cfg: lr, steps, l0w, klw, budget (fraction of the trained groups), la0, calib_bs. Returns deleted dict (binary)."""
    dev = m.device; rng = random.Random(seed); EOS = tok.convert_tokens_to_ids("<|im_end|>"); W = m.lm_head
    for p in m.parameters(): p.requires_grad_(False)
    for d in mods.values(): d.la = torch.full(d.shape, float(cfg.get("la0", 1.0)), device=dev, requires_grad=True)
    labs = label if isinstance(label, list) else [label] * len(ann_prompts); seqs = []
    for q, lb in zip(ann_prompts, labs):
        p = tok(chat(tok, q), add_special_tokens=False).input_ids; g_ = tok(lb, add_special_tokens=False).input_ids + [EOS]; seqs.append((p + g_, len(p)))
    cal_tok = [tok(chat(tok, q), add_special_tokens=False).input_ids for q in calib_prompts]
    def batch(items):   # items: list of token lists -> ids, mask
        L = max(map(len, items)); ids = torch.full((len(items), L), tok.pad_token_id); am = torch.zeros((len(items), L), dtype=torch.long)
        for i, p in enumerate(items): ids[i, :len(p)] = torch.tensor(p); am[i, :len(p)] = 1
        return ids.to(dev), am.to(dev)
    SC = 128.0; lr = cfg["lr"]; l0w = cfg.get("l0w", 0.0); klw = cfg.get("klw", 1.0); hist = []
    for step in range(cfg["steps"]):
        dm = set_train_masks(mods, cfg["budget"]); nd = int(sum(v.sum() for v in dm.values()))
        # CE on annotations (micro-batches of 13)
        ce_tot = 0.0; mb = 13
        for s in range(0, len(seqs), mb):
            sub = seqs[s:s + mb]; ids, am = batch([x[0] for x in sub]); h = m.model(input_ids=ids, attention_mask=am).last_hidden_state
            pos = [(i, j) for i, (x, n) in enumerate(sub) for j in range(n - 1, len(x) - 1)]; tg = torch.tensor([sub[i][0][j + 1] for i, j in pos], device=dev)
            hh = h[torch.tensor([i for i, _ in pos], device=dev), torch.tensor([j for _, j in pos], device=dev)]
            ce = F.cross_entropy(W(hh).float(), tg, reduction="sum") / len(seqs) / 2.0; (ce * SC).backward(); ce_tot += float(ce)
        kl_v = 0.0
        if klw > 0:
            ix = rng.sample(range(len(cal_tok)), min(cfg.get("calib_bs", 16), len(cal_tok))); ids, am = batch([cal_tok[i] for i in ix])
            h = m.model(input_ids=ids, attention_mask=am).last_hidden_state; last = (am.sum(1) - 1); hl = h[torch.arange(len(ix), device=dev), last]
            lp = F.log_softmax(W(hl).float(), -1); ref = ref_logp[ix].float(); kl = (ref.exp() * (ref - lp)).sum(-1).mean(); (klw * kl * SC).backward(); kl_v = float(kl)
        gs = [d.la.grad for d in mods.values()]; ok = all(torch.isfinite(g).all() for g in gs)
        gmax = max(float(g.abs().max()) for g in gs) if ok else 0.0
        if ok and gmax > 0:
            with torch.no_grad():
                for d in mods.values(): d.la -= lr * (d.la.grad / gmax - l0w)
        for d in mods.values(): d.la.grad = None
        hist.append((step, round(ce_tot, 4), round(kl_v, 4), nd))
        if log and (step % 10 == 0 or step == cfg["steps"] - 1): log("step", step, "ce", round(ce_tot, 4), "kl", round(kl_v, 4), "deleted", nd, "gmax", gmax)
    dm = hard_masks(mods, cfg["budget"])
    for d in mods.values(): d.la = None
    return dm, hist
