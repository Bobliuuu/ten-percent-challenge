"""Deletion-mask feedback carrier (L15). Feedback ONLY deletes existing weights: the shipped artifact is a list of (layer, flat index)
positions in the existing MLP `gate_proj` matrices whose weight is set to zero at load. No new value, no new tensor in the forward pass,
nothing in the prompt, annotations not stored.

Method = Deletion Tuning / Bit-Mask Tuning (Lu et al., NeurIPS 2026, arXiv 2611.10866; reference code
github.com/agentic-learning-ai-lab/deleting-weights, src/learnable_sparsity + src/wanda_calibration.py + scripts/train_sft.sh), re-implemented here:
  * one fp32 logit per gate_proj weight, N(0, 0.01^2); Wanda prior: the `init_sparsity` (paper 2^-11 = 0.049%) lowest |W_ij| * ||x_j||_2 weights
    (activations of the feedback sequences) get logit -= prior*std, all others += prior*std (paper prior_strength 3);
  * training forward uses W * sigmoid(sharpness * logit), sharpness rising linearly sharp0 -> sharp1 (paper 100 -> 1000);
  * loss = answer-token cross-entropy on the annotated items  -  weight_reg * (1 - t/T) * sum ||W * mask||^2   (paper weight_reg 1e-5, decays to 0);
    L15 adds an optional protection term beta * CE(model's OWN greedy answers on the shipped unlabeled calibration prompts) (paper: none);
  * AdamW on the logits only (paper lr 5e-4, wd 0, grad-norm clip 1.0); the weights are frozen and never change value;
  * deploy: weight deleted (set to 0) exactly where its logit <= 0 (paper: kept iff logit > 0).
NOTE (rules): the mask is FOUND by gradient descent on mask logits. The weights are never trained (values unchanged, only removed), but the
search for which weights to remove is gradient-based; see REPORT.md (rule 2 "can't train the weights" / checklist 6, flagged GRAY).

On the 1-bit Bonsai pack (weight = sign * group_scale) a deleted weight becomes exactly 0: the matrix becomes {-s, 0, +s} at the deleted
positions only. bits.bin/scales.bin are untouched; the loader zeroes the listed positions. Group-scale zeroing (128 weights at once) is NOT used."""
import io, json, math, os, random, time, zlib
import numpy as np, torch, torch.nn.functional as F

DEF = dict(steps=200, lr=5e-4, sharp0=100.0, sharp1=1000.0, init_sparsity=2.0 ** -11, prior=3.0, weight_reg=1e-5, bs=8, cal_bs=8,
           beta=1.0, layers=0, target="gate_proj", loss_scale=256.0, clip=1.0, seed=0)
# layers=0 -> every layer (paper: gate_proj of all transformer blocks); layers=k -> last k layers

def _chat(tok, u):
    return tok.apply_chat_template([{"role": "user", "content": u}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
def _prompt(row, T):
    return row["query"] if "query" in row else T["prompt"].replace("{t}", row["text"])
def _label(row):
    return row["label"] if "label" in row else row["gold"][0]["name"]
def eos_id(tok):
    e = tok.convert_tokens_to_ids("<|im_end|>"); return e if isinstance(e, int) and e >= 0 else tok.eos_token_id

def ann_seqs(tok, ann, T):
    """annotated item -> (ids, target positions, target tokens): prompt + label word + EOS; the loss is on the label word and EOS."""
    out = []; E = eos_id(tok)
    for r in ann:
        p = tok(_chat(tok, _prompt(r, T)), add_special_tokens=False).input_ids; g = tok(_label(r), add_special_tokens=False).input_ids + [E]
        out.append((p + g, list(range(len(p) - 1, len(p) + len(g) - 1)), g))
    return out

@torch.no_grad()
def calib_seqs(m, tok, prompts, bs=16, mx=6):
    """unlabeled calibration prompt + the model's OWN greedy answer up to EOS -> (ids, positions, targets) (protects old behaviour)."""
    dev = m.device; E = eos_id(tok); P = [tok(_chat(tok, q), add_special_tokens=False).input_ids for q in prompts]; out = [None] * len(P)
    order = sorted(range(len(P)), key=lambda i: len(P[i])); pad = tok.pad_token_id if tok.pad_token_id is not None else E
    for s in range(0, len(order), bs):
        idx = order[s:s + bs]; L = max(len(P[i]) for i in idx); ids = torch.full((len(idx), L), pad); am = torch.zeros((len(idx), L), dtype=torch.long)
        for r, i in enumerate(idx): ids[r, L - len(P[i]):] = torch.tensor(P[i]); am[r, L - len(P[i]):] = 1
        g = m.generate(input_ids=ids.to(dev), attention_mask=am.to(dev), max_new_tokens=mx, do_sample=False, pad_token_id=pad)[:, L:].tolist()
        for r, i in enumerate(idx):
            o = g[r]; o = o[:o.index(E) + 1] if E in o else o
            out[i] = (P[i] + o, list(range(len(P[i]) - 1, len(P[i]) + len(o) - 1)), o)
    return [x for x in out if x[2]]

def _batch(seqs, dev, pad=0):
    L = max(len(s[0]) for s in seqs); ids = torch.full((len(seqs), L), pad, dtype=torch.long); am = torch.zeros((len(seqs), L), dtype=torch.long); rows, cols, tg = [], [], []
    for b, (q, pos, t) in enumerate(seqs):
        ids[b, :len(q)] = torch.tensor(q); am[b, :len(q)] = 1; rows += [b] * len(pos); cols += pos; tg += t
    return ids.to(dev), am.to(dev), torch.tensor(rows, device=dev), torch.tensor(cols, device=dev), torch.tensor(tg, device=dev)

def _ce(m, seqs, dev):
    ids, am, r, c, t = _batch(seqs, dev); h = m.model(input_ids=ids, attention_mask=am).last_hidden_state[r, c]
    return F.cross_entropy(m.lm_head(h).float(), t)

def target_layers(m, layers):
    n = len(m.model.layers); return list(range(n)) if not layers else list(range(n - layers, n))
def _mod(m, li, target): return getattr(m.model.layers[li].mlp, target)

class _MaskedLinear(torch.autograd.Function):
    """y = x @ (W * sigmoid(sharp * g))^T with only x saved (the mask is recomputed in backward); grad flows to x and to the logits g, never to W."""
    @staticmethod
    def forward(ctx, x, g, W, sharp):
        ctx.save_for_backward(x, g); ctx.W = W; ctx.sharp = sharp
        return F.linear(x, W * torch.sigmoid(g.view(W.shape) * sharp).to(W.dtype))
    @staticmethod
    def backward(ctx, gy):
        x, g = ctx.saved_tensors; W = ctx.W; s = ctx.sharp; msk = torch.sigmoid(g.view(W.shape) * s)
        gx = gy @ (W * msk.to(W.dtype))
        gW = gy.reshape(-1, gy.shape[-1]).T.float() @ x.reshape(-1, x.shape[-1]).float()          # [out, in] fp32
        gg = gW.mul_(W.float()).mul_(msk * (1 - msk) * s)
        return gx, gg.view(-1), None, None

@torch.no_grad()
def wanda_act_norms(m, seqs, LS, target):
    """per-input-column L2 norm of the target module's inputs over all tokens of `seqs` (Wanda, Sun et al. 2023), one sequence at a time (no padding)."""
    dev = m.device; acc = {}; hs = []
    for li in LS:
        def hook(mod, inp, li=li):
            x = inp[0].float(); acc[li] = acc.get(li, 0) + x.pow(2).reshape(-1, x.shape[-1]).sum(0)
        hs.append(_mod(m, li, target).register_forward_pre_hook(hook))
    try:
        for q in seqs: m.model(input_ids=torch.tensor([q[0]], device=dev))
    finally:
        for h in hs: h.remove()
    return {li: v.sqrt() for li, v in acc.items()}

def learn_deletions(m, tok, ann, T, calib_prompts, cfg=None, log=None, cal=None, seqs=None):
    """Learn the deletion mask from <= 25 annotated items. Returns {"layers": [...], "idx": [LongTensor flat indices to zero], "stats": {...}}
    or None if there is nothing to learn. The model is left UNCHANGED (original weights, original forward)."""
    c = dict(DEF); c.update(cfg or {}); dev = m.device; t0 = time.time()
    if not ann and not seqs: return None
    torch.manual_seed(c["seed"]); rnd = random.Random(c["seed"])
    A = seqs if seqs is not None else ann_seqs(tok, ann, T); C = (cal if cal is not None else calib_seqs(m, tok, calib_prompts)) if c["beta"] > 0 else []
    LS = target_layers(m, c["layers"]); mods = [_mod(m, li, c["target"]) for li in LS]
    for p in m.parameters(): p.requires_grad_(False)
    an = wanda_act_norms(m, A, LS, c["target"])
    G = []; prior_del = []
    for li, mod in zip(LS, mods):
        W = mod.weight; g = torch.empty(W.numel(), dtype=torch.float32, device=dev).normal_(std=0.01)
        sc = (W.float().abs() * an[li][None, :]).reshape(-1); k = max(1, int(sc.numel() * c["init_sparsity"]))
        thr = sc.kthvalue(k).values.item() if sc.device.type == "cpu" else torch.topk(sc, k, largest=False).values.max().item()
        d = sc < thr; del sc                                                                                   # paper: strictly below the k-th value
        g += torch.where(d, -1.0, 1.0) * g.std() * c["prior"]; g.requires_grad_(True); G.append(g); prior_del.append(d.nonzero()[:, 0].cpu())
    state = {"sharp": c["sharp0"]}; orig_fwd = [mod.forward for mod in mods]
    def mk(mod, g):
        W = mod.weight
        return lambda x: _MaskedLinear.apply(x, g, W, state["sharp"])
    for mod, g in zip(mods, G): mod.forward = mk(mod, g)
    opt = torch.optim.AdamW(G, lr=c["lr"], weight_decay=0.0); hist = []; skipped = 0
    try:
        order = []; corder = []
        for step in range(c["steps"]):
            prog = step / c["steps"]; state["sharp"] = c["sharp0"] + (c["sharp1"] - c["sharp0"]) * prog
            if len(order) < c["bs"]: order += rnd.sample(range(len(A)), len(A))
            ba = [A[i] for i in order[:c["bs"]]]; order = order[c["bs"]:]
            la = _ce(m, ba, dev); loss = la
            lc = torch.zeros((), device=dev)
            if C and c["beta"] > 0:
                if len(corder) < c["cal_bs"]: corder += rnd.sample(range(len(C)), len(C))
                bc = [C[i] for i in corder[:c["cal_bs"]]]; corder = corder[c["cal_bs"]:]
                lc = _ce(m, bc, dev); loss = loss + c["beta"] * lc
            opt.zero_grad(set_to_none=True); (loss * c["loss_scale"]).backward()
            gn2 = 0.0; ok = True
            for g in G:
                g.grad.div_(c["loss_scale"]); s = float(g.grad.pow(2).sum())
                if not math.isfinite(s): ok = False; break
                gn2 += s
            if not ok: skipped += 1; opt.zero_grad(set_to_none=True); continue
            if c["weight_reg"] > 0:     # paper regulariser - weight_reg*(1-t/T)*sum||W*mask||^2, its logit gradient added analytically (saves memory)
                with torch.no_grad():
                    for mod, g in zip(mods, G):
                        W = mod.weight; mk_ = torch.sigmoid(g * state["sharp"]); g.grad.add_(-c["weight_reg"] * (1 - prog) * 2 * W.float().reshape(-1).pow(2) * mk_ * mk_ * (1 - mk_) * state["sharp"])
                gn2 = sum(float(g.grad.pow(2).sum()) for g in G)
            gn = math.sqrt(gn2)
            if gn > c["clip"]:
                for g in G: g.grad.mul_(c["clip"] / (gn + 1e-6))
            opt.step()
            if step % 20 == 0 or step == c["steps"] - 1:
                with torch.no_grad(): nd = sum(int((g <= 0).sum()) for g in G)
                hist.append(dict(step=step, ce_ann=round(float(la), 4), ce_cal=round(float(lc), 4), sharp=state["sharp"], n_del=nd, gnorm=round(gn, 4)))
                if log: log(hist[-1])
    finally:
        for mod, f in zip(mods, orig_fwd): mod.forward = f
        for p in m.parameters(): p.requires_grad_(False)
    idx = [(g.detach() <= 0).nonzero()[:, 0].cpu() for g in G]
    n_tot = sum(mod.weight.numel() for mod in mods); n_del = sum(len(i) for i in idx)
    new_del = sum(int((~torch.isin(i, p)).sum()) for i, p in zip(idx, prior_del)); kept_prior = sum(int((~torch.isin(p, i)).sum()) for i, p in zip(idx, prior_del))
    st = dict(n_deleted=n_del, frac_of_target=n_del / n_tot, frac_of_all_params=n_del / sum(p.numel() for p in m.parameters()), n_prior=sum(len(p) for p in prior_del),
              deleted_not_in_prior=new_del, prior_restored=kept_prior, skipped_steps=skipped, seconds=round(time.time() - t0, 1), hist=hist, cfg=c, layers=LS)
    del G, opt
    return {"layers": LS, "target": c["target"], "shapes": [list(mod.weight.shape) for mod in mods], "idx": idx, "stats": st}

@torch.no_grad()
def apply(m, D):
    """zero the listed existing weights in place; returns the saved original values (experiments only, for exact undo)."""
    saved = []
    for li, ix in zip(D["layers"], D["idx"]):
        w = _mod(m, li, D.get("target", "gate_proj")).weight.data.view(-1); ix = ix.to(w.device); saved.append((li, ix, w[ix].clone())); w[ix] = 0
    return saved

@torch.no_grad()
def undo(m, saved, target="gate_proj"):
    for li, ix, v in saved: _mod(m, li, target).weight.data.view(-1)[ix] = v

def save(D, path):
    """shipped feedback file: per layer the sorted flat indices, delta-coded as uint32 and zlib-compressed (no values are stored)."""
    meta = {"kind": "delete", "target": D["target"], "layers": D["layers"], "shapes": D["shapes"], "n": [int(len(i)) for i in D["idx"]]}
    blobs = [zlib.compress(np.diff(np.sort(i.numpy().astype(np.int64)), prepend=0).astype(np.uint32).tobytes(), 9) for i in D["idx"]]
    hdr = json.dumps(meta).encode()
    with open(path, "wb") as f:
        f.write(len(hdr).to_bytes(4, "little")); f.write(hdr)
        for b in blobs: f.write(len(b).to_bytes(4, "little")); f.write(b)

def load(path):
    with open(path, "rb") as f:
        n = int.from_bytes(f.read(4), "little"); meta = json.loads(f.read(n)); idx = []
        for _ in meta["layers"]:
            k = int.from_bytes(f.read(4), "little"); d = np.frombuffer(zlib.decompress(f.read(k)), dtype=np.uint32).astype(np.int64); idx.append(torch.from_numpy(np.cumsum(d)))
    return {"layers": meta["layers"], "target": meta["target"], "shapes": meta["shapes"], "idx": idx}

def load_model(art, device=None, dtype=None):
    """Bonsai pack (bits.bin) through the L1 loader, otherwise any HF causal LM folder/name (proxy models for smoke tests)."""
    if os.path.exists(os.path.join(art, "bits.bin")):
        import bonsai_loader as BL
        m, _ = BL.load(art, patch=None); return m
    from transformers import AutoModelForCausalLM
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu"); dt = dtype or (torch.float16 if dev == "cuda" else torch.float32)
    return AutoModelForCausalLM.from_pretrained(art, dtype=dt).to(dev).eval()
