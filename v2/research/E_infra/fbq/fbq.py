"""Feedback THROUGH the compression (annotation-weighted GPTQ re-quantization) vs lm_head row edit, on a SIMULATED 4-bit g128 GPTQ base. TOX natural-mistake scenario.
Runs on one box GPU (no Modal).  python fbq.py --model Qwen/Qwen3-8B --k 4 --ndraw 8 --out res_8B.json
Variants (all applied to the QUANTIZED base; the 'artifact' = 4-bit codes + fp16 scales + 4-bit zeros per 128-group):
  B0   lm_head row edit (fb_tox lmhead_cal: LDA direction, mu/Sigma from unlabeled calib hidden states) on the dequantised row, fp ('B0fp') or re-RTN-quantised to the same format ('B0q')
  V2   lm_head LABEL ROWS re-GPTQ'd from the original fp row with Hessian = calib + annotated hidden states, target logit raised for the annotated label (closed form ridge: u = w H^-1 sum d_a x_a)
  V2h  same but NO label target (Hessian-only weighting): pure rounding change -> label info = 0 by construction
  V3   per-128-group SCALES (codes fixed) of the label rows fitted by ridge LS to raise the annotated label's logit (fp16 scales)
  V1   last-k layers' sites (down_proj by default) re-GPTQ'd from fp weights with Hessian = calib + annotated tokens (weight omega) and a one-step activation-gradient target (CE of annotated label at answer position); closed form u = -c G H^-1
  V1h  same, no target
Controls through identical code: empty (artifact untouched, asserted byte-identical), mixed (half labels = base's wrong output), randwrong.
"""
import torch, json, hashlib, random, sys, time, collections, os, math, argparse
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import scorer
from transformers import AutoTokenizer, AutoModelForCausalLM

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen3-1.7B"); ap.add_argument("--k", type=int, default=4); ap.add_argument("--ndraw", type=int, default=8)
ap.add_argument("--nann", type=int, default=25); ap.add_argument("--pool_n", type=int, default=7000); ap.add_argument("--sites", default="down_proj")
ap.add_argument("--omega", type=float, default=10.0); ap.add_argument("--out", default="res.json"); ap.add_argument("--ncal", type=int, default=750)
ap.add_argument("--scenario", default="natural"); ap.add_argument("--variants", default="B0,V2,V3,V1"); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--bits", type=int, default=4)
args = ap.parse_args()
T0 = time.time(); log = lambda *a: print(f"[{time.time()-T0:6.0f}s]", *a, flush=True)
H8 = lambda s: int(hashlib.sha256(s.encode()).hexdigest()[:8], 16)
dev = "cuda"; GS = 128; MAXQ = 2 ** args.bits - 1
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

# ---------------------------------------------------------------- GPTQ
def gptq(W, H, damp=0.01):
    """W [out,in] fp, H [in,in] (sum of x x^T). asym min-max 4-bit, group 128 (= block). returns deq (float32), codes uint8, scales fp16 [out,in/GS], zeros uint8"""
    W = W.float().clone(); out, inn = W.shape; H = H.float().clone()
    dead = torch.diag(H) == 0; H[dead, dead] = 1; W[:, dead] = 0
    H0_ = H; dm = damp
    for _ in range(8):
        H = H0_ + dm * torch.mean(torch.diag(H0_)) * torch.eye(inn, device=W.device)
        L, info = torch.linalg.cholesky_ex(H)
        if int(info) == 0: break
        dm *= 4
    Hinv = torch.linalg.cholesky(torch.cholesky_inverse(L), upper=True)
    Q = torch.zeros_like(W); codes = torch.zeros(out, inn, dtype=torch.uint8, device=W.device); sc = []; zs = []
    for i1 in range(0, inn, GS):
        W1 = W[:, i1:i1 + GS].clone(); Err1 = torch.zeros_like(W1); Hi = Hinv[i1:i1 + GS, i1:i1 + GS]
        xmin = W1.min(1).values.clamp(max=0); xmax = W1.max(1).values.clamp(min=0)
        s = ((xmax - xmin) / MAXQ); s[s == 0] = 1.0; s = s.half().float(); z = torch.round(-xmin / s).clamp(0, MAXQ)
        sc.append(s.half()); zs.append(z.to(torch.uint8))
        for i in range(GS):
            w = W1[:, i]; d = Hi[i, i]
            c = (torch.round(w / s) + z).clamp(0, MAXQ); q = s * (c - z)
            Q[:, i1 + i] = q; codes[:, i1 + i] = c.to(torch.uint8)
            err = (w - q) / d; W1[:, i:] -= err.unsqueeze(1) @ Hi[i, i:].unsqueeze(0); Err1[:, i] = err
        W[:, i1 + GS:] -= Err1 @ Hinv[i1:i1 + GS, i1 + GS:]
    return Q, codes, torch.stack(sc, 1), torch.stack(zs, 1)

def deq(codes, sc, zs):
    o, n = codes.shape; g = n // GS
    return (sc.float().unsqueeze(2) * (codes.view(o, g, GS).float() - zs.float().unsqueeze(2))).view(o, n)

def rtn(W):
    o, n = W.shape; Wg = W.float().view(o, n // GS, GS); xmin = Wg.min(2).values.clamp(max=0); xmax = Wg.max(2).values.clamp(min=0)
    s = ((xmax - xmin) / MAXQ); s[s == 0] = 1.0; s = s.half().float(); z = torch.round(-xmin / s).clamp(0, MAXQ)
    c = (torch.round(Wg / s.unsqueeze(2)) + z.unsqueeze(2)).clamp(0, MAXQ)
    return c.view(o, n).to(torch.uint8), s.half(), z.to(torch.uint8)

def changed_bytes(a, b):
    """a,b = (codes, scales, zeros): number of differing bytes in the packed artifact (2 nibbles/byte for codes+zeros, 2 B per fp16 scale)"""
    ca, sa, za = a; cb, sb, zb = b
    cd = (ca != cb); cd = cd.view(cd.shape[0], -1, 2).any(2).sum().item()
    zd = (za != zb); n = zd.numel(); zd = zd.reshape(-1)
    if n % 2: zd = torch.cat([zd, zd.new_zeros(1)])
    zd = zd.view(-1, 2).any(1).sum().item()
    return cd + zd + 2 * (sa != sb).sum().item()

def art_bytes(c, s, z): return c.numel() // 2 + (z.numel() + 1) // 2 + 2 * s.numel()

# ---------------------------------------------------------------- model + data
tok = AutoTokenizer.from_pretrained(args.model); tok.padding_side = "left"
lm = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, attn_implementation="sdpa").to(dev).eval()
for p in lm.parameters(): p.requires_grad_(False)
NL = lm.config.num_hidden_layers; layers = lm.model.layers; D = lm.config.hidden_size; LAB = list(scorer.LABELS)
FT = {l: tok.encode(l, add_special_tokens=False)[0] for l in LAB + ["selfharm"]}; assert len(set(FT.values())) == 6, FT; tid2lab = {t: l for l, t in FT.items()}
def chat(u):
    try: return tok.apply_chat_template([{"role": "user", "content": u}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError: return tok.apply_chat_template([{"role": "user", "content": u}], tokenize=False, add_generation_prompt=True)
ld = lambda f: [json.loads(l) for l in open(f"{DATA}/{f}.jsonl")]
dev1, dev3, pool1, pool3, cal = ld("s1_dev"), ld("s3_dev"), ld("s1_pool"), ld("s3_pool"), ld("s1_calib")
random.Random(args.seed).shuffle(pool1); random.Random(args.seed).shuffle(pool3)
pool1, pool3 = pool1[:args.pool_n], pool3[:args.pool_n // 2]; cal = cal[:args.ncal]
NEWROWS = []
if args.scenario == "newlabel":
    pool1, pool3 = pool1[:1500], pool3[:1500]; seen = set()
    for r in json.load(open(f"{DATA}/aegis_train.json")):
        t = r["prompt"]
        if r.get("violated_categories") == "Suicide and Self Harm" and r.get("prompt_label") == "unsafe" and t not in seen and 15 < len(t) < 1500:
            seen.add(t); h = hashlib.sha256(t.encode()).hexdigest()[:10]; NEWROWS.append(dict(gold="selfharm", group="nl" + h, id="nl_" + h, text=t))
    log("new-label items", len(NEWROWS))
ALL = dev1 + dev3 + pool1 + pool3 + NEWROWS + cal
role = ["dev1"] * len(dev1) + ["dev3"] * len(dev3) + ["pool1"] * len(pool1) + ["pool3"] * len(pool3) + ["new"] * len(NEWROWS) + ["cal"] * len(cal)
PROMPTS = [chat(scorer.PROMPT.format(t=r["text"])) for r in ALL]
ENC = [tok.encode(p, add_special_tokens=False) for p in PROMPTS]
log("items", len(ALL), "mean tokens", np.mean([len(e) for e in ENC]))

def make_batch(idx):
    """left-padded batch from item indices -> dict(ids, valid, pos)"""
    T = max(len(ENC[i]) for i in idx); ids = torch.full((len(idx), T), tok.pad_token_id, dtype=torch.long); valid = torch.zeros(len(idx), T, dtype=torch.bool)
    for b, i in enumerate(idx): n = len(ENC[i]); ids[b, T - n:] = torch.tensor(ENC[i]); valid[b, T - n:] = True
    ids, valid = ids.to(dev), valid.to(dev); pos = (valid.long().cumsum(1) - 1).clamp(min=0)
    T_ = ids.shape[1]; causal = torch.tril(torch.ones(T_, T_, dtype=torch.bool, device=dev))
    mask = (causal[None, None] & valid[:, None, None, :]) | torch.eye(T_, dtype=torch.bool, device=dev)[None, None]
    return dict(ids=ids, valid=valid, pos=pos, mask=mask)

def layer_fwd(l, h, b, rot):
    o = layers[l](h, attention_mask=b["mask"], position_ids=b["pos"], position_embeddings=rot)
    return o[0] if isinstance(o, tuple) else o

def rope(h, b): return lm.model.rotary_emb(h, b["pos"])

def batches_by_len(idx, bs):
    o = sorted(idx, key=lambda i: -len(ENC[i])); return [o[s:s + bs] for s in range(0, len(o), bs)]

LIN = dict(q_proj="self_attn.q_proj", k_proj="self_attn.k_proj", v_proj="self_attn.v_proj", o_proj="self_attn.o_proj", gate_proj="mlp.gate_proj", up_proj="mlp.up_proj", down_proj="mlp.down_proj")
def getlin(l, name):
    m = layers[l]
    for p in LIN[name].split("."): m = getattr(m, p)
    return m
# ---------------------------------------------------------------- base quantization (sequential GPTQ, calibration = unlabeled s1_calib prompts rendered in task format)
CALIDX = [i for i, x in enumerate(role) if x == "cal"]
TAIL = list(range(NL - args.k, NL)); SITES = args.sites.split(",")
fp_w = {}; base_art = {}; Hc_tail = {}; Hc_lm = None
bs_cal = 32
def collect_hess(l, names, hs_list, bl, extra_w=None):
    """forward batches (cached residual inputs) through layer l with pre-hooks accumulating x^T x on the sites; returns H dict and outputs"""
    Hs = {n: None for n in names}; cur = {}
    hooks = []
    for n in names:
        def mk(n):
            def f(mod, inp):
                x = inp[0][cur["valid"]].float()
                H_ = x.T @ x; Hs[n] = H_ if Hs[n] is None else Hs[n] + H_
            return f
        hooks.append(getlin(l, n).register_forward_pre_hook(mk(n)))
    outs = []
    with torch.no_grad():
        for h, b in zip(hs_list, bl):
            cur["valid"] = b["valid"]; outs.append(layer_fwd(l, h, b, rope(h, b)))
    for hk in hooks: hk.remove()
    return Hs, outs

cal_batches = [make_batch(ix) for ix in batches_by_len(CALIDX, bs_cal)]
with torch.no_grad(): hs_cal = [lm.model.embed_tokens(b["ids"]) for b in cal_batches]
cut_cal = None
for l in range(NL):
    if l == NL - args.k: cut_cal = [h.clone() for h in hs_cal]
    names = list(LIN)
    Hs, _ = collect_hess(l, names, hs_cal, cal_batches)
    for n in names:
        mod = getlin(l, n); Wfp = mod.weight.data.clone()
        Q, c, s, z = gptq(Wfp, Hs[n]); mod.weight.data.copy_(Q.to(torch.bfloat16))
        if l in TAIL and n in SITES:
            fp_w[(l, n)] = Wfp; base_art[(l, n)] = (c, s, z); Hc_tail[(l, n)] = Hs[n]
    del Hs
    with torch.no_grad(): hs_cal = [layer_fwd(l, h, b, rope(h, b)) for h, b in zip(hs_cal, cal_batches)]
    if l % 6 == 0: log("base GPTQ layer", l)
# lm_head: untied copy, quantised
W_lm_fp = lm.lm_head.weight.data.clone()
lm.lm_head = torch.nn.Linear(D, W_lm_fp.shape[0], bias=False).to(dev, torch.bfloat16); lm.lm_head.weight.requires_grad_(False)
Hc_lm = torch.zeros(D, D, device=dev)
with torch.no_grad():
    for h, b in zip(hs_cal, cal_batches):
        x = lm.model.norm(h)[b["valid"]].float(); Hc_lm += x.T @ x
lm_codes, lm_sc, lm_zs = [], [], []
Wq_lm = torch.zeros_like(W_lm_fp, dtype=torch.float32)
for r0 in range(0, W_lm_fp.shape[0], 40000):   # rows are independent in GPTQ given H
    r1 = min(r0 + 40000, W_lm_fp.shape[0]); Q, c, s, z = gptq(W_lm_fp[r0:r1], Hc_lm); Wq_lm[r0:r1] = Q; lm_codes.append(c); lm_sc.append(s); lm_zs.append(z)
lm_art = (torch.cat(lm_codes), torch.cat(lm_sc), torch.cat(lm_zs)); del lm_codes, lm_sc, lm_zs
lm.lm_head.weight.data.copy_(Wq_lm.to(torch.bfloat16)); W0 = lm.lm_head.weight.data.clone()
tot_art = sum(art_bytes(*lm_art) for _ in [0]) + sum(art_bytes(*v) for v in base_art.values())
log("base quantised. tail+lm artifact bytes", tot_art)

# ---------------------------------------------------------------- base pass: final hidden (answer position), preds; cut-layer caches for needed items
@torch.no_grad()
def make_cut(idx, bs=24):
    out = []
    for ix in batches_by_len(idx, bs):
        b = make_batch(ix); h = lm.model.embed_tokens(b["ids"]); rot = rope(h, b)
        for l in range(L0): h = layer_fwd(l, h, b, rot)
        out.append((ix, h, b))
    return out
L0 = NL - args.k
@torch.no_grad()
def tail_eval_h(cut):
    """final answer-position hidden for cached batches under CURRENT tail weights -> dict item->vec"""
    res = {}
    for ix, h, b in cut:
        rot = rope(h, b)
        for l in range(L0, NL): h = layer_fwd(l, h, b, rot)
        hf = lm.model.norm(h)[:, -1].float()
        for j, i in enumerate(ix): res[i] = hf[j]
    return res
idx_all = list(range(len(ALL)))
cut_all = make_cut(idx_all); log("cut cache (all items) built")
rs_ = tail_eval_h(cut_all); Hall = torch.stack([rs_[i] for i in idx_all]); del rs_; log("final hidden", tuple(Hall.shape))
def logits(Hm, Wm): return (Hm.to(torch.bfloat16) @ Wm.T).float()
def preds(Hm, Wm):
    am = []
    for s0 in range(0, len(Hm), 1024): am += logits(Hm[s0:s0 + 1024], Wm).argmax(1).tolist()
    return [tid2lab.get(a) for a in am], None
base_pred, base_lg = preds(Hall, W0); gold = [r["gold"] for r in ALL]; base_ok = np.array([p == g for p, g in zip(base_pred, gold)])
# validate my runner against HF forward on 32 items
with torch.no_grad():
    smp = idx_all[:32]; enc = tok([PROMPTS[i] for i in smp], return_tensors="pt", padding=True, add_special_tokens=False).to(dev)
    hf_h = lm.model(**enc).last_hidden_state[:, -1].float()
    cs = torch.nn.functional.cosine_similarity(hf_h, Hall[:32]).min().item()
log("runner vs HF forward: min cosine of final hidden", cs)
acc = lambda m: float(np.mean(m)) if len(m) else float("nan")
R = dict(model=args.model, k=args.k, sites=SITES, omega=args.omega, bits=args.bits, cos_runner_vs_hf=cs, base_acc={k: acc(base_ok[[i for i, x in enumerate(role) if x == k]]) for k in ("dev1", "dev3", "pool1", "pool3")})
log("quantised-base acc", R["base_acc"])
# ---------------------------------------------------------------- spec (as fb_tox)
devidx = [i for i, x in enumerate(role) if x in ("dev1", "dev3")]; pidx = [i for i, x in enumerate(role) if x in ("pool1", "pool3")]
grp = lambda i: int(H8("g|" + ALL[i]["group"]) % 10)
if args.scenario == "newlabel":
    R["dev_confusions_top"] = None; concept = ("selfharm", None); R["concept"] = concept
    newi = [i for i, x in enumerate(role) if x == "new"]
    R["new_base_pred_hist"] = dict(collections.Counter(str(base_pred[i]) for i in newi))
    ann_c = [i for i in newi if grp(i) in (0, 1) and base_pred[i] in LAB]
    teach1 = sorted([i for i in newi if 2 <= grp(i) <= 5], key=lambda i: H8("t|" + ALL[i]["id"]))[:150]
    teach3 = sorted([i for i in newi if grp(i) >= 6], key=lambda i: H8("t|" + ALL[i]["id"]))[:150]
    nb = sorted([i for i in pidx if role[i] == "pool1" and base_ok[i] and grp(i) >= 6], key=lambda i: H8("n|" + ALL[i]["id"]))[:400]
    nb3 = sorted([i for i in pidx if role[i] == "pool3" and base_ok[i]], key=lambda i: H8("n|" + ALL[i]["id"]))[:150]
    devret = [i for i in devidx if base_ok[i]]
else:
    conf = collections.Counter((gold[i], base_pred[i]) for i in devidx if base_pred[i] != gold[i]); R["dev_confusions_top"] = [[list(k), v] for k, v in conf.most_common(8)]
    def support(pair):
        a = [i for i in pidx if role[i] == "pool1" and (gold[i], base_pred[i]) == pair and grp(i) in (0, 1)]
        t1 = [i for i in pidx if role[i] == "pool1" and (gold[i], base_pred[i]) == pair and 2 <= grp(i) <= 5]; return len(a), len(t1)
    concept = None
    for pair, n in conf.most_common():
        if pair[1] is None: continue
        sa, st = support(pair)
        if sa >= 20 and st >= 25: concept = pair; break
    R["concept"] = concept; log("concept", concept, support(concept) if concept else None)
    if concept is None:
        R["error"] = "no concept"; json.dump(R, open(args.out, "w")); sys.exit(0)
    cg, cp = concept
    ann_c = [i for i in pidx if role[i] == "pool1" and (gold[i], base_pred[i]) == concept and grp(i) in (0, 1)]
    teach1 = sorted([i for i in pidx if role[i] == "pool1" and (gold[i], base_pred[i]) == concept and 2 <= grp(i) <= 5], key=lambda i: H8("t|" + ALL[i]["id"]))[:150]
    teach3 = sorted([i for i in pidx if role[i] == "pool3" and (gold[i], base_pred[i]) == concept], key=lambda i: H8("t|" + ALL[i]["id"]))[:150]
    nb_same = sorted([i for i in pidx if role[i] == "pool1" and gold[i] == cg and base_ok[i] and grp(i) >= 6], key=lambda i: H8("n|" + ALL[i]["id"]))[:220]
    nb_other = []
    for l in [x for x in LAB if x != cg]: nb_other += sorted([i for i in pidx if role[i] == "pool1" and gold[i] == l and base_ok[i] and grp(i) >= 6], key=lambda i: H8("n|" + ALL[i]["id"]))[:150]
    nb = nb_same + nb_other
    nb3 = sorted([i for i in pidx if role[i] == "pool3" and base_ok[i]], key=lambda i: H8("n|" + ALL[i]["id"]))[:150]
    devret = [i for i in devidx if base_ok[i]]
spec = dict(concept=list(concept), ann=[ALL[i]["id"] for i in ann_c], teach1=[ALL[i]["id"] for i in teach1], teach3=[ALL[i]["id"] for i in teach3], nb=[ALL[i]["id"] for i in nb], nb3=[ALL[i]["id"] for i in nb3])
R["spec_sha256"] = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest(); R["spec_n"] = {k: len(v) for k, v in spec.items() if k != "concept"}
log("SPEC_SHA (fixed BEFORE any feedback)", R["spec_sha256"], R["spec_n"])
gs = lambda L: {ALL[i]["group"] for i in L}
R["group_overlap"] = dict(ann_vs_teach=len(gs(ann_c) & (gs(teach1) | gs(teach3))), ann_vs_nb=len(gs(ann_c) & gs(nb)), teach_vs_nb=len(gs(teach1) & gs(nb)))
sets = {k: v for k, v in dict(teach1=teach1, teach3=teach3, nb=nb, nb3=nb3, devret=devret).items() if len(v) > 0}

# ---------------------------------------------------------------- cut caches for tail forward (V1)
need = set(sum(sets.values(), [])) | set(ann_c)
cut_need = [bt for bt in cut_all if any(i in need for i in bt[0])]; log("cut batches needed", len(cut_need), "of", len(cut_all))

# ---------------------------------------------------------------- annotation draws + controls
cand_sorted = sorted(ann_c, key=lambda i: H8("a|" + ALL[i]["id"]))
draws = [random.Random(args.seed * 100 + d).sample(cand_sorted, min(args.nann, len(cand_sorted))) for d in range(args.ndraw)]
def wrong_random(i, rr): return rr.choice([l for l in LAB if l != gold[i]])
def arm_labels(arm, sel, d):
    rr = random.Random(args.seed * 1000 + d); bp = lambda i: base_pred[i] if base_pred[i] in LAB else rr.choice([l for l in LAB if l != gold[i]])
    if arm == "real": return sel, [gold[i] for i in sel]
    if arm == "empty": return [], []
    if arm == "mixed": return sel, [gold[i] if k % 2 == 0 else bp(i) for k, i in enumerate(sel)]
    return sel, [wrong_random(i, rr) for i in sel]
ARMS = ["real", "empty", "mixed", "randwrong"]

def flip_rate(Hs_, labs, Wm, margin=0.0):
    lg = logits(Hs_, Wm); return float(np.mean([int(lg[k].argmax()) == FT[l] for k, l in enumerate(labs)]))

# -------- B0: lm_head row edit (fb_tox edit()) on quantised base
mu = Hall[CALIDX].mean(0); sig = torch.cov(Hall[CALIDX].T) + 1e-6 * torch.eye(D, device=dev)
def edit_B0(sites, labs, requant, ridge=0.1, margin=1.0, frac=0.9):
    if not sites: return W0.clone(), 0
    Wn = W0.clone(); Hs = Hall[sites]; by = collections.defaultdict(list)
    for k, l in enumerate(labs): by[FT[l]].append(k)
    base_l = logits(Hs, W0); nbytes = 0
    for t, ix in by.items():
        d = Hs[ix].mean(0) - mu; d = torch.linalg.solve(sig + ridge * sig.diagonal().mean() * torch.eye(D, device=dev), d); d = d / d.norm()
        wt = W0[t].float(); other = base_l[ix].clone(); other[:, t] = -1e9; mx = other.max(1).values; chosen = 2.0
        for a in [0.05, 0.1, 0.15, 0.2, 0.3, 0.45, 0.7, 1.0, 1.5, 2.0]:
            if float((((Hs[ix] @ (wt + a * wt.norm() * d)) - mx) >= margin).float().mean()) >= frac: chosen = a; break
        w_new = wt + chosen * wt.norm() * d
        if requant:
            c, s, z = rtn(w_new[None]); w_new = deq(c, s, z)[0]; nbytes += changed_bytes((lm_art[0][t:t+1], lm_art[1][t:t+1], lm_art[2][t:t+1]), (c, s, z))
        Wn[t] = w_new.to(Wn.dtype)
    return Wn, nbytes

# -------- V2 / V2h: re-GPTQ label rows from fp with Hessian incl. annotated states
def lm_hess_ann(sites):
    Ha = torch.zeros(D, D, device=dev)
    for i in sites: x = Hall[i]; Ha += args.omega * torch.outer(x, x)
    return Ha
def edit_V2(sites, labs, hess_only=False, from_base=False, margins=(1, 2, 4, 8, 16, 32), frac=0.9):
    if not sites: return W0.clone(), 0, None
    Ha = lm_hess_ann(sites); Ht = Hc_lm + Ha; Htd = Ht + 0.01 * Ht.diagonal().mean() * torch.eye(D, device=dev)
    rows = sorted({FT[l] for l in labs}); Hs = Hall[sites]; base_l = logits(Hs, W0)
    if hess_only:
        Wn = W0.clone(); nb_ = 0
        for t in rows:
            Q, c, s, z = gptq(W_lm_fp[t:t+1], Ht); Wn[t] = Q[0].to(Wn.dtype); nb_ += changed_bytes((lm_art[0][t:t+1], lm_art[1][t:t+1], lm_art[2][t:t+1]), (c, s, z))
        return Wn, nb_, None
    Hinv = torch.linalg.inv(Htd)
    for m in margins:
        Wn = W0.clone(); nb_ = 0
        for t in rows:
            ix = [k for k, l in enumerate(labs) if FT[l] == t]; wt = (W0[t] if from_base else W_lm_fp[t]).float()
            cur = base_l[ix, t]; other = base_l[ix].clone(); other[:, t] = -1e9; mx = other.max(1).values
            delta = (mx - cur + m).clamp(min=0)           # logit increase needed (margin m over best other)
            u = args.omega * (Hinv @ (Hs[ix].T @ delta))  # = w H^-1 sum d_a x_a  (omega already folded: Ht contains omega*x x^T; target term weight omega)
            Q, c, s, z = gptq((wt + u)[None], Ht); Wn[t] = Q[0].to(Wn.dtype); nb_ += changed_bytes((lm_art[0][t:t+1], lm_art[1][t:t+1], lm_art[2][t:t+1]), (c, s, z))
        if flip_rate(Hs, labs, Wn) >= frac: return Wn, nb_, m
    return Wn, nb_, m

# -------- V3: fit per-group scales of label rows (codes fixed)
def edit_V3(sites, labs, margins=(1, 2, 4, 8, 16, 32), frac=0.9, lam=1e-3, dmax=0.3):
    if not sites: return W0.clone(), 0, None
    Hs = Hall[sites]; base_l = logits(Hs, W0); rows = sorted({FT[l] for l in labs}); G = D // GS
    for m in margins:
        Wn = W0.clone(); nb_ = 0
        for t in rows:
            ix = [k for k, l in enumerate(labs) if FT[l] == t]
            c, s, z = (lm_art[0][t], lm_art[1][t], lm_art[2][t])
            contrib = (s.float().view(G, 1) * (c.view(G, GS).float() - z.float().view(G, 1)))          # [G,GS] dequant row pieces
            fa = torch.einsum("ags,gs->ag", Hs[ix].view(len(ix), G, GS), contrib)                      # per-group logit contribution for annotated
            fc = torch.einsum("ags,gs->ag", Hall[CALIDX].view(len(CALIDX), G, GS), contrib)             # for calib (should not move)
            cur = base_l[ix, t]; other = base_l[ix].clone(); other[:, t] = -1e9; mx = other.max(1).values; delta = (mx - cur + m).clamp(min=0)
            A = fc.T @ fc + args.omega * fa.T @ fa; A = A + lam * A.diagonal().mean() * torch.eye(G, device=dev)
            dl = torch.linalg.solve(A, args.omega * fa.T @ delta).clamp(-dmax, dmax)
            s_new = (s.float() * (1 + dl)).half(); w_new = deq(c[None], s_new[None], z[None])[0]; Wn[t] = w_new.to(Wn.dtype)
            nb_ += changed_bytes((c[None], s[None], z[None]), (c[None], s_new[None], z[None]))
        if flip_rate(Hs, labs, Wn) >= frac: return Wn, nb_, m
    return Wn, nb_, m

# -------- V1: tail-layer sites re-GPTQ'd (one-step activation-gradient target)
ann_cut = {i: None for i in ann_c}
def set_tail_weights(Wd):
    for (l, n), w in Wd.items(): getlin(l, n).weight.data.copy_(w.to(torch.bfloat16))
base_tail_w = {key: getlin(*key).weight.data.clone() for key in fp_w}
def ann_forward_grad(sites, labs):
    """tail forward on annotated prompts with current (base) weights; returns per-site (X [Ntok,in], G [Ntok,out], answer-mask) with G = d CE(label)/d site output"""
    items = sorted(range(len(sites)), key=lambda k: -len(ENC[sites[k]]))
    store = {key: dict(X=[], G=[]) for key in fp_w}
    cutq = make_cut([sites[k] for k in items], bs=len(sites)); lab_of = {sites[k]: labs[k] for k in items}
    for ix, h, b in cutq:
        h = h.clone().requires_grad_(True); rot = rope(h, b); outs = {}; hooks = []
        for key in fp_w:
            mod = getlin(*key)
            def mk(key):
                def f(mod, inp, out):
                    out.retain_grad(); outs[key] = (inp[0].detach(), out)
                return f
            hooks.append(mod.register_forward_hook(mk(key)))
        hh = h
        with torch.enable_grad():
            for l in range(L0, NL): hh = layer_fwd(l, hh, b, rot)
            hf = lm.model.norm(hh)[:, -1].float(); lg = hf @ W0.float().T
            tgt = torch.tensor([FT[lab_of[i]] for i in ix], device=dev); loss = torch.nn.functional.cross_entropy(lg, tgt, reduction="sum"); loss.backward()
        for hk in hooks: hk.remove()
        for key, (xin, out) in outs.items():
            v = b["valid"]; store[key]["X"].append(xin[v].float()); store[key]["G"].append(out.grad[v].float())
    return {key: (torch.cat(v["X"]), torch.cat(v["G"])) for key, v in store.items()}, cutq

def edit_V1(sites, labs, hess_only=False, from_base=False, cs=(0.0005, 0.001, 0.002, 0.004, 0.008, 0.016, 0.03, 0.06, 0.12, 0.25, 0.5), frac=0.9):
    if not sites: return None, 0, None
    set_tail_weights(base_tail_w); st, cutq = ann_forward_grad(sites, labs)
    Ht = {}; Hinv = {}
    for key in fp_w:
        X, Gr = st[key]
        Ht[key] = Hc_tail[key] + args.omega * (X.T @ X)
    def build(c):
        newW = {}; nb_ = 0
        for key in fp_w:
            Wfp = base_tail_w[key].float() if from_base else fp_w[key]; X, Gr = st[key]
            if hess_only: Wt = Wfp
            else:
                Hd = Ht[key] + 0.01 * Ht[key].diagonal().mean() * torch.eye(Ht[key].shape[0], device=dev)
                if key not in Hinv: Hinv[key] = torch.linalg.inv(Hd)
                Dlt = -(Gr.T @ X) @ Hinv[key]; Dlt = Dlt * (c * Wfp.float().norm() / (Dlt.norm() + 1e-12)); Wt = Wfp.float() + Dlt
            Q, cc, s, z = gptq(Wt, Ht[key]); newW[key] = Q; nb_ += changed_bytes(base_art[key], (cc, s, z))
        return newW, nb_
    def flips():
        set_tail_weights(newW); res = tail_eval_h(cutq); hs = torch.stack([res[i] for i in [sites[k] for k in range(len(sites))]])
        return flip_rate(hs, labs, W0)
    if hess_only:
        newW, nb_ = build(0.0); return newW, nb_, 0.0
    for c in cs:
        newW, nb_ = build(c); fr = flips()
        if fr >= frac: break
    return newW, nb_, c

def tail_sets_eval(newW):
    """per-set 0/1 correctness arrays under tail weights newW (None -> base)"""
    set_tail_weights(base_tail_w if newW is None else newW)
    res = tail_eval_h(cut_need); out = {}
    for k, idx in sets.items():
        Hm = torch.stack([res[i] for i in idx]); p, _ = preds(Hm, W0); out[k] = np.array([pp == gold[i] for pp, i in zip(p, idx)], float)
    set_tail_weights(base_tail_w); return out

def ev_lm(Wn, idx):
    p, _ = preds(Hall[idx], Wn); return np.array([pp == gold[i] for pp, i in zip(p, idx)], float)
base_sets = {k: ev_lm(W0, v) for k, v in sets.items()}; R["base_on_sets"] = {k: float(v.mean()) for k, v in base_sets.items()}
# tail-eval sanity: base weights through the tail path equal base_sets
chk = tail_sets_eval(None); R["tail_path_matches_base"] = {k: float(np.abs(chk[k] - base_sets[k]).mean()) for k in sets}; log("tail path vs base mismatch", R["tail_path_matches_base"])

# ---------------------------------------------------------------- run
VARS = []
for v in args.variants.split(","):
    if v == "B0": VARS += ["B0fp", "B0q"]
    elif v == "V2": VARS += ["V2", "V2h", "V2b"]
    elif v == "V3": VARS += ["V3"]
    elif v == "V1": VARS += ["V1", "V1h", "V1b"]
per = {v: {a: {k: [] for k in sets} for a in ARMS} for v in VARS}; bytes_ch = {v: {a: [] for a in ARMS} for v in VARS}; sel_par = {v: {a: [] for a in ARMS} for v in VARS}
art_total = tot_art
for d, sel in enumerate(draws):
    for arm in ARMS:
        sites, labs = arm_labels(arm, sel, d)
        for v in VARS:
            if v in ("V2h", "V1h") and arm in ("mixed", "randwrong"): per[v][arm] = per[v]["real"]; bytes_ch[v][arm] = bytes_ch[v]["real"]; continue   # no label used -> identical to real
            if arm == "empty":
                for k in sets: per[v][arm][k].append(base_sets[k])
                bytes_ch[v][arm].append(0); continue
            if v == "B0fp": Wn, nbb = edit_B0(sites, labs, False); pr = None
            elif v == "B0q": Wn, nbb = edit_B0(sites, labs, True); pr = None
            elif v == "V2": Wn, nbb, pr = edit_V2(sites, labs)
            elif v == "V2h": Wn, nbb, pr = edit_V2(sites, labs, hess_only=True)
            elif v == "V2b": Wn, nbb, pr = edit_V2(sites, labs, from_base=True)
            elif v == "V3": Wn, nbb, pr = edit_V3(sites, labs)
            if v in ("V1", "V1h", "V1b"):
                newW, nbb, pr = edit_V1(sites, labs, hess_only=(v == "V1h"), from_base=(v == "V1b"))
                o = tail_sets_eval(newW)
                for k in sets: per[v][arm][k].append(o[k])
            else:
                for k, idx in sets.items(): per[v][arm][k].append(ev_lm(Wn, idx))
            bytes_ch[v][arm].append(nbb); sel_par[v][arm].append(pr)
    log("draw", d, {v: {a: round(float(np.mean(per[v][a]["teach1"][-1])), 3) for a in ARMS} for v in VARS})

def boot(arm_vals, idx, other=None, B=2000):
    m = np.mean(arm_vals, 0) if other is None else np.mean(arm_vals, 0) - np.mean(other, 0)
    gids = [ALL[i]["group"] for i in idx]; ug = sorted(set(gids)); gi = {g: [] for g in ug}
    for kk, g in enumerate(gids): gi[g].append(kk)
    rs = np.random.RandomState(0); out = []
    for _ in range(B):
        pick = rs.choice(len(ug), len(ug)); items = [kk for p in pick for kk in gi[ug[p]]]; out.append(m[items].mean())
    return float(m.mean()), [float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))]
res = {}
for v in VARS:
    res[v] = dict(arms={}, gaps={}, bytes_changed={a: float(np.mean(bytes_ch[v][a])) for a in ARMS}, selected_param={a: sel_par[v][a][:4] for a in ARMS})
    for a in ARMS:
        res[v]["arms"][a] = {k: dict(zip(("mean", "ci"), boot(per[v][a][k], sets[k]))) for k in sets}
    for k in [x for x in ("teach1", "teach3") if x in sets]:
        for ctrl in ("empty", "mixed", "randwrong"):
            m, ci = boot(per[v]["real"][k], sets[k], other=per[v][ctrl][k]); res[v]["gaps"][f"real_minus_{ctrl}|{k}"] = dict(mean=m, ci=ci)
    for k in [x for x in ("nb", "nb3", "devret") if x in sets]:
        m, ci = boot(per[v]["real"][k], sets[k], other=[base_sets[k]] * args.ndraw); res[v]["gaps"][f"real_minus_base|{k}"] = dict(mean=m, ci=ci)
    # neighbour retention of base-correct neighbours per arm (all base-correct, so retention = mean acc)
R.update(results=res, ndraw=args.ndraw, nann=len(draws[0]), artifact_bytes_tail_lm=art_total, seconds=time.time() - T0)
json.dump(R, open(args.out, "w"), indent=1, default=str); log("DONE", args.out)
