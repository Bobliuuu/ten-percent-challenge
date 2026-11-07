"""TOOLSEL new-tool-as-new-label feedback experiment (lm_head closed-form edits). Runs on the GPU box:  python toolsel_fb.py --model Qwen/Qwen3-8B --out results/
Data: ONLY T1 non-locked splits through build_toolsel.py outputs (data/toolsel_{dev,calib,pool}.jsonl). Never touches locked/S2.

Scenario = a PAIR of NEW tools (A,B) taken from xlam_pool (API families disjoint from xlam_dev / xlam_calib by T1 construction).
  The prompt menu = the original menu of a pool item with the gold tool REMOVED (this is how T1 builds its nocall items): the new tool is NOT in the menu.
  Correct answer = the new tool's name (a label the base has never seen in the menu nor produced). Base answers an old tool or `none`.
  arm N: natural name (e.g. cell_density).  arm D: distinctive single-token prefix (Nova_<name> for A, Atlas_<name> for B; 'Nova','Atlas' are single Qwen tokens,
         0 occurrences as first token of any old tool name in the T1 splits).
  V1 headroom check (not the feedback test): menu keeps the gold tool (renamed as arm) -> base accuracy reported to show the base mostly solves "tool IS in the menu".
Splits of each new tool's queries by sha256(id)%10: <4 annotation pool (kept only if base wrong), 4-5 dev (alpha selection for lmhead_lda), >=6 teach (fixed & hashed BEFORE any edit).
Near-duplicate filter: teach/dev queries with word-Jaccard >= 0.5 to any annotation-pool query of the same tool are dropped.
Neighbours: N1 = base-verified-correct old-tool and `none` items from xlam_dev + s3_dev (cross-family, 400: 250 tool + 150 none if available);
            N2 = base-verified-correct tool items from the SAME xlam family as A/B whose gold is another (old) tool, with A,B stripped from their menus.
Evaluation: lm_head-only edits do not change hidden states, so item correctness (greedy decoding == label + EOS) is computed EXACTLY by teacher forcing on cached final hidden states + base bf16 logits
            with the edited rows recomputed. A real HF generate + strict scorer cross-check runs for one config per scenario (verify)."""
import json, os, sys, time, hashlib, random, argparse, collections, math
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_toolsel as B
ap = argparse.ArgumentParser(); ap.add_argument("--model", default="Qwen/Qwen3-8B"); ap.add_argument("--out", default="results"); ap.add_argument("--npairs", type=int, default=5)
ap.add_argument("--draws", type=int, default=20); ap.add_argument("--arms", default="N,D"); ap.add_argument("--verify", type=int, default=1)
A_ = ap.parse_args(); os.makedirs(A_.out, exist_ok=True)
from transformers import AutoTokenizer, AutoModelForCausalLM
T0 = time.time(); log = lambda *a: print(f"[{time.time()-T0:6.0f}s]", *a, flush=True)
tok = AutoTokenizer.from_pretrained(A_.model); tok.padding_side = "left"
lm = AutoModelForCausalLM.from_pretrained(A_.model, dtype=torch.bfloat16).cuda().eval()
for p in lm.parameters(): p.requires_grad_(False)
dev = "cuda"; D_h = lm.config.hidden_size; mname = A_.model.split("/")[-1]; W0 = lm.lm_head.weight.data.clone(); V = W0.shape[0]
h8 = lambda s: int(hashlib.sha256(s.encode()).hexdigest()[:8], 16)
rows = lambda f: [json.loads(l) for l in open(os.path.join(B.OUT, f"{f}.jsonl"))]
def chat(u):
    kw = dict(tokenize=False, add_generation_prompt=True)
    try: return tok.apply_chat_template([{"role": "user", "content": u}], enable_thinking=False, **kw)
    except TypeError: return tok.apply_chat_template([{"role": "user", "content": u}], **kw)
EOS = "<|im_end|>"
# ------------------------------------------------------------------ teacher-forced machinery
@torch.no_grad()
def tf(prompts_labels, bs=16, want_h=True):
    """for each (prompt, label): returns dict(tgt=[ids], pos0=index of first answer-predicting position, h=[n,D] bf16 final-normed hiddens, lg=[n,V] bf16 base logits)"""
    out = [None] * len(prompts_labels); order = sorted(range(len(prompts_labels)), key=lambda i: len(prompts_labels[i][0]))
    for s in range(0, len(order), bs):
        idx = order[s:s + bs]; seqs = []
        for i in idx:
            p, l = prompts_labels[i]; pi = tok(p, add_special_tokens=False).input_ids; gi = tok(l + EOS, add_special_tokens=False).input_ids; seqs.append((pi, gi))
        L = max(len(a) + len(b) for a, b in seqs); ids = torch.full((len(idx), L), tok.pad_token_id, device=dev); am = torch.zeros((len(idx), L), dtype=torch.long, device=dev)
        for r, (a, b) in enumerate(seqs): ids[r, :len(a) + len(b)] = torch.tensor(a + b, device=dev); am[r, :len(a) + len(b)] = 1
        h = lm.model(input_ids=ids, attention_mask=am).last_hidden_state
        for r, (a, b) in enumerate(seqs):
            hs = h[r, len(a) - 1:len(a) + len(b) - 1]; lg = lm.lm_head(hs)
            out[idx[r]] = dict(tgt=b, h=hs.clone(), lg=lg)
    return out
class EvalSet:
    """positions of many items stacked for fast evaluation under lm_head row edits"""
    def __init__(self, name, items, labels=None):
        self.name = name; self.items = items; recs = tf([(it["prompt"], it["label"] if labels is None else labels[i]) for i, it in enumerate(items)])
        self.tgt = torch.tensor([t for r in recs for t in r["tgt"]], device=dev); self.item = torch.tensor([i for i, r in enumerate(recs) for _ in r["tgt"]], device=dev)
        self.h = torch.cat([r["h"] for r in recs]); self.lg = torch.cat([r["lg"] for r in recs]); self.n = len(items)
        base_pred = self.lg.argmax(1); self.base_ok = self._item_ok(base_pred == self.tgt)
    def _item_ok(self, okpos):
        bad = torch.zeros(self.n, device=dev); bad.index_add_(0, self.item, (~okpos).float()); return (bad == 0)
    def ok(self, edits):
        if not edits: return self.base_ok
        E = torch.tensor(sorted(edits), device=dev); Wn = torch.stack([edits[int(t)] for t in E]).to(torch.bfloat16)
        lg = self.lg.clone(); lg[:, E] = torch.nn.functional.linear(self.h, Wn); return self._item_ok(lg.argmax(1) == self.tgt)
# ------------------------------------------------------------------ data
pool = rows("toolsel_pool"); devr = rows("toolsel_dev"); calr = rows("toolsel_calib")
def toks_of(q): return set(q.lower().split())
def jac(a, b): return len(a & b) / max(1, len(a | b))
# full T1 pool items (need the tool schemas to rebuild menus): reload raw jsonl for xlam pool
rawp = {x["id"]: x for x in (json.loads(l) for l in open(os.path.join(B.T1, "s1_pool.jsonl")))}
byid_pool = {}
for r in pool:
    if r["src"] == "xlam_pool": byid_pool[r["id"]] = rawp[r["id"].split("|", 2)[2]]
import re
norm = lambda s: re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", s.lower())).strip()
def clusters(qs, th=0.7):
    """single-linkage components of normalised word-set Jaccard >= th; returns list of component ids per query (component id = smallest member index)"""
    S = [set(q.split()) for q in qs]; par = list(range(len(qs)))
    def f(i):
        while par[i] != i: par[i] = par[par[i]]; i = par[i]
        return i
    for i in range(len(qs)):
        for k in range(i):
            if len(S[i] & S[k]) / max(1, len(S[i] | S[k])) >= th: par[f(i)] = f(k)
    return [f(i) for i in range(len(qs))]
tool_items = collections.defaultdict(list)
for r in pool:
    if r["src"] == "xlam_pool" and r["kind"] == "tool" and len(r["menu"]) >= 2: tool_items[r["answer"]].append(r)
cands = []
for n, L in tool_items.items():
    seen = {}; 
    for r in L: seen.setdefault(norm(r["query"]), r)
    if len(seen) < 30: continue
    cl = clusters(list(seen)); 
    if len(set(cl)) >= 30: cands.append(n)
cands = sorted(cands, key=lambda n: B.sha("pair|" + n))
PAIRS = [(cands[2 * i], cands[2 * i + 1]) for i in range(min(A_.npairs, len(cands) // 2))]
log("pairs", PAIRS, "cand tools", len(cands))
PREF = {0: "Nova", 1: "Atlas"}
gen_prompts = ["Explain why the sky is blue in two sentences.", "What is the capital of Australia?", "Write a haiku about autumn.", "Translate 'good morning' into French.", "Give three tips for a job interview.", "What is 17 times 23?", "Summarize Romeo and Juliet in one sentence.", "Write a Python function that reverses a string.", "What causes tides?", "Define photosynthesis briefly."]
# generic statistics from UNLABELLED text: chat prompts + their base responses + 120 unlabeled calib prompts rendered in the task format (NO labels used)
@torch.no_grad()
def gen(prompts, mx, bs=32):
    outs = [None] * len(prompts); order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
    for s in range(0, len(order), bs):
        idx = order[s:s + bs]; enc = tok([prompts[i] for i in idx], return_tensors="pt", padding=True).to(dev)
        o = lm.generate(**enc, max_new_tokens=mx, do_sample=False, pad_token_id=tok.pad_token_id)
        for j, i in enumerate(idx): outs[i] = o[j, enc.input_ids.shape[1]:].tolist()
    return outs
@torch.no_grad()
def hid(texts_ids):
    xs = []
    for ids in texts_ids:
        h = lm.model(input_ids=torch.tensor([ids], device=dev)).last_hidden_state[0].float(); xs.append(h)
    return xs
GP = [chat(g) for g in gen_prompts]; go = gen(GP, 40)
cal_prompts = [chat(r["user"]) for r in random.Random(0).sample(calr, 120)]
xs = hid([tok(p, add_special_tokens=False).input_ids + o for p, o in zip(GP, go)]) + hid([tok(p, add_special_tokens=False).input_ids for p in cal_prompts])
X = torch.cat(xs); MU = X.mean(0); SIG = torch.cov(X.T)
co = gen(cal_prompts[:100], 40); xp = []
for p, o in zip(cal_prompts[:100], co):
    n = len(tok(p, add_special_tokens=False).input_ids); h = hid([tok(p, add_special_tokens=False).input_ids + o])[0]; xp.append(h[n - 1:-1])
XP = torch.cat(xp); MUP = XP.mean(0); SIGP = torch.cov(XP.T); del X, XP
log("generic stats done")
# ------------------------------------------------------------------ edits (same definitions as feedback_v3 fb6)
INV = {}
def solve_dir(mu, sig, d, ridge=0.1):
    k = id(sig)
    if k not in INV: INV[k] = torch.linalg.inv(sig + ridge * sig.diagonal().mean() * torch.eye(D_h, device=dev))
    d = INV[k] @ d; return d / d.norm()
def make_edits(S, mode, alpha=None):
    """S: list of (token, hidden[D] float, base_logits[V]) at annotated sites where base argmax != token. returns {token: new row float32}"""
    by = collections.defaultdict(list)
    for t, h, lg in S: by[t].append((h, lg))
    ed = {}
    for t, L in by.items():
        Hs = torch.stack([h for h, _ in L]); w = W0[t].float()
        if mode == "plain": d = Hs.mean(0); d = d / d.norm(); a = alpha
        elif mode == "lda": d = solve_dir(MU, SIG, Hs.mean(0) - MU); a = alpha
        else:
            d = solve_dir(MU, SIG, Hs.mean(0) - MU) if mode == "cal" else solve_dir(MUP, SIGP, Hs.mean(0) - MUP)
            lgs = torch.stack([lg.float() for _, lg in L]); lgs[:, t] = -1e9; mx = lgs.max(1).values; a = 2.0
            for aa in [0.05, 0.1, 0.15, 0.2, 0.3, 0.45, 0.7, 1.0, 1.5, 2.0]:
                if float((((Hs @ (w + aa * w.norm() * d)) - mx) >= 1.0).float().mean()) >= 0.9: a = aa; break
        ed[int(t)] = (w + a * w.norm() * d)
    return ed
SITE_CACHE = {}
def sites_for(it, label):
    """annotation (prompt,label) -> list of (token, hidden, logits) at positions where the base argmax != target (teacher forced)"""
    key = (it["id"], label)
    if key not in SITE_CACHE:
        r = tf([(it["prompt"], label)])[0]; pred = r["lg"].argmax(1); SITE_CACHE[key] = [(int(t), r["h"][j].float(), r["lg"][j]) for j, t in enumerate(r["tgt"]) if int(pred[j]) != t]
    return SITE_CACHE[key]
# ------------------------------------------------------------------ build scenario sets
def mk_item(x, name_map, keep_gold, label, tools_override=None, strip=()):
    tools = [t for t in x["tools"] if t["name"] not in strip]
    g = x["gold"][0]["name"]
    if keep_gold: tools = [dict(t, name=name_map.get(t["name"], t["name"])) for t in tools]
    else: tools = [t for t in tools if t["name"] != g]
    return dict(id=x["id"], prompt=chat(B.render(tools, x["query"])), label=label, query=x["query"], menu=[t["name"] for t in tools])
def nm(tool, arm, idx): return tool if arm == "N" else f"{PREF[idx]}_{tool}"
def base_pred_text(items):
    outs = gen([it["prompt"] for it in items], 28); return [tok.decode(o, skip_special_tokens=True).strip() for o in outs]
# neighbours N1 (model-specific base verification below)
devN = [r for r in devr]; random.Random(11).shuffle(devN)
def N1_items(): return [dict(id=r["id"], prompt=chat(r["user"]), label=r["answer"], kind=r["kind"], menu=r["menu"]) for r in devN]
n1_all = N1_items(); log("N1 candidates", len(n1_all))
tfN1 = EvalSet("N1cand", n1_all[:1800])
okN1 = tfN1.base_ok.cpu().numpy(); tool_ok = [it for it, o in zip(tfN1.items, okN1) if o and it["kind"] == "tool"]; none_ok = [it for it, o in zip(tfN1.items, okN1) if o and it["kind"] == "none"]
N1 = tool_ok[:250] + none_ok[:150]; N1 += (tool_ok[250:] + none_ok[150:])[:max(0, 400 - len(N1))]
log("N1 verified", len(N1), "tool", sum(i["kind"] == "tool" for i in N1), "none", sum(i["kind"] == "none" for i in N1), "base acc on candidates", okN1.mean())
EN1 = EvalSet("N1", N1)
def acc_ci(v): return float(np.mean(v))
results = dict(model=A_.model, pairs=PAIRS, N1=len(N1), base_N1cand_acc=float(okN1.mean()), scen=[], files={})
for arm in A_.arms.split(","):
    for pi, (ta, tb) in enumerate(PAIRS):
        names = {ta: nm(ta, arm, 0), tb: nm(tb, arm, 1)}; labels_all = list(names.values())
        sc = f"{mname}_{arm}_pair{pi}"; log("=== scenario", sc, names)
        # queries of A and B
        splits = {"ann": [], "dev": [], "teach": []}; fam = set(); v1 = []; v1split = {}
        for t_ in names:
            seen = {}
            for r_ in tool_items[t_]: seen.setdefault(norm(r_["query"]), r_)      # exact (normalised) duplicate queries removed
            qs = list(seen); cl = clusters(qs)
            for q_, c_ in zip(qs, cl):
                x = byid_pool[seen[q_]["id"]]; h = h8("split|" + t_ + "|" + qs[c_]) % 10      # whole near-duplicate cluster (Jaccard>=0.7) goes to ONE split
                fam.add(x["family"]); sp_ = "ann" if h < 4 else ("dev" if h < 6 else "teach")
                splits[sp_].append(dict(mk_item(x, names, False, names[t_]), tool=t_)); v1.append(dict(mk_item(x, names, True, names[t_]), tool=t_)); v1split[x["id"]] = sp_
        annset = collections.defaultdict(list)
        for it in splits["ann"]: annset[it["tool"]].append(toks_of(norm(it["query"])))
        for k in ("dev", "teach"):
            for it in splits[k]: it["maxjac_ann"] = max([jac(toks_of(norm(it["query"])), a) for a in annset[it["tool"]]] or [0.0])
        teach = splits["teach"][:120]; devi = splits["dev"][:60]
        teach_hash = hashlib.sha256("|".join(sorted(i["id"] for i in teach)).encode()).hexdigest()
        # N2: same-family old-tool items with A,B stripped
        N2c = []
        for iid, x in byid_pool.items():
            if x["family"] not in fam or x["kind"] != "single" or len(x["gold"]) != 1 or x["gold"][0]["name"] in names: continue
            tools = [t for t in x["tools"] if t["name"] not in names]
            if not tools or x["gold"][0]["name"] not in [t["name"] for t in tools]: continue
            N2c.append(dict(id=iid, prompt=chat(B.render(tools, x["query"])), label=x["gold"][0]["name"], menu=[t["name"] for t in tools]))
        random.Random(5).shuffle(N2c)
        # base outputs of ann pool -> keep base-wrong ones
        ann_all = splits["ann"]; bo = base_pred_text(ann_all); ann = [dict(a, wrong=o) for a, o in zip(ann_all, bo) if o != a["label"]]
        EN2c = EvalSet("N2c", N2c[:400]); N2 = [it for it, o in zip(EN2c.items, EN2c.base_ok.cpu().numpy()) if o][:300]; EN2 = EvalSet("N2", N2)
        ET = EvalSet("teach", teach); ED = EvalSet("dev", devi); EDn = EvalSet("devN", N1[:80])
        # V1 headroom: base accuracy when the (renamed) tool IS in the menu
        v1t = [i for i in v1 if v1split[i["id"]] == "teach"][:120]; ev1 = EvalSet("v1", v1t) if v1t else None
        strict_mask = np.array([t["maxjac_ann"] < 0.5 for t in teach]); info = dict(scen=sc, n_teach_strict=int(strict_mask.sum()), teach_strict_idx=np.nonzero(strict_mask)[0].tolist(), teach_maxjac_median=float(np.median([t["maxjac_ann"] for t in teach])), arm=arm, names=names, family=sorted(fam), n_ann_pool=len(ann), n_ann_all=len(ann_all), n_dev=len(devi), n_teach=len(teach), teach_per_tool={t: sum(i["tool"] == t for i in teach) for t in names}, teach_ids_sha256=teach_hash,
                    n_N1=len(N1), n_N2=len(N2), n_N2_cand=len(EN2c.items), base_teach=float(ET.base_ok.float().mean()), base_dev=float(ED.base_ok.float().mean()), base_V1_in_menu_teach_acc=(float(ev1.base_ok.float().mean()) if ev1 else None), n_V1=len(v1t),
                    base_first_outputs=collections.Counter(i["wrong"] for i in ann).most_common(5), first_token_new=[tok.decode(tok(n, add_special_tokens=False).input_ids[:1]) for n in labels_all], ntok_new=[len(tok(n + EOS, add_special_tokens=False).input_ids) for n in labels_all])
        log(json.dumps(info))
        json.dump(dict(scen=sc, names=names, teach_ids=sorted(i["id"] for i in teach), teach_ids_sha256=teach_hash, ann_pool_ids=sorted(i["id"] for i in ann), written_before_any_edit=True), open(f"{A_.out}/PREREG_{sc}.json", "w"))
        if len(teach) < 25 or len(ann) < 25: log("SKIP scenario: too few", len(teach), len(ann)); info["skipped"] = True; results["scen"].append(info); continue
        # ---------------- methods
        # lmhead_lda alpha: chosen on DEV (labelled) data with draw -1 of 25 annotations, subject to dev-neighbour >= 0.98
        def draw_anns(seed, n): r = random.Random(seed); L = list(ann); r.shuffle(L); return L[:n]
        def label_for(a, ctl, r):
            if ctl == "real": return a["label"]
            if ctl == "derange": return [x for x in labels_all if x != a["label"]][0]
            if ctl == "wrong": return r.choice(a["menu"]) if a["menu"] else "none"
            if ctl == "mixed": return a["wrong"] if (r.random() < 0.5 and a["wrong"] and a["wrong"] in set(a["menu"]) | {"none"}) else a["label"]
        def edits_for(anns, ctl, seed, mode, alpha):
            r = random.Random(seed * 7919 + h8(ctl) % 1000); S = []
            for a in anns: S += sites_for(a, label_for(a, ctl, r))
            return make_edits(S, mode, alpha) if S else {}
        def evalall(ed): return dict(teach=ET.ok(ed).float().cpu().numpy(), dev=ED.ok(ed).float().cpu().numpy(), N1=EN1.ok(ed).float().cpu().numpy(), N2=EN2.ok(ed).float().cpu().numpy(), devN=EDn.ok(ed).float().cpu().numpy())
        sweep = []
        for a_ in [0.05, 0.1, 0.2, 0.3, 0.45, 0.7, 1.0, 1.5, 2.0, 3.0, 4.0]:
            g = [];
            for sd in range(3):
                e = evalall(edits_for(draw_anns(1000 + sd, 25), "real", sd, "lda", a_)); g.append((e["dev"].mean() - ED.base_ok.float().mean().item(), e["devN"].mean()))
            sweep.append(dict(alpha=a_, dev_gain=float(np.mean([x[0] for x in g])), devN=float(np.mean([x[1] for x in g]))))
        okl = [s for s in sweep if s["devN"] >= 0.98] or sweep; best = max(okl, key=lambda s: s["dev_gain"])["alpha"]; info["lda_alpha_chosen_on_dev"] = best; info["sweep"] = sweep; log("sweep", sweep, "chosen", best)
        METH = {"lmhead_lda": ("lda", best), "lmhead_plain": ("plain", 0.03), "lmhead_plain_xs": ("plain", 0.01), "lmhead_cal": ("cal", None), "lmhead_cal2": ("cal2", None)}
        draws = []
        for n in (3, 5, 8, 12, 25):
            for d in range(A_.draws):
                anns = draw_anns(d * 131 + n, n)
                for mname_, (mode, al) in METH.items():
                    rec = dict(n=n, draw=d, method=mname_)
                    for ctl in ("real", "wrong", "mixed", "derange"):
                        e = evalall(edits_for(anns, ctl, d, mode, al)); rec[ctl] = dict(teach=float(e["teach"].mean()), teach_items=e["teach"].astype(int).tolist() if ctl == "real" else None, N1=float(e["N1"].mean()), N2=float(e["N2"].mean()))
                    draws.append(rec)
            log(sc, "n", n, "done", {m: round(float(np.mean([r["real"]["teach"] for r in draws if r["n"] == n and r["method"] == m])), 3) for m in METH})
        info["draws"] = draws; info["base_teach_items"] = ET.base_ok.int().cpu().tolist(); results["scen"].append(info)
        # ---------------- real-generation cross-check (one config), strict scorer
        if A_.verify:
            try:
                anns = draw_anns(25, 25); ed = edits_for(anns, "real", 0, "lda", best); fast = ET.ok(ed).cpu().numpy(); fn = EN1.ok(ed).cpu().numpy()
                for t, row in ed.items(): lm.lm_head.weight.data[t] = row.to(torch.bfloat16)
                gt = [tok.decode(o, skip_special_tokens=True) for o in gen([i["prompt"] for i in ET.items], 28)]; gn = [tok.decode(o, skip_special_tokens=True) for o in gen([i["prompt"] for i in EN1.items[:200]], 28)]
                lm.lm_head.weight.data.copy_(W0)
                strict = lambda text, lab: text.strip() == lab and len(text.strip()) > 0
                real_t = np.array([strict(t, i["label"]) for t, i in zip(gt, ET.items)]); real_n = np.array([strict(t, i["label"]) for t, i in zip(gn, EN1.items[:200])])
                info["verify"] = dict(teach_fast=float(fast.mean()), teach_real=float(real_t.mean()), agree_teach=float((fast == real_t).mean()), N1_fast200=float(fn[:200].mean()), N1_real200=float(real_n.mean()), agree_N1=float((fn[:200] == real_n).mean()), sample_outputs=gt[:4])
                log("VERIFY", info["verify"])
            except Exception as e_: log("verify failed", repr(e_)); lm.lm_head.weight.data.copy_(W0)
        assert torch.equal(lm.lm_head.weight.data, W0)
        json.dump(results, open(f"{A_.out}/{mname}_toolsel_fb.json", "w"), default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
log("DONE")
json.dump(results, open(f"{A_.out}/{mname}_toolsel_fb.json", "w"), default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
