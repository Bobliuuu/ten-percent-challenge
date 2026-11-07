#!/usr/bin/env python
"""Build TOX (content moderation) splits. Deterministic. Refuses to overwrite data/LOCK.json.
TASK: input = one user comment/message; real output = the model GENERATES one moderation verdict word from a fixed taxonomy
  {safe, toxic, hate, sexual, violence}. (self-harm is NOT in the taxonomy: no public source gives >=330 clean single-label self-harm
  prompts that survive dedup; Aegis has 265. Stated honestly in REPORT.md.)
Item schema (jsonl): id, source(S1|S2|S3), kind (= class), gold (= class), text, group, hard (bool: lexicon-bearing item), lex [lexicon groups hit].
Sources (three genuinely different platforms / periods / licences):
  S1 = Jigsaw/Google civil_comments (CC0-1.0). News-site reader comments, 2015-2017.   build/dev/calib/pool + LOCKED slice
  S3 = HateXplain (Hate-speech-CNERG, CC-BY-4.0). Gab + Twitter posts, 2019-2020.      OOD DEV PROXY ONLY (dev/pool). never locked
  S2 = NVIDIA Aegis-AI-Content-Safety-Dataset-2.0 (CC-BY-4.0). USER PROMPTS to an LLM (derived from Anthropic HH red-team), 2024. LOCKED OOD block
       (+ a small group-disjoint 's2_probe' slice of the same corpus used ONLY for the rule-8 base-health check of S2 cells; never for tuning)
Label mapping (single-label ground truth; anything ambiguous is DROPPED from all splits, not guessed):
  civil_comments (fractions of raters; thresholds raised from .5 to .6/.7 after a DEV-only look showed .5 labels are too noisy to be measurable: S1/sexual base acc 0.08):
     safe = toxicity==0 ; toxic = toxicity>=.7 and (insult>=.5 or obscene>=.5) and identity_attack,sexual_explicit,threat all <.2 ;
     hate = identity_attack>=.6 and toxicity>=.6 and sexual,threat <.2 ; sexual = sexual_explicit>=.6 and toxicity>=.6 and identity,threat <.2 ; violence = threat>=.6 and toxicity>=.6 and identity,sexual <.2.
  HateXplain: UNANIMOUS 3/3 annotators ; normal->safe, offensive->toxic, hatespeech->hate ; any disagreement dropped.
  Aegis prompts: categories mapped Hate/Identity Hate->hate, Harassment+Profanity->toxic, Sexual->sexual, Violence->violence ; item must have exactly ONE
     mapped taxonomy class and prompt_label=unsafe (categories outside the taxonomy, e.g. Criminal Planning, PII, are ignored when attached to a mapped one);
     safe = prompt_label safe with no category ('Needs Caution' dropped). Single-label items whose only category is outside taxonomy are dropped.
Hard negatives: safe items are split 50/50 'hard' (contain >=1 lexicon hit: profanity/slur lexicon, violence, sexual, identity terms, insults -> news about violence,
  quotes, reclaimed/in-context terms) and 'plain'. Lexicon lists are stratification tools only. A lexicon baseline is reported in diagnostics.
Splitting S1 BY GROUP: group = connected component of (near-duplicate pairs Jaccard>=0.6) U (TF-IDF/SVD/KMeans topic cluster, k=2000 (S1) / 250 (S2) / 200 (S3)). No thread/author id
  exists in the HF civil_comments release, topic clusters are the best available proxy for 'same news article discussion'. Whole groups -> locked/dev/calib/pool.
Dedup: exact normalised text, near-dup Jaccard>=0.6 (whole-token sets; ONE representative kept per near-dup component in every source), cross-source (priority S2 > S1 > S3: the lower-priority copy is dropped).
Touched scan: locked-candidate texts are grepped across ~/ten-percent (json/jsonl/py/md/txt/log); hits dropped from locked.
Locked rows are never printed. Only counts are logged."""
import hashlib, json, os, re, sys, codecs, collections, datetime, subprocess, random, glob
import numpy as np, pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import normalize
HERE = os.path.dirname(os.path.abspath(__file__)); RAW = f"{HERE}/raw"; D = os.environ.get("TOX_BUILD_OUT", f"{HERE}/data")
os.makedirs(D, exist_ok=True)
assert not os.path.exists(f"{D}/LOCK.json"), "LOCK.json exists; never overwrite"
sha = lambda b: hashlib.sha256(b).hexdigest()
H = lambda s: int(hashlib.sha256(s.encode()).hexdigest()[:12], 16)
def log(*a): print(*a, flush=True)
CLASSES = ["safe", "toxic", "hate", "sexual", "violence"]
# targets per (split, pseudo-class); locked cells are the exam cells (>=330 required)
PC = ["safe_hard", "safe_plain", "toxic", "hate", "sexual", "violence"]
TGT = dict(
    S1=dict(locked=dict(safe_hard=230, safe_plain=230, toxic=450, hate=450, sexual=450, violence=450),
            dev=dict(safe_hard=100, safe_plain=100, toxic=200, hate=200, sexual=200, violence=200),
            calib=dict(safe_hard=75, safe_plain=75, toxic=150, hate=150, sexual=150, violence=150)),
    S2=dict(locked=dict(safe_hard=230, safe_plain=230, toxic=450, hate=450, sexual=450, violence=450),
            probe=dict(safe_hard=50, safe_plain=50, toxic=100, hate=100, sexual=100, violence=100)),
    S3=dict(dev=dict(safe_hard=100, safe_plain=100, toxic=200, hate=200)))
OVERSAMPLE = 1.1           # assign this much more than target to locked, then subsample (touched scan / caps)
POOL_CAP = 5000            # per pseudo-class pool rows saved
JT = 0.6
GROUP_CAP_FRAC = 0.2       # max share of a locked cell from one group

# ---------------- lexicon (stratification + diagnostics only). slurs/profanity are rot13-encoded.
R13 = "avttre avttn snttbg snt ergneq ergneqrq xvxr puvax fcvp qlxr genaal phag juber fyhg ovgpu onfgneq shpx shpxvat shpxrq fuvg nffubyr qvpx chffl pbpx"
LEX = {
    "prof": [codecs.decode(w, "rot13") for w in R13.split()],
    "viol": "kill killed killing murder murdered shoot shot shooting bomb stab stabbed attack rape raped terror terrorist terrorists gun guns weapon weapons blood massacre execute hang die dies dead suicide beat punch strangle behead".split(),
    "sex": "sex sexual porn naked nude breast breasts penis vagina erotic prostitute orgasm incest pedophile pedophilia molest".split(),
    "insult": "idiot idiots stupid moron morons dumb loser losers scum trash pathetic ugly fool jerk liar".split(),
    "ident": "gay gays lesbian trans transgender black blacks white muslim muslims islam jew jews jewish christian immigrant immigrants refugee refugees women woman female asian latino hispanic african racist racism sexist homosexual feminist".split(),
}
LEXRE = {k: re.compile(r"\b(?:" + "|".join(sorted(set(v), key=len, reverse=True)) + r")\b", re.I) for k, v in LEX.items()}
STRONG = ("prof", "viol", "sex", "insult")
def lexhits(t): return sorted(k for k, r in LEXRE.items() if r.search(t))
def is_hard(lx): return any(k in lx for k in STRONG) or "ident" in lx
def ws(t): return " ".join(str(t).split())
def nrm(t): return re.sub(r"\W+", " ", t.lower()).strip()

# ---------------- raw pins
def pins():
    P = {}
    for f in sorted(glob.glob(f"{RAW}/civil/data/*.parquet")) + sorted(glob.glob(f"{RAW}/aegis/*.json")) + sorted(glob.glob(f"{RAW}/hatexplain/plain_text/*/*.parquet")):
        P[os.path.relpath(f, RAW)] = dict(sha256=sha(open(f, "rb").read()), bytes=os.path.getsize(f))
    P["_hf_repos"] = dict(civil="google/civil_comments@f2970eb3a55777454c94069077cc8d9b5866312d (cc0-1.0)",
                          aegis="nvidia/Aegis-AI-Content-Safety-Dataset-2.0@d86bb8bedff51d25ac834ab7838f1cc61acb7a2c (cc-by-4.0)",
                          hatexplain="Hate-speech-CNERG/hatexplain@refs/convert/parquet=b0e5676acc1d0dcb31d6ce8e93e7e94109044ed6 (cc-by-4.0)")
    return P
PINS = pins()

# ---------------- load candidates
cand = []   # dict(src, text, gold, hard, lex, _o)
def add(src, text, gold, o=None):
    text = ws(text)
    if not (20 <= len(text) <= 1000): return
    lx = lexhits(text)
    cand.append(dict(src=src, text=text, gold=gold, lex=lx, hard=is_hard(lx), _o=o))
# S1 civil_comments
d = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f"{RAW}/civil/data/*.parquet"))], ignore_index=True)
L = d.text.str.len(); d = d[(L >= 20) & (L <= 1000)].reset_index(drop=True)
t, ia, se, th = d.toxicity.values, d.identity_attack.values, d.sexual_explicit.values, d.threat.values
lab = np.full(len(d), "", dtype=object)
ins_, ob_ = d.insult.values, d.obscene.values
lab[t == 0] = "safe"
lab[(t >= .7) & ((ins_ >= .5) | (ob_ >= .5)) & (ia < .2) & (se < .2) & (th < .2)] = "toxic"
lab[(ia >= .6) & (se < .2) & (th < .2) & (t >= .6)] = "hate"
lab[(se >= .6) & (ia < .2) & (th < .2) & (t >= .6)] = "sexual"
lab[(th >= .6) & (ia < .2) & (se < .2) & (t >= .6)] = "violence"
d["gold"] = lab; d = d[d.gold != ""].reset_index(drop=True)
d["_h"] = [H("s1|" + x) for x in d.text.values]
log("civil labelled (counts only):", dict(collections.Counter(d.gold)))
# candidate sampling (hash order): safe -> compute lex only on 400k to pick 20k hard + 20k plain
safe = d[d.gold == "safe"].sort_values("_h").head(400000)
sl = [(lexhits(x), x) for x in safe.text.values]
sh = [x for lx, x in sl if is_hard(lx)][:22000]; sp = [x for lx, x in sl if not is_hard(lx)][:22000]
for x in sh + sp: add("S1", x, "safe")
for c, cap in [("toxic", 30000), ("hate", 12500), ("sexual", 5000), ("violence", 5000)]:
    for x in d[d.gold == c].sort_values("_h").head(cap).text.values: add("S1", x, c)
del d
log("S1 candidates", collections.Counter(c["gold"] for c in cand if c["src"] == "S1"))
# S3 HateXplain
hx = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f"{RAW}/hatexplain/plain_text/*/*.parquet"))], ignore_index=True)
HXM = {0: "hate", 1: "safe", 2: "toxic"}   # hatespeech, normal, offensive
for _, r in hx.iterrows():
    labs = [int(x) for x in r["annotators"]["label"]]; c = collections.Counter(labs).most_common()
    if c[0][1] < 3: continue     # unanimous only
    add("S3", " ".join(r["post_tokens"]), HXM[c[0][0]], o=r["id"])
log("S3 candidates", collections.Counter(c["gold"] for c in cand if c["src"] == "S3"))
# S2 Aegis prompts
AM = {"Hate/Identity Hate": "hate", "Harassment": "toxic", "Profanity": "toxic", "Sexual": "sexual", "Violence": "violence"}
for f in ["train", "validation", "test"]:
    for r in json.load(open(f"{RAW}/aegis/{f}.json")):
        p = r.get("prompt")
        if not p or p == "REDACTED": continue
        cats = [x.strip() for x in (r["violated_categories"] or "").split(",") if x.strip()]
        if r["prompt_label"] == "safe":
            if cats: continue          # 'Needs Caution' etc: ambiguous, dropped
            g = "safe"
        else:
            m = {AM[x] for x in cats if x in AM}
            if len(m) != 1: continue
            g = next(iter(m))
        add("S2", p, g, o=r["id"])
log("S2 candidates", collections.Counter(c["gold"] for c in cand if c["src"] == "S2"))
N = len(cand)

# ---------------- dedup: exact, near-dup (Jaccard>=0.6), cross-source priority
PRI = dict(S2=0, S1=1, S3=2)
drop = {}      # idx -> reason
seen = {}
for i in sorted(range(N), key=lambda i: (PRI[cand[i]["src"]], H(cand[i]["text"]))):
    k = nrm(cand[i]["text"])
    if k in seen: drop[i] = "exact_dup" if cand[i]["src"] == cand[seen[k]]["src"] else "exact_dup_cross_source"
    else: seen[k] = i
keep0 = [i for i in range(N) if i not in drop]
texts = [cand[i]["text"] for i in keep0]
cv = CountVectorizer(binary=True, token_pattern=r"(?u)\b\w+\b", lowercase=True, dtype=np.float32)
X = cv.fit_transform(texts).tocsr(); sz = np.asarray(X.sum(1)).ravel()
pairs = []; XT = X.T.tocsr()
for a in range(0, X.shape[0], 700):      # EXACT all-pairs Jaccard (no df cut-off, no candidate filter): full token-set sparse product in chunks
    P = (X[a:a + 700] @ XT).tocoo(); i, j, v = P.row + a, P.col, P.data
    m = j > i; i, j, v = i[m], j[m], v[m]; jac = v / (sz[i] + sz[j] - v); k = jac >= JT
    pairs += list(zip(i[k].tolist(), j[k].tolist()))
log("near-dup pairs (J>=0.6):", len(pairs))
# union-find helper
class UF:
    def __init__(s, n): s.p = list(range(n))
    def f(s, x):
        while s.p[x] != x: s.p[x] = s.p[s.p[x]]; x = s.p[x]
        return x
    def u(s, a, b):
        a, b = s.f(a), s.f(b)
        if a != b: s.p[b] = a
ufd = UF(len(keep0))
for a, b in pairs: ufd.u(a, b)
comp = collections.defaultdict(list)
for k in range(len(keep0)): comp[ufd.f(k)].append(k)
cross_dropped = 0; dup_dropped = 0
for r, mem in comp.items():
    if len(mem) == 1: continue
    srcs = {cand[keep0[k]]["src"] for k in mem}; best = min(PRI[s] for s in srcs)
    if len(srcs) > 1:
        for k in mem:
            if PRI[cand[keep0[k]]["src"]] != best: drop[keep0[k]] = "neardup_cross_source"; cross_dropped += 1
    # within same (best) source: keep ONE representative per near-dup component for S2 (locked) and S3; S1 keeps them but group-locks them (same group)
    rest = [k for k in mem if keep0[k] not in drop]
    if rest:
        rest.sort(key=lambda k: H(cand[keep0[k]]["text"]))
        for k in rest[1:]: drop[keep0[k]] = "neardup_within_source"; dup_dropped += 1
log("dropped:", dict(collections.Counter(drop.values())))
alive = [i for i in range(N) if i not in drop]
keepmap = {i: k for k, i in enumerate(keep0)}
# near-dup edges among alive (S1 groups)
edges = [(keep0[a], keep0[b]) for a, b in pairs if keep0[a] not in drop and keep0[b] not in drop]

# ---------------- groups: topic cluster U near-dup, per source
def topic_clusters(idx, K):
    tx = [cand[i]["text"] for i in idx]
    tv = TfidfVectorizer(max_features=40000, min_df=3, sublinear_tf=True, stop_words="english")
    M = tv.fit_transform(tx); Z = normalize(TruncatedSVD(100, random_state=0).fit_transform(M))
    return MiniBatchKMeans(K, random_state=0, n_init=3, batch_size=4096).fit_predict(Z)
GRP = {}
for src, K in [("S1", 2000), ("S2", 250), ("S3", 200)]:
    idx = [i for i in alive if cand[i]["src"] == src]; cl = topic_clusters(idx, K)
    pos = {i: k for k, i in enumerate(idx)}; uf = UF(len(idx)); first = {}
    for k, c in enumerate(cl):
        if c in first: uf.u(first[c], k)
        else: first[c] = k
    for a, b in edges:
        if a in pos and b in pos: uf.u(pos[a], pos[b])
    gs = collections.Counter(uf.f(k) for k in range(len(idx)))
    for i in idx: GRP[i] = f"{src.lower()}g{uf.f(pos[i])}"
    sizes = sorted(gs.values(), reverse=True)
    log(src, "groups", len(gs), "largest", sizes[:5], "n", len(idx))
for i in alive: cand[i]["group"] = GRP[i]
def pcls(c): return c["gold"] if c["gold"] != "safe" else ("safe_hard" if c["hard"] else "safe_plain")

# ---------------- group-level assignment
def assign(src, targets, order):
    """targets: {split: {pc: n}} in priority order. whole groups go to one split; deficit-greedy; big groups that would overshoot go to pool."""
    idx = [i for i in alive if cand[i]["src"] == src]
    gc = collections.defaultdict(collections.Counter)
    for i in idx: gc[cand[i]["group"]][pcls(cand[i])] += 1
    tot = collections.Counter(c for g in gc.values() for c, n in g.items() for _ in range(n)); lk_t = targets[order[0]]
    rare = {c for c in lk_t if tot[c] < 10 * lk_t[c]}
    rem = {s: {c: int(n * (OVERSAMPLE if s == "locked" else 1.0)) for c, n in tg.items()} for s, tg in targets.items()}
    asg = {}
    for g in sorted(gc, key=lambda g: H("grp|" + g)):
        best = None
        for s in order:        # first split (priority order) that still needs these classes and that this group does not overshoot
            if sum(min(n, max(0, rem[s].get(c, 0))) for c, n in gc[g].items()) <= 0: continue
            if any(n > max(0, rem[s].get(c, 0)) + 0.25 * targets[s].get(c, 0) for c, n in gc[g].items() if c in targets[s] and c in rare): continue   # overshoot only matters for RARE classes; common-class surplus is subsampled away later
            best = s; break
        asg[g] = best or "pool"
        if best:
            for c, n in gc[g].items():
                if c in rem[best]: rem[best][c] -= n
    return asg, rem
SPLIT = {}
a1, rem1 = assign("S1", TGT["S1"], ["locked", "dev", "calib"])
a2, rem2 = assign("S2", TGT["S2"], ["locked", "probe"])
a3, rem3 = assign("S3", TGT["S3"], ["dev"])
log("remaining deficits S1", {s: {c: n for c, n in v.items() if n > 0} for s, v in rem1.items()})
log("remaining deficits S2", {s: {c: n for c, n in v.items() if n > 0} for s, v in rem2.items()})
log("remaining deficits S3", {s: {c: n for c, n in v.items() if n > 0} for s, v in rem3.items()})
for i in alive:
    c = cand[i]; c["split"] = {"S1": a1, "S2": a2, "S3": a3}[c["src"]][c["group"]]
    if c["src"] == "S3" and c["split"] == "pool": c["split"] = "pool"

# ---------------- touched scan on locked-assigned candidates (counts only)
lk = [i for i in alive if cand[i]["split"] == "locked"]
import ahocorasick
ROOT = os.path.expanduser("~/ten-percent"); A = ahocorasick.Automaton(); owner = collections.defaultdict(set)
for i in lk:
    t = cand[i]["text"]
    if len(t) >= 25:
        s0 = t[:60]
        for v in {s0, json.dumps(s0, ensure_ascii=False)[1:-1], json.dumps(s0)[1:-1]}:
            owner[v].add(i)
for v in owner: A.add_word(v, v)
A.make_automaton()
hitfiles = []; hit_texts = set(); nfiles = 0
for dp, dn, fn in os.walk(ROOT):
    dn[:] = [x for x in dn if x not in (".git", "node_modules", "TOX", "__pycache__")]
    for f in fn:
        if not f.endswith((".json", ".jsonl", ".py", ".md", ".txt", ".log", ".csv")): continue
        p = os.path.join(dp, f)
        try:
            if os.path.getsize(p) > 600e6: continue
            blob = open(p, errors="ignore").read()
        except Exception: continue
        nfiles += 1; found = False
        for _, v in A.iter(blob):
            hit_texts |= owner[v]; found = True
        if found: hitfiles.append(p)
log("touched-scan: files scanned", nfiles, "files with hits", len(hitfiles))
log("touched-scan: locked candidates dropped:", len(hit_texts))
for i in hit_texts: cand[i]["split"] = "dropped_touched"

# ---------------- locked / dev subsampling to exact targets with per-group cap
def take(src, split, cls, n, cap_frac=None):
    rows = sorted([i for i in alive if cand[i]["src"] == src and cand[i]["split"] == split and pcls(cand[i]) == cls], key=lambda i: H("pick|" + cand[i]["text"]))
    out, pg = [], collections.Counter(); cap = int(cap_frac * n) if cap_frac else None
    for i in rows:
        if cap and pg[cand[i]["group"]] >= cap: continue
        out.append(i); pg[cand[i]["group"]] += 1
        if len(out) >= n: break
    return out, len(rows)
final = collections.defaultdict(list); avail = {}
for src in TGT:
    for split, tg in TGT[src].items():
        for cls, n in tg.items():
            sel, na = take(src, split, cls, n, GROUP_CAP_FRAC if split == "locked" else None)
            final[(src, split)] += sel; avail[f"{src}/{split}/{cls}"] = (len(sel), na, n)
short = {k: v for k, v in avail.items() if v[0] < v[2]}
log("cells short of target:", short)
# pools (non-locked, non-dev leftovers); S2 leftover is NOT saved (S2 stays minimal)
for src, nm in [("S1", "pool"), ("S3", "pool")]:
    used = {i for k, v in final.items() for i in v}
    for cls in PC:
        rows = sorted([i for i in alive if cand[i]["src"] == src and cand[i]["split"] == "pool" and pcls(cand[i]) == cls and i not in used], key=lambda i: H("pool|" + cand[i]["text"]))
        final[(src, "pool")] += rows[:POOL_CAP]
lockedS1_cells = collections.Counter(pcls(cand[i]) for i in final[("S1", "locked")]); lockedS2_cells = collections.Counter(pcls(cand[i]) for i in final[("S2", "locked")])

# ---------------- write
def out_row(i):
    c = cand[i]
    return dict(id=f"{c['src'].lower()}_{sha(c['text'].encode())[:12]}", source=c["src"], kind=c["gold"], gold=c["gold"], text=c["text"], group=c["group"], hard=bool(c["hard"]), lex=c["lex"])
FILES = {"locked": final[("S1", "locked")] + final[("S2", "locked")], "s1_dev": final[("S1", "dev")], "s1_calib": final[("S1", "calib")], "s1_pool": final[("S1", "pool")],
         "s3_dev": final[("S3", "dev")], "s3_pool": final[("S3", "pool")], "s2_probe": final[("S2", "probe")]}
# final cross-split exact check by id/text
allt = collections.Counter()
for f, idx in FILES.items():
    for i in idx: allt[cand[i]["text"]] += 1
assert all(v == 1 for v in allt.values()), "text appears in two files"
# group disjointness within each source across splits
gsp = collections.defaultdict(set)
for f, idx in FILES.items():
    for i in idx: gsp[cand[i]["group"]].add(f)
assert all(len(v) == 1 for v in gsp.values()), "a group spans two files"
lock = dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), version="tox-v1-2026-11-06", rule=__doc__, splits={},
            params=dict(TGT=TGT, OVERSAMPLE=OVERSAMPLE, POOL_CAP=POOL_CAP, JACCARD=JT, GROUP_CAP_FRAC=GROUP_CAP_FRAC, kmeans_k=dict(S1=2000, S2=250, S3=200)),
            raw_inputs=PINS, build_data_py_sha256=sha(open(__file__, "rb").read()),
            audit=dict(dropped=dict(collections.Counter(drop.values())), near_dup_pairs=len(pairs), touched_scan_files_scanned=nfiles, touched_scan_files_with_hits=len(hitfiles), touched_hit_files=sorted(os.path.relpath(p, ROOT) for p in hitfiles), locked_candidates_dropped_touched=len(hit_texts),
                       cells_short_of_target=short, n_candidates=N, n_alive=len(alive), groups_unique_per_file_checked=True, text_unique_across_files_checked=True))
for k, idx in FILES.items():
    rows = sorted([out_row(i) for i in idx], key=lambda r: r["id"])
    p = f"{D}/{k}.jsonl"
    with open(p, "w") as f:
        for r in rows: f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")
    cnt = collections.Counter((r["source"], r["kind"], "hard" if r["hard"] else "plain") for r in rows)
    lexc = collections.defaultdict(collections.Counter); lens = collections.defaultdict(list)
    for r in rows:
        key = f"{r['source']}/{r['kind']}"; lens[key].append(len(r["text"]))
        for g in r["lex"]: lexc[key][g] += 1
    lock["splits"][k] = dict(n=len(rows), cells={f"{a}/{b}/{c}": n for (a, b, c), n in sorted(cnt.items())}, groups=len({r["group"] for r in rows}), file_sha256=sha(open(p, "rb").read()),
                             lexicon_hits_per_cell={kk: dict(v) for kk, v in sorted(lexc.items())}, median_chars_per_cell={kk: int(np.median(v)) for kk, v in sorted(lens.items())})
    if k == "locked": os.chmod(p, 0o444)
cells = collections.Counter((cand[i]["src"], cand[i]["gold"]) for i in FILES["locked"])
lock["locked_totals"] = dict(n=len(FILES["locked"]), source_x_kind={f"{a}/{b}": n for (a, b), n in sorted(cells.items())}, min_cell=min(cells.values()),
                             S1=sum(a == "S1" for (a, b) in cells.elements()), S2=sum(a == "S2" for (a, b) in cells.elements()),
                             safe_hard_share_S1=lockedS1_cells["safe_hard"] / max(1, lockedS1_cells["safe_hard"] + lockedS1_cells["safe_plain"]),
                             safe_hard_share_S2=lockedS2_cells["safe_hard"] / max(1, lockedS2_cells["safe_hard"] + lockedS2_cells["safe_plain"]))
json.dump(lock, open(f"{D}/LOCK.json", "w"), indent=1)
log(json.dumps({k: v for k, v in lock.items() if k not in ("rule", "raw_inputs")}, indent=1, default=str)[:6000])
