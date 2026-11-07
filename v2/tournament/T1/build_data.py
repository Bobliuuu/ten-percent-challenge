#!/usr/bin/env python
"""Build T1 (agentic tool use) splits, v2 (re-lock 2026-11-06, after red team T-REDTEAM). Refuses to overwrite data/LOCK.json.
Item schema (jsonl): id, source(S1|S2|S3), kind(nocall|single|multi), query, tools[{name,description,parameters(JSON schema)}],
  gold[{name, arguments{param: value}}]   (S2: arguments{param:[allowed values]} with '' = may be omitted; gold_alts=true),
  family (API family: S1 = tool-name community id, S2 = BFCL subset, S3 = first tool name), stratum (locked S1/S2 only: unseen|seen).
Sources:
  S1 = xLAM function-calling-60k (Salesforce, CC-BY-4.0; ungated mirror NobodyExistsOnTheInternet/xlam-function-calling-60k, 60000 rows).
       SPLIT BY API FAMILY, NOT PER ITEM. Graph: nodes = tool names, edge weight = number of tool lists in which two names co-occur.
       Families = Louvain communities (resolution 1.0, seed 1) -> 38 families. Every row whose tool list lies inside ONE family goes to the
       split that owns the family; rows whose tool list spans families are dropped from every split ("straddlers").
       Whole families are assigned to locked / dev / calib / pool (sha256-ordered, deficit-greedy), so NO tool name, tool schema or gold
       call is shared across splits. Locked rows additionally avoid every tool name that occurs in the legacy 600-item dev probe and the
       legacy 599-item calibration file (so no model-facing artefact built so far shares an API with the exam).
       kind=nocall is HARD and SYNTHESISED: the gold tool(s) are DELETED from the item's tool list and replaced by 1-2 near-miss tools
       of the same split (TF-IDF nearest to the deleted tool and to the query) that lack at least one parameter the gold call needs
       (a missing capability, not a topic mismatch); at least one >=4-char token must be shared between query and the final tool list.
  S3 = Team-ACE/ToolACE (apache-2.0), OOD DEV PROXY only (h%7==1 -> s3_dev, others s3_pool). First turn of each dialogue only. UNCHANGED split rule.
  S2 = BFCL v3 (gorilla-llm/Berkeley-Function-Calling-Leaderboard, apache-2.0) -> locked OOD block, never used elsewhere.
       nocall = BFCL irrelevance + live_irrelevance (real irrelevance, not synthesised); single/multi as before.
Audit (counts only; locked rows are never printed): exact all-pairs Jaccard>=0.6 query near-duplicate join (prefix-filter, exact, no df cut-off),
  tool-schema (named + anonymous) and gold-call overlap, tool-name overlap; dev/calib/pool/S3 are cross-deduped too.
Locked rows are never inspected here except for counts and aggregate construction diagnostics."""
import hashlib, json, os, re, ast, random, datetime, collections, sys, itertools, math
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); D = os.environ.get("T1_BUILD_OUT", f"{HERE}/data"); RAW = f"{HERE}/raw"
LEGACY = [f"{HERE}/probe_items.jsonl", f"{HERE}/../../compress/dense/t1/calib_t1.jsonl"]
os.makedirs(D, exist_ok=True)
assert not os.path.exists(f"{D}/LOCK.json"), "LOCK.json exists; never overwrite"
H = lambda s: int(hashlib.sha256(s.encode()).hexdigest()[:8], 16)
sha = lambda b: hashlib.sha256(b).hexdigest()
CELL = dict(single=450, multi=450, nocall=400)           # S1 locked targets (floor required: 330 per cell)
CAP_S2 = dict(single=500, multi=1000, nocall=400)        # multi = all parallel BFCL that survive the audit
LOCKED_ELIGIBLE_TARGET = 5200                            # raw closed, legacy-name-free S1 rows in locked families before cleaning
FRAC = dict(dev=0.22, calib=0.22, pool=0.56)             # share of non-locked closed rows
JT = 0.6                                                 # query near-duplicate Jaccard threshold
PER_TOOL_CAP = 3                                         # max locked items per primary (gold / deleted) tool name
FAM_CAP = 0.25                                           # max share of a locked cell from one family
TYPEMAP = dict(str="string", string="string", int="integer", integer="integer", float="number", number="number", double="number", bool="boolean", boolean="boolean",
               list="array", array="array", tuple="array", set="array", dict="object", object="object", any="string", callable="string")
def jtype(t):
    t = str(t).lower().strip()
    if t.startswith(("list", "array", "tuple", "set")): return "array"
    if t.startswith("dict"): return "object"
    t = re.split(r"[,\[ ]", t)[0]
    return TYPEMAP.get(t, "string")
def fix_schema(p):  # BFCL schemas: dict/float/tuple -> JSON schema
    if isinstance(p, dict):
        q = {}
        for k, v in p.items():
            if k == "type" and isinstance(v, str): q[k] = jtype(v)
            else: q[k] = fix_schema(v)
        return q
    if isinstance(p, list): return [fix_schema(x) for x in p]
    return p
def norm(s): return re.sub(r"\W+", " ", s.lower()).strip()
def toks(s, m=4): return {w for w in re.split(r"[^a-z0-9]+", s.lower().replace("_", " ")) if len(w) >= m}
def tool_text(t): return t["name"] + " " + t["description"] + " " + " ".join((t["parameters"].get("properties") or {}))
def overlap(r, m=4):
    q = toks(r["query"], m); t = set()
    for x in r["tools"]: t |= toks(tool_text(x), m)
    return len(q & t)
def cj(o): return json.dumps(o, sort_keys=True, ensure_ascii=False)
def sig_named(t): return sha(cj([t["name"], t["description"], t["parameters"]]).encode())
def sig_anon(t):
    if len(t["description"].strip()) < 15 or not (t["parameters"].get("properties")): return None
    props = sorted((k, str(p.get("type")), norm(p.get("description", ""))) for k, p in t["parameters"]["properties"].items())
    return sha(cj([norm(t["description"]), props]).encode())
def gold_sigs(r): return {sha(cj([g["name"], g["arguments"]]).encode()) for g in r["gold"]}

# ---------------- pins of raw inputs
def raw_pins():
    pins = {}
    files = [f"{RAW}/x/xlam_function_calling_60k.json", f"{RAW}/data.json"] + sorted(
        f"{RAW}/BFCL_v3_{n}.json" for n in ["simple", "multiple", "live_simple", "live_multiple", "parallel", "parallel_multiple", "live_parallel", "live_parallel_multiple", "irrelevance", "live_irrelevance"]) + sorted(
        f"{RAW}/possible_answer/BFCL_v3_{n}.json" for n in ["simple", "multiple", "live_simple", "live_multiple", "parallel", "parallel_multiple", "live_parallel", "live_parallel_multiple"])
    for f in files:
        rel = os.path.relpath(f, RAW); meta = None
        for mp in (f"{os.path.dirname(f)}/.cache/huggingface/download/{os.path.basename(f)}.metadata", f"{RAW}/.cache/huggingface/download/{rel}.metadata"):
            if os.path.exists(mp): meta = open(mp).read().split("\n")[:2]; break
        pins[rel] = dict(sha256=sha(open(f, "rb").read()), bytes=os.path.getsize(f), hf_revision=meta[0] if meta else None, hf_etag=meta[1] if meta else None)
    # xlam has no download metadata on disk: revision verified against the HF LFS oid / blob at the revision below (see LOCK.json notes)
    pins["x/xlam_function_calling_60k.json"]["hf_revision"] = pins["x/xlam_function_calling_60k.json"]["hf_revision"] or "19fcb6385e3ff7bb835a0734bbe0bdd630b9f8b6"
    pins["x/xlam_function_calling_60k.json"]["revision_verified"] = "file sha256 == HF LFS oid of xlam_function_calling_60k.json at this revision (checked 2026-11-06 via /api/datasets/<repo>/revision/<rev>?blobs=true)"
    pins["data.json"]["revision_verified"] = "file sha256 == HF LFS oid of data.json at hf_revision (checked 2026-11-06)"
    pins["x/xlam_function_calling_60k.json"]["hf_repo"] = "NobodyExistsOnTheInternet/xlam-function-calling-60k"
    pins["data.json"]["hf_repo"] = "Team-ACE/ToolACE"
    for k in pins:
        if k.startswith(("BFCL", "possible_answer")): pins[k]["hf_repo"] = "gorilla-llm/Berkeley-Function-Calling-Leaderboard"
    return pins
PINS = raw_pins()

# ---------------- S1 xLAM
def xlam_tool(t):
    props, req = {}, []
    for k, v in t.get("parameters", {}).items():
        ty = v.get("type", "str"); props[k] = dict(type=jtype(ty), description=v.get("description", ""))
        if "default" in v: props[k]["default"] = v["default"]
        if "optional" not in str(ty).lower() and "default" not in v and "default" not in str(ty).lower(): req.append(k)
    return dict(name=t["name"], description=t.get("description", ""), parameters=dict(type="object", properties=props, required=req))
raw = json.load(open(f"{RAW}/x/xlam_function_calling_60k.json"))
S1rows = []
for r in raw:
    ans = json.loads(r["answers"]); tools = json.loads(r["tools"])
    if not ans: continue   # xLAM's 1166 empty-answer rows are ill-posed (matching tools exist)
    S1rows.append(dict(id=f"s1_{r['id']}", source="S1", kind="single" if len(ans) == 1 else "multi", query=r["query"],
                       tools=[xlam_tool(t) for t in tools], gold=[dict(name=a["name"], arguments=a.get("arguments", {})) for a in ans]))
import networkx as nx
Gr = nx.Graph()
for r in S1rows:
    ns = sorted({t["name"] for t in r["tools"]}); Gr.add_nodes_from(ns)
    for a_, b_ in itertools.combinations(ns, 2):
        if Gr.has_edge(a_, b_): Gr[a_][b_]["weight"] += 1
        else: Gr.add_edge(a_, b_, weight=1)
communities = nx.community.louvain_communities(Gr, weight="weight", resolution=1.0, seed=1)
communities = sorted(communities, key=lambda c: min(c))
FAM = {n: i for i, c in enumerate(communities) for n in c}
for r in S1rows:
    fs = {FAM[t["name"]] for t in r["tools"]}
    r["_fam"] = next(iter(fs)) if len(fs) == 1 else None
n_strad = sum(r["_fam"] is None for r in S1rows)
closed = [r for r in S1rows if r["_fam"] is not None]
legacy_rows = []
for p in LEGACY:
    if os.path.exists(p): legacy_rows += [json.loads(l) for l in open(p) if l.strip()]
legacy_names = {t["name"] for r in legacy_rows if r["source"] == "S1" for t in r["tools"]}
legacy_hash = {p: sha(open(p, "rb").read()) for p in LEGACY if os.path.exists(p)}
fam_rows = collections.defaultdict(list)
for r in closed: fam_rows[r["_fam"]].append(r)
fam_elig = {f: sum(not ({t["name"] for t in r["tools"]} & legacy_names) for r in rs) for f, rs in fam_rows.items()}
order = sorted(fam_rows, key=lambda f: H("fam|" + min(communities[f])))
assign = {}; tot = 0
for f in order:    # locked families: sha256 order, skipping families with <150 eligible rows
    if tot >= LOCKED_ELIGIBLE_TARGET: break
    if fam_elig[f] >= 150: assign[f] = "locked"; tot += fam_elig[f]
rest = [f for f in order if f not in assign]
mass = {f: len(fam_rows[f]) for f in rest}; total = sum(mass.values()); got = collections.Counter()
for f in rest:      # deficit-greedy over the remaining families; dev/calib only take families <= 50% of their target mass (>= 3 families each)
    ok = [s for s in FRAC if s == "pool" or mass[f] <= 0.5 * FRAC[s] * total]
    best = max(ok, key=lambda s: FRAC[s] * total - got[s]); assign[f] = best; got[best] += mass[f]
S1 = {k: [] for k in ["locked_all", "dev", "calib", "pool"]}
dropped_legacy_in_locked = 0
for r in closed:
    sp = assign[r["_fam"]]
    if sp == "locked":
        if {t["name"] for t in r["tools"]} & legacy_names: dropped_legacy_in_locked += 1; continue
        S1["locked_all"].append(r)
    else: S1[sp].append(r)

# ---------------- hard nocall (near-miss distractor) construction
from sklearn.feature_extraction.text import TfidfVectorizer
def split_camel(s): return re.sub(r"([a-z])([A-Z])", r"\1 \2", s).replace("_", " ")
def ttxt(t): return split_camel(t["name"]) + " " + t["description"] + " " + " ".join(split_camel(k) for k in (t["parameters"].get("properties") or {}))
def hard_nocall(rows, tag, rate_mod=6, force_ids=None):
    """convert rows (H('nc|'+id)%rate_mod==0, or ids in force_ids) into hard nocall items; returns (rows_out, stats)."""
    uni = {}
    for r in rows:
        for t in r["tools"]: uni.setdefault(t["name"], t)
    names = list(uni); vec = TfidfVectorizer(stop_words="english", token_pattern=r"[A-Za-z][A-Za-z0-9]{2,}", sublinear_tf=True, min_df=1)
    M = vec.fit_transform([ttxt(uni[n]) for n in names]); idx = {n: i for i, n in enumerate(names)}
    out = []; st = collections.Counter()
    sel = [r for r in rows if (force_ids is not None and r["id"] in force_ids) or (force_ids is None and H("nc|" + r["id"]) % rate_mod == 0)]
    selids = {r["id"] for r in sel}
    Q = vec.transform([r["query"] for r in sel]) if sel else None; qi = {r["id"]: i for i, r in enumerate(sel)}
    for r in rows:
        if r["id"] not in selids: out.append(r); continue
        gn = {g["name"] for g in r["gold"]}; g0 = r["gold"][0]; gspec = uni[g0["name"]] if g0["name"] in uni else None
        keep = [t for t in r["tools"] if t["name"] not in gn]
        have = {t["name"] for t in keep}
        gkeys = set(g0["arguments"]) if gspec is not None else set()
        sim_g = (M @ M[idx[g0["name"]]].T).toarray().ravel() if g0["name"] in idx else np.zeros(len(names))
        sim_q = (M @ Q[qi[r["id"]]].T).toarray().ravel()
        score = 0.5 * sim_g + 0.5 * sim_q; cands = []
        for j in np.argsort(-score)[:60]:
            n = names[j]; t = uni[n]
            if n in gn or n in have or sim_g[j] > 0.5 or score[j] <= 0: continue
            if gkeys and gkeys <= set(t["parameters"].get("properties") or {}): continue   # candidate could take the gold arguments: not a missing capability
            cands.append(t)
            if len(cands) >= 4: break
        k = 1 + (H("k|" + r["id"]) % 2) if keep else 2
        for kk in range(min(k, len(cands)), 0, -1):
            tools = keep + cands[:kk]
            if len(tools) >= 1 and overlap(dict(query=r["query"], tools=tools), 4) >= 1:
                rng = random.Random(H("shuf|" + r["id"])); tools = list(tools); rng.shuffle(tools)
                out.append(dict(id=r["id"].replace("s1_", "s1nc_"), source="S1", kind="nocall", query=r["query"], tools=tools, gold=[], _fam=r["_fam"],
                                _deleted=g0["name"], _nm=[t["name"] for t in cands[:kk]])); st["converted"] += 1; break
        else: st["kept_call"] += 1; out.append(r)
    return out, dict(st)
nc_stats = {}
for k in ["dev", "calib", "pool"]: S1[k], nc_stats[k] = hard_nocall(S1[k], k)
# legacy files (dev probe / calib) are left untouched on disk elsewhere; they are only used as exclusion sets here

# ---------------- S3 ToolACE (dev proxy) -- split rule unchanged
def split_top(s, sep=","):
    parts, depth, q, cur = [], 0, None, ""
    i = 0
    while i < len(s):
        c = s[i]
        if q:
            cur += c
            if c == "\\": cur += s[i + 1]; i += 1
            elif c == q: q = None
        elif c in "\"'": q = c; cur += c
        elif c in "([{": depth += 1; cur += c
        elif c in ")]}": depth -= 1; cur += c
        elif c == sep and depth == 0: parts.append(cur); cur = ""
        else: cur += c
        i += 1
    if cur.strip(): parts.append(cur)
    return parts
def pyval(v):
    v = v.strip()
    try: return ast.literal_eval(v)
    except Exception: pass
    try: return json.loads(v)
    except Exception: raise ValueError(v)
def parse_pycalls(a):
    a = a.strip()
    if not (a.startswith("[") and a.endswith("]")): raise ValueError("fmt")
    calls = []
    for part in split_top(a[1:-1]):
        part = part.strip(); m = re.match(r"^([^()=\[\]{}\"']+?)\((.*)\)$", part, re.S)
        if not m: raise ValueError("call")
        args = {}
        for kv in split_top(m.group(2)):
            if not kv.strip(): continue
            k, _, v = kv.partition("=")
            if not _: raise ValueError("kv")
            args[k.strip()] = pyval(v)
        calls.append(dict(name=m.group(1).strip(), arguments=args))
    return calls
S3 = {"dev": [], "pool": []}
for i, r in enumerate(json.load(open(f"{RAW}/data.json"))):
    c = r["conversations"]; sysm = r["system"]
    try:
        j = sysm.index("[", sysm.index("JSON format that you can invoke"))
        tools, _ = json.JSONDecoder().raw_decode(sysm[j:])
        tools = [dict(name=t["name"], description=t.get("description", ""), parameters=fix_schema(t.get("parameters", {"type": "object", "properties": {}}))) for t in tools]
    except Exception: continue
    if c[0]["from"] != "user": continue
    a = c[1]["value"].strip(); q = c[0]["value"]
    if re.search(r"^\[.*\(.*\)\]$", a, re.S):
        try: gold = parse_pycalls(a)
        except Exception: continue
        kind = "single" if len(gold) == 1 else "multi"
    elif not re.search(r"\w\(|\[|\{|<|=", a) and re.search(r"none of|cannot|can't|not (?:possible|able|available)|unable|lack|missing|need(?:s)? (?:more|the)|please provide|could you (?:please )?provide", a, re.I) and len(a) < 600:
        gold, kind = [], "nocall"
    else: continue
    h = H(q + "|" + json.dumps(tools, sort_keys=True))
    S3["dev" if h % 7 == 1 else "pool"].append(dict(id=f"s3_{i}", source="S3", kind=kind, query=q, tools=tools, gold=gold, _h=h, _fam="s3:" + tools[0]["name"] if tools else "s3:"))

# ---------------- S2 BFCL v3 (locked OOD block)
def bfcl(name, kind_fn, answers=True):
    out = []
    qs = [json.loads(l) for l in open(f"{RAW}/BFCL_v3_{name}.json") if l.strip()]
    gt = {}
    if answers:
        for l in open(f"{RAW}/possible_answer/BFCL_v3_{name}.json"):
            if l.strip(): j = json.loads(l); gt[j["id"]] = j["ground_truth"]
    for r in qs:
        if len(r["question"]) != 1 or len(r["question"][0]) != 1 or r["question"][0][0]["role"] != "user": continue
        tools = [dict(name=f["name"], description=f.get("description", ""), parameters=fix_schema(f.get("parameters", {}))) for f in r["function"]]
        if answers:
            if r["id"] not in gt: continue
            g = gt[r["id"]]; gold = [dict(name=list(c)[0], arguments=list(c.values())[0]) for c in g]
        else: gold = []
        out.append(dict(id=f"s2_{r['id']}", source="S2", kind=kind_fn(gold), query=r["question"][0][0]["content"], tools=tools, gold=gold, gold_alts=True, _fam=f"s2:{name}", _h=H(r["id"])))
    return out
S2raw = []
for n in ["simple", "multiple", "live_simple", "live_multiple", "parallel", "parallel_multiple", "live_parallel", "live_parallel_multiple"]:
    S2raw += bfcl(n, lambda g: "single" if len(g) == 1 else "multi")
for n in ["irrelevance", "live_irrelevance"]: S2raw += bfcl(n, lambda g: "nocall", answers=False)

# ---------------- audit: exact all-pairs Jaccard join over query word sets (prefix filter, no df cut-off)
RANK = dict(locked=0, s2=-1, pool=1, s3pool=1, calib=2, dev=3, s3dev=4, legacy=9)
def near_pairs(items, t):
    """items: list of word sets. exact all-pairs (J>=t); AllPairs prefix filtering. returns set of (i,j) i<j."""
    df = collections.Counter(w for s in items for w in s)
    ordered = [sorted(s, key=lambda w: (df[w], w)) for s in items]
    inv = collections.defaultdict(list); pairs = set()
    for i in sorted(range(len(items)), key=lambda i: len(items[i])):
        s = items[i]; n = len(s)
        if n == 0: continue
        plen = n - math.ceil(t * n) + 1
        cand = set()
        for w in ordered[i][:plen]:
            cand.update(inv[w])
        for j in cand:
            m = len(s & items[j])
            if m / (n + len(items[j]) - m) >= t: pairs.add((min(i, j), max(i, j)))
        for w in ordered[i][:plen]: inv[w].append(i)
    return pairs
def brute_pairs(items, t):
    return {(i, j) for i in range(len(items)) for j in range(i + 1, len(items)) if items[i] and items[j] and len(items[i] & items[j]) / len(items[i] | items[j]) >= t}

# locked S1 candidates: hard nocall built from a reserved subset of the eligible rows (never from rows used for single/multi)
loc = sorted(S1["locked_all"], key=lambda r: H("lk|" + r["id"]))
loc_nc_ids = {r["id"] for r in loc if H("lncs|" + r["id"]) % 4 == 0}      # ~25% of eligible rows become nocall
loc_all, nc_stats["locked"] = hard_nocall(loc, "locked", force_ids=loc_nc_ids)
cand = []        # (row, split-label)
for r in loc_all: cand.append((r, "locked"))
for k in ["dev", "calib", "pool"]:
    for r in S1[k]: cand.append((r, k))
for r in S3["dev"]: cand.append((r, "s3dev"))
for r in S3["pool"]: cand.append((r, "s3pool"))
for r in S2raw: cand.append((r, "s2"))
for r in legacy_rows: cand.append((r, "legacy"))
audit = collections.Counter(); dropped = set()
def drop(i, why): dropped.add(i); audit[why] += 1
# (1) exact-duplicate / near-duplicate queries across splits
sets = [set(norm(r["query"]).split()) for r, _ in cand]
pairs = near_pairs(sets, JT)
for i, j in sorted(pairs):
    if i in dropped or j in dropped: continue
    (ri, si), (rj, sj) = cand[i], cand[j]
    if si == sj: continue
    lo, hi = (i, j) if RANK[si] <= RANK[sj] else (j, i)
    if "legacy" in (si, sj) and RANK[cand[lo][1]] > 0: continue
    exact = sets[i] == sets[j]
    drop(lo, f"query_{'exact' if exact else 'near'}:{cand[lo][1]}_vs_{cand[hi][1]}")
# (2) tool-schema (named + anonymous), tool-name and gold-call overlap across splits
key_split = collections.defaultdict(set)
def keys_of(r):
    ks = {("tn", t["name"]) for t in r["tools"]} | {("ts", sig_named(t)) for t in r["tools"]} | {("ga", s) for s in gold_sigs(r)}
    ks |= {("ta", s) for s in (sig_anon(t) for t in r["tools"]) if s}
    return ks
KEYS = [keys_of(r) for r, _ in cand]
for i, (r, s) in enumerate(cand):
    if i in dropped: continue
    for k in KEYS[i]: key_split[k].add(s)
S3G = {"s3dev", "s3pool"}
for i, (r, s) in enumerate(cand):
    if i in dropped or s == "legacy": continue
    for k in sorted(KEYS[i]):
        others = key_split[k] - {s}
        if s == "locked": bad = bool(others - {"s2"})
        elif s == "s2": bad = bool(others)
        else: bad = any(RANK[o] > RANK[s] and o != "legacy" and RANK[o] > 0 and not (k[0] == "tn" and s in S3G and o in S3G) for o in others)
        if bad: drop(i, f"keyoverlap_{k[0]}:{s}"); break
# (3) S2 / locked: anything left that shares a tool name with the legacy probe/calib files is dropped
for i, (r, s) in enumerate(cand):
    if s in ("locked", "s2") and i not in dropped and any(t["name"] in legacy_names for t in r["tools"]): drop(i, f"legacy_name_overlap:{s}")
# (4) exact duplicate (query) inside each locked/S2 pool
seen = set()
for i, (r, s) in enumerate(cand):
    if s in ("locked", "s2") and i not in dropped:
        q = (s, norm(r["query"]))
        if q in seen: drop(i, f"internal_dup:{s}")
        seen.add(q)
clean = {s: [r for i, (r, ss) in enumerate(cand) if ss == s and i not in dropped] for s in ["locked", "dev", "calib", "pool", "s3dev", "s3pool", "s2"]}

# ---------------- locked selection with per-tool / per-family caps
def pick(rows, cell, cap_fam=None):
    out = []; per_tool = collections.Counter(); per_fam = collections.Counter()
    for r in sorted(rows, key=lambda r: H("pick|" + r["id"])):
        pt = r.get("_deleted") or r["gold"][0]["name"] if (r.get("_deleted") or r["gold"]) else r["tools"][0]["name"]
        fam = r.get("_fam")
        if per_tool[pt] >= PER_TOOL_CAP: continue
        if cap_fam and per_fam[fam] >= cap_fam: continue
        out.append(r); per_tool[pt] += 1; per_fam[fam] += 1
        if len(out) >= cell: break
    return out
S1_locked = []
for k, c in CELL.items():
    rows = [r for r in clean["locked"] if r["kind"] == k]
    S1_locked += pick(rows, c, cap_fam=int(FAM_CAP * c))
S2_locked = []
S2_nc = [r for r in clean["s2"] if r["kind"] == "nocall"]
# S2 nocall: prefer items whose query shares >=1 4-char token with the tools (a zero-overlap rule must not solve it), then fill by hash order
S2_nc.sort(key=lambda r: (overlap(r, 4) == 0, H("s2nc|" + r["id"])))
S2_locked += S2_nc[:CAP_S2["nocall"]]
for k in ["single", "multi"]:
    S2_locked += sorted([r for r in clean["s2"] if r["kind"] == k], key=lambda r: r["_h"])[:CAP_S2[k]]
LOCKED = S1_locked + S2_locked
# strata (S1/S2 locked): 'seen' iff any tool name occurs in a non-locked split (dev/calib/pool/S3/legacy)
nonlocked_names = {t["name"] for s in ["dev", "calib", "pool", "s3dev", "s3pool"] for r in clean[s] for t in r["tools"]} | legacy_names
nonlocked_names |= {t["name"] for r in S1rows if r["_fam"] is not None and assign.get(r["_fam"]) != "locked" for t in r["tools"]}
for r in LOCKED: r["stratum"] = "seen" if any(t["name"] in nonlocked_names for t in r["tools"]) else "unseen"
LOCKED.sort(key=lambda r: r["id"])
nonlocked_sets = {"s1_dev": clean["dev"], "s1_calib": clean["calib"], "s1_pool": clean["pool"], "s3_dev": clean["s3dev"], "s3_pool": clean["s3pool"]}
for k, rows in nonlocked_sets.items():
    for r in rows: r.setdefault("_fam", None)
files = dict(locked=LOCKED, **nonlocked_sets)
def out_row(r):
    o = {a: b for a, b in r.items() if not a.startswith("_") and a != "stratum"}
    if r.get("_fam") is not None: o["family"] = f"s1c{r['_fam']}" if isinstance(r["_fam"], int) else r["_fam"]
    if "stratum" in r: o["stratum"] = r["stratum"]
    return o

# ---------------- diagnostics (aggregates only)
def nocall_diag(rows):
    nc = [r for r in rows if r["kind"] == "nocall"]; cl = [r for r in rows if r["kind"] != "nocall"]; d = {}
    if not nc or not cl: return d
    for m in (3, 4, 5):
        d[f"zero_overlap_rule_m{m}"] = dict(recall=sum(overlap(r, m) == 0 for r in nc) / len(nc), false_nocall_rate=sum(overlap(r, m) == 0 for r in cl) / len(cl))
    # tf-idf max cosine(query, any tool) as score: AUC for separating call from nocall (0.5 = no signal)
    v = TfidfVectorizer(stop_words="english", token_pattern=r"[A-Za-z][A-Za-z0-9]{2,}", sublinear_tf=True)
    v.fit([r["query"] for r in rows] + [ttxt(t) for r in rows for t in r["tools"]])
    def s(r):
        q = v.transform([r["query"]]); T = v.transform([ttxt(t) for t in r["tools"]]); return float((T @ q.T).toarray().max())
    sn = np.array([s(r) for r in nc]); sc = np.array([s(r) for r in cl])
    order_ = np.argsort(np.concatenate([sn, sc])); ranks = np.empty(len(order_)); ranks[order_] = np.arange(1, len(order_) + 1)
    auc = (ranks[len(sn):].sum() - len(sc) * (len(sc) + 1) / 2) / (len(sn) * len(sc))   # P(call score > nocall score)
    d["tfidf_maxcos_auc_call_vs_nocall"] = float(auc)
    return d
diag = {}
for k in ["dev", "calib"]:
    diag[f"s1_{k}"] = nocall_diag(random.Random(0).sample(clean[k], min(3000, len(clean[k]))))
diag["S1_locked_construction"] = nocall_diag([r for r in LOCKED if r["source"] == "S1"])
diag["S2_locked_nocall_zero_overlap_m4"] = dict(recall=sum(overlap(r, 4) == 0 for r in LOCKED if r["source"] == "S2" and r["kind"] == "nocall") / max(1, sum(r["source"] == "S2" and r["kind"] == "nocall" for r in LOCKED)))
# brute-force recall check of the prefix-filter join on a random subset (exact vs naive O(n^2))
rs = random.Random(5).sample(range(len(sets)), 3000); sub = [sets[i] for i in rs]
diag["audit_join_recall_vs_bruteforce_3000"] = dict(join=len(near_pairs(sub, JT)), brute=len(brute_pairs(sub, JT)), equal=near_pairs(sub, JT) == brute_pairs(sub, JT))

# ---------------- write
lock = dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), version="v2-relock-2026-11-06", rule=__doc__, splits={},
            params=dict(CELL=CELL, CAP_S2=CAP_S2, LOCKED_ELIGIBLE_TARGET=LOCKED_ELIGIBLE_TARGET, FRAC=FRAC, JACCARD_NEAR_DUP=JT, PER_TOOL_CAP=PER_TOOL_CAP, FAM_CAP=FAM_CAP,
                        louvain=dict(resolution=1.0, seed=1, n_families=len(communities)), nocall_rate_mod=6),
            raw_inputs=PINS, legacy_exclusion_files=legacy_hash, build_data_py_sha256=sha(open(__file__, "rb").read()),
            families=dict(assignment_counts=dict(collections.Counter(assign.values())), locked_families=sorted(f for f, s in assign.items() if s == "locked"),
                          closed_rows=len(closed), straddler_rows_dropped_everywhere=n_strad, locked_family_rows_dropped_for_legacy_name=dropped_legacy_in_locked,
                          rows_per_split_before_audit={k: len(v) for k, v in S1.items()}),
            audit=dict(dropped=dict(audit), nocall_conversion=nc_stats, diagnostics=diag))
for k, rows in files.items():
    p = f"{D}/{k}.jsonl"
    with open(p, "w") as f:
        for r in rows: f.write(json.dumps(out_row(r), ensure_ascii=False, sort_keys=True) + "\n")
    cnt = collections.Counter((r["source"], r["kind"]) for r in rows)
    lock["splits"][k] = dict(n=len(rows), counts={f"{a}/{b}": c for (a, b), c in sorted(cnt.items())}, file_sha256=sha(open(p, "rb").read()))
    os.chmod(p, 0o444)
lc = collections.Counter(r["kind"] for r in LOCKED); st = collections.Counter((r["source"], r["stratum"]) for r in LOCKED)
lock["locked_totals"] = dict(n=len(LOCKED), per_kind=dict(lc), S1=sum(r["source"] == "S1" for r in LOCKED), S2=sum(r["source"] == "S2" for r in LOCKED),
                             strata={f"{a}/{b}": c for (a, b), c in sorted(st.items())}, locked_S1_families=len({r["family"] for r in map(out_row, LOCKED) if r["family"].startswith("s1c")}),
                             name_overlap_with_nonlocked=dict(S1_items_with_any_tool_name_in_nonlocked=sum(any(t["name"] in nonlocked_names for t in r["tools"]) for r in LOCKED if r["source"] == "S1"),
                                                              S2_items_with_any_tool_name_in_nonlocked=sum(any(t["name"] in nonlocked_names for t in r["tools"]) for r in LOCKED if r["source"] == "S2")))
json.dump(lock, open(f"{D}/LOCK.json", "w"), indent=1); os.chmod(f"{D}/LOCK.json", 0o444)
print(json.dumps({k: (v if k not in ("rule", "raw_inputs") else "...") for k, v in lock.items()}, indent=1, default=str))
