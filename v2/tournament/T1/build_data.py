#!/usr/bin/env python
"""Build T1 (agentic tool use) splits. Refuses to overwrite data/LOCK.json.
Item schema (jsonl): id, source(S1|S2|S3), kind(nocall|single|multi), query, tools[{name,description,parameters(JSON schema)}],
  gold[{name, arguments{param: value}}]   (S2: arguments{param:[allowed values]} with '' = may be omitted; gold_alts=true)
Sources:
  S1 = xLAM function-calling-60k (Salesforce, CC-BY-4.0; fetched from ungated mirror NobodyExistsOnTheInternet/xlam-function-calling-60k, 60000 rows)
       split by sha256(query|tools): h%7==0 locked-S1 slice, 1 dev, 2 calib, else pool(build).
       xLAM has no real irrelevance rows (its 1166 empty-answer rows are ill-posed: matching tools exist), so kind=nocall is
       SYNTHESISED: query paired with the tool list of another item of the SAME split whose tools share no content tokens with the query.
  S3 = Team-ACE/ToolACE (apache-2.0), OOD DEV PROXY only (h%7==1 -> s3_dev, others s3_pool). First turn of each dialogue only.
  S2 = BFCL v3 (gorilla-llm/Berkeley-Function-Calling-Leaderboard, apache-2.0) -> locked OOD block, never used elsewhere.
Locked rows are never inspected here except for counts."""
import hashlib, json, os, re, ast, random, datetime, collections, sys
HERE = os.path.dirname(os.path.abspath(__file__)); D = f"{HERE}/data"; RAW = f"{HERE}/raw"
os.makedirs(D, exist_ok=True)
assert not os.path.exists(f"{D}/LOCK.json"), "LOCK.json exists; never overwrite"
H = lambda s: int(hashlib.sha256(s.encode()).hexdigest()[:8], 16)
CAP_S1 = dict(single=500, multi=500, nocall=400)
CAP_S2 = dict(single=500, multi=1000, nocall=300)  # multi = all parallel BFCL
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
def toks(s): return {w for w in re.split(r"[^a-z0-9]+", s.lower().replace("_", " ")) if len(w) >= 5}

# ---------------- S1 xLAM
def xlam_tool(t):
    props, req = {}, []
    for k, v in t.get("parameters", {}).items():
        ty = v.get("type", "str"); props[k] = dict(type=jtype(ty), description=v.get("description", ""))
        if "default" in v: props[k]["default"] = v["default"]
        if "optional" not in str(ty).lower() and "default" not in v and "default" not in str(ty).lower(): req.append(k)
    return dict(name=t["name"], description=t.get("description", ""), parameters=dict(type="object", properties=props, required=req))
raw = json.load(open(f"{RAW}/x/xlam_function_calling_60k.json"))
S1 = {"locked_all": [], "dev": [], "calib": [], "pool": []}
for r in raw:
    ans = json.loads(r["answers"]); tools = json.loads(r["tools"])
    if not ans: continue
    key = r["query"] + "|" + r["tools"]; h = H(key)
    sp = {0: "locked_all", 1: "dev", 2: "calib"}.get(h % 7, "pool")
    S1[sp].append(dict(id=f"s1_{r['id']}", source="S1", kind="single" if len(ans) == 1 else "multi", query=r["query"],
                       tools=[xlam_tool(t) for t in tools], gold=[dict(name=a["name"], arguments=a.get("arguments", {})) for a in ans], _h=h))
def synth_nocall(rows, seed):
    """every item with h2%6==0 becomes a nocall item: own query + tools of a different same-split item with no token overlap."""
    rng = random.Random(seed); out = []; pool = rows
    for r in rows:
        if H("nc|" + r["id"]) % 6 != 0: out.append(r); continue
        qt = toks(r["query"])
        for _ in range(60):
            o = pool[rng.randrange(len(pool))]
            tt = set().union(*[toks(t["name"] + " " + t["description"]) for t in o["tools"]])
            if not (qt & tt) and not ({t["name"] for t in o["tools"]} & {g["name"] for g in r["gold"]}):
                out.append(dict(id=r["id"].replace("s1_", "s1nc_"), source="S1", kind="nocall", query=r["query"], tools=o["tools"], gold=[], _h=r["_h"])); break
        else: out.append(r)
    return out
for k in S1: S1[k] = synth_nocall(S1[k], 7 + H(k) % 1000)

# ---------------- S3 ToolACE (dev proxy)
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
    S3["dev" if h % 7 == 1 else "pool"].append(dict(id=f"s3_{i}", source="S3", kind=kind, query=q, tools=tools, gold=gold, _h=h))

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
        k = kind_fn(gold)
        out.append(dict(id=f"s2_{r['id']}", source="S2", kind=k, query=r["question"][0][0]["content"], tools=tools, gold=gold, gold_alts=True, _h=H(r["id"])))
    return out
S2raw = []
for n in ["simple", "multiple", "live_simple", "live_multiple", "parallel", "parallel_multiple", "live_parallel", "live_parallel_multiple"]:
    S2raw += bfcl(n, lambda g: "single" if len(g) == 1 else "multi")
for n in ["irrelevance", "live_irrelevance"]: S2raw += bfcl(n, lambda g: "nocall", answers=False)
S2raw.sort(key=lambda r: r["_h"])

# ---------------- leakage audit + locked selection (counts only)
def cap(rows, caps):
    c = collections.Counter(); out = []
    for r in sorted(rows, key=lambda r: r["_h"]):
        if c[r["kind"]] < caps[r["kind"]]: out.append(r); c[r["kind"]] += 1
    return out
others = {"S1:" + k: v for k, v in S1.items() if k != "locked_all"} | {"S3:dev": S3["dev"], "S3:pool": S3["pool"]}
def build_index(rows):
    ex = set(); inv = collections.defaultdict(list); T = []
    for r in rows:
        q = norm(r["query"]); ex.add(q); t = set(q.split()); T.append(t)
        for w in t: inv[w].append(len(T) - 1)
    return ex, inv, T
def near(q, idx):
    ex, inv, T = idx; nq = norm(q)
    if nq in ex: return "exact"
    t = set(nq.split()); c = collections.Counter()
    for w in t:
        if len(inv[w]) < 300:
            for i in inv[w]: c[i] += 1
    for i, _ in c.most_common(40):
        if len(t & T[i]) / max(1, len(t | T[i])) >= 0.8: return "near"
    return None
oth_rows = [r for v in others.values() for r in v]
idx_oth = build_index(oth_rows); idx_s1 = build_index([r for k, v in S1.items() for r in v]); idx_s3 = build_index(S3["dev"] + S3["pool"])
audit = {"dropped": collections.Counter(), "locked_candidates": {}}
def clean(rows, idxs, tag):
    seen = set(); out = []
    for r in rows:
        nq = norm(r["query"])
        if nq in seen: audit["dropped"][tag + ":internal_dup"] += 1; continue
        hit = next((n for i in idxs if (n := near(r["query"], i))), None)
        if hit: audit["dropped"][f"{tag}:{hit}_dup_vs_other"] += 1; continue
        seen.add(nq); out.append(r)
    return out
s1_clean = clean(sorted(S1["locked_all"], key=lambda r: r["_h"]), [idx_oth, idx_s3], "S1locked")
# S1 locked vs S1 locked-internal handled by 'seen'; also S1 locked vs other S1 handled by idx_oth
S1["locked"] = cap(s1_clean, CAP_S1)
# S2 vs S1(all incl. other locked rows) and S3
s2_clean = clean(S2raw, [idx_s1, idx_s3], "S2locked")
# also drop S1-locked vs S2 near dups from S1 locked slice
idx_s2 = build_index(s2_clean)
S1["locked"] = [r for r in S1["locked"] if not near(r["query"], idx_s2)]
S2 = cap(s2_clean, CAP_S2)
audit["locked_candidates"] = dict(S1_clean=len(s1_clean), S2_clean=len(s2_clean))
# S2-capped: also remove from non-locked S1/S3 nothing (they are not locked). Final counts:
LOCKED = S1["locked"] + S2
LOCKED.sort(key=lambda r: r["id"])
files = dict(locked=LOCKED, s1_dev=S1["dev"], s1_calib=S1["calib"], s1_pool=S1["pool"], s3_dev=S3["dev"], s3_pool=S3["pool"])
lock = dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), rule=__doc__, splits={}, audit={"dropped": dict(audit["dropped"]), **audit["locked_candidates"]})
for k, rows in files.items():
    p = f"{D}/{k}.jsonl"
    with open(p, "w") as f:
        for r in rows:
            r = {a: b for a, b in r.items() if a != "_h"}; f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")
    cnt = collections.Counter((r["source"], r["kind"]) for r in rows)
    lock["splits"][k] = dict(n=len(rows), counts={f"{a}/{b}": c for (a, b), c in sorted(cnt.items())}, file_sha256=hashlib.sha256(open(p, "rb").read()).hexdigest())
    os.chmod(p, 0o444)
lc = collections.Counter(r["kind"] for r in LOCKED)
lock["locked_totals"] = dict(n=len(LOCKED), per_kind=dict(lc), S2=sum(r["source"] == "S2" for r in LOCKED))
json.dump(lock, open(f"{D}/LOCK.json", "w"), indent=1); os.chmod(f"{D}/LOCK.json", 0o444)
print(json.dumps({k: (v if k != "rule" else "...") for k, v in lock.items()}, indent=1))
