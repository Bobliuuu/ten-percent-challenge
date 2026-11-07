"""INJ (prompt-injection / jailbreak detection) deterministic, leak-proof data build. Pure python (no GPU, no model).
Sources (all pinned by HF commit sha + sha256 of the raw files, see LOCK.json):
 S1 = hendzh/PromptShield (apache-2.0, 2025; train.json + validation.json merged; its 'test' split is NOT used)  -> locked / dev / calib / pool, split by GROUP
 S3 = deepset/prompt-injections (apache-2.0, 2023, EN+DE, hand-made) + Lakera/gandalf_ignore_instructions (mit, injections only)  -> OOD DEV PROXY ONLY
 S2 = walledai/JailbreakHub (mit, in-the-wild jailbreaks vs ordinary prompts from the same Reddit/Discord/FlowGPT communities, 2022-12..2023-12)
      -> LOCKED OOD block (+ a disjoint-group 'audit' slice used ONLY for dataset QA / lexical-baseline / rule-8 proxy, never for any model decision)
Split unit = GROUP = connected component of the within-source near-duplicate graph (word-3-shingle Jaccard >= 0.5); split = sha256(group)%5.
Global dedup (exact + shingle Jaccard >= 0.6) over ALL items of ALL splits/sources, priority locked > dev > calib > pool > audit: a later item is dropped when it
matches an earlier kept item. No text >2000 chars is kept (never truncated: an injection can sit at the tail). Label strings: 'injection' / 'benign'.
Locked & dev samples are length-matched across classes (log-length buckets) and the benign half is half-filled with 'trigger-lexicon' benign items (near-miss text).
Run:  python3 build_data.py   (needs huggingface_hub, pandas, pyarrow). Re-running reproduces byte-identical files."""
import os, re, json, csv, hashlib, random, zlib, collections, sys
from huggingface_hub import hf_hub_download

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "raw"); OUT = os.path.join(HERE, "data")
SRC = {
    "hendzh/PromptShield": ("a5234cb1f5cdb256600cab64b8c961195b5e8404", ["train.json", "validation.json"], "apache-2.0"),
    "deepset/prompt-injections": ("4f61ecb038e9c3fb77e21034b22511b523772cdd", ["data/train-00000-of-00001-9564e8b05b4757ab.parquet", "data/test-00000-of-00001-701d16158af87368.parquet"], "apache-2.0"),
    "Lakera/gandalf_ignore_instructions": ("04737b65e90a6794ec227012e4a255a7def6344b", None, "mit"),
    "walledai/JailbreakHub": ("1a33b351b53167e37a64bd9bf7e286e435ba4a31", ["data/train-00000-of-00001.parquet"], "mit"),
}
MAXC, MINC = 2000, 8
LOCKED_N = {"S1": 800, "S2": 400}        # per class target
DEV_N = {"S1": 400, "S3": 400}           # per class cap for dev (probe uses 300 balanced)
J_GROUP, J_DEDUP = 0.5, 0.6
TRIG = re.compile(r"ignor|disregard|previous instruction|system prompt|jailbreak|\bDAN\b|developer mode|password|secret|overrid|pretend|you are now|act as|bypass|hack|malware|exploit|prompt injection|instruction|forget|reveal|confidential|do anything", re.I)

def sha(b): return hashlib.sha256(b).hexdigest()
def fsha(p): return sha(open(p, "rb").read())
def hsel(s, m=5): return int(sha(s.encode())[:8], 16) % m
def norm(s): return re.sub(r"\W+", " ", s.lower()).strip()

def clean(t):
    t = str(t).replace("\r\n", "\n").replace("\r", "\n")
    t = "\n".join(l.rstrip() for l in t.split("\n")); t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()

def fetch(repo, fn):
    rev = SRC[repo][0]
    p = hf_hub_download(repo, fn, repo_type="dataset", revision=rev, local_dir=os.path.join(RAW, repo.replace("/", "__")))
    return p

def load_all():
    import pandas as pd
    items = []
    # S1 PromptShield
    for fn in SRC["hendzh/PromptShield"][1]:
        d = json.load(open(fetch("hendzh/PromptShield", fn), encoding="utf-8"))
        for i, r in enumerate(d):
            lab = r.get("label", r.get("flag"))
            items.append(dict(src="S1", subset="promptshield_" + fn.split(".")[0], rid=f"{fn.split('.')[0]}-{i}", text=r["prompt"], gold="injection" if int(lab) == 1 else "benign"))
    # S3 deepset
    for fn in SRC["deepset/prompt-injections"][1]:
        d = pd.read_parquet(fetch("deepset/prompt-injections", fn))
        for i, r in enumerate(d.itertuples(index=False)):
            items.append(dict(src="S3", subset="deepset", rid=f"{fn.split('/')[1][:5]}-{i}", text=r.text, gold="injection" if int(r.label) == 1 else "benign"))
    # S3 gandalf (ignore_instructions): every row is an injection attempt; we pull the parquet files via the hub api listing
    from huggingface_hub import HfApi
    rev = SRC["Lakera/gandalf_ignore_instructions"][0]
    for fn in sorted(s.rfilename for s in HfApi().dataset_info("Lakera/gandalf_ignore_instructions", revision=rev).siblings if s.rfilename.endswith(".parquet")):
        d = pd.read_parquet(fetch("Lakera/gandalf_ignore_instructions", fn))
        for i, r in enumerate(d.itertuples(index=False)):
            items.append(dict(src="S3", subset="gandalf", rid=f"{os.path.basename(fn)[:8]}-{i}", text=r.text, gold="injection"))
    # S2 JailbreakHub
    d = pd.read_parquet(fetch("walledai/JailbreakHub", SRC["walledai/JailbreakHub"][1][0]))
    for i, r in enumerate(d.itertuples(index=False)):
        items.append(dict(src="S2", subset=f"jbhub_{r.platform}", rid=f"jbh-{i}", text=r.prompt, gold="injection" if str(r.jailbreak) == "True" else "benign"))
    return items

# ---------------------------------------------------------------- shingles / near-dup (exact Jaccard via prefix filtering)
def shingles(text):
    w = re.findall(r"\w+", text.lower())
    if len(w) < 3: return frozenset([zlib.crc32(" ".join(w).encode())])
    return frozenset(zlib.crc32(" ".join(w[i:i + 3]).encode()) for i in range(len(w) - 2))

def near_pairs(sh, thr, order=None):
    """sh: list of frozensets. yields (i, j) i<j with J>=thr. Exact (prefix filter + verification)."""
    df = collections.Counter(x for s in sh for x in s)
    pre = []
    for s in sh:
        o = sorted(s, key=lambda x: (df[x], x)); k = len(o) - int(-(-thr * len(o) // 1)) + 1
        pre.append(o[:max(1, k)])
    idx = collections.defaultdict(list); out = []
    for i in sorted(range(len(sh)), key=lambda i: len(sh[i])):
        seen = set()
        for x in pre[i]:
            for j in idx[x]:
                if j in seen: continue
                seen.add(j); a, b = sh[i], sh[j]
                if min(len(a), len(b)) < thr * max(len(a), len(b)): continue
                if len(a & b) / len(a | b) >= thr: out.append((min(i, j), max(i, j)))
            idx[x].append(i)
    return out

class UF:
    def __init__(s, n): s.p = list(range(n))
    def f(s, x):
        while s.p[x] != x: s.p[x] = s.p[s.p[x]]; x = s.p[x]
        return x
    def u(s, a, b): s.p[s.f(a)] = s.f(b)

# ---------------------------------------------------------------- sampling
def lb(n): return int(__import__("math").log2(max(2, n)) * 3)

def matched(rows, n, seed):
    """class-balanced, length-matched sample; benign half filled with TRIG-matching items. rows sorted for determinism."""
    rng = random.Random(seed)
    rows = sorted(rows, key=lambda r: r["id"])
    inj = [r for r in rows if r["gold"] == "injection"]; ben = [r for r in rows if r["gold"] == "benign"]
    rng.shuffle(inj); rng.shuffle(ben)
    # choose the smaller class as the anchor so that both reach the same size
    m = min(n, len(inj), len(ben))
    anchor_is_inj = len(inj) <= len(ben)
    A, B = (inj, ben) if anchor_is_inj else (ben, inj)
    A = A[:m]
    cnt = collections.Counter(lb(len(r["text"])) for r in A)
    bk = collections.defaultdict(list)
    for r in B: bk[lb(len(r["text"]))].append(r)
    pick = []
    used = set()
    def take(b, k, prefer_trig):
        got = 0
        cand = bk.get(b, [])
        order = [r for r in cand if r["id"] not in used]
        if prefer_trig:
            tr = [r for r in order if TRIG.search(r["text"])]; ot = [r for r in order if not TRIG.search(r["text"])]
            half = (k + 1) // 2; order = tr[:half] + ot + tr[half:]
            # keep the trig half first, then others, then leftover trig
            order = tr[:half] + ot + tr[half:]
        for r in order[:k]:
            used.add(r["id"]); pick.append(r); got += 1
        return got
    deficit = 0
    trig_pref = True
    # trigger-preferred items only help when benign is the matched class; apply to benign side only
    for b in sorted(cnt):
        got = take(b, cnt[b], trig_pref and anchor_is_inj)
        deficit += cnt[b] - got
    if deficit:  # fill from nearest buckets
        for b in sorted(cnt, key=lambda b: -cnt[b]):
            for d in range(1, 12):
                if deficit == 0: break
                for bb in (b - d, b + d):
                    if deficit and bb in bk: deficit -= take(bb, min(deficit, 3), False)
    # if still deficit (rare) shrink the anchor correspondingly
    if len(pick) < len(A):
        keep = len(pick)
        A = A[:keep]
    res = A + pick
    return sorted(res, key=lambda r: sha((str(seed) + r["id"]).encode()))

def main():
    os.makedirs(RAW, exist_ok=True); os.makedirs(OUT, exist_ok=True)
    items = load_all()
    # clean + length filter
    keep = []
    dropped = collections.Counter()
    for i in items:
        t = clean(i["text"]) if isinstance(i["text"], str) else ""
        if len(t) < MINC: dropped["short_or_empty"] += 1; continue
        if len(t) > MAXC: dropped[f"too_long_{i['src']}_{i['gold']}"] += 1; continue
        i["text"] = t; i["id"] = f"{i['src']}-{i['rid']}"; keep.append(i)
    items = keep
    # exact dup / conflicting-label handling
    by = collections.defaultdict(list)
    for i in items: by[norm(i["text"])].append(i)
    out = []; confl = 0; exd = 0
    for k, v in by.items():
        if len({x["gold"] for x in v}) > 1: confl += len(v); continue
        v.sort(key=lambda x: x["id"]); out.append(v[0]); exd += len(v) - 1
    items = sorted(out, key=lambda x: x["id"])
    dropped["exact_dups"] = exd; dropped["label_conflict_all_copies"] = confl
    # within-source grouping
    for src in ("S1", "S2", "S3"):
        idx = [k for k, i in enumerate(items) if i["src"] == src]
        sh = [shingles(items[k]["text"]) for k in idx]
        for k, s in zip(idx, sh): items[k]["_sh"] = s
        uf = UF(len(idx))
        for a, b in near_pairs(sh, J_GROUP): uf.u(a, b)
        root_min = {}
        for a, k in enumerate(idx):
            r = uf.f(a); root_min[r] = min(root_min.get(r, items[k]["id"]), items[k]["id"])
        for a, k in enumerate(idx): items[k]["group"] = root_min[uf.f(a)]
    # split assignment
    for i in items:
        if i["src"] == "S1":
            i["split"] = {0: "locked", 1: "dev", 2: "calib"}.get(hsel("INJ-S1|" + i["group"]), "pool")
        elif i["src"] == "S3": i["split"] = "dev"
        else: i["split"] = "locked" if hsel("INJ-S2|" + i["group"]) in (0, 1, 2) else "audit"
    # global dedup in priority order
    PRI = {"locked": 0, "dev": 1, "calib": 2, "pool": 3, "audit": 4}
    order = sorted(range(len(items)), key=lambda k: (PRI[items[k]["split"]], items[k]["id"]))
    sh = [items[k]["_sh"] for k in order]
    drop = set(); pair_cnt = collections.Counter()
    pairs = near_pairs(sh, J_DEDUP)
    nb = collections.defaultdict(list)
    for a, b in pairs: nb[max(a, b)].append(min(a, b))   # later one (higher position) vs earlier
    kept = set()
    for pos in range(len(order)):
        k = order[pos]
        early = [e for e in nb.get(pos, []) if e in kept]
        if early:
            drop.add(k); e = early[0]; pair_cnt[(items[k]["split"] + "<-" + items[order[e]]["split"], items[k]["src"] + "<-" + items[order[e]]["src"])] += 1
        else: kept.add(pos)
    dropped.update({"neardup_" + "|".join(k): v for k, v in pair_cnt.items()})
    items = [i for k, i in enumerate(items) if k not in drop]
    # build splits
    S = collections.defaultdict(list)
    for i in items: S[(i["src"], i["split"])].append(i)
    L1 = matched(S[("S1", "locked")], LOCKED_N["S1"], 101)
    L2 = matched(S[("S2", "locked")], LOCKED_N["S2"], 102)
    D1 = matched(S[("S1", "dev")], DEV_N["S1"], 103)
    # S3 dev: gandalf (injection only, trivially lexical) is capped at the number of deepset injections
    s3 = S[("S3", "dev")]
    dsi = [r for r in s3 if r["subset"] == "deepset" and r["gold"] == "injection"]
    gan = sorted([r for r in s3 if r["subset"] == "gandalf"], key=lambda r: sha(("g" + r["id"]).encode()))[:len(dsi)]
    D3 = matched([r for r in s3 if r["subset"] == "deepset"] + gan, DEV_N["S3"], 104)
    out = {"locked_S1": L1, "locked_S2": L2, "dev_S1": D1, "dev_S3": D3,
           "calib_S1": [r for r in S[("S1", "calib")]], "pool_S1": [r for r in S[("S1", "pool")]], "audit_S2": S[("S2", "audit")]}
    # locked group rows that were not sampled are DROPPED from everything (they share locked groups)
    os.makedirs(OUT, exist_ok=True)
    meta = {}
    for name, rows in out.items():
        rows = sorted(rows, key=lambda r: sha(("order" + r["id"]).encode()))
        fp = os.path.join(OUT, name + ".jsonl")
        if os.path.exists(fp): os.chmod(fp, 0o644)
        with open(fp, "w") as f:
            for r in rows:
                f.write(json.dumps(dict(id=r["id"], text=r["text"], gold=r["gold"], source=r["src"], subset=r["subset"], kind=f"{r['src']}_{r['gold']}", group=r["group"], nchar=len(r["text"])), ensure_ascii=False) + "\n")
        meta[name] = dict(n=len(rows), sha256=fsha(fp), by_kind=dict(collections.Counter(f"{r['src']}_{r['gold']}" for r in rows)),
                          n_groups=len({r["group"] for r in rows}), by_subset=dict(collections.Counter(r["subset"] for r in rows)))
    # group disjointness / dedup self-check (counts only)
    gsets = {n: {r["group"] for r in rows} for n, rows in out.items() if n.endswith("S1")}
    names = list(gsets); overlaps = {f"{a}&{b}": len(gsets[a] & gsets[b]) for x, a in enumerate(names) for b in names[x + 1:]}
    audit = dict(dropped=dict(dropped), meta=meta, s1_group_overlaps=overlaps, n_total_locked=len(L1) + len(L2),
                 params=dict(MAXC=MAXC, MINC=MINC, J_GROUP=J_GROUP, J_DEDUP=J_DEDUP, LOCKED_N=LOCKED_N, DEV_N=DEV_N))
    json.dump(audit, open(os.path.join(OUT, "BUILD_AUDIT.json"), "w"), indent=1)
    raw = {}
    for root, _, fs in os.walk(RAW):
        for f in fs:
            if ".cache" in root or f.startswith("."): continue
            p = os.path.join(root, f); raw[os.path.relpath(p, RAW)] = fsha(p)
    lock = dict(created="2026-11-06", note="LOCK for task INJ. locked rows never printed/used for decisions. Files chmod 444.",
                sources={k: dict(revision=v[0], license=v[2]) for k, v in SRC.items()}, raw_sha256=dict(sorted(raw.items())),
                files={n: dict(n=m["n"], sha256=m["sha256"], by_kind=m["by_kind"]) for n, m in meta.items()},
                locked_total=len(L1) + len(L2), build_script_sha256=fsha(__file__))
    json.dump(lock, open(os.path.join(HERE, "LOCK.json"), "w"), indent=1)
    with open(os.path.join(HERE, "SHA256SUMS"), "w") as f:
        for n in sorted(meta): f.write(f"{meta[n]['sha256']}  data/{n}.jsonl\n")
        f.write(f"{fsha(os.path.join(HERE, 'LOCK.json'))}  LOCK.json\n"); f.write(f"{fsha(__file__)}  build_data.py\n")
    for n in out: os.chmod(os.path.join(OUT, n + ".jsonl"), 0o444) if n.startswith("locked") else None
    print(json.dumps({k: (v if k != "meta" else {n: (m["n"], m["by_kind"]) for n, m in v.items()}) for k, v in audit.items()}, indent=1))

if __name__ == "__main__":
    main()
