"""T3 AI-vs-human detection: build hash-split sources on Modal CPU (no GPU, no model scoring).
 S1 = Hello-SimpleAI/HC3 (English; ChatGPT vs human answers; 5 domains; cc-by-sa-4.0) -> locked/dev/calib/pool (split by QUESTION hash)
 S3 = yaful/MAGE test.csv (apache-2.0; 10 domains x many generators) -> OOD DEV PROXY ONLY (dev.jsonl, never locked)
 S2 = liamdugan/raid train.csv (mit; attack=none; newer chat generators gpt4/llama-chat/mistral-chat/cohere-chat/mpt-chat; 8 domains) -> LOCKED OOD block only
Split rule: h=int(sha256(key)[:8],16); h%7==0 locked, ==1 dev, ==2 calib, else pool (S1). S3 -> dev(h%7==1) + pool. S2 -> locked only.
All texts whitespace-normalised, truncated to 1500 chars at a word boundary, min 200 chars. Locked sets are class-balanced by seeded sampling.
Run: modal run --detach build_data.py   -> writes /d/*.jsonl + audit to volume tenpct-t3-exam."""
import modal, json, hashlib, re, random, collections, os
app = modal.App("tenpct-t3-build")
img = modal.Image.debian_slim().pip_install("huggingface_hub", "pandas", "pyarrow")
vol = modal.Volume.from_name("tenpct-t3-exam", create_if_missing=True)
hfv = modal.Volume.from_name("tenpct-v2-hf", create_if_missing=True)
LOCKED_S1 = 800  # per class
LOCKED_S2 = 400  # per class
DEV_S1 = 300; DEV_S3 = 300  # per-source dev items used by probe (balanced); larger dev files also stored
MAXC, MINC = 1500, 200

def hsel(s): return int(hashlib.sha256(s.encode()).hexdigest()[:8], 16)
def clean(t):
    t = " ".join(str(t).split())
    if len(t) > MAXC:
        t = t[:MAXC]; t = t[:t.rfind(" ")] if " " in t else t
    return t

@app.function(image=img, timeout=7200, memory=16384, volumes={"/d": vol, "/hf": hfv}, cpu=4)
def build():
    import pandas as pd, csv
    from huggingface_hub import hf_hub_download
    os.environ["HF_HOME"] = "/hf"
    out = {}  # name -> list of rows
    # ---------------- S1 HC3
    p = hf_hub_download("Hello-SimpleAI/HC3", "all.jsonl", repo_type="dataset")
    s1 = {"locked": [], "dev": [], "calib": [], "pool": []}
    for ln, l in enumerate(open(p, encoding="utf-8")):
        r = json.loads(l); q = r["question"]; h = hsel("S1|" + q)
        sp = {0: "locked", 1: "dev", 2: "calib"}.get(h % 7, "pool")
        for gold, key, gen in [("human", "human_answers", "human"), ("ai", "chatgpt_answers", "chatgpt")]:
            for j, a in enumerate(r[key]):
                t = clean(a)
                if len(t) < MINC: continue
                s1[sp].append(dict(id=f"S1-{ln}-{gold[0]}{j}", text=t, gold=gold, source="S1", domain=r["source"], generator=gen, qhash=h))
    # ---------------- S3 MAGE (OOD dev proxy)
    csv.field_size_limit(10**9)
    p = hf_hub_download("yaful/MAGE", "test.csv", repo_type="dataset")
    s3 = {"dev": [], "pool": []}
    for i, r in enumerate(csv.DictReader(open(p, encoding="utf-8-sig"))):
        t = clean(r["text"])
        if len(t) < MINC: continue
        h = hsel("S3|" + t); gold = "human" if r["label"] == "1" else "ai"
        dom, _, gen = r["src"].partition("_")
        s3["dev" if h % 7 == 1 else "pool"].append(dict(id=f"S3-{i}", text=t, gold=gold, source="S3", domain=dom, generator=("human" if gold == "human" else gen), qhash=h))
    # ---------------- S2 RAID (locked only)
    p = hf_hub_download("liamdugan/raid", "train.csv", repo_type="dataset")
    AIM = {"gpt4", "llama-chat", "mistral-chat", "cohere-chat", "mpt-chat"}
    cand = []
    for ch in pd.read_csv(p, chunksize=100000, usecols=["id", "model", "domain", "attack", "generation"]):
        ch = ch[(ch.attack == "none") & (ch.model.isin(AIM | {"human"}))]
        for r in ch.itertuples(index=False):
            t = clean(r.generation)
            if len(t) < MINC: continue
            h = hsel("S2|" + t)
            if h % 7 != 0: continue
            cand.append(dict(id=f"S2-{r.id}", text=t, gold="human" if r.model == "human" else "ai", source="S2", domain=r.domain, generator=r.model, qhash=h))
    print("S2 locked candidates", len(cand), collections.Counter((c["gold"], c["generator"]) for c in cand))
    # ---------------- balanced locked samples (seeded, stratified round-robin over domain x generator)
    def strat(rows, n, seed):
        rng = random.Random(seed); g = collections.defaultdict(list)
        for r in sorted(rows, key=lambda r: r["id"]): g[(r["domain"], r["generator"])].append(r)
        for v in g.values(): rng.shuffle(v)
        keys = sorted(g); picked = []
        while len(picked) < n and any(g.values()):
            for k in keys:
                if g[k] and len(picked) < n: picked.append(g[k].pop())
        return picked
    def bal(rows, n, seed): return strat([r for r in rows if r["gold"] == "human"], n, seed) + strat([r for r in rows if r["gold"] == "ai"], n, seed + 1)
    L1 = bal(s1["locked"], LOCKED_S1, 11); L2 = bal(cand, LOCKED_S2, 12)
    # shuffle for file order
    rng = random.Random(5); rng.shuffle(L1); rng.shuffle(L2)
    out["locked_S1"] = L1; out["locked_S2"] = L2
    out["dev_S1"] = s1["dev"]; out["calib_S1"] = s1["calib"]; out["pool_S1"] = s1["pool"]
    out["dev_S3"] = s3["dev"]; out["pool_S3"] = s3["pool"]
    # rest of s1 locked-hash rows not sampled are DROPPED from everything (never in pool): they share locked question hashes
    # ---------------- leakage audit: locked vs everything else (exact normalised + 5-word-shingle Jaccard >=0.5)
    norm = lambda s: re.sub(r"\W+", " ", s.lower()).strip()
    def sh(s):
        w = norm(s).split(); return {hash(" ".join(w[i:i + 5])) for i in range(max(1, len(w) - 4))}
    locked = L1 + L2; Lnorm = {norm(r["text"]): r["id"] for r in locked}
    Lsh = [sh(r["text"]) for r in locked]; inv = collections.defaultdict(list)
    for i, s in enumerate(Lsh):
        for x in s: inv[x].append(i)
    audit = {"n_locked": len(locked), "per_split": {}}
    # internal locked dups
    audit["locked_internal_exact_dups"] = len(locked) - len(Lnorm)
    for name, rows in out.items():
        if name.startswith("locked"): continue
        keep = []; ex = nd = 0
        for r in rows:
            if norm(r["text"]) in Lnorm: ex += 1; continue
            s = sh(r["text"]); c = collections.Counter()
            for x in s:
                if len(inv[x]) < 50:
                    for i in inv[x]: c[i] += 1
            near = any(v / len(s | Lsh[i]) if False else (v / max(1, len(s | Lsh[i]))) >= 0.5 for i, v in c.most_common(10))
            if near: nd += 1; continue
            keep.append(r)
        audit["per_split"][name] = dict(before=len(rows), dropped_exact=ex, dropped_near_j50=nd, after=len(keep))
        out[name] = keep
    # locked-internal near dups across S1/S2 (report only)
    # ---------------- write
    os.makedirs("/d/data", exist_ok=True)
    meta = {}
    for name, rows in out.items():
        for r in rows: r["kind"] = f"{r['source']}_{r['gold']}"
        fp = f"/d/data/{name}.jsonl"
        if os.path.exists(fp): os.chmod(fp, 0o644)
        with open(fp, "w") as f:
            for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
        meta[name] = dict(n=len(rows), file_sha256=hashlib.sha256(open(fp, "rb").read()).hexdigest(),
                          by_kind=dict(collections.Counter(r["kind"] for r in rows)),
                          by_domain=dict(collections.Counter(r["domain"] for r in rows)))
    # locked stats are counts only (no text inspection)
    audit["meta"] = meta
    json.dump(audit, open("/d/data/BUILD_AUDIT.json", "w"), indent=1, default=str)
    vol.commit()
    return audit

@app.local_entrypoint()
def main():
    a = build.remote()
    open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs", "build_audit.json"), "w").write(json.dumps(a, indent=1))
    print(json.dumps(a, indent=1))
