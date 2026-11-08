"""I1: assert zero normalised-text overlap among CAL/SEL/REP/GEN/feedback files, and no collision with any LOCK/HASH_PLAN/MANIFEST manifest.
Manifests are read as JSON only (hash strings); locked item files (*locked*.jsonl) are NEVER opened.  Prints + stores a result hash.
  python check_disjoint.py [--write]   (exit 1 on any overlap)"""
import sys, os, re, json, glob, hashlib, itertools
HERE = os.path.dirname(os.path.abspath(__file__)); V2 = os.path.dirname(HERE)
H = lambda s: hashlib.sha256(s.encode()).hexdigest()
norm = lambda t: re.sub(r"[^a-z0-9 ]", "", " ".join(str(t).lower().split()))
def body(it):   # strip the fixed task prompt so texts compare as articles
    u = it["user"]; return u.split("Article:", 1)[1] if "Article:" in u else u
def load(p): return [json.loads(l) for l in open(p)]
def sets():
    sp = {n: load(f"{HERE}/splits/{n}.jsonl") for n in ("cal", "sel", "rep", "gen")}
    fb = {}
    for p in sorted(glob.glob(V2 + "/task_search/NEWS/fb_*.jsonl")): fb["fb:" + os.path.basename(p)] = load(p)
    sp.update(fb); return sp
def manifests():
    out = []
    for pat in ("**/LOCK*.json", "**/HASH_PLAN*.json", "**/*MANIFEST*.json", "**/SHA256SUMS", "**/*SHA256*.json"):
        out += glob.glob(f"{V2}/{pat}", recursive=True)
    return sorted(set(p for p in out if "/exp/splits/" not in p and "archive" not in p and "DO_NOT_USE" not in p))
def hexes(p):
    try: return set(re.findall(r"\b[0-9a-f]{64}\b", open(p).read()))
    except Exception: return set()
def main():
    S = sets(); keys = {n: {} for n in S}
    for n, rows in S.items():
        for r in rows:
            if n == "gen" and r["role"] == "tokens": continue
            keys[n].setdefault(H(norm(body(r))), r["id"])
    bad = []
    for a, b in itertools.combinations(S, 2):
        ov = set(keys[a]) & set(keys[b])
        if ov: bad.append((a, b, len(ov)))
    # id overlap too
    for a, b in itertools.combinations(S, 2):
        ov = {r["id"] for r in S[a]} & {r["id"] for r in S[b]}
        if ov: bad.append((a, b, "id", len(ov)))
    # manifest collisions: any 64-hex in a manifest equal to sha256(norm text)/sha256(id)/sha256(group) of our items
    mf = manifests(); allh = {}
    for p in mf:
        for h in hexes(p): allh.setdefault(h, []).append(os.path.relpath(p, V2))
    coll = []
    for n, rows in S.items():
        for r in rows:
            if n == "gen" and r["role"] == "tokens": continue
            for h in (H(norm(body(r))), H(r["id"]), H(body(r)), H(r["user"])):
                if h in allh: coll.append((n, r["id"], allh[h][0]))
    bad += [("manifest",) + c for c in coll]
    # structural: NEWS locked buckets never used
    nb = 0
    for n in ("cal", "sel", "rep"):
        for r in S[n]:
            b = int(H("news-split-v1|" + r["group"]), 16) % 100
            if (r["src"] == "ag_news" and b >= 80) or (r["src"] == "huffpost" and b >= 20): nb += 1
    if nb: bad.append(("locked-bucket", nb))
    res = dict(sets={n: len(v) for n, v in S.items()}, n_manifests=len(mf), n_manifest_hashes=len(allh), violations=bad,
               split_files={n: hashlib.sha256(open(f"{HERE}/splits/{n}.jsonl", "rb").read()).hexdigest() for n in ("cal", "sel", "rep", "gen")})
    res["result_hash"] = H(json.dumps(res, sort_keys=True, default=str))
    print(json.dumps(res, indent=1, default=str))
    if "--write" in sys.argv: json.dump(res, open(HERE + "/DISJOINT_CHECK.json", "w"), indent=1, default=str)
    sys.exit(1 if bad else 0)
if __name__ == "__main__": main()
