"""X15: build + SEAL the NEWS locked-exam manifest. Hashes only; no text, no model run, nothing scored.
  LOCKED_BUILD_OK=1 python seal_news_lock.py [--write]
Cells: {ag_news, huffpost} x {World,Sports,Business,Technology}. Candidates: AG bucket [80,100), HuffPost bucket [20,100) (bucket = build_news.bucket,
sha256 of first 12 normalised words %100, so a reprint group never straddles splits). Per cell: sort candidates by sha256('news-lock-order|'+id), take first 400.
(Where a cell has < 400 eligible the cell size is reduced to the common minimum >= 300; see report.)
Final open-tail prompt = build_splits.PROMPT (fixed in X01). Manifest stores id, sha256(norm text), sha256(group key), sha256(rendered prompt)."""
import os, sys, json, hashlib, collections
assert os.environ.get("LOCKED_BUILD_OK") == "1", "set LOCKED_BUILD_OK=1 (X15 only)"
HERE = os.path.dirname(os.path.abspath(__file__)); V2 = os.path.dirname(HERE)
sys.path.insert(0, V2 + "/task_search/NEWS"); sys.path.insert(0, HERE)
import build_news as bn
import build_splits as bs
bn.PROMPT = bs.PROMPT
H, norm, bucket = bn.H, bn.norm, bn.bucket
PER = 400
import glob, re
def used_hashes():
    """normalised-body hashes of every row of every jsonl in the repo (any earlier build/dev/calib/pool/lock/feedback file, any design); candidates matching are excluded"""
    ROOT = os.path.dirname(V2); out = set()
    for p in glob.glob(ROOT + "/**/*.jsonl", recursive=True):
        if "LOCK_NEWS" in p: continue
        for l in open(p, errors="ignore"):
            if len(l) < 40: continue
            try: r = json.loads(l)
            except Exception: continue
            if not isinstance(r, dict): continue
            u = r.get("user") or r.get("text") or ""
            if not isinstance(u, str): continue
            b = u.split("Article:", 1)[1] if "Article:" in u else u
            out.add(H(norm(b)))
    return out
def main():
    ag, bbc, hp, _ = [bn.dedupe(x) for x in bn.load_all()]
    cells = {}; used = used_hashes(); excl = collections.Counter()
    for src, rows, lo, hi in (("ag_news", ag, 80, 100), ("huffpost", hp, 20, 100)):
        for k in bn.KINDS:
            cand = [r for r in rows if r[1] == k and lo <= bucket(r[2]) < hi]
            keep = [r for r in cand if H(norm(r[2])) not in used]; excl[f"{src}|{k}"] = len(cand) - len(keep); cand = keep
            cells[(src, k)] = sorted(cand, key=lambda r: H("news-lock-order|" + bn.item(src, "LOCK", r[0], r[1], r[2])["id"]))
    elig = {f"{s}|{k}": len(v) for (s, k), v in cells.items()}
    n = min(PER, min(elig.values())); assert n >= 300, elig
    items, rep = [], {}
    for (src, k), v in cells.items():
        for r in v[:n]:
            d = bn.item(src, "LOCK", r[0], r[1], r[2])
            items.append(dict(id=d["id"], cell=f"{src}|{k}", text_sha256=H(norm(r[2])), group_sha256=H(bn.gkey(r[2])), prompt_sha256=H(d["user"])))
    items.sort(key=lambda i: i["id"])
    # guards
    ids = [i["id"] for i in items]; assert len(set(ids)) == len(ids)
    assert len({i["text_sha256"] for i in items}) == len(items)
    for (src, k), v in cells.items():
        for r in v[:n]:
            b = bucket(r[2]); assert (b >= 80) if src == "ag_news" else (b >= 20)
    # group-safety: no group key is shared with any non-locked row (bucket function is a function of the group key, so by construction; verify on all rows)
    lockedg = {i["group_sha256"] for i in items}
    for src, rows in (("ag_news", ag), ("huffpost", hp), ("bbc_news", bbc)):
        for r in rows:
            if H(bn.gkey(r[2])) in lockedg and not ((src == "ag_news" and bucket(r[2]) >= 80) or (src == "huffpost" and bucket(r[2]) >= 20)):
                raise SystemExit("group straddles splits: " + r[0])
    idans = "\n".join(sorted(f"{i['id']}|{i['cell'].split('|')[1]}" for i in items))
    m = dict(version=1, task="NEWS", created_utc=None, builder="v2/exp/seal_news_lock.py", builder_sha256=hashlib.sha256(open(__file__, "rb").read()).hexdigest(),
             bucket_fn="task_search/NEWS/build_news.py:bucket", lock_order="sha256('news-lock-order|'+id) ascending, first n per cell",
             prompt=bs.PROMPT, prompt_sha256=H(bs.PROMPT), eligible_per_cell=elig, excluded_prior_repo_use_per_cell=dict(excl), per_cell=n, n_items=len(items),
             counts=dict(collections.Counter(i["cell"] for i in items)),
             per_kind=dict(collections.Counter(i["cell"].split("|")[1] for i in items)), per_source=dict(collections.Counter(i["cell"].split("|")[0] for i in items)),
             id_answer_sorted_sha256=H(idans), items=items)
    m["items_sha256"] = H(json.dumps(items, sort_keys=True))
    return m
if __name__ == "__main__":
    m = main()
    print({k: v for k, v in m.items() if k != "items"})
    if "--write" in sys.argv:
        import datetime
        m["created_utc"] = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        out = HERE + "/LOCK_NEWS_MANIFEST.json"; json.dump(m, open(out, "w"), indent=1, sort_keys=True)
        os.chmod(out, 0o444); print("manifest file sha256", hashlib.sha256(open(out, "rb").read()).hexdigest())
