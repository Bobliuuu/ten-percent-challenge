"""X15 verification: locked manifest vs every NEWS-ish jsonl in the repo (text hash and id), by normalised text; prints counts only."""
import json, glob, os, re, hashlib, sys
H=lambda s: hashlib.sha256(s.encode()).hexdigest()
norm=lambda t: re.sub(r"[^a-z0-9 ]","", " ".join(str(t).lower().split()))
V2=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
m=json.load(open(V2+"/exp/LOCK_NEWS_MANIFEST.json")); th={i["text_sha256"] for i in m["items"]}; gh={i["group_sha256"] for i in m["items"]}; ids={i["id"] for i in m["items"]}
tot=0; hits=[]; files=0
for p in glob.glob(V2+"/**/*.jsonl", recursive=True):
    if "LOCK_NEWS" in p: continue
    files+=1
    for l in open(p, errors="ignore"):
        try: r=json.loads(l)
        except Exception: continue
        if not isinstance(r,dict): continue
        u=r.get("user") or r.get("text") or ""
        if not isinstance(u,str): continue
        body=u.split("Article:",1)[1] if "Article:" in u else u
        tot+=1
        if r.get("id") in ids or H(norm(body)) in th: hits.append((os.path.relpath(p,V2), r.get("id")))
print(json.dumps(dict(jsonl_files=files, rows=tot, hits=len(hits), hit_files=sorted({h[0] for h in hits})[:10])))
sys.exit(1 if hits else 0)
