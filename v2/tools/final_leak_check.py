"""Final exact-text exam-leak check. Exam request texts (every *_exam.jsonl in v2/exam/data, text after the last 'Request:'/'Message:'/'Email:' header, whitespace/case-normalised)
vs every request text found in annotation/teach/calibration/selection files (name contains ann, teach, calib, cal., sel, neigh-free) anywhere under v2 except v2/exam itself and exam copies.
Exact match only (normalised). usage: python final_leak_check.py > LEAK_CHECK.txt"""
import os, re, json, glob, sys, collections
V = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
norm = lambda s: re.sub(r"\s+", " ", s.strip().lower())
def req(u):
    m = re.split(r"(?:Request|Message|Email|Text|Query|Input):\s*", u)
    return m[-1] if len(m) > 1 else u
def strings(o):
    if isinstance(o, str): yield o
    elif isinstance(o, dict):
        for v in o.values(): yield from strings(v)
    elif isinstance(o, list):
        for v in o: yield from strings(v)
ex = {}
for f in glob.glob(V + "/exam/data/*_exam.jsonl"):
    for l in open(f):
        r = json.loads(l); ex.setdefault(norm(req(r["user"])), []).append((os.path.basename(f), r["id"]))
print("exam texts:", len(ex), "from", sorted(os.path.basename(f) for f in glob.glob(V + "/exam/data/*_exam.jsonl")))
pat = re.compile(r"(ann|teach|calib|cal[._]|_cal|sel[._]|_sel|fb_)", re.I); skip = re.compile(r"(exam[._]|_exam|EXAM|/results/|\.git/|__pycache__|V_vast/bundles|probe)")
tot = 0; hits = collections.defaultdict(list); nf = 0; nrows = 0
for root, _, fs in os.walk(V):
    if "/.git" in root: continue
    for fn in fs:
        p = os.path.join(root, fn)
        if not fn.endswith((".jsonl", ".json")) or not pat.search(fn) or skip.search(p) or p.startswith(V + '/exam/'): continue
        try: txt = open(p).read()
        except Exception: continue
        rows = []
        if fn.endswith(".jsonl"):
            for l in txt.splitlines():
                try: rows.append(json.loads(l))
                except Exception: pass
        else:
            try: j = json.loads(txt)
            except Exception: continue
            rows = j if isinstance(j, list) else [j]
        nf += 1
        for r in rows:
            nrows += 1
            for s in strings(r):
                k = norm(req(s))
                if len(k) >= 8 and k in ex: hits[p.replace(V + "/", "")].append((k[:80], ex[k][0][1]))
print("files scanned:", nf, "rows:", nrows)
print("FILES WITH EXACT EXAM-TEXT OVERLAP:", len(hits))
for p, h in sorted(hits.items()): print(f"  {p}: {len(h)} hits, e.g. {h[0]}")
