"""Collect normalised texts of every NON-locked data file in the repo (dev/calib/sel/feedback/bundles) so exam builders can exclude anything already used.
Skips any path containing exam|lock|seal|x15|never (other lanes' sealed files: never read) and pure raw public-dataset dumps (/raw/, L5 b77_train/test) and the TOX routing `reserve` split (documented untouched). Skips .git and files > 80MB.
Usage: python3 -I leakscan.py OUT.txt [--lenient]   (one normalised text per line; not committed, only the counts are)"""
import os, re, sys, json
from multiprocessing import Pool
SKIP = re.compile(r"exam|lock|seal|x15|never|/\.git/|/raw/|b77_train|b77_test|routing/data/reserve", re.I)
LENIENT_SKIP = re.compile(r"tournament/TOX/routing/data/(dev|train|probe_dev)\.jsonl")   # --lenient: older routing-queue task whose TF-IDF baseline/probe used these; no quantiser lane calibrated on them
MARK = re.compile(r"(?:Request|Message|Utterance|Question|Text|Ticket|Article|Email|Query): (.*)$", re.S)
STR = re.compile(r'"((?:[^"\\]|\\.){8,1500})"')
norm = lambda t: re.sub(r"[^a-z0-9 ]", "", " ".join(str(t).lower().split()))
def scan(p):
    out = set()
    try:
        s = open(p, encoding="utf-8", errors="ignore").read()
    except Exception: return out
    for m in STR.finditer(s):
        t = m.group(1)
        try: t = json.loads('"' + t + '"')
        except Exception: pass
        k = norm(t)
        if k: out.add(k)
        mm = MARK.search(t)
        if mm:
            x = mm.group(1).removesuffix(" ...")
            k = norm(x)
            if k: out.add(k)
    return out
def files(root):
    for d, ds, fs in os.walk(root):
        ds[:] = [x for x in ds if x != ".git"]
        for f in fs:
            p = os.path.join(d, f)
            if f.endswith((".json", ".jsonl")) and not SKIP.search(p) and not (LENIENT and LENIENT_SKIP.search(p)) and "/v2/exam/" not in p:
                try:
                    if os.path.getsize(p) <= 80e6: yield p
                except OSError: pass
LENIENT = "--lenient" in sys.argv
if __name__ == "__main__":
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    fl = list(files(root)); print(len(fl), "files", flush=True)
    U = set()
    with Pool(6) as pl:
        for s in pl.imap_unordered(scan, fl, chunksize=4): U |= s
    open(sys.argv[1], "w").write("\n".join(sorted(U))); print(len(U), "norm strings")
