"""Entry32B dev/calib builder (dev-only; locked_S1/locked_S2 are NEVER read). Deterministic. Output files are git-ignored (dataset text).
Sources (all from tournament/INJ/data, built by tournament/INJ/build_data.py, group-disjoint across splits):
  DEV  S1 = dev_S1 (800) + pool_S1 balanced 1000/1000 (hash order)   [PromptShield 2025]
  DEV  S3 = dev_S3 (766)                                              [deepset 2023 + gandalf]
  DEV  S2 = audit_S2: all injection (247) + 500 benign (hash order)   [JailbreakHub 2022-12..2023-12; OOD proxy, labels noisy]
  CALIB   = 700 items of calib_S1 only (PromptShield, disjoint groups from every dev/locked item). S2/S3 never used for calibration.
python3 build_dev.py"""
import json, hashlib, os, sys
H = os.path.dirname(os.path.abspath(__file__)); D = H + "/../tournament/INJ/data/"; sys.path.insert(0, H)
from prompt import user_message
rd = lambda f: [json.loads(l) for l in open(D + f)]
key = lambda r: hashlib.sha256(("e32" + r["id"]).encode()).hexdigest()
def rows(rs, role): return [dict(id=r["id"], role=role, source=r["source"], kind=r["kind"], gold=r["gold"], group=r["group"], query=user_message(r["text"])) for r in rs]
pool = rd("pool_S1.jsonl"); pi = sorted([r for r in pool if r["gold"] == "injection"], key=key)[:1000]; pb = sorted([r for r in pool if r["gold"] == "benign"], key=key)[:1000]
a = rd("audit_S2.jsonl"); ai = [r for r in a if r["gold"] == "injection"]; ab = sorted([r for r in a if r["gold"] == "benign"], key=key)[:500]
dev = rows(rd("dev_S1.jsonl"), "S1") + rows(pi + pb, "S1") + rows(rd("dev_S3.jsonl"), "S3") + rows(ai + ab, "S2")
cal = sorted(rd("calib_S1.jsonl"), key=key)
ci = [r for r in cal if r["gold"] == "injection"][:300]; cb = [r for r in cal if r["gold"] == "benign"][:400]
calr = [dict(id=r["id"], user=user_message(r["text"]), answer="Injection" if r["gold"] == "injection" else "Benign") for r in ci + cb]
ids = {r["id"] for r in dev}; assert not ids & {r["id"] for r in calr}
gd = {r["group"] for r in dev}; assert not gd & {r["group"] for r in ci + cb}
open(H + "/dev_inj.jsonl", "w").write("".join(json.dumps(r) + "\n" for r in dev)); open(H + "/calib_inj.jsonl", "w").write("".join(json.dumps(r) + "\n" for r in calr))
import collections; print(len(dev), collections.Counter(r["kind"] for r in dev), len(calr))
