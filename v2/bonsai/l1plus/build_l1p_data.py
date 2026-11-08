"""DEV pools for L1+: continual-feedback skills from MASSIVE scenarios email / takeaway (never in the SKILL7 menu, not in the exam, which only has the 7 menu kinds + Recipe teach).
python build_l1p_data.py --out DIR  -> skill_email.jsonl skill_takeaway.jsonl (same item format as build_skill.py; buckets < 70 only)"""
import json, hashlib, re, argparse, os
from datasets import load_dataset
H = lambda s: hashlib.sha256(s.encode()).hexdigest(); norm = lambda t: re.sub(r"[^a-z0-9 ]", "", " ".join(str(t).lower().split()))
bucket = lambda t: int(H("easy-split-v1|SKILL7|" + norm(t)), 16) % 100
ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
EASY = os.path.dirname(os.path.abspath(__file__)) + "/../../task_search/EASY/"
P = json.load(open(EASY + "PLAN.json"))["SKILL7"]["prompt"]
ms = load_dataset("mteb/amazon_massive_scenario", name="en", split="train"); seen = set(); R = {"email": [], "takeaway": []}
for x in ms:
    k = norm(x["text"])
    if x["label_text"] in R and k not in seen and bucket(x["text"]) < 70:
        seen.add(k); R[x["label_text"]].append(dict(id=f"SKILL7|massive|m{x['id']}", src="massive", kind=x["label_text"].capitalize(), user=P.replace("{X}", " ".join(x["text"].split())[:900])))
for n, L in R.items():
    L = sorted(L, key=lambda r: H("l1p" + r["id"])); open(f"{a.out}/skill_{n}.jsonl", "w").write("".join(json.dumps(r) + "\n" for r in L)); print(n, len(L))
