"""verify_loader <task>: run the SHIPPED answer.py (subprocess) on the first 200 SEL items from the packed artifact in /art/<task> and compare with the in-memory dev path outputs (devpath_outs_<task>_SEL.json)."""
import sys, os, json, subprocess, shutil, time
tn = sys.argv[1]; ART = f"/art/{tn}"
for f in ("task.json", "calib_prompts.jsonl"): shutil.copy(f"/root/l3/pkg_{tn}/{f}", f"{ART}/{f}")
task = json.load(open(f"{ART}/task.json")); PFX = task["prompt"].split("{X}")[0]
rows = [json.loads(l) for l in open(f"/root/data/{tn}_sel.jsonl")]; dp = json.load(open(f"/art/results/devpath_outs_{tn}_SEL.json")); assert dp["ids"] == [r["id"] for r in rows]
sub = list(range(0, len(rows), max(1, len(rows) // 200)))[:200]; texts = [rows[i]["user"][len(PFX):].replace("\n", " ") for i in sub]
open("/tmp/msgs.txt", "w").write("\n".join(texts) + "\n"); t = time.time()
p = subprocess.run([sys.executable, "/root/l3/answer.py", ART, "/tmp/msgs.txt"], capture_output=True, text=True); print(p.stderr[-1500:]); out = p.stdout.strip().split("\n")
assert len(out) == len(sub), (len(out), len(sub))
same = sum(o == dp["outs"][i] for o, i in zip(out, sub)); gold = sum(o == rows[i]["answer"] for o, i in zip(out, sub)); gdev = sum(dp["outs"][i] == rows[i]["answer"] for i in sub)
R = dict(task=tn, n=len(sub), identical=same, packed_gold=gold, devpath_gold=gdev, secs=round(time.time() - t), mismatches=[(rows[i]["id"], o, dp["outs"][i]) for o, i in zip(out, sub) if o != dp["outs"][i]][:20])
json.dump(R, open(f"/art/results/verify_loader_{tn}.json", "w")); print("RESULT_JSON", json.dumps(R))
