import sys, json, re, subprocess, collections, numpy as np
app = sys.argv[1]; txt = subprocess.run(["modal", "app", "logs", app], capture_output=True, text=True).stdout
runs = [json.loads(l.split("RESULT ", 1)[1]) for l in txt.splitlines() if "] RESULT {" in l]
json.dump(runs, open(sys.argv[2], "w"))
by = collections.defaultdict(lambda: collections.defaultdict(list))
for r in runs: by[r["tag"]][r["arm"]].append(r)
rows = []
for t, d in by.items():
    re_ = d.get("real", []); sh = d.get("shuffled", [])
    if not re_: continue
    rows.append((np.mean([x["teach_sel"] for x in re_]), t, np.mean([x["neigh_sel"] for x in re_]), np.mean([x["teach_sel"] for x in sh]) if sh else None, np.mean([x["frac"] for x in re_]) * 100, np.mean([x["bytes"] for x in re_]), len(re_)))
for r in sorted(rows, reverse=True)[:int(sys.argv[3]) if len(sys.argv) > 3 else 40]: print("teach %.3f %-60s neigh %.4f shuf %s del%% %.3f bytes %d n%d" % (r[0], r[1], r[2], None if r[3] is None else round(r[3], 3), r[4], r[5], r[6]))
print(len(rows), "configs")
