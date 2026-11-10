"""python3 summ_b6.py file.json [...]: per grid x cond summary of fbdev/fblock result JSONs."""
import sys, json, collections
g = collections.defaultdict(list); emp = {}
for f in sys.argv[1:]:
    r = json.load(open(f)); emp[f] = r["empty"]
    for x in r["runs"]: g[(json.dumps(x["grid"], sort_keys=True), x["cond"])].append(x)
print("empty:", {f.split('/')[-2]: (round(e["teach_acc"], 3), round(e["neigh_ret"], 4)) for f, e in emp.items()})
for (grid, cond), v in sorted(g.items()):
    t = [x["teach_acc"] for x in v]; n = [x["neigh_ret"] for x in v]; lk = [x.get("neigh_leak", x.get("new_label_leak_on_exam")) for x in v]
    ex = [x["exam_ret"] for x in v if "exam_ret" in x]; km = [x["exam_kind_min"] for x in v if "exam_kind_min" in x]
    print(f"{grid} {cond:8s} n={len(v)} teach mean {sum(t)/len(t):.3f} [{min(t):.2f}-{max(t):.2f}] neigh mean {sum(n)/len(n):.4f} [{min(n):.4f}-{max(n):.4f}] leak {lk}" + (f" exam {min(ex):.4f}-{max(ex):.4f} kindmin {min(km):.3f}" if ex else ""))
