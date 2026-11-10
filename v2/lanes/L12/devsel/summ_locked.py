import json, glob, collections
runs = []; emp = []
for f in sorted(glob.glob("results/locked/locked*.json")): d = json.load(open(f)); runs += d["runs"]; emp.append(d["empty"])
print("empty", [(e["neigh_ret"], e["exam_ret"], e["exam_kind_min"]) for e in emp])
by = collections.defaultdict(list)
for r in runs: by[r["cond"]].append(r)
for c, rs in by.items():
    rs.sort(key=lambda r: r["draw"]); t = [r["teach_acc"] for r in rs]
    print(c, "n", len(rs), "teach", " ".join("%.3f" % x for x in t), "mean %.3f" % (sum(t) / len(t)), "| neigh %.3f-%.3f" % (min(r["neigh_ret"] for r in rs), max(r["neigh_ret"] for r in rs)),
          "| exam %.4f-%.4f lo %.4f-%.4f kmin %.3f-%.3f leak %s" % (min(r["exam_ret"] for r in rs), max(r["exam_ret"] for r in rs), min(r["exam_wilson_lo"] for r in rs), max(r["exam_wilson_lo"] for r in rs), min(r["exam_kind_min"] for r in rs), max(r["exam_kind_min"] for r in rs), [r["new_label_leak_on_exam"] for r in rs]), "bytes", rs[0]["carrier_bytes"])
