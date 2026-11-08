"""Score the shared locked exam run: python final_report.py DS TASK NAME RESDIR  -> JSON report via v2/exam/loader.report (retention of base-correct, per kind/src, never-used source, clean_only)."""
import sys, json, re, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "exam")); import loader as L
ds, task, name, d = sys.argv[1:5]
norm = lambda t: re.sub(r"^\s*category\s*:\s*", "", (t or "").strip(), flags=re.I).strip()
items = L.load(task, "exam"); b = json.load(open(f"{d}/base_{ds}_exam_1.json")); o = json.load(open(f"{d}/outs_{ds}_{name}_exam.json")); assert b["ids"] == o["ids"]
ans = {i["id"]: i["answer"] for i in items}
base_ok = {i: norm(x) == ans[i] for i, x in zip(b["ids"], b["outs"])}; q_ok = {i: norm(x) == ans[i] for i, x in zip(o["ids"], o["outs"])}
rep = L.report(items, base_ok, q_ok); clean = [i for i in items if i.get("tier", "clean") == "clean"]; rep["clean_only"] = L.report(clean, base_ok, q_ok)
meta = json.load(open(f"{d}/{ds}_{name}.json")); rep["bytes"] = meta["bytes"]; rep["quant_acc"] = sum(q_ok.values()) / len(q_ok)
json.dump(rep, open(f"{d}/final_{ds}_{name}.json", "w"), indent=1)
def f(x): return f"{x['ret']:.4f} [lo {x['wilson_lo']:.4f}] n={x['n']}"
print(ds, name, "size %.2f%%" % (100 * meta["bytes"]["ratio"]), "base_acc %.4f quant_acc %.4f" % (rep["base_acc"], rep["quant_acc"]))
print(" overall", f(rep["overall"]), "gate98", rep["gate_overall_98"], "gate_kind95", rep["gate_kind_95"])
for k, v in rep["by_kind"].items(): print("  kind", k, f(v))
print(" never-used", rep["never_used_source_items"], f(rep["never_used_retention"])); print(" clean_only overall", f(rep["clean_only"]["overall"]), "min kind", min(v["ret"] for v in rep["clean_only"]["by_kind"].values()))
