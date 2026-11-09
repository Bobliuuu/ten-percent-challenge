"""score the saved locked-exam outputs with the shared loader (no model, CPU). usage: python3 exam_score.py  -> results/exam_report_b6_vq_k160.json"""
import sys, os, json
H = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(H, "..", "..", "..", ".."))
sys.path.insert(0, f"{ROOT}/v2/exam"); sys.path.insert(0, f"{ROOT}/v2/lanes/L9b")
from loader import load, report, verify
import core
verify(); rows = load("BANK6", "exam"); b = json.load(open(f"{H}/results/exam_base3_BANK6.json")); p = json.load(open(f"{H}/results/exam_packed_b6_vq_k160_BANK6.json"))
assert b["ids"] == p["ids"] == [r["id"] for r in rows]
ok = lambda outs: {r["id"]: c for r, c in zip(rows, core.correct(outs, rows))}
q = ok(p["outs"]); out = dict(n=len(rows), base_runs_identical=all(x == b["runs"][0] for x in b["runs"]), base_acc=[sum(ok(r).values()) / len(rows) for r in b["runs"]], packed_acc=sum(q.values()) / len(rows))
out["report_vs_base_run0"] = report(rows, ok(b["runs"][0]), q)
out["report_clean_only_vs_base_run0"] = report([r for r in rows if r.get("tier", "clean") == "clean"], ok(b["runs"][0]), q)
if not out["base_runs_identical"]: out["report_vs_base_run1"] = report(rows, ok(b["runs"][1]), q); out["report_vs_base_run2"] = report(rows, ok(b["runs"][2]), q)
json.dump(out, open(f"{H}/results/exam_report_b6_vq_k160.json", "w"), indent=1)
R = out["report_vs_base_run0"]; f = lambda v: "%d/%d = %.2f%% [lo %.2f]" % (v["kept"], v["n"], 100 * v["ret"], 100 * v["wilson_lo"])
print("base identical:", out["base_runs_identical"], "base acc", out["base_acc"], "packed acc", out["packed_acc"]); print("overall", f(R["overall"]), "gates 98/95:", R["gate_overall_98"], R["gate_kind_95"])
for k, v in R["by_kind"].items(): print(" kind", k, f(v))
for k, v in R["by_src"].items(): print(" src", k, f(v))
print(" never-used", R["never_used_source_items"], f(R["never_used_retention"])); print(" rule3", R["rule3"])
c = out["report_clean_only_vs_base_run0"]; print("clean only overall", f(c["overall"]), c["gate_overall_98"], c["gate_kind_95"])
