"""Run the shared loader.report on packed-artifact EXAM outputs. usage: python3 exam_report.py skill7 ART [ART...]"""
import sys, json
sys.path.insert(0, "../../exam"); import loader
pre = sys.argv[1]; T = pre.upper(); items = loader.load(T, "exam"); base = json.load(open(f"results/base_{pre}_EXAM.json"))
bok = {i["id"]: c for i, c in zip(items, base["cor"])}; out = {}
for a in sys.argv[2:]:
    q = json.load(open(f"results/outs_{a}_EXAM.json")); qok = {i["id"]: c for i, c in zip(items, q["cor"])}
    r = loader.report(items, bok, qok); r["clean_only"] = loader.report([i for i in items if i.get("tier", "clean") == "clean"], bok, qok)["overall"]; out[a] = r
    o = r["overall"]; print(a, "base_acc", round(r["base_acc"], 4), "ret", round(o["ret"], 4), "lo", round(o["wilson_lo"], 4), "n", o["n"], "gates", r["gate_overall_98"], r["gate_kind_95"], "rule3", r["rule3"],
          "kinds", {k: round(v["ret"], 3) for k, v in r["by_kind"].items()}, "never-used", round(r["never_used_retention"]["ret"], 4), "clean_only", round(r["clean_only"]["ret"], 4), round(r["clean_only"]["wilson_lo"], 4))
json.dump(out, open(f"results/exam_report_{pre}.json", "w"), indent=1)
