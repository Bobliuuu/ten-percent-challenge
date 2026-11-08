"""score_exam.py BASE.json CAND.json OUT.json : shared locked SKILL7 exam via v2/exam/loader.report (run ONCE per model)."""
import sys, json, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
from v2.exam.loader import load, report
w = lambda o: (o.split() or [""])[0].strip(".,:;!\"'").lower()
B, C = json.load(open(sys.argv[1])), json.load(open(sys.argv[2]))
for nm, clean in (("all", False), ("clean_only", True)):
    ex = load("SKILL7", "exam", clean_only=clean)
    ok = lambda R: {i["id"]: w(R[i["id"]]) == i["answer"].lower() for i in ex}
    r = report(ex, ok(B), ok(C)); r["diff_outputs"] = sum(B[i["id"]] != C[i["id"]] for i in ex)
    json.dump(r, open(sys.argv[3].replace(".json", f"_{nm}.json"), "w"), indent=1)
    print(nm, "n", r["n_items"], "base_acc %.4f" % r["base_acc"], "overall", {k: round(v, 4) if isinstance(v, float) else v for k, v in r["overall"].items()})
    print("  by_kind", {k: round(v["ret"], 3) for k, v in r["by_kind"].items()}); print("  by_src", {k: (round(v["ret"], 3), v["n"]) for k, v in r["by_src"].items()})
    print("  never_used", r["never_used_retention"], "gates", r["gate_overall_98"], r["gate_kind_95"], r["rule3"], "outputs differing", r["diff_outputs"])
