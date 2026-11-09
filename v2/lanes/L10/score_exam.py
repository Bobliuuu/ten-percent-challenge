"""python3 score_exam.py results/exam_skill7_omni7.json -> results/EXAM_SCORE.json (uses v2/exam/loader.report). Exact label match of REAL greedy generations."""
import sys, json, collections
sys.path.insert(0, "../../exam"); import loader
loader.verify(); D = json.load(open(sys.argv[1])); ex = loader.load("SKILL7", "exam"); gold = {r["id"]: r["answer"].strip() for r in ex}
ok = lambda O: {i: (O.get(i, "").strip() == g) for i, g in gold.items()}
base, quant = ok(D["base"]), ok(D["quant"]); out = {}
out["quant_vs_base"] = loader.report(ex, base, quant)
out["quant_vs_base_clean_only"] = loader.report(loader.load("SKILL7", "exam", clean_only=True), base, quant)
out["after_feedback_vs_base"] = loader.report(ex, base, ok(D["real_exam"]))   # exam retention AFTER feedback (old behaviour kept)
nb = set(loader.load("SKILL7", "neighbours")); exn = [r for r in ex if r["id"] in nb]
out["neighbours_quant"] = loader.report(exn, base, quant)["overall"]; out["neighbours_after_feedback"] = loader.report(exn, base, ok(D["real_neigh"]) if False else {i: (D["real_neigh"].get(i, "").strip() == gold[i]) for i in gold})["overall"]
teach = loader.load("SKILL7", "teach"); N = loader.new_label("SKILL7")
tr = lambda O: dict(n=len(teach), hit=sum(O[r["id"]].strip() == N for r in teach), rate=sum(O[r["id"]].strip() == N for r in teach) / len(teach), wilson_lo=loader.wilson_lo(sum(O[r["id"]].strip() == N for r in teach), len(teach)))
out["teach"] = {k: tr(D[k + "_teach"]) for k in ("quant", "real", "shuffled", "empty")}
out["teach_by_src"] = {s: {k: sum(D[k + "_teach"][r["id"]].strip() == N for r in teach if r["src"] == s) / sum(1 for r in teach if r["src"] == s) for k in ("quant", "real", "shuffled")} for s in sorted({r["src"] for r in teach})}
out["ann"] = dict(n=len(D["ann_ids"]), quant_said=collections.Counter(D["ann_pred"]).most_common(5), real_bytes=D["real_bytes"], cfg=D["cfg"])
json.dump(out, open(sys.argv[2] if len(sys.argv) > 2 else "results/EXAM_SCORE.json", "w"), indent=1, default=str)
o = out["quant_vs_base"]; print("base acc", o["base_acc"], "overall", o["overall"], "\nkinds", {k: (round(v["ret"], 4), round(v["wilson_lo"], 4), v["n"]) for k, v in o["by_kind"].items()}, "\nsrc", {k: (round(v["ret"], 4), v["n"]) for k, v in o["by_src"].items()})
print("gates", o["gate_overall_98"], o["gate_kind_95"], o["rule3"], "never-used", o["never_used_retention"]); print("clean-only", out["quant_vs_base_clean_only"]["overall"], out["quant_vs_base_clean_only"]["by_kind"])
print("after fb", out["after_feedback_vs_base"]["overall"], {k: round(v["ret"], 4) for k, v in out["after_feedback_vs_base"]["by_kind"].items()}); print("neigh", out["neighbours_quant"], out["neighbours_after_feedback"]); print("teach", out["teach"], out["teach_by_src"], out["ann"])
