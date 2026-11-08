"""Local scoring of the ONE exam run: python3 -I final_report.py <skill7x|bank6x> <dir with base_*_EXAM_1.json, outs_*_EXAM.json, *_fb.json>  -> JSON via v2/exam/loader.report (also clean_only)."""
import sys, json, os
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")); sys.path.insert(0, ROOT + "/v2/exam"); import loader
ds, d = sys.argv[1], sys.argv[2]; task = {"skill7x": "SKILL7", "bank6x": "BANK6"}[ds]
items = loader.load(task, "exam"); by = {i["id"]: i for i in items}
base = json.load(open(f"{d}/base_{ds}_EXAM_1.json")); pre = json.load(open(f"{d}/outs_{ds}_L3_final_{ds}_EXAM.json"))
bok = {i: o == by[i]["answer"] for i, o in zip(base["ids"], base["outs"])}; qok = {i: o == by[i]["answer"] for i, o in zip(pre["ids"], pre["outs"])}
out = dict(task=task, before_feedback=loader.report(items, bok, qok), before_feedback_clean_only=loader.report([i for i in items if i.get("tier", "clean") == "clean"], bok, qok))
fb = json.load(open(f"{d}/fb_{ds}.json")) if os.path.exists(f"{d}/fb_{ds}.json") else None
if fb:
    real = fb["draws"][0]["real"]
    if "neigh_outs" in real:
        aok = {i: o == by[i]["answer"] for i, o in real["neigh_outs"].items()}
        for i in bok: aok.setdefault(i, False)
        out["after_feedback"] = loader.report(items, bok, aok)
    out["fb"] = {k: fb["draws"][0][k] for k in fb["draws"][0] if k in ("real", "shuffled", "wrong")}; out["fb_empty"] = fb["empty"]
    for k in ("real", "shuffled", "wrong"):
        out["fb"][k] = {a: b for a, b in out["fb"].get(k, {}).items() if a not in ("neigh_outs", "teach_outs", "cells")}
print(json.dumps(out, indent=1))
