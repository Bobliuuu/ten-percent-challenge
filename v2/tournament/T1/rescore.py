"""Re-score stored dev-probe outputs with scorer v2 (tournament/T1/scorer.py). Dev only: probe_items.jsonl (legacy 600-item dev probe, built from the PRE-relock s1_dev/s3_dev).
Targets: tournament/T1/results/<model>.json (rows[].output full text) and compress/dense/results/t1/t1_*.json (t1[].output clipped to 600 chars, padded with EOS tokens).
In place: rows get correct_v1/parse_fail_v1 (original verdict, written once) and correct/parse_fail (v2). Clipped outputs whose content reaches the 600-char clip
without an EOS token cannot be re-scored reliably: they keep the v1 verdict and are counted as 'unrescorable'. Writes tournament/T1/rescore_reports/RESCORE_REPORT.json.
Usage: python rescore.py"""
import json, glob, os, re, collections
import scorer, scorer_v1
HERE = os.path.dirname(os.path.abspath(__file__)); V2 = os.path.abspath(f"{HERE}/../..")
items = {}
for l in open(f"{HERE}/probe_items.jsonl"):
    r = json.loads(l); items[r["id"]] = r
EOS = re.compile(r"<\|im_end\|>|<\|endoftext\|>|<\|eot_id\|>|<\|end_of_text\|>|<\|return\|>|<\|end\|>|<\|call\|>")
def process(path, key, clipped):
    d = json.load(open(path)); rows = d[key]; flips = []; unres = 0; empties = 0; counts = collections.Counter()
    for r in rows:
        it = items[r["id"]]
        r.setdefault("correct_v1", bool(r["correct"]) if not isinstance(r["correct"], str) else r["correct"] == "True")
        r.setdefault("parse_fail_v1", bool(r["parse_fail"]) if not isinstance(r["parse_fail"], str) else r["parse_fail"] == "True")
        txt = r["output"]
        if clipped:
            m = EOS.search(txt)
            if not m and len(txt) >= 600: unres += 1; r["rescored"] = "unrescorable_clipped"; r["correct"] = r["correct_v1"]; r["parse_fail"] = r["parse_fail_v1"]; continue
            if m: txt = txt[:m.end()]
        s = scorer.score_item(txt, it); r["rescored"] = "v2"; r["correct"] = bool(s["correct"]); r["parse_fail"] = bool(s["parse_fail"]); empties += bool(s["empty"])
        if r["correct"] != r["correct_v1"]:
            flips.append(dict(id=r["id"], source=r["source"], kind=r["kind"], v1=r["correct_v1"], v2=r["correct"], empty=bool(s["empty"]))); counts[f"{r['source']}/{r['kind']}:{'lost' if r['correct_v1'] else 'gained'}"] += 1
    d["rescored_with"] = "tournament/T1/scorer.py (v2)"; json.dump(d, open(path, "w"), default=str)
    return dict(file=os.path.relpath(path, V2), n=len(rows), flips=len(flips), lost=sum(f["v1"] for f in flips), gained=sum(not f["v1"] for f in flips), unrescorable_clipped=unres, empty_outputs=empties,
                acc_v1=sum(r["correct_v1"] for r in rows) / len(rows), acc_v2=sum(r["correct"] for r in rows) / len(rows), by_cell=dict(counts), flip_ids=flips)
rep = []
for f in sorted(glob.glob(f"{HERE}/results/*.json")):
    if os.path.basename(f).startswith(("SUMMARY", "RESCORE")): continue
    rep.append(process(f, "rows", False))
for f in sorted(glob.glob(f"{V2}/compress/dense/results/t1/t1_*.json")): rep.append(process(f, "t1", True))
json.dump(rep, open(f"{HERE}/rescore_reports/RESCORE_REPORT.json", "w"), indent=1)
for r in rep: print(f"{r['file']:70s} n={r['n']} flips={r['flips']} (lost {r['lost']}, gained {r['gained']}) unrescorable={r['unrescorable_clipped']} empty={r['empty_outputs']} acc {r['acc_v1']:.3f}->{r['acc_v2']:.3f}")
print("TOTAL flips", sum(r["flips"] for r in rep), "of", sum(r["n"] for r in rep), "rows")
