"""score_cli.py : teach/neighbour numbers of the shipped CLIs (feedback.py -> feedback.pt -> answer.py), vs the start model (ternary base) and, informationally, the ancestor."""
import sys, json, os, collections
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
from v2.exam.loader import load, wilson_lo
w = lambda o: (o.split() or [""])[0].strip(".,:;!\"'").lower()
R = os.path.dirname(os.path.abspath(__file__)) + "/results/"
teach = load("SKILL7", "teach"); nb = set(load("SKILL7", "neighbours")); ex = {i["id"]: i for i in load("SKILL7", "exam")}
base = json.load(open(R + "ex_base.json")); par = json.load(open(R + "ex_parent.json"))
out = {}
for arm in ("empty", "real", "wrong"):
    C = json.load(open(R + f"cli2/cli_{arm}.json")); r = dict(teach=sum(w(C[t["id"]]) == "recipe" for t in teach) / len(teach), n_teach=len(teach))
    for nm, B in (("vs_start_model", base), ("vs_ancestor_info", par)):
        ids = [i for i in nb if w(B[i]) == ex[i]["answer"].lower()]; k = sum(w(C[i]) == ex[i]["answer"].lower() for i in ids)
        cells = collections.defaultdict(lambda: [0, 0])
        for i in ids: cells[ex[i]["kind"]][1] += 1; cells[ex[i]["kind"]][0] += w(C[i]) == ex[i]["answer"].lower()
        r[nm] = dict(n=len(ids), kept=k, ret=k / len(ids), wilson_lo=wilson_lo(k, len(ids)), worst_kind=min((v[0] / v[1], c) for c, v in cells.items()))
    out[arm] = r; print(arm, json.dumps(r))
json.dump(out, open(R + "cli2_summary.json", "w"), indent=1)
