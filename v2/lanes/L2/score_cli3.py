"""score_cli3.py TAG: score the m16 CLI end-to-end arms (results/cli3/<tag>/cli_{empty,real,shuf,wrong}.json + rows.jsonl) -> results/cli3/<tag>/summary.json.
TAG recipe: teach = locked Recipe teach set (exam), neighbours = locked 357 base-correct neighbours scored vs the start model's own outputs (results/ex_base.json) and vs gold.
TAG reminder: teach = CLINC reminder teach items (not in exam), same neighbours."""
import sys, json, os, collections
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
from v2.exam.loader import load, wilson_lo
w = lambda o: (o.split() or [""])[0].strip(".,:;!\"'").lower()
tag = sys.argv[1]; D = os.path.dirname(os.path.abspath(__file__)) + f"/results/cli3/{tag}/"; R = os.path.dirname(os.path.abspath(__file__)) + "/results/"
new = {"recipe": "recipe", "reminder": "reminder"}[tag]
rows = [json.loads(l) for l in open(D + "rows.jsonl")]; ids = [r["id"] for r in rows]
nb = set(load("SKILL7", "neighbours")); ex = {i["id"]: i for i in load("SKILL7", "exam")}; base = json.load(open(R + "ex_base.json"))
teach_ids = [i for i in ids if i not in nb and i not in ex]; teach_ids = [i for i in ids if i not in ex or i not in nb] if tag == "recipe" else teach_ids
teach_ids = [i for i in ids if i not in nb]
out = {}
for arm in ("empty", "real", "shuf", "wrong"):
    C = json.load(open(D + f"cli_{arm}.json")); r = dict(teach=sum(w(C[i]) == new for i in teach_ids) / len(teach_ids), n_teach=len(teach_ids))
    keep = [i for i in nb if w(base[i]) == ex[i]["answer"].lower()]; k = sum(w(C[i]) == ex[i]["answer"].lower() for i in keep)
    cells = collections.defaultdict(lambda: [0, 0])
    for i in keep: cells[ex[i]["kind"]][1] += 1; cells[ex[i]["kind"]][0] += w(C[i]) == ex[i]["answer"].lower()
    r["neighbours"] = dict(n=len(keep), kept=k, ret=k / len(keep), wilson_lo=wilson_lo(k, len(keep)), worst_kind=min((v[0] / v[1], c) for c, v in cells.items()), new_word_on_neigh=sum(w(C[i]) == new for i in nb) / len(nb))
    out[arm] = r; print(arm, json.dumps(r))
json.dump(out, open(D + "summary.json", "w"), indent=1)
