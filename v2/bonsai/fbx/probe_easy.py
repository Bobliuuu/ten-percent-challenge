"""Bonsai (shipped answer path, real greedy generation) on the EASY task dev items. python probe_easy.py ART items_dev.jsonl PLAN.json OUT.json [tasks...]"""
import sys, os, json, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/../ship")
import bonsai_loader as BL, common
from transformers import AutoTokenizer
art, ip, pp, out = sys.argv[1:5]; tasks = sys.argv[5:] or ["SKILL7", "BANKFAM", "SPAM2"]
R = [json.loads(l) for l in open(ip)]; R = [r for r in R if r["task"] in tasks]; m, _ = BL.load(art, patch=None); tok = AutoTokenizer.from_pretrained(art)
o = common.generate(m, tok, [r["user"] for r in R]); res = {}
for t in tasks:
    rs = [(r, x) for r, x in zip(R, o) if r["task"] == t]; cell = collections.defaultdict(list)
    for r, x in rs: cell[(r["src"], r["kind"])].append(int(x == r["kind"]))
    res[t] = dict(n=len(rs), acc=sum(int(x == r["kind"]) for r, x in rs) / len(rs), cells={f"{k[0]}|{k[1]}": [round(sum(v) / len(v), 3), len(v)] for k, v in sorted(cell.items())},
                  off_menu=collections.Counter(x for r, x in rs if x not in json.load(open(pp))[t]["labels"]).most_common(5))
    print(t, res[t]["n"], round(res[t]["acc"], 3), res[t]["cells"], res[t]["off_menu"])
json.dump(dict(res=res, preds={r["id"]: x for r, x in zip(R, o)}), open(out, "w"))
