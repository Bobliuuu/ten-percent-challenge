"""retention analysis with SEL/REP split. usage: python analyze_repair.py CAND_ct.json [CAND2 ...]
Base = bf16 32B ct rows (INJ rows from the _inj file, NEWS rows from the main file). Split: md5(id)%2==0 -> SEL, else REP.
Prints per task: overall ret (+Wilson lo), worst cell, #cells <95%, for ALL / SEL / REP."""
import json, sys, math, hashlib, collections, os
V = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vq", "results")
def wil(k, n, z=1.96):
    if n == 0: return float("nan")
    p = k / n; d = 1 + z * z / n; return (p + z * z / (2 * n) - z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / d
B = {}
for r in json.load(open(V + "/ct_Qwen2.5-32B-Instruct_b32_bf16.json"))["rows"]:
    if r["task"] == "NEWS": B[r["id"]] = r
for r in json.load(open(V + "/ct_Qwen2.5-32B-Instruct_b32_bf16_inj.json"))["rows"]: B[r["id"]] = r
split = lambda i: "SEL" if int(hashlib.md5(i.encode()).hexdigest(), 16) % 2 == 0 else "REP"
def analyse(path, quiet=False):
    C = {r["id"]: r for r in json.load(open(path))["rows"]}
    out = {}
    for task in ("INJ", "NEWS"):
        for part in ("ALL", "SEL", "REP"):
            cells = collections.defaultdict(lambda: [0, 0])
            for i, b in B.items():
                if b["task"] != task or not b["correct"] or i not in C: continue
                if part != "ALL" and split(i) != part: continue
                for key in ("ALL", b["src"] + "_" + b["kind"]):
                    cells[key][1] += 1; cells[key][0] += int(C[i]["correct"])
            a, n = cells.pop("ALL"); worst = min(cells.items(), key=lambda kv: kv[1][0] / kv[1][1])
            nb = sum(1 for k, (x, m) in cells.items() if x / m < 0.95)
            out[(task, part)] = dict(ret=a / n, lo=wil(a, n), n=n, worst=worst[0], worst_ret=worst[1][0] / worst[1][1], worst_n=worst[1][1], below95=nb, ncell=len(cells), cells={k: v[0] / v[1] for k, v in cells.items()})
    return out
if __name__ == "__main__":
    for p in sys.argv[1:]:
        o = analyse(p); print("==", os.path.basename(p))
        for (task, part), d in o.items():
            print(f"  {task:4s} {part:3s} ret {d['ret']:.4f} lo {d['lo']:.4f} n={d['n']} | worst {d['worst']} {d['worst_ret']:.3f} (n={d['worst_n']}) | cells<95%: {d['below95']}/{d['ncell']}")
