"""TASK_SEARCH2 analysis (REAL greedy generation only). python analyze3.py <model e.g. Qwen2.5-32B-Instruct>  reads results/raw3/ct_<m>_t_*.json
Scoring: whole output must equal gold. JSON-output tasks (PIIDET, NER, TOOLARG): parse both as JSON and compare (whitespace-insensitive, content/order strict);
all other tasks: exact string after strip. SQL also reported whitespace/semicolon-insensitive is NOT applied (strict)."""
import json, sys, glob, os, collections, numpy as np
from analyze2 import wil
JSONT = {"PIIDET", "NER", "TOOLARG"}
def sc(r):
    if r["task"] in JSONT:
        try: return json.loads(r["out"]) == json.loads(r["gold"])
        except Exception: return False
    return r["out"].strip() == r["gold"].strip()
def load(m):
    R = {}
    for f in glob.glob(f"results/raw3/ct_{m}_t_*.json"): R[os.path.basename(f)[len(f"ct_{m}_"):-5]] = json.load(open(f))
    return R
if __name__ == "__main__":
    m = sys.argv[1]; R = load(m); base = {r["id"]: r for r in R["t_bf16"]["rows"]}
    cf = sorted([c for c in R if c != "t_bf16"], key=lambda c: R[c]["ratio"]); tasks = sorted(set(r["task"] for r in base.values()))
    print(f"### {m}: " + ", ".join(f"{c}={R[c]['ratio']*100:.2f}% of fp16 / {R[c]['body_bpw']:.2f} bpw" for c in cf))
    print("| task | n | base acc | base acc by src | " + " | ".join(f"ret {c} [Wilson lo] / worst src x kind (n)" for c in cf) + " |")
    print("|---|---|---|---|" + "---|" * len(cf))
    for t in tasks + ["ALL"]:
        ids = [i for i, r in base.items() if t == "ALL" or r["task"] == t]; bc = [i for i in ids if sc(base[i])]
        srcs = sorted(set(base[i]["src"] for i in ids)); bs = "/".join(f"{np.mean([sc(base[i]) for i in ids if base[i]['src']==s]):.2f}" for s in srcs)
        line = f"| {t} | {len(ids)} | {len(bc)/len(ids):.3f} | {bs} | "
        for c in cf:
            cr = {r["id"]: r for r in R[c]["rows"]}; k = sum(sc(cr[i]) for i in bc); lo = wil(k, len(bc))[0]; cells = collections.defaultdict(lambda: [0, 0])
            for i in bc:
                key = (base[i]["task"], base[i]["src"], base[i]["kind"]) if t == "ALL" else (base[i]["src"], base[i]["kind"]); cells[key][1] += 1; cells[key][0] += sc(cr[i])
            w = min(((v[0] / v[1], k_, v[1]) for k_, v in cells.items() if v[1] >= 8), default=None)
            line += f"{k/len(bc):.3f} [{lo:.3f}] (n={len(bc)}) / " + (f"{w[0]:.2f} {'/'.join(w[1][-2:])} ({w[2]})" if w else "-") + " | "
        print(line)
