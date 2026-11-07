"""Item retention for INJ (rule 2): P(candidate correct | base correct), overall and per kind (source x class) and per source, with Wilson CI and GROUP-cluster bootstrap CI.
  python retention.py --base results/Qwen__Qwen3-14B.json --cand results/cand.json [--cand_format auto|probe|dense_t1] [--noise results/base_run2.json]
Accepts probe_base.py / eval_inj.py json ({items:[{id,correct,kind,source}]}) or dense_stream/dense.py t1 json ({t1:[{id,correct,kind,source}]}).
Optional --noise: a second independent run of the BASE; its retention vs base is the noise floor to report beside the candidate (red-team finding 12). DEV ONLY by default."""
import json, argparse, os, math, collections, random
H = os.path.dirname(os.path.abspath(__file__))
def rows(p):
    d = json.load(open(p)); r = d.get("items") or d.get("t1"); return {x["id"]: x for x in r}
def wilson(k, n, z=1.96):
    if n == 0: return (float("nan"),) * 2
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); w = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)); return ((c - w) / d, (c + w) / d)
def groups():
    g = {}
    for f in os.listdir(H + "/data"):
        if f.startswith(("dev_", "locked_")) and False: pass
    for n in ("dev_S1", "dev_S3"):
        for l in open(f"{H}/data/{n}.jsonl"): r = json.loads(l); g[r["id"]] = r["group"]
    return g
def retention(base, cand, G=None, B=2000, seed=0):
    ids = [i for i in base if base[i]["correct"] in (True, "True") and i in cand]
    def stat(sel):
        if not sel: return None
        k = sum(cand[i]["correct"] in (True, "True") for i in sel); return k, len(sel)
    cells = collections.defaultdict(list)
    for i in ids: cells["ALL"].append(i); cells[base[i]["kind"]].append(i); cells["src:" + base[i]["source"]].append(i)
    out = {}
    rng = random.Random(seed)
    for c, sel in sorted(cells.items()):
        k, n = stat(sel); lo, hi = wilson(k, n); d = dict(ret=k / n, k=k, n=n, wilson=[round(lo, 4), round(hi, 4)])
        if G:
            gs = collections.defaultdict(list)
            for i in sel: gs[G.get(i, i)].append(i)
            keys = list(gs); bs = []
            for _ in range(B):
                kk = nn = 0
                for gk in rng.choices(keys, k=len(keys)):
                    for i in gs[gk]: nn += 1; kk += cand[i]["correct"] in (True, "True")
                bs.append(kk / nn)
            bs.sort(); d["cluster_ci"] = [round(bs[int(.025 * B)], 4), round(bs[int(.975 * B)], 4)]; d["n_groups"] = len(keys)
        out[c] = d
    return out
if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--base", required=True); ap.add_argument("--cand", required=True); ap.add_argument("--noise"); ap.add_argument("--out")
    a = ap.parse_args(); b = rows(a.base); c = rows(a.cand); G = groups()
    res = {"candidate": retention(b, c, G)}
    if a.noise: res["base_vs_base_noise_floor"] = retention(b, rows(a.noise), G)
    print(json.dumps(res, indent=1))
    if a.out: json.dump(res, open(a.out, "w"), indent=1)
