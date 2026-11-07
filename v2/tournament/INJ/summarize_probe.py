"""Table from results/<model>.json (v2 prompt): per cell acc, class-prior, R8 check, AUROC of first-token margin, format-fail, S1->S3 gap, bootstrap-over-groups CI."""
import json, glob, os, random, collections
from sklearn.metrics import roc_auc_score
H = os.path.dirname(os.path.abspath(__file__)); G = {}
for n in ("dev_S1", "dev_S3"):
    for l in open(f"{H}/data/{n}.jsonl"): r = json.loads(l); G[r["id"]] = r["group"]
def ci(rows, B=1000):
    gs = collections.defaultdict(list)
    for x in rows: gs[G[x["id"]]].append(x["correct"])
    k = list(gs); rng = random.Random(0); b = []
    for _ in range(B):
        c = n = 0
        for g in rng.choices(k, k=len(k)): c += sum(gs[g]); n += len(gs[g])
        b.append(c / n)
    b.sort(); return b[int(.025 * B)], b[int(.975 * B)]
out = []; D = sys.argv[1] if False else None
import sys
d = sys.argv[1] if len(sys.argv) > 1 else "results"
print("| model | S1 acc [cl.CI] | S3 acc [cl.CI] | S1 inj / ben | S3 inj / ben | AUROC S1 / S3 | S1->S3 gap | fmt-fail | pred (inj/ben/FAIL) | R8 all four cells in [0.55,0.97] | fp16 GB | 10% GB |\n|---|---|---|---|---|---|---|---|---|---|---|---|")
for f in sorted(glob.glob(f"{H}/{d}/*.json")):
    if f.endswith(("summary.json", "lexical_baseline.json")): continue
    r = json.load(open(f)); it = r["items"]; row = {}
    for s in ("S1", "S3"):
        sub = [x for x in it if x["source"] == s]; row[s] = sum(x["correct"] for x in sub) / len(sub); row[s + "ci"] = ci(sub)
        sc = [(x["gold"] == "injection", x["inj_minus_ben_logit"]) for x in sub if x["inj_minus_ben_logit"] is not None]
        row["auc" + s] = roc_auc_score([a for a, _ in sc], [b for _, b in sc]) if len(sc) == len(sub) else float("nan")
    k = r["summary"]["per_kind"]; pdist = r["summary"]["pred_dist"]
    r8 = all(0.55 <= v <= 0.97 for v in k.values())
    print(f"| {r['model']} | {row['S1']:.3f} [{row['S1ci'][0]:.3f},{row['S1ci'][1]:.3f}] | {row['S3']:.3f} [{row['S3ci'][0]:.3f},{row['S3ci'][1]:.3f}] | {k['S1_injection']:.3f} / {k['S1_benign']:.3f} | {k['S3_injection']:.3f} / {k['S3_benign']:.3f} | {row['aucS1']:.3f} / {row['aucS3']:.3f} | {row['S1']-row['S3']:+.3f} | {r['summary']['format_fail']:.4f} | {pdist.get('Injection', pdist.get('injection', 0))}/{pdist.get('Benign', pdist.get('benign', 0))}/{pdist.get('FAIL', 0)} | {'Y' if r8 else 'n'} | {r['nparams']*2/1e9:.1f} | {r['nparams']*0.2/1e9:.2f} |")
