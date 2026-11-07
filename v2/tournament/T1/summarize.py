"""(scorer v2 numbers; per-row correct_v1 kept in the result files, flips in rescore_reports/RESCORE_REPORT.json) Summarise results/*.json -> results/SUMMARY.md + SUMMARY.json. Dev-only (S1-dev, S3-dev); locked never touched."""
import json, glob, os, collections
HERE = os.path.dirname(os.path.abspath(__file__))
# bf16 params from HF safetensors metadata (queried 2026-11-06). gpt-oss-20b: 20.91B params, shipped MXFP4 (~13.8 GB on disk); fp16-equiv = 2*params.
PARAMS = {"Qwen3-30B-A3B-Instruct-2507": 30532122624, "Qwen2.5-32B-Instruct": 32763876352, "Qwen3-14B": 14768307200, "Qwen3-8B": 8190735360,
          "Llama-3.1-8B-Instruct": 8030261248, "gpt-oss-20b": 20914757184}
out = {}; lines = []
hdr = "| model | fp16 GB | 10% budget GB | S1 all | S1 nocall/single/multi | S3 all | S3 nocall/single/multi | parse-fail | mean | R8 (S1 & S3 in 60-92%) |"
lines += [hdr, "|" + "---|" * hdr.count("|")[:0 or 0] if False else "|" + "---|" * (hdr.count("|") - 1)]
for f in sorted(glob.glob(f"{HERE}/results/*.json")):
    n = os.path.basename(f)[:-5]
    if n.startswith(("SUMMARY", "RESCORE")): continue
    r = json.load(open(f)); rows = r["rows"]; c = collections.defaultdict(list)
    for x in rows:
        c[(x["source"], x["kind"])].append(x["correct"]); c[(x["source"], "all")].append(x["correct"])
    a = lambda s, k: sum(c[(s, k)]) / len(c[(s, k)])
    pf = sum(x["parse_fail"] for x in rows) / len(rows); fp16 = PARAMS[n] * 2
    o = dict(model=n, fp16_bytes=fp16, budget_10pct_bytes=fp16 // 10, parse_fail_rate=pf, n=len(rows), secs=r["secs"],
             acc={f"{s}/{k}": a(s, k) for (s, k) in sorted(c)}, acc_all_rows_v1=sum(x.get('correct_v1', x['correct']) for x in rows) / len(rows), acc_all_rows_v2=sum(x['correct'] for x in rows) / len(rows), finish_length_truncations=sum(x["finish"] == "length" for x in rows),
             R8_in_60_92=all(0.60 <= a(s, "all") <= 0.92 for s in ("S1", "S3")))
    o["mean_S1_S3"] = (a("S1", "all") + a("S3", "all")) / 2; out[n] = o
    lines.append(f"| {n} | {fp16/1e9:.1f} | {fp16/1e10:.2f} | {a('S1','all'):.3f} | {a('S1','nocall'):.2f}/{a('S1','single'):.2f}/{a('S1','multi'):.2f} | {a('S3','all'):.3f} | "
                 f"{a('S3','nocall'):.2f}/{a('S3','single'):.2f}/{a('S3','multi'):.2f} | {pf:.3f} | {o['mean_S1_S3']:.3f} | {'YES' if o['R8_in_60_92'] else 'no'} |")
open(f"{HERE}/results/SUMMARY.md", "w").write("\n".join(lines) + "\n"); json.dump(out, open(f"{HERE}/results/SUMMARY.json", "w"), indent=1)
print("\n".join(lines))
