"""Pull results from volume, build table.md. usage: python collect.py"""
import json, subprocess, glob, os
os.makedirs("results", exist_ok=True)
subprocess.run("modal volume get --force tenpct-v2-hf moe_res results/ && rm -rf results/vol && mv results/moe_res results/vol", shell=True)
subprocess.run("modal volume get --force tenpct-v2-hf moe_data/base_noise.json results/base_noise.json", shell=True)
rows = []
for f in sorted(glob.glob("results/vol/*.json")):
    r = json.load(open(f)); rows.append(r) if isinstance(r, dict) else None
def g(r, k, d="-"):
    v = r.get(k); return d if v is None else v
L = ["| name | k | exp bpw tgt | calib | bytes | %fp16 | em8 A | em16 A | em64 A | top1 A | KL A | em16 B(shift) | top1 B | KL B | retention agnews/sst2/json (A) |", "|" + "---|" * 15]
for r in rows:
    c = r["cfg"]; ret = r.get("retention_A", {})
    rt = "/".join(f"{ret[x]['retention']:.2f}" if x in ret and ret[x]["retention"] is not None else "-" for x in ("agnews", "sst2", "json"))
    f = lambda k, p=3: ("%.*f" % (p, r[k])) if r.get(k) is not None else "-"
    L.append(f"| {r['name']} | {c.get('k',128)} | {c.get('exp_bits')} | {c.get('calib','A')} | {r['bytes']/1e9:.3f}G | {100*r['ratio']:.2f}% | {f('em8_A')} | {f('em16_A')} | {f('em64_A')} | {f('top1_A')} | {f('kl_A',4)} | {f('em16_B')} | {f('top1_B')} | {f('kl_B',4)} | {rt} |")
open("results/table.md", "w").write("\n".join(L)); print("\n".join(L))
