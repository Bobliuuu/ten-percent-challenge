import json
from gen_cfgs import *
out = [dict(name="base", ds="skill7", sets=["SEL"], ncal=2000)]
for b in (1.5, 1.0):
    k = keep_for(b + 0.01)
    out += [cfg(f"ecsq_had_b{b}", "ecsq_had", b, g=256), cfg(f"ecsq_rdo_had_b{b}", "ecsq_had", b, g=256, lam=0.05), cfg(f"vq4_had_b{b}", "vq_had", b, D=4), cfg(f"vq8_had_b{b}", "vq_had", b, D=8),
            cfg(f"rvq8_had_b{b}", "rvq_had", b, D=8)]
out += [cfg("tcq_had_b1.5", "tcq_had", 1.5), cfg("e8_had_b1.375", "vq_had", 1.375, D=8, cb="e8"), cfg("vq4_had_b2_nokeep", "vq_had", 2, D=4, keep=87)]
# group size / scale coding at 2 bit ECSQ
for g in (64, 128, 512): out.append(cfg(f"ecsq_had_b2_g{g}", "ecsq_had", 2, g=g))
out.append(cfg("ecsq_had_b2_s8", "ecsq_had", 2, g=256, sbits=8))
# keep-vs-bits at ECSQ: more experts at lower bits and fewer at higher
for b in (2.5, 2.25, 1.75): out.append(cfg(f"ecsq_had_b{b}", "ecsq_had", b, g=256))
out.append(cfg("ecsq_had_b3_k80", "ecsq_had", 3, g=256, keep=80))
# mixed by routing mass (ecsq): top-30% experts 3 bit, rest 1.5
out.append(dict(cfg("mix_ecsq_h30_3_1.5", "ecsq_had", 2, g=256, keep=keep_for(2.0)), mix=dict(hi_frac=0.3, hi=3, lo=1.5)))
out += [cfg("gptq_clip_had_b2_k81", "gptq_clip_had", 2), cfg("vq4_had_b2.5", "vq_had", 2.5, D=4), cfg("vq4_had_b1.75", "vq_had", 1.75, D=4), cfg("vq4_had_b2.25", "vq_had", 2.25, D=4)]
json.dump(out, open("cfgs_sweep2.json", "w"), indent=0)
names = [c["name"] for c in out][1:]; sh = {str(i): [] for i in range(3)}
for i, n in enumerate(names): sh[str(i % 3)].append(n)
json.dump(sh, open("shards2.json", "w")); print(len(out), {k: len(v) for k, v in sh.items()}); [print(c["name"], c.get("keep"), c.get("nom_bpw")) for c in out]
