"""Cluster-bootstrap analysis of fb_inj.py output. Two-level bootstrap: resample annotation draws (clusters of 25 annotations) AND item GROUPS within the teach / neighbour sets.
usage: python3 analyze_fb.py results/fb_v2/Qwen__Qwen3-8B_generic.json [...]   -> prints a markdown table and writes <file>.analysis.json
Gates (REDESIGN G2 + red-team): teach gain >= +0.15 with CI lo > 0; neighbour retention >= 0.98 (lower CI reported; n=400 cannot certify 0.98 unless <=1 break: reported honestly);
real - control >= +0.10 with CI lo > 0 for control in {mixed, allflip, shuffle}; empty byte-identical (asserted inside the job)."""
import json, sys, random, collections, numpy as np
def boot(d, key, groups_key, fn, B=1000, seed=0):
    rng = random.Random(seed); draws = d["draws"]; G = d["groups"][groups_key]; gi = collections.defaultdict(list)
    for i, g in enumerate(G): gi[g].append(i)
    keys = list(gi); out = []
    for _ in range(B):
        ds = [draws[rng.randrange(len(draws))] for _ in draws]; sel = [i for g in rng.choices(keys, k=len(keys)) for i in gi[g]]
        out.append(np.mean([fn(x, sel) for x in ds]))
    out.sort(); return float(np.mean([fn(x, list(range(len(G)))) for x in draws])), float(out[int(.025 * B)]), float(out[int(.975 * B)])
def run(path):
    d = json.load(open(path)); res = {}
    base = {k: np.array(v) for k, v in d["base_eval"].items()}
    for setname, gk in (("teach", "teach"), ("teach_ood", "teach_ood"), ("neigh", "neigh")):
        if not d["groups"][gk]: continue
        acc = lambda c: (lambda x, s: float(np.array(x["res"][c][setname])[s].mean()))
        for c in ("real", "empty", "mixed", "allflip", "shuffle"):
            res[f"{setname}/{c}"] = boot(d, setname, gk, acc(c))
        for c in ("empty", "mixed", "allflip", "shuffle"):
            res[f"{setname}/real-{c}"] = boot(d, setname, gk, lambda x, s, c=c: float((np.array(x["res"]["real"][setname])[s] - np.array(x["res"][c][setname])[s]).mean()))
    return d, res
for p in sys.argv[1:]:
    d, r = run(p); f = lambda t: f"{t[0]:+.3f} [{t[1]:+.3f},{t[2]:+.3f}]"
    print(f"\n### {d['model']} stat={d['stat']} tied_lm_head={d['tied']} base acc (calib+pool S1 / S3-dev) = {d['base_acc']['pool']:.3f} / {d['base_acc']['s3']:.3f}; n_teach={len(d['groups']['teach'])} n_teach_ood={len(d['groups']['teach_ood'])} n_neigh={len(d['groups']['neigh'])} draws={len(d['draws'])}")
    print("| set | real | empty | mixed | allflip | shuffle | real-empty | real-mixed | real-allflip | real-shuffle |\n|---|---|---|---|---|---|---|---|---|---|")
    for s in ("teach", "teach_ood", "neigh"):
        if f"{s}/real" in r: print(f"| {s} | " + " | ".join(f"{r[f'{s}/{c}'][0]:.3f} [{r[f'{s}/{c}'][1]:.3f},{r[f'{s}/{c}'][2]:.3f}]" for c in ("real", "empty", "mixed", "allflip", "shuffle")) + " | " + " | ".join(f(r[f'{s}/real-{c}']) for c in ("empty", "mixed", "allflip", "shuffle")) + " |")
    g_teach = r["teach/real-empty"][0] >= .15 and r["teach/real-empty"][1] > 0
    g_neigh = r["neigh/real"][0] >= .98
    g_ctrl = all(r[f"teach/real-{c}"][0] >= .10 and r[f"teach/real-{c}"][1] > 0 for c in ("mixed", "allflip", "shuffle"))
    print(f"gates: teach_gain>=.15&lo>0: {g_teach} | neighbours>=.98: {g_neigh} | controls>=.10&lo>0: {g_ctrl}  => G2 {'PASS' if g_teach and g_neigh and g_ctrl else 'FAIL'}")
    json.dump(r, open(p.replace(".json", ".analysis.json"), "w"), indent=1)
