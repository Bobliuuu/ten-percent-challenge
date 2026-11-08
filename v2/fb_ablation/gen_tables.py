"""gen_tables.py -> tables.md (auto tables from results/modal/*.json, results/cli/*.json). Prose lives in ../FEEDBACK_ABLATION.md."""
import json, glob, collections, numpy as np, os
from analyze import table, md, load, f3
def get(names): return load([f"results/modal/{n}.json" for n in names if os.path.exists(f"results/modal/{n}.json")])
out = []
def sec(t, s): out.append(f"### {t}\n\n{s}\n")
main = get(["A1", "A1b", "A1c", "A2", "B1", "B2", "G"]); rows = [r for r in table(main, 12)]
sec("T1. Qwen2.5-7B-Instruct bf16 body, SKILL7-derived, new label Jukebox, n=12 annotations (ranked by teach - shuffled, then neighbours)", md(rows))
ns = []
for n in (5, 25):
    for r in table(main, n):
        ns.append(r | dict(n=n))
if ns:
    h = "| method | n | teach | shuffled | neigh | min cell | MB dense |\n|---|---|---|---|---|---|---|\n"
    for r in sorted(ns, key=lambda r: (r["tag"], r["n"])): h += f"| {r['tag']} | {r['n']} | {f3(r['teach'])} | {f3(r['shuf'])} | {f3(r['neigh'])} | {f3(r['mincell'])} | {f3(r['dense_mb'])} |\n"
    sec("T2. n sweep (5 / 25 annotations) for the core methods", h)
q = get(["C", "G4"]); sec("T3. 4-bit RTN g128 body (Qwen2.5-7B-Instruct), edited tensors as fp16 override vs re-quantised onto the same 4-bit grid (0 extra bytes)", md(table(q, 12)) if q else "pending")
e = get(["E", "F", "F2"]); sec("T4. Qwen3-14B (E = 4-bit RTN body, F = bf16 body)", md(table(e, 12)) if e else "pending")
cont = []
for R in get(["D"]):
    cc = collections.defaultdict(list)
    for c in R["cont"]: cc[(c["m"], c["mode"])].append(c)
    h = "| method | 2nd feedback | after 1st: Jukebox teach | 1st-round annotated kept | after 2nd: Jukebox teach | 1st-round annotated kept | 2nd label (or same) annotated | Briefing teach (new label) | neighbours kept |\n|---|---|---|---|---|---|---|---|---|\n"
    for (m, md_), L in sorted(cc.items()):
        f = lambda k, rr: np.mean([np.mean(list(map(int, x[rr][k]))) for x in L])
        h += f"| {m} | {'new label Briefing' if md_=='new' else 'more Jukebox'} | {f('teach','r1'):.3f} | {f('ann1','r1'):.3f} | {f('teach','r2'):.3f} | {f('ann1','r2'):.3f} | {f('ann2','r2'):.3f} | {f('teachN','r2'):.3f} | {f('neigh','r2'):.3f} |\n"
    sec("T5. Continual: feedback applied twice (5 draws x 12 annotations each)", h)
val = ["| run | method | set | teacher-forced acc | real greedy generation acc | agreement |\n|---|---|---|---|---|---|"]
for n in ["A1", "A2", "B1", "B2", "C"]:
    p = f"results/modal/{n}.json"
    if os.path.exists(p):
        for v in json.load(open(p)).get("val", []): val.append(f"| {n} | {v['m']} | {v['set']} | {v['acc_tf']:.3f} | {v['acc_gen']:.3f} | {v['agree']:.3f} |")
sec("T6. Real greedy generation vs teacher-forced evaluation (2 draws x 96-100 items)", "\n".join(val))
for task in ("SKILL7", "BANK6"):
    for b in ("bf16", "q4"):
        p = f"results/modal/X_{task}_{b}.json"
        if os.path.exists(p):
            R = load([p]); rows = table(R, 12); base = R[0]["base"]
            sec(f"T7. SHARED LOCKED-EXAM SPLITS: {task} new label {R[0]['meta']['new']}, Qwen2.5-7B-Instruct {b}", f"base teach {base['teach']:.3f}; neighbour set {base['n_neigh']} exam items ({base['n_neigh_ok']} base-correct); annotations from ann_pool only ({base['ann_base_labels']}). 'xsrc' columns are not used here (annotations are tuned-on sources, teach includes never-used sources; per-source teach below).\n\n" + md(rows))
            ts = R[0]["meta"]["teach_src"]; srcs = sorted(set(ts)); h = "| method | " + " | ".join(f"teach {s}" for s in srcs) + " |\n|---|" + "---|" * len(srcs) + "\n"
            for t in sorted({r["m"] for r in R[0]["runs"]}):
                rr = [r for r in R[0]["runs"] if r["m"] == t and r["c"] == "real"]
                h += f"| {t} | " + " | ".join(f"{np.mean([np.mean([int(b_) for b_, s_ in zip(r['teach'], ts) if s_ == s]) for r in rr]):.3f}" for s in srcs) + " |\n"
            sec(f"T8. per-source teach ({task} {b})", h)
            cc = R[0].get("cont", [])
            if cc:
                g = collections.defaultdict(list)
                for c in cc: g[c["m"]].append(c)
                h = "| method | teach after 1st | annotated-1 kept after 2nd | teach after 2nd | neigh after 2nd |\n|---|---|---|---|---|\n"
                for m, L in g.items(): h += f"| {m} | {np.mean([c['r1']['teach'] for c in L]):.3f} | {np.mean([c['r2']['ann1'] for c in L]):.3f} | {np.mean([c['r2']['teach'] for c in L]):.3f} | {np.mean([c['r2']['neigh'] for c in L]):.3f} |\n"
                sec(f"T9. continual on shared exam splits ({task} {b})", h)
for p in sorted(glob.glob("results/cli/cli2.json")):
    h = ""
    for s in json.load(open(p)): h += f"- `{s['cmd'][:55]}...` {s['secs']} s rc={s['rc']}\n```\n{s['stdout'][-1500:]}\n```\n"
    sec("T10. End-to-end CLI test (answer.py / feedback.py, safetensors shards rewritten in place, feedback applied twice)", h)
open("tables.md", "w").write("\n".join(out)); print(len(out), "sections")
