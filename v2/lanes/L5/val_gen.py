"""Real-greedy-generation validation of the feedback edit on the packed-body 32B (rule 5/4d): for draw d, n=25 real annotations, apply the fixed primary carrier and GENERATE the answer text for the 164 locked teach items and the neighbour items; compare with shuffled/wrong/empty. usage: python val_gen.py CFG.json OUT.json (run in the fb dir with fblib.py + exam_data)"""
import json, sys, random, tempfile, collections, numpy as np
sys.path.insert(0, ".")
from fblib import Env
cfg = json.load(open(sys.argv[1])); TASK = cfg["task"]; DD = json.load(open(f"exam_data/{TASK}.json")); NEW = DD["new"]; LABS = DD["labels"]
cal = DD["cal"]; random.Random(0).shuffle(cal); td = tempfile.mkdtemp()
open(f"{td}/calib_old.jsonl", "w").write("".join(json.dumps(dict(text=r["user"], gold=r["kind"])) + "\n" for r in cal[:300]))
env = Env(cfg["model"], td, q4=False, qbits=4, raw=True, ncal=300, log=print)
mk = lambda rows: [dict(text=r["user"], gold=r["kind"], src=r["src"], id=r["id"], pid=env.pid(r["user"])) for r in rows]
ann = mk(DD["ann_pool"]); teach = mk(DD["teach"]); neigh = mk(DD["neigh"]); shuf_pool = mk(cal[300:700])
mt = [m for m in cfg["methods"] if m["tag"] == "o_prot_c1024_m32"][0]
base_n = env.gen_text([r["pid"] for r in neigh], 6); NE_OK = [i for i, r in enumerate(neigh) if base_n[i] == r["gold"]]
ANN_BASE = env.gen_text([r["pid"] for r in ann], 6); out = dict(method=mt, n_neigh_ok=len(NE_OK), runs=[])
def evaluate(tag, d):
    gt = env.gen_text([r["pid"] for r in teach], 6); gn = env.gen_text([neigh[i]["pid"] for i in NE_OK], 6)
    rec = dict(cond=tag, draw=d, teach_gen=float(np.mean([g == NEW for g in gt])), teach_same=float(np.mean([g == NEW for g, r in zip(gt, teach) if r["src"] in ("massive", "clinc150")])), teach_cross=float(np.mean([g == NEW for g, r in zip(gt, teach) if r["src"] in ("mtop", "hwu64")])),
               neigh_gen=float(np.mean([g == neigh[i]["gold"] for g, i in zip(gn, NE_OK)])), teach_outputs=dict(collections.Counter(gt)))
    print(rec, flush=True); out["runs"].append(rec); json.dump(out, open(sys.argv[2], "w"), indent=1)
evaluate("empty", 0)
for d in range(3):
    rr = random.Random(100 * 25 + d); ix = rr.sample(range(len(ann)), 25)
    conds = {"real": [(ann[i], NEW) for i in ix], "wrong": [(ann[i], rr.choice([l for l in LABS if l != ANN_BASE[i]])) for i in ix], "shuffle": [(shuf_pool[i], NEW) for i in rr.sample(range(len(shuf_pool)), 25)]}
    for c in ("real", "shuffle", "wrong"):
        env.restore(); env.apply(mt, conds[c]); evaluate(c, d); env.restore()
