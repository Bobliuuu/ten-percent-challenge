"""Deterministic generator of the default stage-2 feedback sets (reads ONLY C2 pool/calib + dev for choosing hard tools/pairs' eval support; never the locked file).
Writes: ann_newtool_k5.json, ann_pair_k1.json, ann_combined_k6.json (annotation rows: inst,text,names,answer; answer 4 = none fits),
        memory_k5_newtool.json, memory_k1_pair.json, memory_k6_combined.json, feedback_plan.json
Memory is produced by predict.ingest_feats (the same function predict.ingest calls after its forward pass) from the decoded-artifact features of the pool rows
(feats/pool.npz extracted by feat_c2.py). GPU path predict.ingest(art, ann, None, mem) gives the same file up to bf16 batch noise (checked in verify_final.py).
python make_feedback_memory.py [out_dir]"""
import sys, os, json, importlib.util, collections, numpy as np
D = os.path.dirname(os.path.abspath(__file__)); OUT = sys.argv[1] if len(sys.argv) > 1 else D + "/feedback_sets"; os.makedirs(OUT, exist_ok=True)
SEED = 20261106
spec = importlib.util.spec_from_file_location("predict_mod", D + "/artifact_code/predict.py"); P = importlib.util.module_from_spec(spec); spec.loader.exec_module(P)
tn = P.tool_name; R = D + "/../commercial/C2/"
def ld(sp):
    rows = [json.loads(l) for l in open(R + sp + ".jsonl")]
    for r in rows: r.pop("tools", None); r.pop("gold_calls", None)
    return rows
pool, dev = ld("pool"), ld("dev")
zp = np.load(D + "/feats/pool.npz"); zd = np.load(D + "/feats/dev.npz")
pp = zp["L"][:, :5].argmax(1); pd = zd["L"][:, :5].argmax(1)       # plain (stage-1) decisions
yp = np.array([r["answer"] for r in pool]); yd = np.array([r["answer"] for r in dev])
rng = np.random.default_rng(SEED)
# S2: hard new tool = plain dev recall <= .6, >=5 gold pool items, >=5 gold dev items
gp = collections.defaultdict(list); gd = collections.defaultdict(list)
for i, r in enumerate(pool):
    if yp[i] != 4: gp[tn(r["names"][yp[i]])].append(i)
for i, r in enumerate(dev):
    if yd[i] != 4: gd[tn(r["names"][yd[i]])].append(i)
cand = sorted(t for t in gp if len(gp[t]) >= 5 and len(gd[t]) >= 5 and (pd[gd[t]] == yd[gd[t]]).mean() <= 0.6)
tool = cand[rng.integers(len(cand))]; idx_t = sorted(rng.choice(gp[tool], 5, replace=False).tolist())
# S3: pair error pattern (gold A, plain pred B) present in pool and >=2 times in dev
pat_p = collections.defaultdict(list); pat_d = collections.defaultdict(list)
for i, r in enumerate(pool):
    if yp[i] != 4 and pp[i] != 4 and pp[i] != yp[i]: pat_p[(tn(r["names"][yp[i]]), tn(r["names"][pp[i]]))].append(i)
for i, r in enumerate(dev):
    if yd[i] != 4 and pd[i] != 4 and pd[i] != yd[i]: pat_d[(tn(r["names"][yd[i]]), tn(r["names"][pd[i]]))].append(i)
cp = sorted(k for k in pat_p if len(pat_d.get(k, [])) >= 2)
pair = cp[rng.integers(len(cp))]; idx_p = [int(rng.choice(pat_p[pair]))]
def ann(idx): return [dict(inst=pool[i]["inst"], text=pool[i]["text"], names=pool[i]["names"], answer=int(pool[i]["answer"]), id=pool[i].get("id")) for i in idx]
sets = dict(newtool_k5=idx_t, pair_k1=idx_p, combined_k6=idx_t + idx_p)
H = P.Head(D + "/artifact_code")
for name, idx in sets.items():
    a = ann(idx); json.dump(a, open(f"{OUT}/ann_{name}.json", "w"), indent=0)
    M = P.Mem(); P.ingest_feats(H, M, a, zp["L"][idx], zp["FO"][idx])
    mp = f"{OUT}/memory_{name}.json"; M.dump(mp); print(name, "bytes", os.path.getsize(mp))
json.dump(dict(seed=SEED, new_tool=tool, new_tool_pool_idx=idx_t, new_tool_dev_eval_n=len(gd[tool]), new_tool_dev_plain_recall=float((pd[gd[tool]] == yd[gd[tool]]).mean()),
               pair_gold_pred=list(pair), pair_pool_idx=idx_p, pair_dev_err_n=len(pat_d[pair]), pair_dev_err_idx=pat_d[pair], new_tool_dev_idx=gd[tool]), open(f"{OUT}/feedback_plan.json", "w"), indent=1)
print(tool, pair)
