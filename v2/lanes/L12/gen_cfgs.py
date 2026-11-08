import json
NQ = dict(attn_b=4, emb_b=4, head_b=4)   # realistic non-expert quant for budget configs
A = []; B = []
def add(L, name, **k): L.append(dict(name=name, **k))
# A1 pruning scores (no quantisation of anything: isolates the score)
for keep in (0.25, 0.4):
    for sc in ("freq", "mass", "reap", "reapsum", "gatefree", "rand", "kl"):
        add(A, f"A1_{sc}_k{int(keep*100)}", score=sc, keep=keep)
# A2 layer-wise adaptive counts (score reap, keep 0.3)
for ad, extra in (("uniform", {}), ("global", {}), ("cover", dict(cover=0.95)), ("kl", dict(gamma=0.5)), ("kl", dict(gamma=1.0))):
    add(A, f"A2_{ad}{extra.get('gamma','')}_k30", score="reap", keep=0.3, adapt=ad, **extra)
# A3 top-k reduction (zero bytes)
for tk in (8, 6, 4, 3, 2): add(A, f"A3_full_top{tk}", keep=1.0, topk=tk)
for tk in (6, 4): add(A, f"A3_reap40_top{tk}", score="reap", keep=0.4, topk=tk)
# A4 merging vs pruning
for keep in (0.25, 0.4):
    k = int(keep * 100)
    add(A, f"A4_prune_k{k}", score="reap", keep=keep)
    add(A, f"A4_mergeavg_k{k}", score="mass", keep=keep, merge="avg")
    add(A, f"A4_mergealign_k{k}", score="mass", keep=keep, merge="align")
    add(A, f"A4_mergealign_nolse_k{k}", score="mass", keep=keep, merge="align", merge_router=False)
# A5 attention/embedding/head quant alone
add(A, "A5_nonexp_q4", keep=1.0, **NQ); add(A, "A5_nonexp_q3", keep=1.0, attn_b=3, emb_b=4, head_b=4)
# A6 RTN on all experts
for b in (4, 3, 2): add(A, f"A6_rtn{b}_all", keep=1.0, eb=b, **NQ)
# A7 shared subspace on kept experts (flag: adapter-like)
for r in (1024, 512, 256):
    add(A, f"A7_sub_r{r}_k40", score="reap", keep=0.4, subspace=dict(r=r, r_down=r // 2 if r > 256 else 128, cb=4, bb=4), **NQ)
add(A, "A7_sub_r512_cb3_k40", score="reap", keep=0.4, subspace=dict(r=512, r_down=256, cb=3, bb=4), **NQ)
add(A, "A7_sub_r1024_k100", keep=1.0, subspace=dict(r=1024, r_down=512, cb=3, bb=4), **NQ)
# B GPTQ+Hadamard
for b in (4, 3, 2): add(B, f"B1_gptq{b}_all", keep=1.0, eb=b, quant="gptq", **NQ)
for keep, b in ((0.7, 2), (0.5, 3), (0.45, 3), (0.35, 4), (0.4, 3), (0.3, 3), (0.25, 4), (0.3, 4), (0.5, 2), (0.35, 3)):
    add(B, f"B2_gptq{b}_reap_k{int(keep*100)}", score="reap", keep=keep, eb=b, quant="gptq", **NQ)
for keep, b in ((0.5, 2), (0.4, 3)):
    add(B, f"B3_rtn{b}_reap_k{int(keep*100)}", score="reap", keep=keep, eb=b, **NQ)
json.dump(A, open("cfgsA.json", "w"), indent=1); json.dump(B, open("cfgsB.json", "w"), indent=1); print(len(A), len(B))
