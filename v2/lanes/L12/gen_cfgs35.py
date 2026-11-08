import json
NQ = dict(attn_b=4, emb_b=4, head_b=4, shared_b=4)
C = []
def add(name, **k): C.append(dict(name=name, **k))
for sc in ("freq", "mass", "reap", "gatefree", "rand"):
    add(f"Q35_A1_{sc}_k40", score=sc, keep=0.4)
add("Q35_A1_reap_k25", score="reap", keep=0.25); add("Q35_A1_mass_k25", score="mass", keep=0.25)
for tk in (6, 4, 2): add(f"Q35_A3_full_top{tk}", keep=1.0, topk=tk)
add("Q35_A4_mergealign_k25", score="mass", keep=0.25, merge="align"); add("Q35_A4_mergealign_k40", score="mass", keep=0.4, merge="align")
add("Q35_A2_global_k30", score="reap", keep=0.3, adapt="global"); add("Q35_A2_uniform_k30", score="reap", keep=0.3)
add("Q35_A5_nonexp_q4", keep=1.0, **NQ)
for b in (4, 3): add(f"Q35_A6_rtn{b}_all", keep=1.0, eb=b, **NQ)
add("Q35_A7_sub_r512_k40", score="reap", keep=0.4, subspace=dict(r=512, r_down=256, cb=4, bb=4), **NQ)
for keep, b in ((0.7, 2), (0.5, 3), (0.4, 3), (0.35, 4), (0.3, 3), (0.5, 2), (0.25, 4), (0.2, 4)):
    add(f"Q35_B2_gptq{b}_reap_k{int(keep*100)}", score="reap", keep=keep, eb=b, quant="gptq", **NQ)
add("Q35_B1_gptq3_all", keep=1.0, eb=3, quant="gptq", **NQ)
json.dump(C, open("cfgsQ35.json", "w"), indent=1); print(len(C))
