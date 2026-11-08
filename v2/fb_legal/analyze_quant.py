import json,sys,numpy as np,collections
A=lambda s:np.array(list(map(int,s)))
for p in sys.argv[1:]:
    R=json.load(open(p)); print("==",p,"quant_base",R.get("quant_base"))
    g=collections.defaultdict(list)
    for q in R.get("quant",[]): g[(q["b"],q["rot"],q["c"],q["mode"])].append((A(q["teach"]).mean(),A(q["neigh"]).mean(),q["nrows"]))
    for k,v in sorted(g.items()): print(k,"teach %.3f neigh %.4f rows %.1f n=%d"%(np.mean([x[0] for x in v]),np.mean([x[1] for x in v]),np.mean([x[2] for x in v]),len(v)))
