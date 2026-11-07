"""usage: python analyze.py BASE.json CAND.json [CAND2.json ...]  (dense_stream outputs, key 't1'). Strict-gate report:
overall retention + Wilson lo, per-(source,kind) cell retention, worst cell, size ratio, KL if present."""
import json,sys,os,numpy as np
sys.path.insert(0,os.path.expanduser("~/ten-percent/v2/harness")); from scorecard import retention
b=json.load(open(sys.argv[1]))["t1"]; bc=np.array([r["correct"] for r in b]); src=np.array([r["source"] for r in b]); kd=np.array([r["kind"] for r in b])
print(f"base acc {bc.mean():.3f} n={len(bc)}  "+" ".join(f"{s}:{bc[src==s].mean():.3f}" for s in sorted(set(src))))
for f in sys.argv[2:]:
    d=json.load(open(f)); cc=np.array([r["correct"] for r in d["t1"]]); assert [r["id"] for r in d["t1"]]==[r["id"] for r in b]
    o=retention(bc,cc); cells={f"{s}/{k}":retention(bc,cc,(src==s)&(kd==k)) for s in sorted(set(src)) for k in sorted(set(kd))}
    w=min(cells,key=lambda c:cells[c]["ret"])
    ok=o["ret"]>=.98 and o["ret_lo"]>=.98 and all(c["ret"]>=.95 for c in cells.values())
    print(f"\n{os.path.basename(f)} ratio={d.get('ratio',float('nan')):.4f} bytes={d.get('bytes',0):,.0f} acc={cc.mean():.3f} RET={o['ret']:.4f} (Wilson lo {o['ret_lo']:.4f}; {o['kept']}/{o['n_base_correct']}) KL={d.get('tf_kl')} top1={d.get('tf_top1')} STRICT_PASS={ok}")
    for c,v in cells.items(): print(f"   {c:14s} ret {v['ret']:.3f} (lo {v['ret_lo']:.3f}) n={v['n_base_correct']}")
    print("   binding cell:",w, f"{cells[w]['ret']:.3f}")
    for s in sorted(set(src)): print(f"   {s} overall ret {retention(bc,cc,src==s)['ret']:.3f}")
