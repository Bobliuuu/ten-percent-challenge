"""Aggregate fb4 results (results/*_{first,last}*.json) with scorecard.paired_boot. python analyze4.py > table4.md"""
import json,glob,sys,os,numpy as np
sys.path.insert(0,os.path.expanduser("~/ten-percent/v2/harness"))
from scorecard import paired_boot
rows=[]
print("| model | scenario | fmt | method (cfg) | base teach | teach gain (95% lo) | neighbors | real-shuf (lo) | G2 |\n|---|---|---|---|---|---|---|---|---|")
for f in sorted(glob.glob("results/*_first*.json")+glob.glob("results/*_last*.json")):
    R=json.load(open(f)); bt=np.array(R["base_teach"],float)
    for meth in ["lmhead_lda","blend","memit_down"]:
        F=[x for x in R["final"] if x["method"]==meth]
        if len(F)<3: continue
        g=paired_boot(np.concatenate([np.array(x["real"]["teach"],float)-bt for x in F])); sh=paired_boot(np.concatenate([np.array(x["real"]["teach"],float)-np.array(x["shuffled"]["teach"],float) for x in F]))
        nb=float(np.mean([v for x in F for v in x["real"]["neigh"]])); ok=g[1]>0 and g[0]>=.15 and nb>=.98 and sh[1]>0 and sh[0]>=.10
        c=F[0]["cfg"]; cs=",".join(f"{k}={v}" for k,v in c.items() if k!="m")
        print(f"| {R['model'].split('/')[-1]} | {R['scen']} | {R['fmt']} | {meth} ({cs}) | {bt.mean():.2f} | {g[0]:+.2f} ({g[1]:+.2f}) | {nb:.3f} | {sh[0]:+.2f} ({sh[1]:+.2f}) | {'PASS' if ok else 'FAIL'} |")
print("\n### lm_head codec test (lm_head 4-bit g128 RTN as stage-1; edited rows: fp16 kept / re-quantised 4-bit own scales / snapped to original scales)\n| model | scenario | variant | teach acc | gain vs quantised base | neighbors | rows edited | bytes added |\n|---|---|---|---|---|---|---|---|")
for f in sorted(glob.glob("results/*_first*.json")+glob.glob("results/*_last*.json")):
    R=json.load(open(f))
    if not R.get("codec"): continue
    qb=np.array(R["codec_base"]["teach"],float)
    print(f"| {R['model'].split('/')[-1]} | {R['scen']} | quantised base (no edit) | {qb.mean():.2f} | | {np.mean(R['codec_base']['neigh']):.3f} | 0 | 0 |")
    for v in ["fp16row","req4_newscale","snap_orig"]:
        C=[x for x in R["codec"] if x["var"]==v]
        if C: print(f"| {R['model'].split('/')[-1]} | {R['scen']} | {v} | {np.mean([np.mean(x['teach']) for x in C]):.2f} | {np.mean([np.mean(np.array(x['teach'],float)-qb) for x in C]):+.2f} | {np.mean([np.mean(x['neigh']) for x in C]):.3f} | {np.mean([x['rows_edited'] for x in C]):.0f} | {np.mean([x['bytes_added'] for x in C]):.0f} |")
print("\n### lmhead_lda / blend sweep on dev (alpha, ridge): dev gain / neighbor-dev\n| model | scen | fmt | cfg | dev gain | neigh_dev |\n|---|---|---|---|---|---|")
for f in sorted(glob.glob("results/*_first*.json")+glob.glob("results/*_last*.json")):
    R=json.load(open(f))
    for s in R["sweep"]:
        if "err" in s: continue
        c=s["cfg"]; print(f"| {R['model'].split('/')[-1]} | {R['scen']} | {R['fmt']} | {c['m']} {','.join(f'{k}={v}' for k,v in c.items() if k!='m')} | {s['gain']:+.2f} | {s['neigh_dev']:.2f} |")
