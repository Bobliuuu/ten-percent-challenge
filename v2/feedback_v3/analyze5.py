"""fb5 aggregation: cluster bootstrap (annotation draw -> item), controls empty/mixed/random-tool. python analyze5.py > table5.md"""
import json,glob,numpy as np,math
g=np.random.default_rng(0)
def cboot(M,B=2000):
    """M: draws x items matrix of per-item values. two-stage bootstrap of the grand mean."""
    D,T=M.shape; out=np.empty(B)
    for b in range(B):
        di=g.integers(0,D,D); out[b]=np.mean([M[d][g.integers(0,T,T)].mean() for d in di])
    return float(M.mean()),float(np.quantile(out,.025)),float(np.quantile(out,.975))
def wilson_lo(k,n,z=1.96):
    if n==0: return 0.0
    p=k/n; d=1+z*z/n; return (p+z*z/(2*n)-z*math.sqrt(p*(1-p)/n+z*z/(4*n*n)))/d
print("| model | scenario | method | draws | ann tools | base teach | gain vs empty (lo) | real-mixed (lo) | real-randtool (lo) | neighbors (n, lo) | G2 |\n|---|---|---|---|---|---|---|---|---|---|---|")
for f in sorted(glob.glob("results/*_nat_*.json")+glob.glob("results/*_newtool*.json")+glob.glob("results/*_pairS*.json")+glob.glob("results/*_nl_*.json")):
    R=json.load(open(f))
    if "draws" not in R: continue
    bt=np.array(R["base_teach"],float)
    for m in ["lmhead_lda","lmhead_plain","lmhead_plain_xs","lmhead_cal","lmhead_cal2","memit_down"]:
        Dm=[d for d in R["draws"] if d["method"]==m]
        if len(Dm)<3: continue
        real=np.array([d["real"]["teach"] for d in Dm],float); mix=np.array([d["mixed"]["teach"] for d in Dm],float); rnd=np.array([d["randtool"]["teach"] for d in Dm],float)
        nb=np.array([d["real"]["neigh"] for d in Dm],float)
        g0=cboot(real-bt); g1=cboot(real-mix); g2=cboot(real-rnd); nbm=nb.mean(); n=nb.size
        ok=g0[1]>0 and g0[0]>=.15 and g1[1]>0 and g1[0]>=.10 and g2[1]>0 and g2[0]>=.10 and nbm>=.98
        cfgs_=Dm[0]['cfg']; lab=m+(f" a={cfgs_.get('alpha')}" if 'alpha' in cfgs_ else '')
        print(f"| {R['model'].split('/')[-1]} | {R['scen']} | {lab} | {len(Dm)} | {R.get('ntools_ann')} | {bt.mean():.2f} | {g0[0]:+.2f} ({g0[1]:+.2f}) | {g1[0]:+.2f} ({g1[1]:+.2f}) | {g2[0]:+.2f} ({g2[1]:+.2f}) | {nbm:.3f} (n={R['sizes']['neigh']}, lo {wilson_lo(int(nb.sum()),n):.3f}) | {'PASS' if ok else 'FAIL'} |")
print("\n### real-codec stand-in (rotation + L=8 grid per-128 scale + ideal-rANS entropy bytes), lm_head only\n| model | scenario | variant | teach gain vs quantised base | neighbors | rows edited | artifact bytes before | after | delta bytes |\n|---|---|---|---|---|---|---|---|---|")
for f in sorted(glob.glob("results/*_nat_*.json")+glob.glob("results/*_newtool*.json")+glob.glob("results/*_pairS*.json")+glob.glob("results/*_nl_*.json")):
    R=json.load(open(f)); C=R.get("codec_real")
    if not C or not C["runs"]: continue
    qb=np.array(C["base_teach"],float)
    for v in ["reencode_refit","reencode_orig_scale"]:
        r=[x for x in C["runs"] if x["var"]==v]
        if r: print(f"| {R['model'].split('/')[-1]} | {R['scen']} | {v} | {np.mean([np.mean(np.array(x['teach'],float)-qb) for x in r]):+.2f} | {np.mean([np.mean(x['neigh']) for x in r]):.3f} | {np.mean([x['rows'] for x in r]):.0f} | {C['artifact_bytes_before']:.0f} ({C['bpw']:.3f} bpw; fp16 {C['fp16_bytes']}) | {np.mean([x['bytes_after'] for x in r]):.0f} | {np.mean([x['bytes_after'] for x in r])-C['artifact_bytes_before']:+.0f} |")
print("\n### sweep (draw 0, dev)\n| scenario | cfg | dev gain | neigh_dev |\n|---|---|---|---|")
for f in sorted(glob.glob("results/*_nat_*.json")+glob.glob("results/*_newtool*.json")+glob.glob("results/*_pairS*.json")+glob.glob("results/*_nl_*.json")):
    R=json.load(open(f))
    for s in R.get("sweep",[]): print(f"| {R['scen']} | {s['cfg']} | {s['dev_gain']} | {s['neigh_dev']} |")
