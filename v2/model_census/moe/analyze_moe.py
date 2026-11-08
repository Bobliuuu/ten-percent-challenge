import json,glob,os,sys
rows=[];conc=[];arch=[]
for f in sorted(glob.glob('results/*_v2.json')):
    d=json.load(open(f)); a=d['arch']; m=a['model']
    P=a['total_params'];X=a['expert_params'];N=P-X
    def bp(f): return (1.6*P-6.25*N)/(f*X)
    arch.append(f"| {m} | {a['moe_layers']} | {a['experts']} | {a['topk']} | {P/1e9:.1f} | {100*a['expert_share']:.1f}% | {2*P/1e9:.1f} | {0.2*P/1e9:.2f} | "+" / ".join(f"{bp(f):.2f}" for f in (1,.75,.5,.4,.3))+" |")
    for t,s in d['conc']['summary'].items():
        if t=='NEWS': continue
        conc.append(f"| {m} | {t} | {s['n90']:.2f} / {s['n95']:.2f} / {s['n99']:.2f} | {s['entropy_norm']:.2f} | {s['mass_in_own_top50']:.2f} | {s['mass_in_own_top30']:.2f} | {', '.join(f'{u} {v:.2f}' for u,v in s['cross_top50'].items())} |")
    for c in d['cfgs']:
        wc=c['worst_cell']; rows.append((m,c['task'],c['f'],c['bits'],c['est_ratio'],d['base'][c['task']]['acc'],c['ret'],c['ret_lo'],wc[0] if wc else None,wc[1] if wc else '',c['n_bc']))
out=["| model | task | kept f | bits | est size % fp16 | base acc | retention | lower | worst cell | cell | n |","|---|---|---|---|---|---|---|---|---|---|---|"]
for r in rows: out.append(f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} | {100*r[4]:.1f} | {r[5]:.3f} | {r[6]:.3f} | {r[7]:.3f} | {r[8] if r[8] is None else round(r[8],2)} | {r[9]} | {r[10]} |")
hdr=["| model | task | experts needed 90/95/99% (fraction, layer mean) | routing entropy / ln E | mass in own top-50% | mass in own top-30% | other task's mass in top-50% |","|---|---|---|---|---|---|---|"]
open('results/TABLES.md','w').write("### Architecture and byte budget (non-expert tensors charged at 6.25 bpw; expert bpw includes scales)\n| model | MoE layers | experts | top-k | params B | expert share | fp16 GB | 10% budget GB | expert bpw if keep 100/75/50/40/30% |\n|---|---|---|---|---|---|---|---|---|\n"+"\n".join(arch)+"\n\n### Routing concentration (dev calibration text)\n"+"\n".join(hdr+conc)+"\n\n### Prune + RTN probe (real greedy generation)\n"+"\n".join(out)+"\n")
print(open('results/TABLES.md').read())
