import re,subprocess,json,glob
subprocess.run(["python3","analyze_moe.py"],capture_output=True)
T=open('results/TABLES.md').read()
# best-point ranking: per model, the cheapest cfg (lowest est ratio <=0.115) with both tasks; rank by min over tasks of retention lower bound
R={}
for f in sorted(glob.glob('results/*_v2.json')):
    d=json.load(open(f)); m=d['arch']['model']
    by={}
    for c in d['cfgs']: by.setdefault((c['f'],c['bits']),{})[c['task']]=c
    for k,v in by.items():
        if len(v)<2: continue
        est=max(x['est_ratio'] for x in v.values()); lo=min(x['ret_lo'] for x in v.values()); ret=min(x['ret'] for x in v.values())
        wc=min((x['worst_cell'][0] if x['worst_cell'] else 1) for x in v.values())
        R.setdefault(m,[]).append((k,est,ret,lo,wc,{t:round(x['ret'],3) for t,x in v.items()},{t:round(d['base'][t]['acc'],3) for t in v}))
rank=[]
for m,L in R.items():
    ok=[x for x in L if x[1]<=0.115]
    best=max(ok,key=lambda x:x[3]) if ok else None
    rank.append((m,best))
rank.sort(key=lambda r:-(r[1][3] if r[1] else -1))
rt=["| rank | model | best point (keep f, bits) with est size <= 11.5% | est size | retention SKILL7 / BANK8 | min Wilson lower | worst cell | base acc S7 / B8 |","|---|---|---|---|---|---|---|---|"]
for i,(m,b) in enumerate(rank,1):
    if b is None: rt.append(f"| {i} | {m} | none reaches <=11.5% | | | | | |"); continue
    k,est,ret,lo,wc,rr,ba=b
    rt.append(f"| {i} | {m} | f={k[0]}, {k[1]}-bit | {100*est:.1f}% | {rr.get('SKILL7')} / {rr.get('BANK8')} | {lo:.3f} | {wc:.2f} | {ba.get('SKILL7')} / {ba.get('BANK8')} |")
sec=f"""## MoE census (fork B): routing concentration and expert-prune + RTN probe (SKILL7 + BANK8 dev, real greedy generation)
Code `v2/model_census/moe/moe_probe2.py` (transformers 5.22 fused-expert layout; every router returns (logits, weights, indices); pruned experts are removed by re-selecting among kept experts with the reference's own scoring/renorm, so the kept experts get exactly the weights a physically pruned model would give). Per task: experts kept per layer by routing mass on that task's CALIBRATION text (disjoint from the dev items), expert weights RTN asymmetric g128 (fake-quant, bf16 dequantised), real greedy generation of the label (<= 8 tokens), retention = P(correct | base correct) on <= 400 dev items per task (SKILL7 = task_search/EASY dev, BANK8 = rival3 bank+clin SEL first 400 each by md5). Raw per-config rows: `v2/model_census/moe/results/*_v2.json`. Attention, embeddings and head are left bf16 in the probe (not quantised); the size column ASSUMES them at 6.25 bpw (6-bit g128 incl. scale) plus fp16 router rows, so the real size is an estimate, not a packed file. No GPTQ (plain RTN: a lower bound for retention). Dev only, n=400/task: a 99% point has Wilson lower ~98%, i.e. this is a ranking probe, not pass evidence.

### Ranked MoE bases (best point under 11.5% estimated size)
{chr(10).join(rt)}

{T}
Reading: concentration = fraction of experts (layer mean) needed to cover 90/95/99% of the task's routing mass; the lower, the more prunable. Fine-grained models with many experts (Qwen3.5-35B-A3B 256, Qwen3-30B-A3B 128) concentrate most (90% of mass in 27-34% of experts) and survive keeping 30-50%; 64-expert models (OLMoE, DS-V2-Lite, ERNIE) need 56-63%+ and collapse below keep 50%. Cross-task overlap (other task's mass inside this task's top-50%) is 0.73-0.94, so a SKILL7-pruned model keeps most of BANK8's routing mass but BANK8 needs more experts than SKILL7 (more labels, finer boundaries).
"""
open('SECTION.md','w').write(sec)
def inject(path,marker,text):
    s=open(path).read(); a=f"<!-- {marker} BEGIN -->"; b=f"<!-- {marker} END -->"
    blk=f"{a}\n{text}\n{b}"
    if a in s: s=re.sub(re.escape(a)+".*?"+re.escape(b),lambda m:blk,s,flags=re.S)
    else: s=s.rstrip()+"\n\n"+blk+"\n"
    open(path,'w').write(s)
inject('../../MODEL_CENSUS.md','MOE-CENSUS-B',sec)
# near-miss rows
rows=["## MoE census rows (fork B; prune-by-task-routing-mass + RTN g128 on experts only, est size assumes non-expert at 6.25 bpw; retention vs base on dev, n<=400/task)","| model | task | keep f | bits | est size | retention [Wilson lo] | worst cell (n>=8) | rule gaps |","|---|---|---|---|---|---|---|---|"]
for f in sorted(glob.glob('results/*_v2.json')):
    d=json.load(open(f)); m=d['arch']['model']
    for c in d['cfgs']:
        if c['est_ratio']>0.125: continue
        wc=c['worst_cell']; rows.append(f"| {m} | {c['task']} | {c['f']} | {c['bits']} | {100*c['est_ratio']:.1f}% | {100*c['ret']:.1f} [{100*c['ret_lo']:.1f}] | {('%.2f %s'%(wc[0],wc[1])) if wc else ''} | probe only: size estimated, no GPTQ/real pack, no locked exam, no feedback |")
inject('../../NEAR_MISSES.md','MOE-CENSUS-B',"\n".join(rows))
print(len(rows))
