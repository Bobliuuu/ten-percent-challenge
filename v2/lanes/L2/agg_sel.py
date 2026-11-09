"""agg_sel.py: dev-selection table from results/sel/*.jsonl: per (carriers, margin) real25 / mshuf teach, gap, neighbours (mean over draws and the 2 dev labels)."""
import json,glob,collections
R=[json.loads(l) for f in glob.glob('results/sel/sel_*.jsonl') for l in open(f)]
tk=lambda r:[v for k,v in r.items() if k.startswith('teach_')][0]
G=collections.defaultdict(list)
for r in R: G[(r['job'].split('_')[1],int(r['m']),r['arm'])].append(r)
av=lambda L,f:sum(f(x) for x in L)/len(L)
print("carr  m  real25 shuf gap | neigh_r25 minCell shufNeigh | real12 neigh12")
for (c,m) in sorted({(k[0],k[1]) for k in G}):
    g=lambda a:G.get((c,m,a))
    if g('real25') and g('mshuf'):
        t=av(g('real25'),tk);s=av(g('mshuf'),tk)
        print(f"{c:4s} {m:3d} {t:.3f} {s:.3f} {t-s:+.3f} | {av(g('real25'),lambda x:x['neigh']):.4f} {av(g('real25'),lambda x:x['min_cell']):.3f} {av(g('mshuf'),lambda x:x['neigh']):.3f} | {av(g('real12'),tk):.3f} {av(g('real12'),lambda x:x['neigh']):.4f}")
