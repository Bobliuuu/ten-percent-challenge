import json,glob,collections,sys,numpy as np
G=collections.defaultdict(list)
for f in glob.glob('results/dev_*.json'):
    if 'smoke' in f: continue
    R=json.load(open(f))
    for r in R['rows']:
        if r.get('kk') and 'wrong_teach' in r: G[(R['layers'],r['rule'],r['mu'],r['nu'],r['k'])].append(r)
rows=[]
for k,v in G.items():
    g=lambda key: float(np.mean([x[key] for x in v]))
    rows.append((k,len(v),g('teach'),g('neigh'),g('shuffled'),g('wrong_teach') if 'wrong_teach' in v[0] else -1,g('wrong_neigh') if 'wrong_neigh' in v[0] else -1,g('loo_hit'),g('calib_flip'),min(x['neigh_by_kind']['Alarm'] for x in v if 'neigh_by_kind' in x) if 'neigh_by_kind' in v[0] else -1))
rows.sort(key=lambda r:-(r[2]-5*r[4]-3*max(0,.99-r[3])))
print("layers,rule,mu,nu,k | n | teach neigh shuf wrongT wrongN loo flip minAlarm")
for r in rows[:int(sys.argv[1]) if len(sys.argv)>1 else 20]: print(r[0],r[1],*[round(x,3) for x in r[2:]])
