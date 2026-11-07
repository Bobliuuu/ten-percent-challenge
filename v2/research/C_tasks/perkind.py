import json,glob,sys,os,math,collections
def wil(k,n,z=1.96):
    if n==0: return (0,0)
    p=k/n;d=1+z*z/n;c=p+z*z/(2*n);h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n));return ((c-h)/d,(c+h)/d)
model=sys.argv[1]; tasks=sys.argv[2].split(','); cfgs=sys.argv[3].split(',')
R={c:{r['id']:r for r in json.load(open(f'results/ct_{model}_{c}.json'))['rows']} for c in ['bf16']+cfgs if os.path.exists(f'results/ct_{model}_{c}.json')}
for t in tasks:
    for c in cfgs:
        if c not in R: continue
        print(f"{model} {t} {c}:",end=" ")
        g=collections.defaultdict(lambda:[0,0])
        for i,b in R['bf16'].items():
            if b['task']==t and b['correct']:
                for key in (b['role'],b['kind']):
                    g[key][1]+=1; g[key][0]+=R[c][i]['correct']
        print("; ".join(f"{k} {v[0]}/{v[1]}={v[0]/v[1]:.3f} [{wil(*v)[0]:.2f}]" for k,v in sorted(g.items())))
