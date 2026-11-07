import json,sys
def L(n): return {r['id']:r for r in json.load(open(f'results/res_{n}.json'))['rows']}
base=L(sys.argv[1])
for n in sys.argv[2:]:
    v=L(n); bc=[i for i,r in base.items() if r['correct']]
    ret=sum(v[i]['correct'] for i in bc)/len(bc)
    same=sum(v[i]['output']==base[i]['output'] for i in base)/len(base)
    acc=sum(r['correct'] for r in v.values())/len(v)
    ks={}
    for k in ('nocall','single','multi'):
        b=[i for i in bc if base[i]['kind']==k]; ks[k]=round(sum(v[i]['correct'] for i in b)/len(b),3)
    ss={}
    for s in ('S1','S3'):
        b=[i for i in bc if base[i]['source']==s]; ss[s]=round(sum(v[i]['correct'] for i in b)/len(b),3)
    print(n,'acc',round(acc,3),'retention',round(ret,3),'identical-output',round(same,3),ks,ss,'n_base_correct',len(bc))
