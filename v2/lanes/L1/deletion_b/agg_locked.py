import json,glob,numpy as np,collections
D=[]
for f in sorted(glob.glob('results/locked_d*.json')): D+=json.load(open(f))['draws']
for arm in ('real','shuffled','wrong'):
    x=[d for d in D if d['arm']==arm]; print(arm,len(x),'teach',round(np.mean([d['teach'] for d in x]),4),[round(d['teach'],3) for d in x],'k',[d['k'] for d in x],'bitmapB',int(np.mean([d['bitmap_bytes'] for d in x])))
    if arm!='shuffled':
        n=[d['neigh_kept'] for d in x]; print('  neigh',round(np.mean(n),4),'min',round(min(n),4),[round(v,3) for v in n])
        kinds=collections.defaultdict(list)
        for d in x:
            for k,v in d['neigh_by_kind'].items(): kinds[k].append(v)
        print('  by kind',{k:round(float(np.mean(v)),3) for k,v in kinds.items()})
    src=collections.defaultdict(list)
    for d in x:
        for k,v in d['teach_by_src'].items(): src[k].append(v[0])
    print('  teach by src',{k:round(float(np.mean(v)),3) for k,v in src.items()})
    for d in x:
        if 'exam_after' in d: e=d['exam_after']; print('  exam_after d',d['d'],'acc',round(d['exam_acc_after'],4),'ret',round(e['overall']['ret'],4),'wilson',round(e['overall']['wilson_lo'],4),'gate98',e['gate_overall_98'],'kind95',e['gate_kind_95'],'recipe_leak',d['recipe_leak_exam'],'offmenu',d['off_menu_exam'],{k:round(v['ret'],3) for k,v in e['by_kind'].items()})
