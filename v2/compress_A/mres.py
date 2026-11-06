import sys,json
for l in sys.stdin:
    if 'RESULT_JSON ' not in l: continue
    r=json.loads(l.split('RESULT_JSON ',1)[1]); print(r['cfg']['name'],'base',round(r['base_acc'],3),'acc',round(r['acc'],3),'| bias: base',round(r['base_acc_bias'] or 0,3),'q',round(r['acc_bias'] or 0,3),'agree',round(r['agree'],3),'ratio',round(r['ratio_bf16'],4))
