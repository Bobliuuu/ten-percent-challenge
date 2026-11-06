import json,sys
r=json.load(open(sys.argv[1])); print(len(r))
for x in r:
    if 'cfg' not in x: print(x['error'][:200]); continue
    c=x['cfg']; print(c['model'][11:-9],c['name'],'base',round(x['base_acc'],3),'acc',round(x['acc'],3),'agree',round(x['agree'],4),'bf16',round(x['ratio_bf16'],3),'fp32',round(x['ratio_fp32'],3),'bf16ent',round(x['ratio_bf16_ent'],3),'bpw',round(x['body_bits_per_w'],2))
