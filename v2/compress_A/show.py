import json
for l in open("/Users/jhome/ten-percent/v2/compress_A/results/modal_results.jsonl"):
    r=json.loads(l); print(r['cfg']['name'],'base',round(r['base_acc'],3),'acc',round(r['acc'],3),'| bias: base',round(r['base_acc_bias'] or 0,3),'q',round(r['acc_bias'] or 0,3),'agree',round(r['agree'],3),'ratio',round(r['ratio_bf16'],4))
