#!/bin/bash
# usage: harvest.sh app_id ; appends new RESULT_JSON lines to results/modal_results.jsonl (dedupe by cfg name)
cd /Users/jhome/ten-percent/v2/compress_A
modal app logs $1 2>&1 | grep "RESULT_JSON" | sed 's/^.*RESULT_JSON //' > /tmp/h_$1.txt
python3 - $1 <<'P'
import json,sys,os
f="results/modal_results.jsonl"; have=set()
if os.path.exists(f): have={json.loads(l)["cfg"]["name"] for l in open(f)}
with open(f,"a") as o:
    for l in open(f"/tmp/h_{sys.argv[1]}.txt"):
        try: r=json.loads(l)
        except Exception: continue
        if r["cfg"]["name"] not in have: o.write(json.dumps(r)+"\n"); have.add(r["cfg"]["name"])
P
python3 - <<'P'
import json
for l in open("results/modal_results.jsonl"):
    r=json.loads(l); print(r['cfg']['name'],'base',round(r['base_acc'],3),'acc',round(r['acc'],3),'| bias: base',round(r['base_acc_bias'] or 0,3),'q',round(r['acc_bias'] or 0,3),'agree',round(r['agree'],3),'ratio',round(r['ratio_bf16'],4))
P
