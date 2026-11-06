cd ~/w; for f in out*/*.json; do python3 -c "
import json,sys;r=json.load(open('$f'));print(r['cfg']['name'],'base',round(r['base_acc'],4),'acc',round(r['acc'],4),'agree',round(r['agree'],4),'ratio',round(r['ratio_bf16'],4),'bpw',round(r.get('body_bpw',0),2))"; done
