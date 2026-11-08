timeout 40 modal app logs ap-8jd8k1x8XtUuX3XLtyMGpq 2>&1 | grep -E "Traceback|Error|RESULT_JSON" | python3 -c "
import sys,json
for l in sys.stdin:
    if 'RESULT_JSON' in l and 'faith' in l:
        d=json.loads(l.split('RESULT_JSON ',1)[1]); print(d['name'],d['K'],round(d['ratio'],4),'acc',round(d['acc'],4),'faith',round(d['faith'],4),{k:round(v,3) for k,v in d['faith_kind'].items()},{k:round(v,3) for k,v in d['faith_src'].items()})
    elif 'Traceback' in l or 'Error' in l: print(l[:300])
"
