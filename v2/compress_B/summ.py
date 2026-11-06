import json,subprocess,sys
out=subprocess.run("modal app logs ap-UsSTn86RRIJ8e3GahrSVbH",shell=True,capture_output=True,text=True,timeout=100).stdout
rows=[json.loads(l.split('RESULT_JSON ',1)[1]) for l in out.splitlines() if 'RESULT_JSON ' in l]
json.dump(rows,open('results1.json','w'))
for d in sorted(rows,key=lambda d:d['cfg']['name']):
    print(d['cfg']['name'],{k:round(d[k],4) for k in('base_acc','base_acc_bias','base_auc','acc','acc_bias','auc','agree','ratio_bf16','body_bpw')})
