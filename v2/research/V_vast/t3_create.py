import json,subprocess,sys,time,datetime as dt
label,minv,maxdph=sys.argv[1],int(sys.argv[2]),float(sys.argv[3])
q=f"num_gpus=1 gpu_ram>={minv} gpu_ram<=100 reliability>0.97 inet_down>=300 disk_space>=200 rentable=true compute_cap>=800 cuda_vers>=12.8"
offers=[o for o in json.loads(subprocess.run(["vastai","search","offers",q,"-o","dph","--raw"],capture_output=True,text=True).stdout) if "CMP" not in o["gpu_name"] and o["dph_total"]<=maxdph]
for o in offers[:30]:
    r=subprocess.run(["vastai","create","instance",str(o["id"]),"--image","pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel","--disk","200","--label","tenpct-"+label,"--ssh","--direct","--cancel-unavail","--raw"],capture_output=True,text=True)
    try: j=json.loads(r.stdout)
    except Exception: continue
    if j.get("success"):
        line=f"- {dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%dT%H:%MZ')} vast id {j['new_contract']} label tenpct-{label} GPU {o['gpu_name']} ${o['dph_total']:.3f}/h started"
        open("/Users/jhome/ten-percent/RUNS.md","a").write(line+"\n"); print(line); sys.exit(0)
sys.exit(1)
