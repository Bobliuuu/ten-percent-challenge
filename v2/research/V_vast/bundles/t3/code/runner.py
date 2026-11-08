import json,sys,os
here=os.path.dirname(os.path.abspath(__file__)); data=os.path.join(here,"..","data"); res=os.path.join(here,"..","results")
os.makedirs(res+"/dense",exist_ok=True)
os.environ.update(HF_HOME="/workspace/hf",DENSE_HF=res,DENSE_DATA=data+"/dense_data.json",CT_ITEMS=data+"/items.jsonl",CT_CALIB=data+"/calib.jsonl",DENSE_GPU="1",HF_HUB_ENABLE_HF_TRANSFER="0")
sys.path.insert(0,here); import dense_core
jobs=json.load(open(data+"/jobs.json")); sel=[int(x) for x in sys.argv[1].split(",")]
for k in sel:
    job=jobs[k]
    if os.environ.get("DENSE_GPU") and not job.get("gpu_load",True): os.environ.pop("DENSE_GPU")
    cfgs=[dict(model=job["model"],ctasks=1,calib_ct=1,gen=True,**c) if not c["name"].endswith("bf16") else dict(model=job["model"],name=c["name"],mode="ctbase",gen=True,gen_bs=c.get("gen_bs",32),ct_bs=c.get("ct_bs",8)) for c in job["cfgs"]]
    r=dense_core._run(dict(multi=cfgs)); print("JOB_DONE",k,[x for x in r if isinstance(x,dict) and "error" in x],flush=True)
open(res+"/DONE","w").write("ok")
