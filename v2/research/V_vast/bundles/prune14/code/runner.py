import json,sys,os
here=os.path.dirname(os.path.abspath(__file__)); data=os.path.join(here,"..","data"); res=os.path.join(here,"..","results")
os.makedirs(res+"/dense",exist_ok=True)
os.environ.update(HF_HOME="/workspace/hf",DENSE_HF=res,DENSE_DATA=data+"/dense_data.json",CT_ITEMS=data+"/items2.jsonl",CT_CALIB=data+"/calib2.jsonl",DENSE_GPU="1",HF_HUB_ENABLE_HF_TRANSFER="0")
sys.path.insert(0,here); import dense_core
jobs=json.load(open(data+"/jobs.json")); sel=[int(x) for x in sys.argv[1].split(",")]
for k in sel:
    job=jobs[k]
    if job.get("injr"): os.environ.update(CT_ITEMS=data+"/injr_items.jsonl",CT_CALIB=data+"/injr_calib.jsonl")
    if job.get("news"): os.environ.update(CT_ITEMS=data+"/news_items.jsonl",CT_CALIB=data+"/news_calib.jsonl")
    cfgs=[dict(model=job["model"],ctasks=1,calib_ct=1,**c) for c in job["cfgs"]]
    r=dense_core._run(dict(multi=cfgs)); print("JOB_DONE",k,[x for x in r if isinstance(x,dict) and "error" in x],flush=True)
open(res+"/DONE","w").write("ok")
