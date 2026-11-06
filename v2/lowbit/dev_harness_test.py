"""DEV-only test of the artifact directory through the harness contract (worker.py logic), no network, no base weights.
TAGS=a,b modal run --detach dev_harness_test.py   (never touches locked files)"""
import modal, json, os
app = modal.App("tenpct-lowbit-devtest")
img = (modal.Image.debian_slim().pip_install("torch","transformers==4.57.1","numpy","safetensors","constriction","accelerate")
       .add_local_file("mmlu_dev.jsonl","/root/mmlu_dev.jsonl").add_local_file("mdc_dev.jsonl","/root/mdc_dev.jsonl")
       .add_local_dir("artifact_code","/root/code").add_local_dir("../harness","/root/harness",ignore=["__pycache__","receipts","locks","tests"]))
vol = modal.Volume.from_name("lowbit-data", create_if_missing=True)
@app.function(image=img, gpu=["A100-40GB","A100-80GB","H100"], cpu=8, memory=65536, timeout=7200, volumes={"/d":vol}, block_network=True)
def run(tag, base_bytes=4825641380):
    import sys, shutil, time, socket, glob, importlib.util, subprocess, numpy as np
    A=f"/tmp/art_{tag}"; shutil.rmtree(A,ignore_errors=True); shutil.copytree(f"/d/art/{tag}",A,ignore=shutil.ignore_patterns("*.npy","result.json"))
    shutil.copy("/root/code/predict.py",A+"/predict.py")
    sys.path.insert(0,"/root/harness"); import size_check
    sz=size_check.measure(A,None,base_bytes); print("SIZE",json.dumps({k:v for k,v in sz.items() if k!="files"}),flush=True)
    ev={}
    try: socket.create_connection(("1.1.1.1",443),timeout=5).close(); ev["network"]="REACHABLE"
    except Exception as e: ev["network"]="blocked ("+type(e).__name__+")"
    ev["hf_cache_files"]=len(glob.glob(os.path.expanduser("~/.cache/huggingface/**/*"),recursive=True))
    spec=importlib.util.spec_from_file_location("ap",A+"/predict.py"); mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    t0=time.time(); f=mod.load(A,None); tl=time.time()-t0
    res=dict(tag=tag,size=dict(total=sz["candidate_bytes"],ratio=sz["byte_ratio"],within=sz["within_budget"],n_files=sz["n_files"],files=sz["files"]),evidence=ev,load_secs=tl)
    M="microsoft/bitnet-b1.58-2B-4T"
    for name,fl in (("mmlu","/root/mmlu_dev.jsonl"),("mdc","/root/mdc_dev.jsonl")):
        rows=[json.loads(x) for x in open(fl)]
        view=[{k:r[k] for k in r if k!="answer"} for r in rows]
        pred=f(view); L="ABCDEFGHIJKLMN"; y=[L[r["answer"]] for r in rows]
        acc=float(np.mean([p==t for p,t in zip(pred,y)]))
        base=np.load(f"/d/{M.replace('/','_')}.{name}.npy").argmax(1)
        agree=float(np.mean([p==L[b] for p,b in zip(pred,base)])); bacc=float(np.mean([L[b]==t for b,t in zip(base,y)]))
        res[name]=dict(acc=acc,base_packed_acc=bacc,agree=agree,n=len(rows))
    print("RESULT_JSON "+json.dumps(res)); 
    os.makedirs("/d/devtest",exist_ok=True); json.dump(res,open(f"/d/devtest/{tag}.json","w")); vol.commit(); return res
@app.local_entrypoint()
def main():
    for t in os.environ["TAGS"].split(","): run.spawn(t)
