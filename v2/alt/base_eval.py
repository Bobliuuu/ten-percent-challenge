"""Score base models on MDC-6 DEV (zero-shot letter-logit). Never touches locked.jsonl.
Run: MODEL=Qwen/Qwen2.5-3B-Instruct modal run --detach base_eval.py   -> RESULT_JSON in app logs; also saves dev logits to Volume alt-data:/base/<model>.npy"""
import modal, json, os
app = modal.App("tenpct-alt-base")
img = (modal.Image.debian_slim().pip_install("torch","transformers==4.51.3","numpy","accelerate")
       .add_local_file("dev.jsonl","/root/dev.jsonl"))
vol = modal.Volume.from_name("alt-data", create_if_missing=True)
hf = modal.Volume.from_name("hf-cache", create_if_missing=True)
def fmt(r):
    s=r["inst"]+"\n\nText: "+r["text"]+"\n\nOptions:\n"
    for l,c in zip("ABCDEFGHIJKLMN",r["names"]): s+=f"{l}. {c}\n"
    return s+"Answer:"
@app.function(image=img, gpu="A100-40GB", timeout=3600, volumes={"/hf":hf,"/d":vol})
def run(M):
    import torch, numpy as np, time, collections
    os.environ["HF_HOME"]="/hf"
    from transformers import AutoTokenizer, AutoModelForCausalLM
    t0=time.time(); tok=AutoTokenizer.from_pretrained(M)
    model=AutoModelForCausalLM.from_pretrained(M,torch_dtype=torch.bfloat16).cuda().eval()
    n_params=sum(p.numel() for p in model.parameters())
    ids=[tok.encode(" "+l)[0] for l in "ABCDEFGHIJKLMN"]
    rows=[json.loads(x) for x in open("/root/dev.jsonl")]
    lg=np.full((len(rows),14),-1e9,dtype=np.float32)
    with torch.no_grad():
        for i,r in enumerate(rows):
            x=torch.tensor([tok.encode(fmt(r))[-1024:]]).cuda()
            h=model.model(input_ids=x).last_hidden_state[0,-1]
            full=(h@model.get_output_embeddings().weight[ids].T).float().cpu().numpy()
            k=len(r["names"]); lg[i,:k]=full[:k]
    y=np.array([r["answer"] for r in rows]); pred=lg.argmax(1)
    per=collections.defaultdict(list)
    for r,p,yy in zip(rows,pred,y): per[r["task"]].append(int(p==yy))
    srt=np.sort(lg,1); margin=(srt[:,-1]-srt[:,-2])
    res=dict(model=M,params=n_params,fp16_bytes=n_params*2,n=len(rows),acc=float((pred==y).mean()),
      macro_acc=float(np.mean([np.mean(v) for v in per.values()])),per_task={k:[float(np.mean(v)),len(v)] for k,v in per.items()},
      margin_mean=float(margin.mean()),margin_p10=float(np.percentile(margin,10)),secs=time.time()-t0)
    os.makedirs("/d/base",exist_ok=True); np.save(f"/d/base/{M.replace('/','_')}.npy",lg); vol.commit()
    print("RESULT_JSON "+json.dumps(res)); return res
@app.local_entrypoint()
def main():
    for m in os.environ["MODEL"].split(","): run.spawn(m)   # detached: containers outlive client; RESULT_JSON in app logs
