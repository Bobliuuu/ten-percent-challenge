"""Score native-ternary models on MMLU dev + MDC-6 dev (letter-logit). Never touches locked files.
MODELS=a,b modal run --detach eval_base.py ; RESULT_JSON in logs + Volume lowbit-data:/base/*.json"""
import modal, json, os
app = modal.App("tenpct-lowbit-base")
img = (modal.Image.debian_slim().pip_install("torch","transformers==4.57.1","numpy","accelerate","sentencepiece","protobuf")
       .add_local_file("mmlu_dev.jsonl","/root/mmlu_dev.jsonl").add_local_file("mdc_dev.jsonl","/root/mdc_dev.jsonl"))
vol = modal.Volume.from_name("lowbit-data", create_if_missing=True)
hf = modal.Volume.from_name("hf-cache", create_if_missing=True)
def fmt_mmlu(r):
    s="The following are multiple choice questions (with answers) about "+r["subject"].replace("_"," ")+".\n\n"+r["question"]+"\n"
    for l,c in zip("ABCD",r["choices"]): s+=f"{l}. {c}\n"
    return s+"Answer:"
def fmt_mdc(r):
    s=r["inst"]+"\n\nText: "+r["text"]+"\n\nOptions:\n"
    for l,c in zip("ABCDEFGHIJKLMN",r["names"]): s+=f"{l}. {c}\n"
    return s+"Answer:"
@app.function(image=img, gpu="A100-40GB", timeout=7200, volumes={"/hf":hf,"/d":vol})
def run(M):
    import torch, numpy as np, time, traceback
    os.environ["HF_HOME"]="/hf"
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        t0=time.time(); tok=AutoTokenizer.from_pretrained(M)
        model=AutoModelForCausalLM.from_pretrained(M,torch_dtype=torch.bfloat16).cuda().eval()
        n_params=sum(p.numel() for p in model.parameters())
        ids=[tok.encode(" "+l,add_special_tokens=False)[-1] for l in "ABCDEFGHIJKLMN"]
        W=model.get_output_embeddings().weight
        out=dict(model=M,params_loaded=n_params,label_ids=ids,cfg=str(type(model)))
        for name,fn,f in (("mmlu",fmt_mmlu,"/root/mmlu_dev.jsonl"),("mdc",fmt_mdc,"/root/mdc_dev.jsonl")):
            rows=[json.loads(x) for x in open(f)]; K=4 if name=="mmlu" else 14
            lg=np.full((len(rows),K),-1e9,dtype=np.float32)
            with torch.no_grad():
                for i,r in enumerate(rows):
                    x=torch.tensor([tok.encode(fn(r))[-1024:]]).cuda()
                    h=model.model(input_ids=x).last_hidden_state[0,-1]
                    full=(h@W[ids[:K]].T).float().cpu().numpy()
                    k=len(r["choices"] if name=="mmlu" else r["names"]); lg[i,:k]=full[:k]
            y=np.array([r["answer"] for r in rows]); pred=lg.argmax(1)
            out[name+"_acc"]=float((pred==y).mean()); out[name+"_n"]=len(rows)
            np.save(f"/d/{M.replace('/','_')}.{name}.npy",lg)
        out["secs"]=time.time()-t0
    except Exception as e:
        out=dict(model=M,error=repr(e),tb=traceback.format_exc()[-1500:])
    os.makedirs("/d/base",exist_ok=True); json.dump(out,open(f"/d/base/{M.replace('/','_')}.json","w")); vol.commit()
    print("RESULT_JSON "+json.dumps(out)); return out
@app.local_entrypoint()
def main():
    for m in os.environ["MODELS"].split(","): run.spawn(m)
