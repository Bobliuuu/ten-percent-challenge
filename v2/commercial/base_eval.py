"""Score base models on commercial suites A/B/C DEV (never touches locked). Zero-shot and k-shot, letter-logit; plus C generative exact tool-call match (first 300 dev).
Run: MODEL=Qwen/Qwen2.5-1.5B-Instruct,Qwen/Qwen2.5-3B-Instruct,Qwen/Qwen2.5-7B-Instruct modal run --detach base_eval.py -> RESULT_JSON in app logs; logits in Volume commercial-data:/base/"""
import modal, json, os
app = modal.App("tenpct-commercial-base")
img = (modal.Image.debian_slim().pip_install("torch","transformers==4.51.3","numpy","accelerate","scikit-learn")
       .add_local_file("A/dev.jsonl","/root/A_dev.jsonl").add_local_file("B/dev.jsonl","/root/B_dev.jsonl").add_local_file("C/dev.jsonl","/root/C_dev.jsonl")
       .add_local_file("A/calib.jsonl","/root/A_calib.jsonl").add_local_file("B/calib.jsonl","/root/B_calib.jsonl").add_local_file("C/calib.jsonl","/root/C_calib.jsonl")
       .add_local_file("C2/dev.jsonl","/root/C2_dev.jsonl").add_local_file("C2/calib.jsonl","/root/C2_calib.jsonl"))
vol = modal.Volume.from_name("commercial-data", create_if_missing=True)
hf = modal.Volume.from_name("hf-cache", create_if_missing=True)
L="ABCDEFGHIJKLMN"
def block(r, ans=None):
    s=r["inst"]+"\n\nText: "+r["text"]+"\n\nOptions:\n"
    for l,c in zip(L,r["names"]): s+=f"{l}. {c}\n"
    return s+"Answer:"+(f" {L[ans]}" if ans is not None else "")
def shots(T, calib):
    import random
    rng=random.Random(0); by={}
    for r in calib:
        if len(r["text"])<1200: by.setdefault(r["answer"],[]).append(r)
    ks=sorted(by); out=[]
    if T=="A": out=[rng.choice(by[k]) for k in ks]            # one per class
    elif T=="B": out=[rng.choice(by[k]) for k in ks for _ in range(2)]
    else: out=[rng.choice(by[k]) for k in ks[:5]]
    rng.shuffle(out); return out
@app.function(image=img, gpu=["A100-40GB","L40S","A10G","L4"], timeout=7200, volumes={"/hf":hf,"/d":vol})
def run(M, tasks="A,B,C"):
    import torch, numpy as np, time, collections
    os.environ["HF_HOME"]="/hf"
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from sklearn.metrics import roc_auc_score
    t0=time.time(); tok=AutoTokenizer.from_pretrained(M)
    model=AutoModelForCausalLM.from_pretrained(M,torch_dtype=torch.bfloat16).cuda().eval()
    W=model.get_output_embeddings().weight
    ids=[tok.encode(" "+l)[0] for l in L]
    res=dict(model=M,params=sum(p.numel() for p in model.parameters()),tasks={})
    os.makedirs("/d/base",exist_ok=True)
    for T in tasks.split(","):
        rows=[json.loads(x) for x in open(f"/root/{T}_dev.jsonl")]; calib=[json.loads(x) for x in open(f"/root/{T}_calib.jsonl")]
        sh=shots(T,calib); pre="".join(block(s,s["answer"])+"\n\n" for s in sh)
        y=np.array([r["answer"] for r in rows]); out={}
        for mode,prefix in [("zs",""),("fs",pre)]:
            lg=np.full((len(rows),len(L)),-1e9,dtype=np.float32)
            with torch.no_grad():
                for i,r in enumerate(rows):
                    x=torch.tensor([tok.encode(prefix+block(r))[-6000:]]).cuda()
                    h=model.model(input_ids=x).last_hidden_state[0,-1]
                    f=(h@W[ids].T).float().cpu().numpy(); k=len(r["names"]); lg[i,:k]=f[:k]
            pred=lg.argmax(1); m=dict(acc=float((pred==y).mean()),n=len(rows),pred_dist=dict(collections.Counter(pred.tolist())))
            if T=="A":
                m["binary_pii_acc"]=float(((pred==7)==(y==7)).mean())
                m["per_class_recall"]={int(c):float((pred[y==c]==c).mean()) for c in np.unique(y)}
                lang=collections.defaultdict(list)
                for r,p,yy in zip(rows,pred,y): lang[r["lang"]].append(int(p==yy))
                m["per_lang"]={k:float(np.mean(v)) for k,v in lang.items()}
                m["chance_majority"]=float(np.bincount(y).max()/len(y))
            if T=="B":
                d=lg[:,1]-lg[:,0]; m["auroc"]=float(roc_auc_score(y,d))
                dom=collections.defaultdict(list)
                for r,p,yy in zip(rows,pred,y): dom[r["src"].split("_")[0]].append(int(p==yy))
                m["per_domain"]={k:float(np.mean(v)) for k,v in dom.items()}; m["chance_majority"]=float(max(y.mean(),1-y.mean()))
            if T in("C","C2"):
                irr=np.array([r["irrelevant"] for r in rows]); m["acc_relevant"]=float((pred==y)[~irr].mean()); m["acc_irrelevant"]=float((pred==y)[irr].mean()); m["chance_majority"]=float(np.bincount(y).max()/len(y))
            out[mode]=m; np.save(f"/d/base/{T}_{mode}_{M.replace('/','_')}.npy",lg)
        if T=="C":  # generative exact tool-call match
            n=300; ok=0; namok=0; outs=[]
            for r in rows[:n]:
                if r["irrelevant"]: pass
                tools=json.loads(r["tools"]); gold=json.loads(r["gold_calls"])
                msgs=[{"role":"system","content":"You are a function-calling assistant. Available tools (JSON):\n"+json.dumps(tools)+"\nRespond ONLY with a JSON list of tool calls in the form [{\"name\": ..., \"arguments\": {...}}]."},{"role":"user","content":r["text"]}]
                x=tok.apply_chat_template(msgs,add_generation_prompt=True,return_tensors="pt").cuda()
                with torch.no_grad(): g=model.generate(x,max_new_tokens=300,do_sample=False)
                txt=tok.decode(g[0,x.shape[1]:],skip_special_tokens=True).strip()
                if txt.startswith("```"): txt=txt.strip("`").lstrip("json").strip()
                try: pc=json.loads(txt)
                except Exception: pc=None
                norm=lambda cs:sorted(json.dumps({"name":c["name"],"arguments":{k:str(v) for k,v in c.get("arguments",{}).items()}},sort_keys=True) for c in cs)
                try: ok+=int(norm(pc)==norm(gold)); namok+=int(sorted(c["name"] for c in pc)==sorted(c["name"] for c in gold))
                except Exception: pass
            out["gen_exact"]=dict(n=n,exact_call_match=ok/n,name_match=namok/n)
        res["tasks"][T]=out
        print("PARTIAL_JSON "+json.dumps(dict(model=M,task=T,r=out)),flush=True)
    res["secs"]=time.time()-t0; vol.commit()
    print("RESULT_JSON "+json.dumps(res)); return res
@app.local_entrypoint()
def main():
    ms=os.environ["MODEL"].split(",")
    for r in run.map(ms): print("DONE",r["model"])   # detached entrypoint stays alive; RESULT_JSON printed in app logs by each run
