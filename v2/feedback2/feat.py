"""Feature extraction for feedback2 (no-RAG feedback). Frozen LM, chat prompt, letter logits + last-token hiddens.
modal run --detach feat.py --model Qwen/Qwen2.5-1.5B-Instruct --tag q15
Writes /vol/f2/<tag>_<set>.npz to Volume tenpct-fb2 and prints RESULT_JSON."""
import modal, json
app = modal.App("tenpct-fb2-feat")
R="/Users/jhome/ten-percent/v2/"
img = (modal.Image.debian_slim().pip_install("torch","transformers==4.46.3","numpy","accelerate")
       .add_local_file(R+"alt/pool.jsonl","/root/a_pool.jsonl").add_local_file(R+"alt/dev.jsonl","/root/a_dev.jsonl")
       .add_local_file(R+"alt/calib.jsonl","/root/a_calib.jsonl")
       .add_local_file(R+"base/pool.jsonl","/root/m_pool.jsonl").add_local_file(R+"base/dev.jsonl","/root/m_dev.jsonl")
       .add_local_file(R+"base/calib.jsonl","/root/m_calib.jsonl"))
vol = modal.Volume.from_name("tenpct-fb2", create_if_missing=True)
hf = modal.Volume.from_name("hf-cache", create_if_missing=True)
HOLD={"ag_news":3,"dbpedia":12,"emotion":2}   # held-out (new) label per task
@app.function(image=img, gpu="A10G", timeout=14400, volumes={"/hf":hf,"/vol":vol})
def run(model="Qwen/Qwen2.5-1.5B-Instruct", tag="q15"):
    import os, time, numpy as np, torch
    os.environ["HF_HOME"]="/hf"
    from transformers import AutoTokenizer, AutoModelForCausalLM
    t0=time.time()
    tok=AutoTokenizer.from_pretrained(model); tok.padding_side="left"
    lm=AutoModelForCausalLM.from_pretrained(model,torch_dtype=torch.bfloat16).cuda().eval()
    NL=lm.config.num_hidden_layers; LAY=[NL//2,(3*NL)//4,NL]
    LID=[tok.encode(l)[0] for l in "ABCDEFGHIJKLMNO"]
    ld=lambda f:[json.loads(l) for l in open(f)]
    def chat(u): return tok.apply_chat_template([{"role":"user","content":u}],tokenize=False,add_generation_prompt=True)
    def mdc_prompt(r,hold=None):
        names=list(r["names"]); idx=list(range(len(names)))
        if hold is not None: idx=[i for i in idx if i!=hold]
        opts=[names[i] for i in idx]+(["Other / none of these"] if hold is not None else [])
        u=r["inst"]+"\n\nText: "+r["text"]+"\n\nOptions:\n"+"".join(f"{'ABCDEFGHIJKLMNO'[j]}. {o}\n" for j,o in enumerate(opts))+"Answer with the letter only."
        return chat(u), idx
    def mmlu_prompt(r):
        u=f"The following is a multiple choice question about {r['subject'].replace('_',' ')}.\n\n{r['question']}\n"+"".join(f"{l}. {c}\n" for l,c in zip("ABCD",r['choices']))+"Answer with the letter only."
        return chat(u)
    @torch.no_grad()
    def embed(prompts,nl,bs=16):
        N=len(prompts); order=np.argsort([len(p) for p in prompts]); D=lm.config.hidden_size
        L=np.full((N,15),-1e4,np.float32); H=np.zeros((N,len(LAY),D),np.float16); M=np.zeros((N,D),np.float16)
        for i in range(0,N,bs):
            idx=order[i:i+bs]; enc=tok([prompts[j] for j in idx],return_tensors="pt",padding=True,truncation=True,max_length=900).to("cuda")
            o=lm.model(**enc,output_hidden_states=True); hs=o.hidden_states
            lg=(o.last_hidden_state[:,-1].float()@lm.lm_head.weight[LID].float().T).cpu().numpy()
            m=enc.attention_mask.unsqueeze(-1).float(); mm=((hs[NL//2].float()*m).sum(1)/m.sum(1)).cpu().numpy()
            hh=torch.stack([hs[l][:,-1].float() for l in LAY],1).cpu().numpy()
            for a,j in enumerate(idx):
                n=nl[j]; L[j,:n]=lg[a,:n]; H[j]=hh[a]; M[j]=mm[a]
        return L,H,M
    summ={}
    def save(name,rows,prompts,nl,extra):
        L,H,M=embed(prompts,nl); y=np.array([e for e in extra["y"]])
        np.savez(f"/vol/f2/{tag}_{name}.npz",L=L,H=H,M=M,**{k:np.array(v) for k,v in extra.items()})
        summ[name]=dict(n=len(rows),acc=float((L.argmax(1)==y).mean()),secs=time.time()-t0); print(name,summ[name],flush=True)
    os.makedirs("/vol/f2",exist_ok=True)
    for sp in ("dev","pool","calib"):
        rows=ld(f"/root/a_{sp}.jsonl")
        P=[mdc_prompt(r)[0] for r in rows]; nl=[len(r["names"]) for r in rows]
        save(f"mdc_{sp}",rows,P,nl,dict(y=[r["answer"] for r in rows],task=[r["task"] for r in rows]))
        for tn,h in HOLD.items():
            rr=[r for r in rows if r["task"]==tn]; PP=[];nn=[]
            for r in rr:
                p,idx=mdc_prompt(r,h); PP.append(p); nn.append(len(idx)+1)
            # y in held-out label space: position in remaining list, or Other (= n-1)
            y=[]
            for r in rr:
                idx=[i for i in range(len(r["names"])) if i!=h]; y.append(idx.index(r["answer"]) if r["answer"]!=h else len(idx))
            save(f"hold_{tn}_{sp}",rr,PP,nn,dict(y=y,ytrue=[r["answer"] for r in rr]))
        vol.commit()
    for sp in ("dev","pool","calib"):
        rows=ld(f"/root/m_{sp}.jsonl"); save(f"mmlu_{sp}",rows,[mmlu_prompt(r) for r in rows],[4]*len(rows),dict(y=[r["answer"] for r in rows],subj=[r["subject"] for r in rows]))
        vol.commit()
    print("RESULT_JSON",json.dumps(summ))

@app.local_entrypoint()
def main(model:str="Qwen/Qwen2.5-1.5B-Instruct",tag:str="q15"):
    run.remote(model,tag)
