"""Forward pass from the PACKED artifact only (sign/trit codes + scales, dequantized on the fly to the exact fp16 values).
usage: probe_packed.py <orig_dir(config+tokenizer)> <art_dir> <mode 1bit|tern> <name>  [env LIMIT, BS, MNT, REF=1 to use original fp16 weights instead]"""
import json,sys,os,time,glob,torch,numpy as np,torch.nn as nn,torch.nn.functional as F
sys.path.insert(0,"/opt/dlami/nvme/work/nat")
from scorer import score_item
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
orig,art,mode,name=sys.argv[1:5]; G=128
LIMIT=int(os.environ.get("LIMIT","100000")); BS=int(os.environ.get("BS","16")); MNT=int(os.environ.get("MNT","400"))
items=[json.loads(l) for l in open("/opt/dlami/nvme/work/nat/probe_items.jsonl")][:LIMIT]
tok=AutoTokenizer.from_pretrained(orig); tok.padding_side="left"
prompts=[tok.apply_chat_template([{"role":"user","content":it["query"]}],tools=[{"type":"function","function":t} for t in it["tools"]],add_generation_prompt=True,tokenize=False,enable_thinking=False) for it in items]
SH=torch.arange(8,device="cuda")
def deq(codes,scale):   # codes uint8 [N,B], scale fp16 [N] -> fp16 [N,128]
    if mode=="1bit":
        b=((codes[:,:,None]>>(7-SH))&1).reshape(codes.shape[0],G); v=(b.half()*2-1)
    else:
        c=((codes[:,:,None]>>(2*(3-SH[:4])))&3).reshape(codes.shape[0],G); v=(c.half()-1)
    return v*scale[:,None]
class PL(nn.Module):
    def __init__(s,codes,scale,shape): super().__init__(); s.codes=codes; s.scale=scale; s.shape=shape
    def weight(s): return deq(s.codes,s.scale).reshape(s.shape)
    def forward(s,x): return F.linear(x,s.weight())
class PE(nn.Module):
    def __init__(s,codes,scale,shape): super().__init__(); s.codes=codes; s.scale=scale; s.shape=shape; s.pg=shape[1]//G
    def forward(s,ids):
        V,D=s.shape; B=s.codes.shape[1]
        c=s.codes.view(V,s.pg,B)[ids.reshape(-1)].reshape(-1,B); sc=s.scale.view(V,s.pg)[ids.reshape(-1)].reshape(-1)
        return deq(c,sc).reshape(*ids.shape,D)
cfg=AutoConfig.from_pretrained(orig)
if os.environ.get("REF"):
    model=AutoModelForCausalLM.from_pretrained(orig,torch_dtype=torch.float16,attn_implementation=os.environ.get("ATTN","sdpa")).cuda().eval()
else:
    with torch.device("meta"): model=AutoModelForCausalLM.from_config(cfg,torch_dtype=torch.float16,attn_implementation=os.environ.get("ATTN","sdpa"))
    small=torch.load(art+"/small.pt"); index=json.load(open(art+"/index.json"))
    bits=torch.from_numpy(np.fromfile(art+"/bits.bin",dtype=np.uint8)).cuda(); sc=torch.from_numpy(np.fromfile(art+"/scales.bin",dtype=np.float16)).cuda()
    B=16 if mode=="1bit" else 32
    model=model.to_empty(device="cuda")
    sd=model.state_dict(); ob=os_=0
    for k,t in small.items():
        if k in sd: sd[k].copy_(t.cuda())
    mods=dict(model.named_modules()); tied=[]
    for k,shape in index:
        n=shape[0]*shape[1]; ng=n//G
        c=bits[ob:ob+ng*B].view(ng,B); s=sc[os_:os_+ng]; ob+=ng*B; os_+=ng
        mname=k[:-len(".weight")]; pp,_,child=mname.rpartition("."); parent=model.get_submodule(pp)
        setattr(parent,child,(PE if "embed_tokens" in mname else PL)(c,s,tuple(shape)))
    model.config.tie_word_embeddings=False
    # rotary buffers were emptied by to_empty: recompute
    from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding
    model.model.rotary_emb=Qwen3RotaryEmbedding(config=cfg).cuda()
    del sd,mods; import gc; gc.collect(); torch.cuda.empty_cache(); model.eval()
if os.environ.get("CMP"):
    ref=AutoModelForCausalLM.from_pretrained(orig,torch_dtype=torch.float16,attn_implementation="sdpa").cuda().eval()
    Wp=model.lm_head.weight(); print("DBG head weight equal",torch.equal(Wp,ref.lm_head.weight),Wp.shape,Wp.stride(),ref.lm_head.weight.stride(),Wp.dtype,ref.lm_head.weight.dtype,flush=True)
    H1={};H2={}
    for nm,m in model.named_modules():
        if (nm in ("model.norm","lm_head") or (nm.startswith("model.layers.") and nm.count(".")==2)) and not nm=="" :
            m.register_forward_hook((lambda nm: lambda mod,i,o: H1.__setitem__(nm,o))(nm))
    for nm,m in ref.named_modules():
        if (nm in ("model.norm","lm_head") or (nm.startswith("model.layers.") and nm.count(".")==2)) and not nm=="":
            m.register_forward_hook((lambda nm: lambda mod,i,o: H2.__setitem__(nm,o))(nm))
    mx=0;eq=0;n=0
    for b in range(0,int(os.environ["CMP"]),2):
        enc=tok(prompts[b:b+2],return_tensors="pt",padding=True,add_special_tokens=False).to("cuda")
        with torch.no_grad(): a=model(**enc,logits_to_keep=1).logits; r=ref(**enc,logits_to_keep=1).logits
        if b==0:
            for nm in H1:
                x,y=H1[nm],H2.get(nm)
                if isinstance(x,tuple): x,y=x[0],y[0]
                print("HOOK",nm,torch.equal(x,y) if y is not None else None,flush=True)
        if b==0:
            xin=H2["model.norm"]; import torch.nn.functional as F
            o1=F.linear(xin,ref.lm_head.weight); o2=F.linear(xin,Wp); o3=model.lm_head(xin); o4=ref.lm_head(xin)
            xs=H2["model.norm"][:, -1:]
            print("DBG4 sliced: pk-vs-ref",torch.equal(model.lm_head(xs),ref.lm_head(xs)),"contig:",torch.equal(model.lm_head(xs.contiguous()),ref.lm_head(xs.contiguous())),"a-vs-pk",torch.equal(a,model.lm_head(xs)),"r-vs-ref",torch.equal(r,ref.lm_head(xs)),"norm_eq",torch.equal(H1["model.norm"],H2["model.norm"]),flush=True)
            print("DBG3",torch.equal(H1["lm_head"],H2["lm_head"]),torch.equal(a,r),(H1["lm_head"].float()-H2["lm_head"].float()).abs().max().item(),flush=True); print("DBG2 ref-vs-deq",torch.equal(o1,o2),"ref-vs-reflayer",torch.equal(o1,o4),"packedlayer-vs-ref",torch.equal(o3,o4),"hookout-vs",torch.equal(H1["lm_head"],o3),torch.equal(H2["lm_head"],o4),xin.shape,H1["lm_head"].shape,flush=True)
        eq+=int(torch.equal(a,r)); n+=1; mx=max(mx,(a.float()-r.float()).abs().max().item())
        print("batch",b,"equal",torch.equal(a,r),"maxabs",(a.float()-r.float()).abs().max().item(),flush=True)
    print("LOGITS_EQUAL_BATCHES",eq,"of",n,"maxabs",mx); sys.exit()
order=sorted(range(len(items)),key=lambda i:len(prompts[i])); outs={}; t0=time.time(); lg={}
for b in range(0,len(order),BS):
    idx=order[b:b+BS]
    enc=tok([prompts[i] for i in idx],return_tensors="pt",padding=True,add_special_tokens=False).to("cuda")
    with torch.no_grad():
        g=model.generate(**enc,max_new_tokens=MNT,do_sample=False,pad_token_id=tok.pad_token_id or tok.eos_token_id)
    for j,i in enumerate(idx): outs[i]=tok.decode(g[j,enc.input_ids.shape[1]:],skip_special_tokens=False)
    print(b,time.time()-t0,flush=True)
rows=[]
for i,it in enumerate(items):
    txt=outs[i].split("<|im_end|>")[0].replace("<|endoftext|>",""); s=score_item(txt,it)
    rows.append(dict(id=it["id"],source=it["source"],kind=it["kind"],correct=bool(s["correct"]),parse_fail=bool(s["parse_fail"]),n_calls=s["n_calls"],finish="stop",output=outs[i]))
json.dump(dict(model=name,path=art,secs=time.time()-t0,prompt_example=prompts[0],rows=rows),open(f"/opt/dlami/nvme/work/nat/res_{name}.json","w"))
print("DONE",name,sum(r["correct"] for r in rows)/len(rows),time.time()-t0)
