import json,sys,os,time,torch
sys.path.insert(0,"/opt/dlami/nvme/work/nat")
from scorer import score_item
from transformers import AutoTokenizer, AutoModelForCausalLM
path,name=sys.argv[1],sys.argv[2]; limit=int(os.environ.get("LIMIT","100000"))
items=[json.loads(l) for l in open("/opt/dlami/nvme/work/nat/probe_items.jsonl")][:limit]
tok=AutoTokenizer.from_pretrained(path); tok.padding_side="left"
if tok.pad_token is None: tok.pad_token=tok.eos_token
SYS=("# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>{T}\n</tools>\n\n"
 "For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call>\nIf none of the functions is relevant to the request, do not call any function and answer normally.")
def build(it):
    T="".join("\n"+json.dumps({"type":"function","function":t}) for t in it["tools"])
    msgs=[{"role":"system","content":SYS.replace("{T}",T)},{"role":"user","content":it["query"]}]
    try: return tok.apply_chat_template(msgs,add_generation_prompt=True,tokenize=False)
    except Exception: return tok.apply_chat_template([{"role":"user","content":msgs[0]["content"]+"\n\n"+it["query"]}],add_generation_prompt=True,tokenize=False)
prompts=[build(it) for it in items]
model=AutoModelForCausalLM.from_pretrained(path,torch_dtype=torch.bfloat16).cuda().eval()
EMB=os.environ.get("EMB")
if EMB:
    kind,bits,g=EMB.split(":"); bits=float(bits); g=int(g)
    W=model.get_input_embeddings().weight.data; V,D=W.shape; x=W.float().reshape(-1,g)
    if kind=="asym":
        lv=2**int(bits)-1; mn=x.amin(1,keepdim=True); mx=x.amax(1,keepdim=True); sc=((mx-mn)/lv).clamp_min(1e-8)
        q=((x-mn)/sc).round().clamp(0,lv)*sc+mn
    elif kind=="sym":
        lv=2**(int(bits)-1)-1; sc=(x.abs().amax(1,keepdim=True)/lv).clamp_min(1e-8); q=(x/sc).round().clamp(-lv,lv)*sc
    elif kind=="tern":
        sc=x.abs().mean(1,keepdim=True).clamp_min(1e-8); q=(x/sc).round().clamp(-1,1)*sc
    elif kind=="bin":
        sc=x.abs().mean(1,keepdim=True); q=torch.where(x>=0,sc,-sc)
    err=((q-x)**2).sum()/ (x**2).sum(); print("EMB",EMB,"rel_mse",err.item(),flush=True)
    W.copy_(q.reshape(V,D).to(W.dtype))
    if model.lm_head.weight.data_ptr()!=W.data_ptr(): print("WARN untied head",flush=True)
order=sorted(range(len(items)),key=lambda i:len(prompts[i])); outs={}
t0=time.time(); BS=8
for b in range(0,len(order),BS):
    idx=order[b:b+BS]
    enc=tok([prompts[i] for i in idx],return_tensors="pt",padding=True,add_special_tokens=False).to("cuda")
    with torch.no_grad(): g=model.generate(**enc,max_new_tokens=int(os.environ.get("MNT","300")),do_sample=False,pad_token_id=tok.pad_token_id)
    for j,i in enumerate(idx): outs[i]=tok.decode(g[j,enc.input_ids.shape[1]:],skip_special_tokens=False)
rows=[]
for i,it in enumerate(items):
    txt=outs[i].replace("<|eot_id|>","").replace("<|im_end|>","").replace("<|end_of_text|>",""); s=score_item(txt,it)
    rows.append(dict(id=it["id"],source=it["source"],kind=it["kind"],correct=bool(s["correct"]),parse_fail=bool(s["parse_fail"]),n_calls=s["n_calls"],finish="stop",output=outs[i]))
json.dump(dict(model=name,path=path,secs=time.time()-t0,prompt_example=prompts[0],rows=rows),open(f"/opt/dlami/nvme/work/nat/res_{name}.json","w"))
print("DONE",name,sum(r["correct"] for r in rows)/len(rows),time.time()-t0)
