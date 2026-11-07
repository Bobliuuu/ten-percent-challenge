import json,sys,os,time
sys.path.insert(0,"/opt/dlami/nvme/work/nat")
from scorer import score_item
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
path,name=sys.argv[1],sys.argv[2]
kw=json.loads(sys.argv[3]) if len(sys.argv)>3 else {}
items=[json.loads(l) for l in open("/opt/dlami/nvme/work/nat/probe_items.jsonl")]
tok=AutoTokenizer.from_pretrained(path)
prompts=[tok.apply_chat_template([{"role":"user","content":it["query"]}],tools=[{"type":"function","function":t} for t in it["tools"]],add_generation_prompt=True,tokenize=False,**kw) for it in items]
llm=LLM(path,max_model_len=16000,gpu_memory_utilization=float(os.environ.get("GMU","0.3")),enable_prefix_caching=True,seed=0,dtype="float16",enforce_eager=True)
t0=time.time()
outs=llm.generate(prompts,SamplingParams(temperature=0,max_tokens=1024,skip_special_tokens=False),use_tqdm=False)
rows=[]
for it,o in zip(items,outs):
    txt=o.outputs[0].text; s=score_item(txt,it)
    rows.append(dict(id=it["id"],source=it["source"],kind=it["kind"],correct=bool(s["correct"]),parse_fail=bool(s["parse_fail"]),n_calls=s["n_calls"],finish=o.outputs[0].finish_reason,output=txt))
json.dump(dict(model=name,path=path,kw=kw,secs=time.time()-t0,prompt_example=prompts[0],rows=rows),open(f"/opt/dlami/nvme/work/nat/res_{name}.json","w"))
print("DONE",name,sum(r["correct"] for r in rows)/len(rows),time.time()-t0)
