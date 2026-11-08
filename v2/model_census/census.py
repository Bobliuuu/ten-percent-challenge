import json,sys
from huggingface_hub import HfApi, hf_hub_download
from models import LIST
api=HfApi(); out=[]
def cfg(mid):
    try:
        p=hf_hub_download(mid,"config.json"); return json.load(open(p))
    except Exception as e: return None
for mid,kind in LIST:
    r=dict(id=mid,kind=kind)
    try:
        i=api.model_info(mid,files_metadata=True)
        st=i.safetensors
        r["params"]=st.total if st else None; r["dtypes"]=dict(st.parameters) if st else None
        r["repo_bytes"]=sum(s.size or 0 for s in i.siblings if s.rfilename.endswith((".safetensors",".bin",".gguf")))
        r["gated"]=bool(i.gated)
    except Exception as e: r["err"]=type(e).__name__
    c=cfg(mid) if "err" not in r else None
    if c:
        t=c.get("text_config",c)
        r["arch"]=dict(model_type=c.get("model_type"),L=t.get("num_hidden_layers"),d=t.get("hidden_size"),V=t.get("vocab_size"),tie=t.get("tie_word_embeddings",c.get("tie_word_embeddings")),
            experts=t.get("num_experts") or t.get("n_routed_experts") or t.get("num_local_experts"),topk=t.get("num_experts_per_tok"),kv=t.get("num_key_value_heads"),ff=t.get("intermediate_size"))
    out.append(r); print(mid,r.get("params"),r.get("err"),flush=True)
json.dump(out,open("census.json","w"),indent=1)
