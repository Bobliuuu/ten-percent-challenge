import json,sys
from safetensors import safe_open
from safetensors.torch import save_file
d=sys.argv[1]; idx=json.load(open(d+"/model.safetensors.index.json")); wm=idx["weight_map"]
bad=[k for k in wm if k.startswith("vision_tower")]; print(len(bad))
for fn in sorted(set(wm[k] for k in bad)):
    t={}
    with safe_open(d+"/"+fn,"pt") as g:
        for k in g.keys():
            if not k.startswith("vision_tower"): t[k]=g.get_tensor(k)
    save_file(t,d+"/"+fn,metadata={"format":"pt"})
for k in bad: del wm[k]
json.dump(idx,open(d+"/model.safetensors.index.json","w")); print("fixed")
