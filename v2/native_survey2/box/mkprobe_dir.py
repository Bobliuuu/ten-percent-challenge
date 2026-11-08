"""mkprobe_dir.py LOADED VARIANT : make LOADED probe-able in vLLM (copy tokenizer/config from VARIANT, add vision tensors from VARIANT)."""
import sys, json, shutil, os
from safetensors import safe_open
from safetensors.torch import save_file
L, V = sys.argv[1], sys.argv[2]
for fn in os.listdir(V):
    if fn.endswith((".json", ".jinja", ".txt")) and fn != "model.safetensors.index.json": shutil.copyfile(V + "/" + fn, L + "/" + fn)
vi = json.load(open(V + "/model.safetensors.index.json"))["weight_map"]; li = json.load(open(L + "/model.safetensors.index.json"))
vis = {}
for fn in sorted(set(v for k, v in vi.items() if k.startswith("model.visual"))):
    with safe_open(V + "/" + fn, "pt") as g:
        for k in g.keys():
            if k.startswith("model.visual"): vis[k] = g.get_tensor(k)
save_file(vis, L + "/vision.safetensors", metadata={"format": "pt"}); li["weight_map"].update({k: "vision.safetensors" for k in vis})
json.dump(li, open(L + "/model.safetensors.index.json", "w")); print("MKPROBE ok", len(vis))
