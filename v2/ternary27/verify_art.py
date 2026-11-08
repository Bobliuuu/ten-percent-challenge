import sys, json, torch
from safetensors import safe_open
a, b = sys.argv[1], sys.argv[2]
wa = json.load(open(a + "/model.safetensors.index.json"))["weight_map"]; wb = json.load(open(b + "/model.safetensors.index.json"))["weight_map"]
miss = [k for k in wb if k not in wa and not k.startswith("model.visual")]; print("missing_in_loader", miss[:5], len(miss))
nbad = 0; maxd = 0.0; n = 0; byfile = {}
for k, fn in wa.items(): byfile.setdefault(fn, []).append(k)
for fn, ks in byfile.items():
    with safe_open(a + "/" + fn, "pt") as fa:
        for k in ks:
            if k not in wb: print("extra", k); continue
            with safe_open(b + "/" + wb[k], "pt") as fb:
                x = fa.get_tensor(k).float(); y = fb.get_tensor(k).float()
            n += 1
            if x.shape != y.shape: print("shape", k, x.shape, y.shape); nbad += 1; continue
            d = float((x - y).abs().max()); maxd = max(maxd, d)
            if d > 0: nbad += 1; print("diff", k, d) if nbad < 10 else None
print("VERIFY tensors", n, "differing", nbad, "maxdiff", maxd)
