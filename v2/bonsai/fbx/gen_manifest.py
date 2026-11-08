"""tensor_manifest.json = {name: [dtype, shape]} from the safetensors headers of the base snapshot (same format as v2/bonsai/h3/checks.arch_of_dir). python gen_manifest.py SNAPDIR OUT.json"""
import sys, glob, json, struct
out = {}
for f in sorted(glob.glob(sys.argv[1] + "/*.safetensors")):
    with open(f, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]; h = json.loads(fh.read(n))
    for k, v in h.items():
        if k != "__metadata__": out[k] = [v["dtype"], v["shape"]]
json.dump(out, open(sys.argv[2], "w")); print(len(out), "tensors")
