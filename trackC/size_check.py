#!/usr/bin/env python
"""Size gate: true on-disk bytes of artifact (+ memory file) vs 10% of base fp32 safetensors.
Usage: size_check.py ARTIFACT [--memory FILE] [--base-safetensors F] [--json]"""
import argparse, json, os, struct, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

BASE_PARAMS_FALLBACK = None


def safetensors_numel(p):
    with open(p, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        h = json.loads(f.read(n))
    return sum(int(np.prod(v["shape"])) if v["shape"] else 1 for k, v in h.items() if k != "__metadata__")


def numel(p):
    """Stored tensor elements in a weights file (0 if not a recognised tensor format)."""
    e = os.path.splitext(p)[1].lower()
    try:
        if e == ".safetensors": return safetensors_numel(p)
        if e == ".npy": return int(np.load(p, mmap_mode="r", allow_pickle=False).size)
        if e == ".npz":
            z = np.load(p, allow_pickle=False); return int(sum(z[k].size for k in z.files))
        if e in (".pt", ".pth", ".bin"):
            import torch
            o = torch.load(p, map_location="cpu", weights_only=True)
            if isinstance(o, dict): return int(sum(v.numel() for v in o.values() if hasattr(v, "numel")))
    except Exception as ex:
        print(f"warn: cannot count params in {p}: {ex}", file=sys.stderr)
    return 0


def measure(artifact, memory=None, base_st=None):
    base_st = base_st or C.find_base_safetensors()
    base_bytes = os.path.getsize(base_st)
    base_params = safetensors_numel(base_st)
    files = C.tree_files(artifact) + ([("MEMORY:" + os.path.basename(memory), os.path.abspath(memory))] if memory else [])
    rows, apparent, alloc, params = [], 0, 0, 0
    for rel, ap in files:
        st = os.stat(ap)
        k = 0 if rel.startswith("MEMORY:") else numel(ap)
        rows.append(dict(file=rel, bytes=st.st_size, allocated_bytes=st.st_blocks * 512, tensor_elems=k))
        apparent += st.st_size; alloc += st.st_blocks * 512; params += k
    budget = C.BUDGET_FRACTION * base_bytes
    return dict(
        base_safetensors=base_st, base_bytes=base_bytes, base_params=base_params,
        budget_bytes=budget, candidate_bytes=apparent, candidate_allocated_bytes=alloc,
        byte_ratio=apparent / base_bytes, allocated_ratio=alloc / base_bytes,
        param_ratio=params / base_params, candidate_tensor_elems=params,
        memory_bytes=os.path.getsize(memory) if memory else 0,
        within_10pct=bool(apparent <= budget), files=rows,
        note="byte_ratio (apparent st_size of every file incl. code, tokenizer, metadata, memory) is binding; "
             "param_ratio counts stored tensor elements only and ignores bit-width.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("artifact"); ap.add_argument("--memory"); ap.add_argument("--base-safetensors"); ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    r = measure(a.artifact, a.memory, a.base_safetensors)
    if a.json:
        print(json.dumps(r, indent=2))
    else:
        for f in r["files"]: print(f"  {f['bytes']:>12,d} B  {f['file']}")
        print(f"candidate  : {r['candidate_bytes']:,d} B (allocated {r['candidate_allocated_bytes']:,d} B), memory {r['memory_bytes']:,d} B")
        print(f"base fp32  : {r['base_bytes']:,d} B   budget(10%): {r['budget_bytes']:,.0f} B")
        print(f"byte ratio : {100*r['byte_ratio']:.3f}%   param ratio: {100*r['param_ratio']:.3f}% ({r['candidate_tensor_elems']:,d}/{r['base_params']:,d})")
        print("RESULT     :", "PASS (<=10%)" if r["within_10pct"] else "FAIL (>10%)")
    sys.exit(0 if r["within_10pct"] else 1)
