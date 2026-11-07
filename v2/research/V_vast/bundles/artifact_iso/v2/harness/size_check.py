#!/usr/bin/env python
"""Size gate. Budget = BUDGET_FRACTION x (FP16 bytes of the base = 2 x params), recorded in LOCK.json.
EVERY file of the artifact directory counts (weights, code, tokenizer, metadata, codebooks) plus the memory file.
Usage: size_check.py ARTIFACT [--memory F] (--lock LOCK.json | --base-fp16-bytes N) [--json]"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C


def measure(artifact, memory, base_fp16_bytes, fraction=C.BUDGET_FRACTION):
    files = C.tree_files(artifact) + ([("MEMORY:" + os.path.basename(memory), os.path.abspath(memory))] if memory else [])
    rows = [dict(file=rel, bytes=os.stat(ap).st_size) for rel, ap in files]
    total = sum(r["bytes"] for r in rows)
    budget = fraction * base_fp16_bytes
    return dict(base_fp16_bytes=base_fp16_bytes, fraction=fraction, budget_bytes=budget, candidate_bytes=total,
                memory_bytes=os.path.getsize(memory) if memory else 0, n_files=len(rows),
                byte_ratio=total / base_fp16_bytes, within_budget=bool(total <= budget), files=rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("artifact"); ap.add_argument("--memory"); ap.add_argument("--lock"); ap.add_argument("--base-fp16-bytes", type=int)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    if a.lock:
        L = json.load(open(a.lock)); base, frac = L["base_fp16_bytes"], L["budget_fraction"]
    else:
        base, frac = a.base_fp16_bytes, C.BUDGET_FRACTION
    r = measure(a.artifact, a.memory, base, frac)
    if a.json:
        print(json.dumps(r, indent=2))
    else:
        for f in r["files"]: print(f"  {f['bytes']:>14,d} B  {f['file']}")
        print(f"candidate {r['candidate_bytes']:,d} B | base fp16 {base:,d} B | budget {r['budget_bytes']:,.0f} B | ratio {100*r['byte_ratio']:.3f}%")
        print("RESULT:", "PASS" if r["within_budget"] else "FAIL (over budget)")
    sys.exit(0 if r["within_budget"] else 1)
