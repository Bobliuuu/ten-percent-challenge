"""Proposed harness helper: greedy-decode nondeterminism floor for the 98% rule.
Usage:  python noise_floor.py runA.json runB.json [runC.json ...]   (each: list of 0/1 per item, or {"correct":[...]})
Protocol (recommended): run the ORIGINAL bf16 model K>=3 times with different batch sizes / engines / padding side on the SAME locked items.
  * define base_correct as the item-wise MAJORITY (or intersection, stricter) of the K runs;
  * report retention(base_run_i -> base_run_j) for all ordered pairs = the noise floor; the candidate must be compared against the same majority bar;
  * pass/fail for a candidate should use  retention_cand >= 0.98  AND  retention_cand >= floor - 0.005 (candidate not worse than base-vs-base).
"""
import json, sys, itertools
import numpy as np

def retention(b, c):
    b, c = np.asarray(b, bool), np.asarray(c, bool); nb = b.sum(); return float((b & c).sum() / nb) if nb else 0.0

def floor(runs):
    runs = [np.asarray(r, bool) for r in runs]
    pairs = {f"{i}->{j}": retention(runs[i], runs[j]) for i, j in itertools.permutations(range(len(runs)), 2)}
    maj = np.mean(runs, 0) >= 0.5 if len(runs) >= 3 else runs[0]
    inter = np.all(runs, 0)
    return dict(pairs=pairs, min_pair=min(pairs.values()), mean_pair=float(np.mean(list(pairs.values()))), n_items=len(runs[0]),
                flip_items=int((np.any(runs, 0) & ~np.all(runs, 0)).sum()), majority_acc=float(maj.mean()), intersection_acc=float(inter.mean()))

if __name__ == "__main__":
    R = []
    for p in sys.argv[1:]:
        d = json.load(open(p)); R.append(d["correct"] if isinstance(d, dict) else d)
    print(json.dumps(floor(R), indent=1))
