#!/usr/bin/env python
"""Feedback ingestion -> memory file. No weight is touched: artifact is only used as a frozen logit source.
  python feedback.py --artifact artifact --mem0 mem/mem0.bin --mem1 mem/mem1.bin [--n-anchors 1000]
MEM0 = replay anchors only, no residual weights (stage-1 memory; behaviour identical to artifact).
MEM1 = MEM0 + consolidation (trackB lexical hinge perceptron) on pending feedback = the artifact's mistakes in feedback[n_anchors:].
Uses splits.json 'feedback' (SST-2 train rows) only. Stand-in for the real feedback phase (real feedback = labelled exam-distribution items)."""
import argparse, json, os, sys, numpy as np
H = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, H + "/feedback"); sys.path.insert(0, H + "/../trackA")
ap = argparse.ArgumentParser(); ap.add_argument("--artifact", default=H + "/artifact"); ap.add_argument("--mem0", default=H + "/mem/mem0.bin"); ap.add_argument("--mem1", default=H + "/mem/mem1.bin")
ap.add_argument("--n-anchors", type=int, default=1000); a = ap.parse_args()
sys.path.insert(0, a.artifact); import predict as P
from lex import Memory
from datasets import load_dataset
sp = json.load(open(H + "/../splits.json"))["feedback"]; d = load_dataset("nyu-mll/glue", "sst2")["train"].select(sp)
tx, y = d["sentence"], np.array(d["label"])
M = P.Model(a.artifact); lg = M.logits(tx); T = M.toks(tx)
mem = Memory(max_df=30)
for i in range(a.n_anchors): mem.add_anchor(T[i], y[i], lg[i])
P.write_memory(a.mem0, {}, mem.anc); print("MEM0", os.path.getsize(a.mem0), "bytes")
wrong = [i for i in range(a.n_anchors, len(tx)) if lg[i].argmax() != y[i]]
for i in wrong: mem.add(T[i], y[i], lg[i])
print("pending feedback items (artifact errors in feedback tail):", len(wrong))
mem.fit_pending()
P.write_memory(a.mem1, mem.w, mem.anc); print("MEM1", os.path.getsize(a.mem1), "bytes;", len(mem.w), "token weights")
