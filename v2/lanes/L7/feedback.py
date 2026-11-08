"""python feedback.py ANN.jsonl [--out state] : ANN lines {"request": str, "label": str}. Closed-form (no gradients) minimum-norm edit of the OUTPUT rows of the labels' first tokens
(untied head rows of the compressed model), solved from the model's own final hidden states of the annotated requests. Writes state/head_rows.npz (token ids + fp16 rows) + state/labels.json (label NAMES only).
GRAY: this is a few-row output edit (maintainer ruling 4b/4d: high risk); weakest rule R4/R5 legality. Feedback is NOT placed in the prompt (only label names are listed in the task definition)."""
import sys, json, os, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from l7lib import *
ann = [json.loads(l) for l in open(sys.argv[1])]; out = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else "state"; os.makedirs(out, exist_ok=True)
labels = LABELS[:]; 
if os.path.exists(out + "/labels.json"): labels = json.load(open(out + "/labels.json"))
for a in ann:
    if a["label"] not in labels: labels.append(a["label"])
S = Sys(); Hs = S.hidden([a["request"] for a in ann], labels); ed = S.edit(Hs, [S.first_tok(a["label"]) for a in ann])
W = S.model.get_output_embeddings().weight.data; toks = sorted(ed)
np.savez(out + "/head_rows.npz", tok=np.array(toks), rows=np.stack([W[t].float().cpu().numpy().astype(np.float16) for t in toks])); json.dump(labels, open(out + "/labels.json", "w"))
print("edited rows", len(toks), "labels", labels, "bytes", len(toks) * 2560 * 2)
