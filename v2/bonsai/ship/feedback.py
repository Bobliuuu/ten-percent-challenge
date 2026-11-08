"""ONE COMMAND THAT TAKES ANNOTATED ITEMS AS FEEDBACK.
  user mode:     python feedback.py --artifact DIR --task task.json --ann ann.jsonl      ann rows: {"text": ..., "label": <correct label word, may be a NEW word>}
  harness mode:  python feedback.py --artifact DIR --annotations ann.jsonl               ann rows: {id, query (full prompt), tools, gold:[{"name": label}]}
At most 25 rows. A closed-form edit of EXISTING lm_head rows; nothing is trained, nothing is looked up, no tensor/part is added, annotations are not stored.
For each target token t (a row of the existing output matrix) at an annotated answer position where the ORIGINAL model's argmax differs from t, with final-normed hidden states H:
    delta_t = M^-1 H^T (H M^-1 H^T + lam I)^-1 g,   g_i = (best competitor logit - logit_t at site i) + margin,   M = E[h h^T] over the shipped UNLABELED calibration prompts
    (feedback_calib.jsonl, or --calib) at the answer position and over the model's own greedy continuations. Touched rows are stored as fp16 in DIR/patch.pt (a 1-bit row cannot take a small step).
Always starts from the original weights (patch removed first). Zero rows => patch.pt removed => artifact byte-identical to the base artifact."""
import sys, os, json, argparse
import torch
sys.dont_write_bytecode = True   # never write into the artifact dir
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bonsai_loader as BL, common
from transformers import AutoTokenizer
a = argparse.ArgumentParser(); a.add_argument("--artifact", "--dir", dest="art", default=os.path.dirname(os.path.abspath(__file__))); a.add_argument("--task"); a.add_argument("--ann", "--annotations", dest="ann", required=True)
a.add_argument("--calib"); a.add_argument("--margin", type=float, default=4.0); a.add_argument("--ridge", type=float, default=0.05); a = a.parse_args()
pf = a.art + "/patch.pt"; T = common.task(a.task) if a.task else None
ann = [json.loads(l) for l in open(a.ann) if l.strip()]; assert len(ann) <= 25, "at most 25 annotated items"
if os.path.exists(pf): os.remove(pf)                       # always edit from the ORIGINAL weights
if not ann: print(json.dumps({"patched_rows": [], "patch_bytes": 0})); sys.exit(0)
m, _ = BL.load(a.art, patch=None); tok = AutoTokenizer.from_pretrained(a.art); W = m.lm_head.weight; dev = m.device
EOS = tok.convert_tokens_to_ids("<|im_end|>")
lab = lambda r: r["label"] if "label" in r else r["gold"][0]["name"]
@torch.no_grad()
def hid(ids, pos): return m.model(input_ids=torch.tensor([ids], device=dev)).last_hidden_state[0, pos].float()
pid = lambda q: tok(common.chat(tok, q), add_special_tokens=False).input_ids
sites = {}
for r in ann:
    p = pid(common.prompt_of(r, T)); g = tok(lab(r), add_special_tokens=False).input_ids + [EOS]; h = hid(p + g, list(range(len(p) - 1, len(p) + len(g) - 1)))
    am = (h.to(W.dtype) @ W.T).argmax(-1)
    for j, t in enumerate(g):
        if int(am[j]) != t: sites.setdefault(t, []).append(h[j])
cp = a.calib or a.art + "/feedback_calib.jsonl"; cal = [common.prompt_of(json.loads(l), T) for l in open(cp) if l.strip()]
CH = [hid(pid(q), [len(pid(q)) - 1]) for q in cal]
for q, o in zip(cal[:200], common.generate(m, tok, cal[:200], mx=6)):
    p = pid(q); g = tok(o, add_special_tokens=False).input_ids
    if len(g) > 1: CH.append(hid(p + g, list(range(len(p), len(p) + len(g) - 1))))
CH = torch.cat(CH); D = CH.shape[1]; M = CH.T @ CH / len(CH); M = M + a.ridge * M.diagonal().mean() * torch.eye(D, device=dev)
rows, vals = [], []
for t, hs in sorted(sites.items()):
    Hs = torch.stack(hs); wt = W.data[t].float(); bl = (Hs.to(W.dtype) @ W.T).float(); z = Hs @ wt; bl[:, t] = -1e9
    g = (bl.max(1).values - z + a.margin).clamp(min=0); MiH = torch.linalg.solve(M, Hs.T); G = Hs @ MiH; lam = 1e-3 * G.diagonal().mean()
    dl = MiH @ torch.linalg.solve(G + lam * torch.eye(len(Hs), device=dev), g); rows.append(t); vals.append((wt + dl).half().cpu())
if rows: torch.save({"rows": torch.tensor(rows), "vals": torch.stack(vals)}, pf)
print(json.dumps({"patched_rows": rows, "n_sites": {t: len(h) for t, h in sites.items()}, "patch_bytes": os.path.getsize(pf) if rows else 0}))
