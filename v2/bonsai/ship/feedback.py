"""ONE COMMAND THAT TAKES ANNOTATED ITEMS AS FEEDBACK.   python feedback.py --dir <artifact dir> --task tasks/news.json --ann ann.jsonl [--calib calib.txt] [--margin 8]
ann.jsonl: <=25 lines {"text": ..., "label": <the correct label word, may be new>}. Closed form, no gradient, no optimizer, no added tensor, no stored annotation:
for each target token t (a row of the EXISTING lm_head) where the ORIGINAL model's argmax at the teacher-forced answer position differs from t, take final-normed hidden states H and
  delta_t = M^-1 H^T (H M^-1 H^T + lam I)^-1 g,  g_i = (best competitor logit - logit_t) + margin,   M = E[h h^T] over the shipped unlabeled calibration texts (+ model's own greedy continuations).
The touched rows are stored as fp16 in <dir>/patch.pt (the 1-bit lm_head rows cannot take a small step). Empty annotation file => patch.pt is removed => artifact byte-identical to base."""
import sys, os, json, argparse, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bonsai_loader as BL, common
from transformers import AutoTokenizer
a = argparse.ArgumentParser(); a.add_argument("--dir", required=True); a.add_argument("--task", required=True); a.add_argument("--ann", required=True)
a.add_argument("--calib", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "calib.txt")); a.add_argument("--margin", type=float, default=8.0); a.add_argument("--ridge", type=float, default=0.05); a = a.parse_args()
pf = a.dir + "/patch.pt"; ann = [json.loads(l) for l in open(a.ann) if l.strip()]
assert len(ann) <= 25, "at most 25 annotated items"
if not ann:
    if os.path.exists(pf): os.remove(pf)
    print("empty feedback: no patch, artifact identical to base"); sys.exit(0)
if os.path.exists(pf): os.remove(pf)               # always edit from the ORIGINAL weights
m, _ = BL.load(a.dir, patch=None); tok = AutoTokenizer.from_pretrained(a.dir); T = common.task(a.task); W = m.lm_head.weight; dev = m.device
EOS = tok.convert_tokens_to_ids("<|im_end|>")
@torch.no_grad()
def hid(ids, pos): return m.model(input_ids=torch.tensor([ids], device=dev)).last_hidden_state[0, pos].float()
pid = lambda t: tok(common.chat(tok, T["prompt"].replace("{t}", t)), add_special_tokens=False).input_ids
sites = {}
for r in ann:
    p = pid(r["text"]); g = tok(r["label"], add_special_tokens=False).input_ids + [EOS]; h = hid(p + g, list(range(len(p) - 1, len(p) + len(g) - 1)))
    am = (h.to(W.dtype) @ W.T).argmax(-1)
    for j, t in enumerate(g):
        if int(am[j]) != t: sites.setdefault(t, []).append(h[j])
calt = [l.rstrip("\n") for l in open(a.calib) if l.strip()]
CH = [hid(pid(t), [len(pid(t)) - 1]) for t in calt]
outs = common.answer(m, tok, T, calt[:200], mx=6)
for t, o in zip(calt[:200], outs):
    p = pid(t); g = tok(o, add_special_tokens=False).input_ids
    if len(g) > 1: CH.append(hid(p + g, list(range(len(p), len(p) + len(g) - 1))))
CH = torch.cat(CH); D = CH.shape[1]; M = CH.T @ CH / len(CH); M = M + a.ridge * M.diagonal().mean() * torch.eye(D, device=dev)
rows, vals = [], []
for t, hs in sorted(sites.items()):
    Hs = torch.stack(hs); wt = W.data[t].float(); bl = (Hs.to(W.dtype) @ W.T).float(); z = Hs @ wt; bl[:, t] = -1e9
    g = (bl.max(1).values - z + a.margin).clamp(min=0); MiH = torch.linalg.solve(M, Hs.T); G = Hs @ MiH; lam = 1e-3 * G.diagonal().mean()
    dl = MiH @ torch.linalg.solve(G + lam * torch.eye(len(Hs), device=dev), g); rows.append(t); vals.append((wt + dl).half().cpu())
if rows:
    torch.save({"rows": torch.tensor(rows), "vals": torch.stack(vals)}, pf)
print(json.dumps({"patched_rows": rows, "n_sites": {t: len(h) for t, h in sites.items()}, "patch_bytes": os.path.getsize(pf) if rows else 0}))
