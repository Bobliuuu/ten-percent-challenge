"""python feedback.py <artifact_dir> annotations.jsonl [--m 32]   annotations: {"text": request, "label": skill name}, <= 25 lines (marking the model's real mistakes).
Closed-form diffuse edit of the EXISTING o_proj tensors of the last 8 layers (no optimiser, no new tensor, no stored feedback): writes edit_o_proj.pt (replacement fp16 tensors, counted in MANIFEST) next to the
weights; runtime.load applies it. New labels need no menu entry: the label tokens are existing vocabulary and the unmodified answer path generates them.
Math: teacher-force each annotated (prompt,label); at answer positions where argmax != label token take the final residual x and the target dx = rms(x)(gap+m) u/(u.u), u = gamma (W_U[t]-W_U[c]);
ascending over layers, layer l delivers 1/(layers left) of the CURRENT remaining dx through o_proj: dW = R^T K (lam (C + awt C_ans) + K^T K)^-1, C from the shipped calibration prompts (calib_prompts.jsonl)."""
import sys, json, torch, os
import runtime
art, ann = sys.argv[1], sys.argv[2]; M = float(sys.argv[sys.argv.index("--m") + 1]) if "--m" in sys.argv else 32.0
LAM, AWT, LAY = 1000.0, 10.0, list(range(40, 48))
rows = [json.loads(l) for l in open(ann) if l.strip()]; assert len(rows) <= 25, "at most 25 annotated items"
for f in os.listdir(art):
    if f.startswith("edit_"): os.remove(f"{art}/{f}")           # edits are recomputed from the base artifact + the annotations (empty feedback = no edit file)
if not rows: print("empty feedback: nothing to do"); sys.exit(0)
model, tok, task = runtime.load(art); dev = "cuda"; L = model.model.layers; EOS = "<|im_end|>"
chat = lambda t: tok(tok.apply_chat_template([{"role": "user", "content": task["prompt"].replace("{X}", t)}], tokenize=False, add_generation_prompt=True), add_special_tokens=False).input_ids
tg = lambda lab: tok(lab + EOS, add_special_tokens=False).input_ids
cap = {}; hooks = [L[l].self_attn.o_proj.register_forward_pre_hook(lambda m_, a, l=l: cap.__setitem__(l, a[0].detach())) for l in LAY]
hooks.append(model.model.norm.register_forward_pre_hook(lambda m_, a: cap.__setitem__("fin", a[0].detach())))
WU = model.lm_head.weight.float(); G = model.model.norm.weight.float(); eps = model.config.rms_norm_eps
@torch.no_grad()
def fwd(seqs):
    n = max(len(s) for s in seqs); ids = torch.full((len(seqs), n), tok.pad_token_id); am = torch.zeros((len(seqs), n), dtype=torch.long)
    for r, s in enumerate(seqs): ids[r, :len(s)] = torch.tensor(s); am[r, :len(s)] = 1
    model.model(input_ids=ids.to(dev), attention_mask=am.to(dev), use_cache=False)
with torch.no_grad():
    cal = [json.loads(l) for l in open(f"{art}/calib_prompts.jsonl")]; Cs = {l: torch.zeros(4096, 4096, device=dev) for l in LAY}; Ca = {l: torch.zeros(4096, 4096, device=dev) for l in LAY}; nt = na = 0
    for s0 in range(0, len(cal), 16):
        bt = cal[s0:s0 + 16]; seqs = [chat(r["text"]) + tg(r["label"]) for r in bt]; fwd(seqs)
        for l in LAY:
            for b, r in enumerate(bt):
                k = cap[l][b, :len(seqs[b])].float(); Cs[l] += k.T @ k; ka = k[len(chat(r["text"])) - 1:]; Ca[l] += ka.T @ ka
        nt += sum(len(s) for s in seqs); na += sum(len(tg(r["label"])) for r in bt)
    for l in LAY: Cs[l] /= nt; Ca[l] /= na
    items = [(chat(r["text"]), r["label"]) for r in rows]; seqs = [i + tg(lab) for i, lab in items]
    def measure():
        fwd(seqs); x = cap["fin"].float(); out = []
        for b, (ids, lab) in enumerate(items):
            for j, tj in enumerate(tg(lab)):
                p = len(ids) - 1 + j; xx = x[b, p]; out.append((b, p, tj, (G * xx / torch.sqrt((xx ** 2).mean() + eps)) @ WU.T))
        return x, out
    x0, rws = measure(); S = []
    for b, p, tj, lg in rws:
        c = int(torch.argmax(lg))
        if c != tj: xx = x0[b, p]; u = G * (WU[tj] - WU[c]); rms = torch.sqrt((xx ** 2).mean() + eps); S.append((b, p, xx + rms * (float(lg[c] - lg[tj]) + M) * u / (u @ u)))
    print("sites", len(S), "of", len(rws), "answer positions need a change")
    new = {}
    for i, l in enumerate(LAY):
        x, _ = measure(); K = torch.stack([cap[l][b, p].float() for b, p, _ in S]); R = torch.stack([t - x[b, p] for b, p, t in S]) / (len(LAY) - i)
        A = LAM * (Cs[l] + AWT * Ca[l]) + K.T @ K + 1e-4 * torch.eye(4096, device=dev) * (LAM * (Cs[l] + AWT * Ca[l])).diagonal().mean()
        W = L[l].self_attn.o_proj.weight; W.data.add_(torch.linalg.solve(A, (R.T @ K).T).T.to(W.dtype)); new[f"model.layers.{l}.self_attn.o_proj.weight"] = W.data.half().cpu()
torch.save(new, f"{art}/edit_o_proj.pt"); print("wrote", f"{art}/edit_o_proj.pt", sum(v.numel() * 2 for v in new.values()), "bytes")
