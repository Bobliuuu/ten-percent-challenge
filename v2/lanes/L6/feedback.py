"""L6 feedback: diffuse closed-form edit (no optimizer, no gradients) of the shared-expert down_proj of the last NL layers of the pruned+quantised MoE.
Scenario (dev only): the shipped skill-router prompt lists 6 skills (NEW label 'Lists' is NOT in the prompt's label list). The user gives <=25 annotated mistakes
(request text, correct skill = 'Lists'). The edit changes W_down so the model's OWN forward pass emits the word 'Lists' on such requests.
usage: PYTHONPATH=/workspace/tf4 python feedback.py MODEL_DIR OUT.json [--new Lists --n 20 --draws 5 --layers 4 --lam 1.0 --margin 3.0]
Closed form per layer l: dW_l = R K^T (lam*C + K K^T)^-1, K = keys (input of down_proj at the last prompt token) of the annotated items, C = mean key covariance over calibration
tokens (task-format SKILL7 calib prompts, disjoint from eval), R = (final-residual change that lifts the new-label logit over the best competitor by `margin`)/NL, the same for every layer.
Stored as two thin factors per layer (A: d x n, B: n x d_ff, fp16) = 'feedback.npz' (counted in MANIFEST); applied at load: W += A B. Controls: real labels, shuffled labels, empty (no edit)."""
import sys, os, json, re, random, argparse, time, collections
import numpy as np, torch
D = os.environ.get("NM_DATA", "/workspace/data")
ap = argparse.ArgumentParser(); ap.add_argument("model"); ap.add_argument("out"); ap.add_argument("--new", default="Lists"); ap.add_argument("--n", type=int, default=20)
ap.add_argument("--draws", type=int, default=5); ap.add_argument("--layers", type=int, default=4); ap.add_argument("--lam", type=float, default=1.0); ap.add_argument("--margin", type=float, default=3.0)
ap.add_argument("--save", default=None)
A = ap.parse_args()
from transformers import AutoTokenizer, AutoModelForCausalLM
T0 = time.time(); log = lambda *a: print(f"[{time.time()-T0:5.0f}s]", *a, flush=True)
tok = AutoTokenizer.from_pretrained(A.model); tok.padding_side = "left"
lm = AutoModelForCausalLM.from_pretrained(A.model, dtype=torch.bfloat16, device_map={"": 0}).eval()
LABS = ["Alarm", "Weather", "Music", "Calendar", "Lists", "News", "Transport"]; OLD = [l for l in LABS if l != A.new]
dev = [json.loads(l) for l in open(D + "/skill_dev.jsonl")]; cal = [json.loads(l) for l in open(D + "/skill_cal.jsonl")]
def prompt(text): return ("You are the router of a voice assistant. Decide which skill should handle the user's request.\nAnswer with exactly one word from: " + ", ".join(OLD) + ".\n\nRequest: " + text)
def req(it): return it["user"].split("Request: ", 1)[1]
def chat(it): return tok.apply_chat_template([{"role": "user", "content": prompt(req(it))}], add_generation_prompt=True, tokenize=False)
def norm(o): return re.sub(r"[^A-Za-z]", "", o.split("\n")[0].strip())
@torch.no_grad()
def gen(items, bs=48):
    outs = []
    for i in range(0, len(items), bs):
        b = tok([chat(x) for x in items[i:i + bs]], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        o = lm.generate(**b, max_new_tokens=6, do_sample=False); outs += [norm(t) for t in tok.batch_decode(o[:, b.input_ids.shape[1]:], skip_special_tokens=True)]
    return outs
L = len(lm.model.layers); EL = list(range(L - A.layers, L)); mods = {l: lm.model.layers[l].mlp.shared_expert.down_proj for l in EL}
W0 = {l: mods[l].weight.data.clone() for l in EL}
import torch.nn.functional as F
def set_delta(d):
    """edited weight = fp16(W0 + AB) applied in fp32 (bf16 rounding would erase the small edit); d=None restores the original forward"""
    for l in EL:
        m = mods[l]
        if d is None:
            m.__dict__.pop("forward", None)
        else:
            Wf = (W0[l].float() + d[l]).half().float()
            m.forward = (lambda x, Wf=Wf: F.linear(x.float(), Wf).to(x.dtype))
# keys at last prompt position
keyrec = {}
def mk(l):
    def h(m, inp): keyrec[l] = inp[0].detach().float().reshape(-1, inp[0].shape[-1])
    return h
hs = [mods[l].register_forward_pre_hook(mk(l)) for l in EL]
@torch.no_grad()
def keys_last(items):
    out = {l: [] for l in EL}; extra = []
    for it in items:
        ids = tok(chat(it), return_tensors="pt", add_special_tokens=False).to("cuda"); o = lm(**ids, output_hidden_states=True, use_cache=False)
        for l in EL: out[l].append(keyrec[l][-1])
        extra.append((o.hidden_states[-2][0, -1].float(), o.logits[0, -1].float()))   # hidden_states[-1] is post-norm in HF; [-2] is the input of the last layer, use model.norm input below
    return {l: torch.stack(out[l]).T for l in EL}, extra
# covariance C over all token positions of calibration prompts (task format)
C = {l: torch.zeros(W0[l].shape[1], W0[l].shape[1], device="cuda") for l in EL}; nt = 0
with torch.no_grad():
    for it in cal[:120]:
        ids = tok(chat(it), return_tensors="pt", add_special_tokens=False).to("cuda"); lm(**ids, use_cache=False)
        for l in EL: k = keyrec[l]; C[l] += k.T @ k
        nt += ids.input_ids.shape[1]
for l in EL: C[l] /= nt
gam = lm.model.norm.weight.float(); Wout = lm.lm_head.weight.float()
tid = lambda lab: tok(lab, add_special_tokens=False).input_ids[0]
for lab in LABS: assert tok(lab, add_special_tokens=False).input_ids[0] != tok("Alarm", add_special_tokens=False).input_ids[0] or lab == "Alarm"
@torch.no_grad()
def final_pre_norm(it):
    ids = tok(chat(it), return_tensors="pt", add_special_tokens=False).to("cuda"); cap = {}
    h = lm.model.norm.register_forward_pre_hook(lambda m, i: cap.__setitem__("x", i[0][0, -1].float())); o = lm(**ids, use_cache=False); h.remove()
    return cap["x"], o.logits[0, -1].float()
@torch.no_grad()
def make_delta(ann, targets):
    K, extra = keys_last(ann); n = len(ann); R = torch.zeros(lm.config.hidden_size, n, device="cuda")
    for i, (it, lab) in enumerate(zip(ann, targets)):
        x, lg = final_pre_norm(it); rms = x.pow(2).mean().sqrt(); t = tid(lab); lg2 = lg.clone(); lg2[t] = -1e9; c = int(lg2.argmax())
        gap = float(lg[c] - lg[t]); u = gam * (Wout[t] - Wout[c]); R[:, i] = rms * (max(gap, 0) + A.margin) * u / (u @ u)
    R = R / len(EL); d = {}; fac = {}
    for l in EL:
        Kl = K[l]; M = A.lam * C[l] + Kl @ Kl.T; M = M + 1e-4 * M.diagonal().mean() * torch.eye(M.shape[0], device="cuda")
        Bm = torch.linalg.solve(M, Kl).T            # n x d_ff
        d[l] = R @ Bm
        fac[l] = (R.half().cpu().numpy(), Bm.half().cpu().numpy())
    return d, fac
# data split: annotations / teach / neighbours
newitems = [x for x in dev if x["kind"] == A.new]; neigh_all = [x for x in dev if x["kind"] != A.new]
log("new-class dev items", len(newitems), "neighbours", len(neigh_all))
base_neigh = gen(neigh_all); base_ok = [i for i, (o, x) in enumerate(zip(base_neigh, neigh_all)) if o.lower() == x["answer"].lower()]
neigh = [neigh_all[i] for i in base_ok]; log("base-correct neighbours (6-label prompt)", len(neigh), "/", len(neigh_all))
res = dict(model=A.model, new=A.new, n=A.n, lam=A.lam, margin=A.margin, layers=EL, base_neigh_acc=len(base_ok) / len(neigh_all), draws=[])
for d in range(A.draws):
    rng = random.Random(d); idx = list(range(len(newitems))); rng.shuffle(idx); ann = [newitems[i] for i in idx[:A.n]]; teach = [newitems[i] for i in idx[A.n:]]
    row = dict(draw=d, n_teach=len(teach))
    for cond in ("empty", "real", "shuffle"):
        if cond == "empty": set_delta(None)
        else:
            tg = [A.new] * len(ann) if cond == "real" else [rng.choice(OLD) for _ in ann]     # shuffled: labels decoupled from the requests (random old skills)
            dl, fac = make_delta(ann, tg); set_delta(dl)
            if cond == "real" and d == 0 and A.save: np.savez_compressed(A.save, **{f"A{l}": fac[l][0] for l in EL}, **{f"B{l}": fac[l][1] for l in EL}); row["feedback_bytes"] = os.path.getsize(A.save if A.save.endswith('.npz') else A.save + '.npz')
        to = gen(teach); no = gen(neigh); tr = gen(ann)
        row[cond] = dict(train_ann=float(np.mean([o.lower() == A.new.lower() for o in tr])), teach=float(np.mean([o.lower() == A.new.lower() for o in to])), neigh_kept=float(np.mean([o.lower() == x["answer"].lower() for o, x in zip(no, neigh)])),
                         neigh_by_class={k: float(np.mean([o.lower() == x["answer"].lower() for o, x in zip(no, neigh) if x["kind"] == k])) for k in OLD},
                         teach_outputs=collections.Counter(to).most_common(4))
        log(d, cond, json.dumps({k: v for k, v in row[cond].items() if k != "neigh_by_class"}))
    res["draws"].append(row); json.dump(res, open(A.out, "w"), indent=1)
set_delta(None)
