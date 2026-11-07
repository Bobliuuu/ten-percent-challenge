"""Feedback-story smoke test on a REAP-pruned MoE (no training): closed-form covariance-whitened lm_head row edit at 'natural mistake' sites.
Pruned model (mask) -> cache final-norm hidden h at gold-response positions (teacher forced) for a pool of dev items. Items where argmax!=gold at some position = natural mistakes.
Split pool items: teach (mistake items, half), test-mistakes (other half), neighbors (items fully correct under the pruned model). Edit: W[gold] += a * Sigma^-1 h  (Sigma from generic text, ridge), optional W[pred] -= b * Sigma^-1 h.
Metrics (teacher-forced, all response tokens argmax==gold): gain on held-out mistake items, neighbor retention, shuffled-gold control, 0 added bytes."""
import sys, json, random, torch
sys.path.insert(0, "/opt/dlami/nvme/work/l2/code"); sys.path.insert(0, "/opt/dlami/nvme/work/l2/data")
from moe_lib import *
from transformers import AutoTokenizer, AutoModelForCausalLM
model_id, salf, spec, npool = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
tok = AutoTokenizer.from_pretrained(model_id); model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16, device_map="cuda:0").eval(); patch(model)
sal = torch.load(salf); method, frac = spec.split(":"); frac = float(frac); E = sal["cnt"].shape[1]
if frac > 0:
    STATE["mask"] = {}
    for l in range(sal["cnt"].shape[0]):
        sc = sal[{"reap": "reap", "freq": "cnt"}[method]][l].clone(); sc[sal["cnt"][l] == 0] = -1
        pr = torch.zeros(E, dtype=torch.bool); pr[sc.argsort()[:int(round(E * frac))]] = True; STATE["mask"][l] = pr
items = [json.loads(l) for f in ("s1_dev", "s3_dev") for l in open(f"/opt/dlami/nvme/work/l2/data/{f}.jsonl")]
random.Random(7).shuffle(items); items = [i for i in items if i["gold"]][:npool]
def hid(texts_ids):
    out = []
    with torch.no_grad():
        for p, r in texts_ids:
            ids = torch.tensor([p + r], device="cuda:0")
            h = model.model(input_ids=ids).last_hidden_state[0, len(p) - 1:len(p) + len(r) - 1]
            out.append(h.float().cpu())
    return out
data = []
for it in items:
    p = tok(chat_prompt(tok, it), add_special_tokens=False).input_ids; r = tok(gold_qwen(it) + "<|im_end|>", add_special_tokens=False).input_ids; data.append((p, r))
H = hid(data)
gen = [tok(json.loads(l)["text"], add_special_tokens=False, truncation=True, max_length=512, return_tensors="pt").input_ids[0].tolist() for l in open("/opt/dlami/nvme/work/l2/data/general.jsonl")][:150]
with torch.no_grad():
    G = torch.cat([model.model(input_ids=torch.tensor([g], device="cuda:0")).last_hidden_state[0].float().cpu() for g in gen])
W0 = model.lm_head.weight.float().cpu(); V, d = W0.shape
Sig = (G.T @ G) / len(G); Sig += 0.05 * Sig.diag().mean() * torch.eye(d); Sinv = torch.linalg.inv(Sig)
gold = [torch.tensor(r) for p, r in data]
def okitems(W, idx):
    return [bool(((H[i] @ W.T).argmax(-1) == gold[i]).all()) for i in idx]
ok0 = okitems(W0, range(len(items)))
mis = [i for i, o in enumerate(ok0) if not o]; nb = [i for i, o in enumerate(ok0) if o]
random.Random(1).shuffle(mis); teach, testm = mis[:len(mis)//2], mis[len(mis)//2:]
print(f"pool {len(items)} ok0 {sum(ok0)} mistakes {len(mis)} teach {len(teach)} test-mistakes {len(testm)} neighbors {len(nb)}", flush=True)
def edit(W, idx, alpha, shuffle=False, beta=0.0):
    W = W.clone(); rng = random.Random(5)
    for i in idx:
        am = (H[i] @ W.T).argmax(-1)
        for t in range(len(gold[i])):
            if am[t] != gold[i][t]:
                tgt = int(gold[i][t]) if not shuffle else rng.randrange(V)
                v = Sinv @ H[i][t]; v = v / v.norm() * W[tgt].norm() * alpha
                W[tgt] += v
                if beta: W[am[t]] -= beta * v
    return W
def rep(name, W, ctrl=None):
    a = sum(okitems(W, testm)) / max(len(testm), 1); b = sum(okitems(W, nb)) / len(nb); c = sum(okitems(W, teach)) / max(len(teach), 1)
    print(f"{name}: held-out mistakes fixed {a:.3f} | teach fixed {c:.3f} | neighbors kept {b:.4f}", flush=True)
rep("base (no edit)", W0)
for alpha in (0.02, 0.05, 0.1, 0.2, 0.4):
    for beta in (0.0, 0.5):
        rep(f"edit a={alpha} b={beta}", edit(W0, teach, alpha, beta=beta))
        rep(f"  SHUFFLED ctrl a={alpha} b={beta}", edit(W0, teach, alpha, shuffle=True, beta=beta))
