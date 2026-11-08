"""L7 dev experiment: BitNet-2B @ vocab74k/3-bit emb (9.9%), SKILL7 + NEW skill 'Cooking'. real / shuffled(3 seeds) / empty feedback. Prints RESULT_JSON, writes results/l7_fb.json"""
import json, random, sys, os, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from l7lib import *
H = os.path.dirname(os.path.abspath(__file__)); N = int(os.environ.get("L7_N", 74167)); BITS = int(os.environ.get("L7_BITS", 3))
S = Sys(N, BITS, 64)
sk = [json.loads(l) for l in open(H + "/data/skill7_dev.jsonl")]; req = lambda r: r["user"].split("Request: ", 1)[1]
cook = json.load(open(H + "/data/cook.json")); L8 = LABELS + ["Cooking"]
fbp = [r for i, r in enumerate(sk) if i % 4 == 0]; nb = [r for i, r in enumerate(sk) if i % 4 != 0]
nb_g = [r["answer"] for r in nb]; nb_q = [req(r) for r in nb]
def acc(qs, gold, labels): o = S.answer(qs, labels); return [a.lower() == g.lower() for a, g in zip(o, gold)], o
base7, _ = acc(nb_q, nb_g, LABELS)          # old 7-label prompt, no edit
emp8, o8 = acc(nb_q, nb_g, L8)               # EMPTY feedback = new prompt (label registry has Cooking), no edit
teach_q = cook["teach_unseen"]; teach_g = ["Cooking"] * len(teach_q); emp_t, ot = acc(teach_q, teach_g, L8)
# annotated mistakes: 13 cooking + 12 old-label items the unedited model got wrong (fbp)
fb_cook = cook["fb25"][:13]; fq = [req(r) for r in fbp]; fok, _ = acc(fq, [r["answer"] for r in fbp], L8)
wrong = [(q, r["answer"]) for q, r, ok in zip(fq, fbp, fok) if not ok][:12]
ann = [(q, "Cooking") for q in fb_cook] + wrong; Hs = S.hidden([a[0] for a in ann], L8); ft = {l: S.first_tok(l) for l in L8}
print("ann", len(ann), "old-label mistakes", len(wrong), "tokens", {l: S.tok_len if False else len(S.tk.encode(l, add_special_tokens=False).ids) for l in L8}, flush=True)
def run(targets, tag):
    S.reset_head(); ed = S.edit(Hs, [ft[t] for t in targets]); a_nb, _ = acc(nb_q, nb_g, L8); a_t, o_t = acc(teach_q, teach_g, L8); S.reset_head()
    keep = [i for i, ok in enumerate(emp8) if ok]; ret = sum(a_nb[i] for i in keep) / max(1, len(keep))
    return {"cond": tag, "teach_acc": sum(a_t) / len(a_t), "nb_acc": sum(a_nb) / len(a_nb), "nb_retention_vs_empty": ret, "teach_sample_out": o_t[:5], "edited_rows": len(set(ft[t] for t in targets))}
R = {"N": N, "bits": BITS, "n_neighbour": len(nb), "n_teach": len(teach_q), "n_ann": len(ann), "nb_acc_7label_prompt": sum(base7) / len(base7), "empty": {"teach_acc": sum(emp_t) / len(emp_t), "nb_acc": sum(emp8) / len(emp8), "teach_sample_out": ot[:5]},
     "real": run([a[1] for a in ann], "real")}
labs = [a[1] for a in ann]; R["shuffled"] = []
for sd in range(3): p = labs[:]; random.Random(sd).shuffle(p); R["shuffled"].append(run(p, f"shuffled{sd}"))
for m in (1.0, 4.0):   # margin sensitivity (real labels)
    S.reset_head(); ed = S.edit(Hs, [ft[a[1]] for a in ann], margin=m); a_nb, _ = acc(nb_q, nb_g, L8); a_t, _ = acc(teach_q, teach_g, L8); S.reset_head(); R[f"real_margin{m}"] = {"teach_acc": sum(a_t) / len(a_t), "nb_acc": sum(a_nb) / len(a_nb)}
R["bytes"] = {"emb": S.emb_bytes, "tokenizer": S.tok_bytes, "feedback_rows_fp16_per_edited_row": 2 * 2560}
os.makedirs(H + "/results", exist_ok=True); json.dump(R, open(H + "/results/l7_fb.json", "w"), indent=1); print("RESULT_JSON", json.dumps(R))
