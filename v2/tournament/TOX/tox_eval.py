"""TOX evaluation helpers, importable by research/C_tasks/dense_c.py, research/D_scale/dense_stream.py and compress/dense/dense.py (they hand in an already-built
HF-style causal LM -- fake-quant or QLinear streamed -- plus tokenizer; no file format of the artifact is assumed).
  * t_items(rows)            : rows of s1_dev/s3_dev/s2_probe/locked -> CT_ITEMS records (id, task, role, src, kind, user, answer) for dense_c.run_ct
                               (teacher-forced answer+<|im_end|> margins), user = scorer.PROMPT with the comment inserted.
  * generate_eval(model,tok,rows,bs): real greedy generation (max_new_tokens=8, chat template, enable_thinking False) -> rows with out, pred (strict scorer), correct,
                               lp of the 5 label first-tokens and first-token margin (gold vs best other label).
  * retention(base_rows, cand_rows): item retention conditioned on base-correct, per (source x kind) cell + overall, Wilson CI, with the rule-2 gates (>=98% overall, >=95% per cell lower bound reported).
CLI: python tox_eval.py gen MODEL ITEMS.jsonl OUT.json   |   python tox_eval.py ret BASE.json CAND.json   |   python tox_eval.py titems ITEMS.jsonl OUT.jsonl
Only ever pass dev/probe files for tuning; locked.jsonl is for the single final exam."""
import json, sys, os, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import scorer
def t_items(rows):
    return [dict(id=r["id"], task="TOX", role=r["source"], src=r["source"], kind=r["kind"], user=scorer.PROMPT.format(t=r["text"]), answer=r["gold"]) for r in rows]
def _chat(tok, text):
    try: return tok.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True, tokenize=False, enable_thinking=False)
    except TypeError: return tok.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True, tokenize=False)
def generate_eval(model, tok, rows, bs=16, max_new=8):
    import torch
    tok.padding_side = "left"
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    ft = {l: tok.encode(l, add_special_tokens=False)[0] for l in scorer.LABELS}
    prompts = [_chat(tok, scorer.PROMPT.format(t=r["text"])) for r in rows]; order = sorted(range(len(rows)), key=lambda i: -len(prompts[i])); out = [None] * len(rows)
    dev = next(model.parameters()).device if hasattr(model, "parameters") else "cuda"
    for b in range(0, len(order), bs):
        idx = order[b:b + bs]; enc = tok([prompts[i] for i in idx], return_tensors="pt", padding=True, add_special_tokens=False).to(dev)
        with torch.no_grad():
            g = model.generate(**enc, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.pad_token_id, output_scores=True, return_dict_in_generate=True, temperature=None, top_p=None, top_k=None)
        lp0 = torch.log_softmax(g.scores[0].float(), -1)
        for j, i in enumerate(idx):
            txt = tok.decode(g.sequences[j, enc.input_ids.shape[1]:], skip_special_tokens=True)
            lp = {l: float(lp0[j, t]) for l, t in ft.items()}
            out[i] = (txt, lp)
    res = []
    for r, (txt, lp) in zip(rows, out):
        pred = scorer.parse_verdict(txt); oth = max(v for l, v in lp.items() if l != r["gold"])
        res.append(dict(id=r["id"], source=r["source"], kind=r["kind"], gold=r["gold"], out=txt, pred=pred, correct=pred == r["gold"], lp=lp, m_first=lp[r["gold"]] - oth))
    return res
def retention(base, cand):
    b = {r["id"]: r for r in base}; c = {r["id"]: r for r in cand}; cells = collections.defaultdict(lambda: [0, 0])
    for i, r in b.items():
        if not r["correct"] or i not in c: continue
        for k in (f"{r['source']}/{r['kind']}", f"{r['source']}/ALL", "ALL/ALL"):
            cells[k][1] += 1; cells[k][0] += bool(c[i]["correct"])
    out = {}
    for k, (a, n) in sorted(cells.items()):
        lo, hi = scorer.wilson(a, n); out[k] = dict(kept=a, base_correct=n, retention=a / n if n else None, lo=lo, hi=hi)
    cell_keys = [k for k in out if not k.endswith("/ALL") and k != "ALL/ALL"]
    out["_gates"] = dict(overall_ge_98=out["ALL/ALL"]["retention"] >= .98, worst_cell=min(cell_keys, key=lambda k: out[k]["retention"]), worst_cell_ret=min(out[k]["retention"] for k in cell_keys),
                         all_cells_ge_95=all(out[k]["retention"] >= .95 for k in cell_keys), all_cell_lo_ge_95=all(out[k]["lo"] >= .95 for k in cell_keys))
    return out
if __name__ == "__main__":
    m = sys.argv[1]
    if m == "titems":
        rows = [json.loads(l) for l in open(sys.argv[2])]
        with open(sys.argv[3], "w") as f:
            for it in t_items(rows): f.write(json.dumps(it, ensure_ascii=False) + "\n")
    elif m == "gen":
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        tok = AutoTokenizer.from_pretrained(sys.argv[2]); model = AutoModelForCausalLM.from_pretrained(sys.argv[2], torch_dtype=torch.bfloat16, device_map="cuda").eval()
        rows = [json.loads(l) for l in open(sys.argv[3])]; json.dump(generate_eval(model, tok, rows), open(sys.argv[4], "w"))
    elif m == "ret":
        A = json.load(open(sys.argv[2])); B = json.load(open(sys.argv[3]))
        print(json.dumps(retention(A["items"] if isinstance(A, dict) else A, B["items"] if isinstance(B, dict) else B), indent=1))
