#!/usr/bin/env python
"""`feedback` entry point of a TPQ1 artifact: closed-form in-place edit of existing output-embedding rows (v2/feedback_v3 fb5 `lmhead`, modes mean / lda), then
the touched rows are re-quantised on the tensor's own grid and re-encoded INTO model.tpq (same file, same probability tables, no new tensor, no new file).
   python feedback.py --artifact DIR --annotations ann.jsonl [--method lda|mean] [--alpha 0.2] [--ridge 0.1] [--stats-items f.jsonl]
ann rows = {id, query, tools, gold:[{name, arguments}]}.  Zero rows, or no site where the model is wrong => nothing is opened for writing.  Single forward passes only."""
import argparse, json, os, sys, time
#@@LOADER@@

GENERIC = ["Hello, how are you today?", "Explain what a hash table is.", "Write a short poem about autumn.", "What is the capital of France?",
           "Summarise the plot of Romeo and Juliet in two sentences.", "How do I reverse a list in Python?", "Give me three tips for better sleep.",
           "Translate 'good morning' into Spanish.", "What causes rainbows?", "List the planets of the solar system."]


def edit_mean(H_by_tok, W, alpha, stats, ridge):
    out = {}
    for t, hs in H_by_tok.items():
        d = torch.stack(hs).mean(0) - stats["mu"]; d = d / d.norm(); w = W[t].float(); out[t] = (w + alpha * w.norm() * d).to(W.dtype)
    return out


def edit_lda(H_by_tok, W, alpha, stats, ridge):
    S = stats["sig"]; A = S + ridge * S.diagonal().mean() * torch.eye(S.shape[0], device=S.device); out = {}
    for t, hs in H_by_tok.items():
        d = torch.linalg.solve(A, torch.stack(hs).mean(0) - stats["mu"]); d = d / d.norm(); w = W[t].float(); out[t] = (w + alpha * w.norm() * d).to(W.dtype)
    return out


EDIT_FNS = {"mean": edit_mean, "lda": edit_lda}      # pluggable: fn(H_by_token, lm_head_weight, alpha, stats{mu,sig}, ridge) -> {row: new bf16 row}


def gold_text(g): return "\n".join("<tool_call>\n" + json.dumps(dict(name=x["name"], arguments=x["arguments"])) + "\n</tool_call>" for x in g)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--artifact", default=os.path.dirname(os.path.abspath(__file__))); ap.add_argument("--annotations", required=True)
    ap.add_argument("--method", default="lda"); ap.add_argument("--alpha", type=float, default=0.5); ap.add_argument("--ridge", type=float, default=0.1)
    ap.add_argument("--stats-items"); ap.add_argument("--device"); ap.add_argument("--workers", type=int)
    a = ap.parse_args(); anns = [json.loads(l) for l in open(a.annotations) if l.strip()]
    if not anns: print("FEEDBACK_NOOP no annotations"); return
    dev = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.artifact); model, _ = load_model(a.artifact, dev, a.workers, log=lambda *x: print(*x, file=sys.stderr, flush=True))
    head_name = "lm_head.weight" if "lm_head.weight" in model.state_dict() and not model.config.tie_word_embeddings else "model.embed_tokens.weight"
    W = model.lm_head.weight.data; sites = []
    with torch.no_grad():
        for it in anns:
            if not it.get("gold"): continue
            p = tok(render(tok, it), add_special_tokens=False).input_ids; g = tok(gold_text(it["gold"]) + "<|im_end|>", add_special_tokens=False).input_ids
            ids = torch.tensor([p + g], device=dev); h = model.model(ids).last_hidden_state[0]; lg = model.lm_head(h).float()
            for j, t in enumerate(g):
                pos = len(p) + j - 1
                if int(lg[pos].argmax()) != t: sites.append((t, h[pos].float().clone()))
        if not sites: print("FEEDBACK_NOOP model already right on every annotation token"); return
        hs = []
        texts = [tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True) for q in GENERIC]
        if a.stats_items: texts += [render(tok, json.loads(l)) for l in open(a.stats_items) if l.strip()][:120]
        for tx in texts:
            ids = tok(tx, return_tensors="pt", add_special_tokens=False).input_ids.to(dev); hs.append(model.model(ids).last_hidden_state[0].float())
        X = torch.cat(hs); stats = dict(mu=X.mean(0), sig=torch.cov(X.T))
        by = {}
        for t, h in sites: by.setdefault(t, []).append(h)
        edits = EDIT_FNS[a.method](by, W, a.alpha, stats, a.ridge)
    del model; torch.cuda.empty_cache() if dev == "cuda" else None
    r = replace_rows(a.artifact, head_name, {int(t): row.cpu() for t, row in edits.items()}, dev)
    print("FEEDBACK_DONE", json.dumps(dict(rows_edited=len(edits), sites=len(sites), tensor=head_name, **r)))


if __name__ == "__main__": main()
