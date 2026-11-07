#!/usr/bin/env python
"""`answer` entry point of a TPQ1 artifact.  python answer.py --artifact DIR --in view.json --out out.json [--dump-prompts p.json]
view.json = [{id, query, tools}] (or .jsonl) ; out.json = [{id, text}] (same order).  Weights are decoded from model.tpq (rANS) and dequantised to bf16 on the GPU
at load time; generation is greedy with the base architecture's own chat template.  Reads only inside the artifact dir; writes nothing there."""
import argparse, json, os, sys, time
#@@LOADER@@


def read_items(p):
    t = open(p, encoding="utf-8").read()
    return json.loads(t) if t.lstrip().startswith("[") else [json.loads(l) for l in t.splitlines() if l.strip()]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--artifact", default=os.path.dirname(os.path.abspath(__file__))); ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--dump-prompts"); ap.add_argument("--max-new-tokens", type=int, default=512); ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--device"); ap.add_argument("--workers", type=int); ap.add_argument("--timings")
    a = ap.parse_args(); dev = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    from transformers import AutoTokenizer
    items = read_items(a.inp); tok = AutoTokenizer.from_pretrained(a.artifact); tok.padding_side = "left"
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model, load_s = load_model(a.artifact, dev, a.workers, log=lambda *x: print(*x, file=sys.stderr, flush=True))
    texts, tm = generate_with(model, tok, items, a.max_new_tokens, a.bs, a.dump_prompts, dev, log=lambda m: print(m, file=sys.stderr, flush=True))
    rows = [dict(id=r["id"], text=t) for r, t in zip(items, texts)]
    if a.out.endswith(".jsonl"): open(a.out, "w").write("".join(json.dumps(r) + "\n" for r in rows))
    else: json.dump(rows, open(a.out, "w"))
    tm.update(dict(TIMERS)); tm.update(load_seconds=load_s, n_items=len(items), tok_per_s=tm["new_tokens"] / max(tm["gen_seconds"], 1e-9))
    print("TIMINGS " + json.dumps(tm), file=sys.stderr, flush=True)
    if a.timings: json.dump(tm, open(a.timings, "w"))


if __name__ == "__main__": main()
