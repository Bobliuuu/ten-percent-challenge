"""Evaluate the PACKED artifact through runtime.load (not the in-memory simulation): python eval_art.py <artifact> <jsonl> [n] [base_outs.json]. Prints accuracy and retention on base-correct items."""
import sys, json, runtime, collections
art, f = sys.argv[1], sys.argv[2]; n = int(sys.argv[3]) if len(sys.argv) > 3 else 10**9
rows = [json.loads(l) for l in open(f)][:n]
model, tok, task = runtime.load(art)
outs = runtime.route(model, tok, task, [r["text"] for r in rows], bs=32)
acc = sum(o == r["answer"] for o, r in zip(outs, rows)) / len(rows); res = dict(n=len(rows), acc=acc)
if len(sys.argv) > 4:
    b = json.load(open(sys.argv[4]))["outs"][:len(rows)]; bc = [x == r["answer"] for x, r in zip(b, rows)]; kept = sum(1 for o, r, y in zip(outs, rows, bc) if y and o == r["answer"])
    res.update(base_acc=sum(bc) / len(rows), n_bc=sum(bc), kept=kept, ret=kept / max(1, sum(bc)))
print("EVAL_ART", json.dumps(res)); json.dump(outs, open("/tmp/eval_art_outs.json", "w"))
