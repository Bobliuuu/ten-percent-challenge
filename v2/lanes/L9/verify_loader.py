"""Verify the standalone packed loader vs the ORIGINAL path (fake-quant outputs of core235 on dev SEL, results/outs_skill7_k88_b2g128_mse_SEL.json) on the first N SEL items; also exercises answer.py's prompt.
usage: python verify_loader.py ART OUT_JSON [N=200]"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import packed_loader as PL
ART, OUT = sys.argv[1:3]; N = int(sys.argv[3]) if len(sys.argv) > 3 else 200
HERE = os.path.dirname(os.path.abspath(__file__))
rows = [json.loads(l) for l in open(f"{HERE}/data/skill7_sel.jsonl")]
orig = json.load(open(f"{HERE}/results/outs_skill7_k88_b2g128_mse_SEL.json")); base = json.load(open(f"{HERE}/results/outs_skill7_base_SEL.json"))
assert orig["ids"] == [r["id"] for r in rows]
t = time.time(); m, tok = PL.build(ART); tl = time.time() - t
t = time.time(); outs = PL.generate(m, tok, [r["user"] for r in rows[:N]]); tg = time.time() - t
agree = sum(a == b for a, b in zip(outs, orig["outs"][:N])); acc = sum(o == r["answer"] for o, r in zip(outs, rows)) / N; oacc = sum(o == r["answer"] for o, r in zip(orig["outs"][:N], rows)) / N
diff = [(r["id"], o, p) for r, o, p in zip(rows, outs, orig["outs"]) if o != p]
res = dict(n=N, agree=agree, agree_frac=agree / N, packed_acc=acc, orig_acc=oacc, load_s=round(tl), gen_s=round(tg), diffs=diff[:20])
print("RESULT_JSON", json.dumps(res)); json.dump(dict(res, outs=outs), open(OUT, "w"))
