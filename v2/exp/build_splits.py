"""I1: three-way NEWS split + GEN set (EXPERIMENT_QUEUE section 1.1). Deterministic, hash-pinned.
  python build_splits.py            -> writes v2/exp/splits/{cal,sel,rep,gen}.jsonl + v2/exp/SPLITS.json
Uses the bucket/normalisation/dedupe functions of task_search/NEWS/build_news.py unchanged (bucket = sha256 of first 12 normalised words, %100).
CAL  : AG buckets [20,50) only, class-balanced task prompts (CAL-v1 text pool; CAL-v2 answers are generated later by the base from this file)
SEL  : AG [70,80) class-balanced 750/class (3000) + BBC half A
REP  : AG [60,70) class-balanced 625/class (2500) + BBC half B + HuffPost buckets [0,20)   (SEALED: only analyze_pair --stage unseal reads REP preds)
GEN  : 1000 MMLU + 500 ARC-C + 500 generic chat prompts + ~100k generic tokens (wikitext-2 test text), fixed by hash order
Locked buckets (AG [80,100), HuffPost [20,100)) are never touched; asserted below."""
import sys, os, json, hashlib, collections, subprocess, random
HERE = os.path.dirname(os.path.abspath(__file__)); V2 = os.path.dirname(HERE)
sys.path.insert(0, V2 + "/task_search/NEWS")
import build_news as bn
# FINAL NEWS PROMPT (fixed in X01 before any data; queue C1): open tail so the shipped answer command can emit a class outside the four.
PROMPT = "Assign the following news article to exactly one topic.\nAnswer with exactly one word from: World, Sports, Business, Technology, or, if none fits, a different one-word topic.\n\nArticle: {X}"
bn.PROMPT = PROMPT
H, norm, bucket, item = bn.H, bn.norm, bn.bucket, bn.item
OUT = HERE + "/splits"
CAL_PER, SEL_PER, REP_PER = 2000, 750, 625

def pick(rows, per, tag, lo, hi, klist=bn.KINDS):
    by = collections.defaultdict(list)
    for r in rows:
        if lo <= bucket(r[2]) < hi: by[r[1]].append((H(tag + r[0]), r))
    return [r for k in klist for _, r in sorted(by[k], key=lambda x: x[0])[:per]]

def mk(rows, role, src, split):
    out = []
    for r in rows:
        d = item(src, role, r[0], r[1], r[2]); d["split"] = split; d["group"] = bn.gkey(r[2]); out.append(d)
    return out

def build_gen():
    from datasets import load_dataset
    gen = []
    def hsort(rows, tag): return sorted(rows, key=lambda r: H(tag + r[0]))
    mm = [(f"mmlu{i}", x) for i, x in enumerate(load_dataset("cais/mmlu", "all", split="test"))]
    for rid, x in hsort(mm, "gen-mmlu|")[:1000]:
        gen.append(dict(id=f"GEN|mmlu|{rid}", task="GEN", role="mmlu", src="mmlu", kind="mmlu", user=x["question"], choices=x["choices"], answer=int(x["answer"]), group=norm(x["question"])[:120]))
    ar = [(f"arc{i}", x) for i, x in enumerate(load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")) if x["answerKey"] in x["choices"]["label"]]
    for rid, x in hsort(ar, "gen-arc|")[:500]:
        gen.append(dict(id=f"GEN|arc|{rid}", task="GEN", role="arc", src="arc_challenge", kind="arc", user=x["question"], choices=x["choices"]["text"], answer=x["choices"]["label"].index(x["answerKey"]), group=norm(x["question"])[:120]))
    al = [(f"alp{i}", x) for i, x in enumerate(load_dataset("tatsu-lab/alpaca", split="train")) if not x["input"].strip() and 20 < len(x["instruction"]) < 400]
    for rid, x in hsort(al, "gen-chat|")[:500]:
        gen.append(dict(id=f"GEN|chat|{rid}", task="GEN", role="chat", src="alpaca", kind="chat", user=x["instruction"], answer=None, group=norm(x["instruction"])[:120]))
    wt = "\n".join(load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")["text"])
    gen.append(dict(id="GEN|tokens|wikitext2_test", task="GEN", role="tokens", src="wikitext2", kind="tokens", user=wt, answer=None, group="wikitext2_test"))
    return gen

def dump(name, rows):
    p = f"{OUT}/{name}.jsonl"; open(p, "w").write("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows)); return p
def fsha(p): return hashlib.sha256(open(p, "rb").read()).hexdigest()

def main():
    os.makedirs(OUT, exist_ok=True)
    ag, bbc, hp, _ = [bn.dedupe(x) for x in bn.load_all()]
    cal = mk(pick(ag, CAL_PER, "exp-cal|", 20, 50), "CAL", "ag_news", "CAL")
    sel_ag = mk(pick(ag, SEL_PER, "exp-sel|", 70, 80), "SEL", "ag_news", "SEL")
    half = lambda r: int(H("exp-bbc-half|" + bn.gkey(r[2])), 16) % 2
    sel_bbc = mk([r for r in sorted(bbc) if half(r) == 0], "SEL", "bbc_news", "SEL")
    rep_ag = mk(pick(ag, REP_PER, "exp-rep|", 60, 70), "REP", "ag_news", "REP")
    rep_bbc = mk([r for r in sorted(bbc) if half(r) == 1], "REP", "bbc_news", "REP")
    rep_hp = mk(pick(hp, 10**9, "exp-rephp|", 0, 20), "REP", "huffpost", "REP")
    sel, rep = sel_ag + sel_bbc, rep_ag + rep_bbc + rep_hp
    for r in cal + sel_ag + rep_ag: assert r["src"] == "ag_news"
    # structural guards: locked buckets never used
    for rows, nm in ((cal, "cal"), (sel, "sel"), (rep, "rep")):
        for r in rows:
            b = int(H("news-split-v1|" + r["group"]), 16) % 100
            if r["src"] == "ag_news": assert b < 80, (nm, r["id"])
            if r["src"] == "huffpost": assert b < 20, (nm, r["id"])
    gen = build_gen()
    files = {n: dump(n, rows) for n, rows in (("cal", cal), ("sel", sel), ("rep", rep), ("gen", gen))}
    try: commit = subprocess.check_output(["git", "-C", V2, "rev-parse", "HEAD"], text=True).strip()
    except Exception: commit = None
    cnt = lambda rows: {f"{k[0]}|{k[1]}": v for k, v in sorted(collections.Counter((r["src"], r["kind"]) for r in rows).items())}
    sp = dict(version=1, builder="v2/exp/build_splits.py", builder_sha256=fsha(__file__), builder_commit=commit,
              bucket_fn="task_search/NEWS/build_news.py:bucket", prompt=bn.PROMPT,
              files={n: dict(path=f"v2/exp/splits/{n}.jsonl", sha256=fsha(p), n=sum(1 for _ in open(p))) for n, p in files.items()},
              counts={n: cnt(rows) for n, rows in (("cal", cal), ("sel", sel), ("rep", rep))},
              base_correct={}, floors={}, notes="base_correct filled by X01 (v2/exp/results/X01) per model")
    sp["split_hash"] = hashlib.sha256(json.dumps({n: f["sha256"] for n, f in sp["files"].items()}, sort_keys=True).encode()).hexdigest()
    json.dump(sp, open(HERE + "/SPLITS.json", "w"), indent=1); print(json.dumps(dict(files=sp["files"], counts=sp["counts"], split_hash=sp["split_hash"]), indent=1))
if __name__ == "__main__": main()
