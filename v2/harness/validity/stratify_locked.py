"""Re-slice the already-run locked C2 receipts (no GPU, no new exam use): does the gain over base survive on items
that cannot be memorised (unseen gold tool, unseen toolset, no near-dup query)?"""
import json, re, os, collections, random
H = os.path.dirname(os.path.abspath(__file__)); R = H + "/../../"
ld = lambda p: [json.loads(l) for l in open(R + p)]
L = ld("commercial/C2/locked.jsonl"); seen = ld("commercial/C2/dev.jsonl") + ld("commercial/C2/pool.jsonl") + ld("commercial/C2/calib.jsonl")
norm = lambda s: re.sub(r"\W+", " ", s.lower()).strip(); toks = lambda s: set(norm(s).split())
gold = lambda r: r["names"][r["answer"]].split(":")[0].strip()
seen_gold = set(gold(r) for r in seen if r["names"][r["answer"]].split(":")[0].strip() != "None of these tools can do this")
seen_tools = set(o.split(":")[0].strip() for r in seen for o in r["names"])
seen_sets = set(frozenset(o.split(":")[0].strip() for o in r["names"]) for r in seen)
seen_q = set(norm(r["text"]) for r in seen)
inv = collections.defaultdict(list); St = [toks(r["text"]) for r in seen]
for i, t in enumerate(St):
    for w in t: inv[w].append(i)
def neardup(r):
    t = toks(r["text"]); c = collections.Counter()
    for w in t:
        if len(inv[w]) < 300:
            for i in inv[w]: c[i] += 1
    return any(len(t & St[i]) / max(1, len(t | St[i])) >= 0.8 for i, _ in c.most_common(40))
s2 = json.load(open(R + "harness/receipts/REAL_final/stage2.json")); s1 = json.load(open(R + "harness/receipts/REAL_final/stage1.json"))
y = ["ABCDE"[r["answer"]] for r in L]
def acc(p, idx): return sum(p[i] == y[i] for i in idx) / max(1, len(idx))
none = lambda r: r["names"][r["answer"]].startswith("None of these")
masks = {
 "all": lambda r: True,
 "gold_tool_unseen_in_train_splits(non-None)": lambda r: not none(r) and gold(r) not in seen_gold,
 "ALL_candidate_tools_unseen": lambda r: all(o.split(":")[0].strip() not in seen_tools for o in r["names"]),
 "toolset_unseen": lambda r: frozenset(o.split(":")[0].strip() for o in r["names"]) not in seen_sets,
 "not_exact_dup": lambda r: norm(r["text"]) not in seen_q,
 "not_neardup(jac<.8)": lambda r: not neardup(r),
 "gold_is_None": lambda r: none(r), "gold_not_None": lambda r: not none(r),
}
random.seed(0); out = {}
for st, rc in (("stage1", s1), ("stage2", s2)):
    b, c = rc["_base_pred"], rc["_cand_pred"]; out[st] = {}
    for k, f in masks.items():
        idx = [i for i, r in enumerate(L) if f(r)]
        if len(idx) < 20: out[st][k] = dict(n=len(idx)); continue
        d = [(c[i] == y[i]) - (b[i] == y[i]) for i in idx]; bs = sorted(sum(random.choice(d) for _ in d) / len(d) for _ in range(2000))
        out[st][k] = dict(n=len(idx), base=round(acc(b, idx), 4), cand=round(acc(c, idx), 4), diff=round(sum(d) / len(d), 4), diff_ci95=[round(bs[50], 4), round(bs[1949], 4)])
print(json.dumps(out, indent=1)); json.dump(out, open(H + "/stratified_result.json", "w"), indent=1)
