"""Leakage audit: C2 locked exam vs every split/data the artifact could have seen. CPU only, read-only."""
import json, re, sys, hashlib, collections, os
R = os.path.dirname(os.path.abspath(__file__)) + "/../../"
def load(p): return [json.loads(l) for l in open(R + p)]
def norm(s): return re.sub(r"\W+", " ", s.lower()).strip()
def toks(s): return set(norm(s).split())
def jac(a, b): return len(a & b) / max(1, len(a | b))
S = {k: load(f"commercial/C2/{k}.jsonl") for k in ["locked", "dev", "pool", "calib"]}
fb = os.path.join(R, "c2_fb/feedback_sets")
for f in sorted(os.listdir(fb)):
    if f.startswith("ann_"):
        d = json.load(open(f"{fb}/{f}")); S["fb:" + f] = d if isinstance(d, list) else d.get("annotations", d.get("items", []))
def qtext(r): return r["text"] if isinstance(r, dict) and "text" in r else (r.get("query") if isinstance(r, dict) else str(r))
def toolset(r):
    try: return frozenset(o.split(":")[0].strip() for o in r["names"])
    except Exception: return frozenset()
def gold(r):
    try: return r["names"][r["answer"]].split(":")[0].strip()
    except Exception: return None
L = S["locked"]; out = {"n_locked": len(L)}
Lq = {norm(qtext(r)): r for r in L}; Lt = [toks(qtext(r)) for r in L]
inv = collections.defaultdict(list)
for i, t in enumerate(Lt):
    for w in t: inv[w].append(i)
Lsets = collections.Counter(toolset(r) for r in L); Lgold = collections.Counter(gold(r) for r in L)
for k, rows in S.items():
    if k == "locked": continue
    ex = nd = ts = 0; toolov = 0; golds_in_locked = 0; near = []
    for r in rows:
        if not isinstance(r, dict) or "text" not in r and "query" not in r: continue
        q = norm(qtext(r)); ex += q in Lq
        if toolset(r) and Lsets.get(toolset(r)): ts += 1
        g = gold(r)
        if g and Lgold.get(g): golds_in_locked += 1
        t = toks(qtext(r)); cand = collections.Counter()
        for w in t:
            if len(inv[w]) < 200:
                for i in inv[w]: cand[i] += 1
        best = max((jac(t, Lt[i]) for i, _ in cand.most_common(30)), default=0)
        if best >= 0.8: nd += 1
    out[k] = dict(n=len(rows), exact_dup_query_in_locked=ex, near_dup_jac80=nd, identical_toolset_as_a_locked_item=ts, gold_tool_name_also_gold_in_locked=golds_in_locked)
# locked internal dup + artifact code scan for locked strings
out["locked_internal_dup_queries"] = len(L) - len(Lq)
bad = 0
for root in [R + "c2_fb/artifact_code", os.path.expanduser("~/ten-percent-artifacts/artC2_fb_fixed")]:
    for dp, _, fs in os.walk(root):
        for f in fs:
            if f.endswith((".py", ".json")):
                txt = open(os.path.join(dp, f), errors="ignore").read()
                bad += sum(1 for r in L[:2000] if len(qtext(r)) > 30 and qtext(r) in txt)
out["locked_query_strings_found_in_artifact_text_files"] = bad
print(json.dumps(out, indent=1)); json.dump(out, open(os.path.dirname(os.path.abspath(__file__)) + "/leakage_result.json", "w"), indent=1)
