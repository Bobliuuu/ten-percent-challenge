"""analyze2.py BASE CAND [PARENT] : large dev + multilingual/code retention. Files: out2/<tag>.json (+ .ml_ok.json). Items from devlarge/devlarge.jsonl + ml_items.jsonl.
Prints/writes per-cell retention of CAND vs BASE (P(cand ok | base ok)) with Wilson lower bounds and counts."""
import sys, json, os, re, collections, math, importlib.util
V2 = os.path.expanduser("~/ten-percent/v2"); HERE = os.path.dirname(os.path.abspath(__file__)); RES = HERE + "/results2"
def load(p, n):
    s = importlib.util.spec_from_file_location(n, p); m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
T1 = load(V2 + "/tournament/T1/scorer.py", "t1s"); TOX = load(V2 + "/tournament/TOX/scorer.py", "toxs"); INJ = load(V2 + "/tournament/INJ/scorer.py", "injs")
sys.argv_saved = sys.argv; sys.argv = ["x"]
A = load(HERE + "/analyze.py", "an") if False else None
def xml2json(txt):
    def conv(m):
        fm = re.search(r"<function=([^>\n]+)>(.*?)</function>", m.group(1), re.S)
        if not fm: return m.group(0)
        args = {}
        for pm in re.finditer(r"<parameter=([^>\n]+)>\n?(.*?)\n?</parameter>", fm.group(2), re.S):
            v = pm.group(2)
            try: v = json.loads(v)
            except Exception: pass
            args[pm.group(1)] = v
        return "<tool_call>\n" + json.dumps({"name": fm.group(1).strip(), "arguments": args}) + "\n</tool_call>"
    return re.sub(r"<tool_call>(.*?)</tool_call>", conv, txt, flags=re.S)
items = {}
for f in (HERE + "/devlarge/devlarge.jsonl", RES + "/ml_items.jsonl"):
    if os.path.exists(f):
        for l in open(f): r = json.loads(l); items[r["id"]] = r
def ok_map(tag):
    out = json.load(open(f"{RES}/{tag}.json")); ml = json.load(open(f"{RES}/{tag}.ml_ok.json")) if os.path.exists(f"{RES}/{tag}.ml_ok.json") else {}
    ok = {}
    for i, t in out.items():
        it = items.get(i)
        if it is None: continue
        k = it["task"]
        if k == "tox": ok[i] = TOX.correct(t, it["gold"])
        elif k == "inj": ok[i] = INJ.correct(t, it["gold"])
        elif k == "news": ok[i] = ((t.split() or [""])[0].strip(".,:;!\"'*").lower() == it["gold"].lower())
        elif k == "t1":
            try: ok[i] = bool(T1.score_item(xml2json(t.split("<|im_end|>")[0]), dict(gold=it["gold"], tools=it["tools"], query=it["user"], kind=it["kind"], source=it["source"]))["correct"])
            except Exception: ok[i] = False
        elif i in ml: ok[i] = bool(ml[i])
    return ok
def wlo(k, n, z=1.96):
    if n == 0: return 0.0
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)); return (c - h) / d
def cmp(B, C, label):
    cells = collections.defaultdict(lambda: [0, 0]); tasks = collections.defaultdict(lambda: [0, 0]); allc = [0, 0]
    for i, b in B.items():
        if not b or i not in C: continue
        it = items[i]; k = (it["task"], it["source"], it["kind"]); cells[k][1] += 1; cells[k][0] += C[i]; tasks[it["task"]][1] += 1; tasks[it["task"]][0] += C[i]
        allc[1] += 1; allc[0] += C[i]
    rows = [dict(cell="|".join(k), n=n, ret=a / n, lo=wlo(a, n)) for k, (a, n) in sorted(cells.items())]
    return dict(label=label, overall=dict(n=allc[1], ret=allc[0] / max(1, allc[1]), lo=wlo(*allc)), tasks={t: dict(n=n, ret=a / n, lo=wlo(a, n)) for t, (a, n) in tasks.items()}, cells=rows)
def main():
    tags = sys.argv_saved[1:]; M = {t: ok_map(t) for t in tags}; B = M[tags[0]]
    print("BASE acc by task:", {t: round(sum(v for i, v in B.items() if items[i]["task"] == t) / max(1, sum(1 for i in B if items[i]["task"] == t)), 3) for t in sorted(set(items[i]["task"] for i in B))})
    res = {}
    for t in tags[1:]:
        r = cmp(B, M[t], f"{t} vs {tags[0]}"); res[t] = r
        print(f"== {r['label']}: overall {r['overall']['ret']:.4f} lo {r['overall']['lo']:.3f} n={r['overall']['n']}"); print("  tasks:", {k: (round(v['ret'], 4), round(v['lo'], 3), v['n']) for k, v in r['tasks'].items()})
        print("  cells >=300:", sum(1 for c in r['cells'] if c['n'] >= 300), "of", len(r['cells']), "min lo among n>=300:", min([round(c['lo'], 3) for c in r['cells'] if c['n'] >= 300] or [None]))
    json.dump(res, open(f"{RES}/analysis2_{'_'.join(tags)}.json", "w"), indent=1)
if __name__ == "__main__": main()
