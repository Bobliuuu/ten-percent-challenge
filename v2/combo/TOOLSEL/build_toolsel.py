"""TOOLSEL exam builder (deterministic function of the T1 split files).
Task: given a menu of tools and a user request, output exactly the tool name, or `none`.
Label derivation (documented rule, identical for every split):
  kind == nocall                       -> answer `none`
  kind == single and len(gold) == 1    -> answer gold[0].name   (must be a member of the menu, menu names unique, no tool called none)
  kind == multi                        -> DROPPED (the first call of a multi-call item is not a well-defined "the" tool; using it would
                                          make half of the correct answers ambiguous for a name-only router).
  prompt rendering is byte-identical to research/C_tasks/build_tasks.py (description cut at 160 chars, <=8 param names); items whose rendered menu
  exceeds MAXMENU chars are dropped (0 of the dev rows are affected at 5000; the same cap is applied to the locked rows).
Modes
  python build_toolsel.py dev            -> writes data/toolsel_{dev,calib,pool}.jsonl + data/eval_dev_items.jsonl + data/calib_items.jsonl (+ MANIFEST_dev.json). Reads ONLY s1_dev s1_calib s1_pool s3_dev s3_pool.
  LOCKED_BUILD_OK=1 python build_toolsel.py locked OUT.jsonl
                                         -> reads data/locked.jsonl, applies the same pure function, writes OUT, prints ONLY counts per cell + sha256s.
                                            Refuses to run without the env var. NEVER run during development. Run once, after the model/feedback artifact is frozen,
                                            and commit MANIFEST_locked.json (sha256 of OUT and of the sorted id|answer list) BEFORE any model is scored on it.
Cells (source x kind): S1 tool, S1 none, S2 tool, S2 none (S2 = BFCL v3). Requirement: every cell >= 330 and S2 >= 500 in total; the locked file contains S1/single 450,
S1/nocall 400, S2/single 500, S2/nocall 400 (counts from LOCK.json), so the expected output is up to 450/400/500/400 = 1750 items with no sampling (take ALL eligible rows,
order = sha256(id) ascending). If a cell falls under 330 after the length/uniqueness filters the build fails loudly rather than back-filling from multi.
"""
import json, os, sys, hashlib, collections
HERE = os.path.dirname(os.path.abspath(__file__))
T1 = os.path.expanduser("~/ten-percent/v2/tournament/T1/data")
OUT = os.path.join(HERE, "data"); os.makedirs(OUT, exist_ok=True)
MAXMENU = 5000
INSTR = "You are a tool router. Given the available tools and the user request, choose the single tool that should be called, or `none` if no tool applies.\nAnswer with only the tool name (or none).\n\nTools:\n{T}\n\nUser request: {Q}"

def menu_text(tools):
    return "\n".join(f"- {t['name']}: {t.get('description','')[:160]} (params: {', '.join(list((t.get('parameters') or {}).get('properties', {}).keys())[:8])})" for t in tools)

def render(tools, query):
    return INSTR.replace("{T}", menu_text(tools)).replace("{Q}", query)

def derive(x):
    """returns (kind_label in {'tool','none'}, answer) or None if dropped"""
    names = [t["name"] for t in x["tools"]]
    if len(set(names)) != len(names) or any(n.strip().lower() == "none" or not n.strip() for n in names): return None
    if len(menu_text(x["tools"])) > MAXMENU: return None
    if x["kind"] == "nocall": return ("none", "none")
    if x["kind"] == "single" and len(x["gold"]) == 1 and x["gold"][0]["name"] in names: return ("tool", x["gold"][0]["name"])
    return None

def make(x, src_tag, role):
    d = derive(x)
    if d is None: return None
    k, ans = d
    return dict(id=f"TOOLSEL|{src_tag}|{x['id']}", task="TOOLSEL", role=role, src=src_tag, kind=k, user=render(x["tools"], x["query"]), answer=ans,
                menu=[t["name"] for t in x["tools"]], family=x.get("family"), query=x["query"], stratum=x.get("stratum"))

def sha(s): return hashlib.sha256(s.encode()).hexdigest()
def hkey(i): return sha("toolsel-order|" + i)
def load(p): return [json.loads(l) for l in open(p)]
def write(p, rows):
    with open(p, "w") as f:
        for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return hashlib.sha256(open(p, "rb").read()).hexdigest()

def build_dev():
    man = {}
    spec = {"toolsel_dev": [("s1_dev", "S1", "xlam_dev"), ("s3_dev", "S3", "s3_dev")], "toolsel_calib": [("s1_calib", "S1", "xlam_calib")],
            "toolsel_pool": [("s1_pool", "S1", "xlam_pool"), ("s3_pool", "S3", "s3_pool")]}
    allr = {}
    for name, srcs in spec.items():
        rows = []
        for f, role, tag in srcs:
            for x in load(f"{T1}/{f}.jsonl"):
                r = make(x, tag, role)
                if r: rows.append(r)
        rows.sort(key=lambda r: hkey(r["id"])); allr[name] = rows
        man[name] = dict(n=len(rows), sha256=write(f"{OUT}/{name}.jsonl", rows), cells={f"{a}/{b}": c for (a, b), c in sorted(collections.Counter((r["src"], r["kind"]) for r in rows).items())})
    # compression eval inputs in the research/C_tasks items.jsonl format: <=300 per (src,kind) from dev (hash order), calibration <=60 per kind from s1_calib
    ev, cal = [], []
    per = collections.defaultdict(list)
    for r in allr["toolsel_dev"]: per[(r["src"], r["kind"])].append(r)
    for k, v in sorted(per.items()): ev += v[:300]
    pc = collections.defaultdict(list)
    for r in allr["toolsel_calib"]: pc[r["kind"]].append(r)
    for k, v in sorted(pc.items()): cal += v[:60]
    keep = ("id", "task", "role", "src", "kind", "user", "answer")
    man["eval_dev_items"] = dict(n=len(ev), sha256=write(f"{OUT}/eval_dev_items.jsonl", [{k: r[k] for k in keep} for r in ev]), cells={f"{a}/{b}": min(300, len(v)) for (a, b), v in sorted(per.items())})
    man["calib_items"] = dict(n=len(cal), sha256=write(f"{OUT}/calib_items.jsonl", [{k: r[k] for k in keep} for r in cal]))
    man["builder_sha256"] = hashlib.sha256(open(__file__, "rb").read()).hexdigest()
    man["t1_file_sha256"] = {f: hashlib.sha256(open(f"{T1}/{f}.jsonl", "rb").read()).hexdigest() for f in ("s1_dev", "s1_calib", "s1_pool", "s3_dev", "s3_pool")}
    json.dump(man, open(f"{HERE}/MANIFEST_dev.json", "w"), indent=1); print(json.dumps(man, indent=1))

def build_locked(outp):
    if os.environ.get("LOCKED_BUILD_OK") != "1": sys.exit("refusing: locked build is a one-shot step run only after the artifact is frozen (set LOCKED_BUILD_OK=1)")
    lock = json.load(open(f"{T1}/LOCK.json")); want = lock["splits"]["locked"]["file_sha256"]
    got = hashlib.sha256(open(f"{T1}/locked.jsonl", "rb").read()).hexdigest()
    assert got == want, "locked.jsonl does not match LOCK.json"
    rows = []
    for x in load(f"{T1}/locked.jsonl"):
        r = make(x, "locked_" + x["source"], x["source"])
        if r: r.pop("query", None); rows.append(r)
    rows.sort(key=lambda r: hkey(r["id"]))
    cells = collections.Counter((r["src"], r["kind"]) for r in rows)
    ok = all(cells[(f"locked_{s}", k)] >= 330 for s in ("S1", "S2") for k in ("tool", "none")) and sum(v for (s, k), v in cells.items() if s == "locked_S2") >= 500
    print("cells", dict(cells), "OK" if ok else "FAIL")
    if not ok: sys.exit(1)
    h = write(outp, rows); ids = sha("\n".join(sorted(r["id"] + "|" + r["answer"] for r in rows)))
    m = dict(out_sha256=h, id_answer_sha256=ids, n=len(rows), cells={f"{a}/{b}": c for (a, b), c in sorted(cells.items())}, builder_sha256=hashlib.sha256(open(__file__, "rb").read()).hexdigest(), locked_file_sha256=got)
    json.dump(m, open(f"{HERE}/MANIFEST_locked.json", "w"), indent=1); print(json.dumps(m, indent=1))

if __name__ == "__main__":
    if sys.argv[1] == "dev": build_dev()
    elif sys.argv[1] == "locked": build_locked(sys.argv[2])
