"""Shared loader for the locked exam + splits (SKILL7, BANK6). Every lane: `from v2.exam.loader import load, report, verify` (or run with sys.path pointing here).
  load(task, split, clean_only=False) -> list of dict(id, task, role, src, kind, user, answer, [tier])
  splits: exam | teach | ann_pool | neighbours (ids) | dev | cal | sel | cal_big(SKILL7)
Exam/teach/ann/cal(BANK6)/sel(BANK6) files are sha256-checked against MANIFEST.json on every load. NEVER use `exam` to build, tune or choose; score it once, at the end.
dev/cal/sel for SKILL7 are the lanes' EXISTING files (unchanged: dev results stay comparable); BANK6 dev/cal are the existing EASY batch-2 files, bigger cal/sel are new."""
import json, os, hashlib, math, collections
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
MAN = json.load(open(f"{HERE}/MANIFEST.json"))
PLAN = {"SKILL7": ("v2/task_search/EASY/PLAN.json", "SKILL7"), "BANK6": ("v2/task_search/EASY/PLAN2.json", "BANK6")}
EXISTING = {  # (task, split) -> list of (relative file, task filter or None)
    ("SKILL7", "dev"): [("v2/task_search/EASY/items_dev.jsonl", "SKILL7")],
    ("SKILL7", "cal"): [("v2/task_search/EASY/calib.jsonl", "SKILL7")],
    ("SKILL7", "cal_big"): [("v2/moe_quant/data/skill7_cal.jsonl", None)],
    ("SKILL7", "sel"): [("v2/moe_quant/data/skill7_sel.jsonl", None)],
    ("BANK6", "dev"): [("v2/task_search/EASY/items2_dev.jsonl", "BANK6")],
    ("BANK6", "cal"): [("v2/task_search/EASY/calib2.jsonl", "BANK6")]}
def _sha(p): return hashlib.sha256(open(p, "rb").read()).hexdigest()
def verify():
    bad = []
    for sec in ("files", "neighbours"):
        for nm, f in MAN[sec].items():
            if _sha(f"{ROOT}/{f['path']}") != f["sha256"]: bad.append(nm)
    if bad: raise RuntimeError(f"exam files changed vs MANIFEST: {bad}")
    return True
def prompt(task): return json.load(open(f"{ROOT}/{PLAN[task][0]}"))[PLAN[task][1]]["prompt"]
def labels(task): return json.load(open(f"{ROOT}/{PLAN[task][0]}"))[PLAN[task][1]]["labels"]
def new_label(task): return {"SKILL7": "Recipe", "BANK6": "Security"}[task]   # never in the prompt menu
def load(task, split, clean_only=False):
    t = task.upper()
    if split == "neighbours":
        f = MAN["neighbours"][f"{t.lower()}_neighbours"]; assert _sha(f"{ROOT}/{f['path']}") == f["sha256"]; return json.load(open(f"{ROOT}/{f['path']}"))
    key = f"{t.lower()}_{split}"
    if key in MAN["files"]:
        f = MAN["files"][key]; p = f"{ROOT}/{f['path']}"
        if _sha(p) != f["sha256"]: raise RuntimeError(f"{key} does not match MANIFEST")
        rows = [json.loads(l) for l in open(p)]
    elif (t, split) in EXISTING:
        rows = []
        for fn, tf in EXISTING[(t, split)]:
            for l in open(f"{ROOT}/{fn}"):
                j = json.loads(l)
                if tf is None or j.get("task") == tf: j.setdefault("task", t); j.setdefault("role", split); rows.append(j)
    else: raise KeyError(f"no split {split} for {task}")
    if clean_only: rows = [r for r in rows if r.get("tier", "clean") == "clean"]
    return rows
def wilson_lo(k, n, z=1.96):
    if n == 0: return 0.0
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)); return (c - m) / d
def report(items, base_ok, quant_ok):
    """items: exam rows; base_ok/quant_ok: dict id->bool (exact label match from real greedy generation).
    Returns retention of base-correct items overall / per kind / per source with Wilson lower bounds, plus rule 2 gates (>=98% overall, >=95% per kind) and rule 3 counts."""
    def ret(S):
        S = [i for i in S if base_ok.get(i["id"])]; k = sum(bool(quant_ok.get(i["id"])) for i in S); return dict(n=len(S), kept=k, ret=k / len(S) if S else None, wilson_lo=wilson_lo(k, len(S)))
    out = dict(overall=ret(items), by_kind={}, by_src={}, base_acc=sum(bool(base_ok.get(i["id"])) for i in items) / len(items), n_items=len(items),
               n_per_kind=dict(collections.Counter(i["kind"] for i in items)))
    for k in sorted({i["kind"] for i in items}): out["by_kind"][k] = ret([i for i in items if i["kind"] == k])
    for s in sorted({i["src"] for i in items}): out["by_src"][s] = ret([i for i in items if i["src"] == s])
    nu = [i for i in items if i["src"] in ("hwu64", "mtop", "bitext_bank")]; out["never_used_source_items"] = len(nu); out["never_used_retention"] = ret(nu)
    out["gate_overall_98"] = out["overall"]["ret"] is not None and out["overall"]["ret"] >= 0.98
    out["gate_kind_95"] = all(v["ret"] is not None and v["ret"] >= 0.95 for v in out["by_kind"].values())
    out["rule3"] = dict(total_ge_2000=len(items) >= 2000, per_kind_ge_300=all(v >= 300 for v in out["n_per_kind"].values()), never_used_ge_500=len(nu) >= 500)
    return out
if __name__ == "__main__":
    verify(); print("MANIFEST ok")
    for t in ("SKILL7", "BANK6"):
        for sp in ("exam", "teach", "ann_pool", "dev", "cal", "sel"):
            try: r = load(t, sp); print(t, sp, len(r), dict(collections.Counter(x["kind"] for x in r)))
            except KeyError: pass
