"""I4: dispatcher. Prints the next launch commands for experiments whose trigger holds; NEVER launches GPUs and never spends.
  python dispatch.py [--status] [--budget]
Predicates in trigger.start_when / cancel_when (very small grammar): 'now' | 'after X01' | 'done(X01)' | 'exists(path)' | 'splits_ok' ; free text = manual (never auto-ready).
A split-hash mismatch (SPLITS.json vs queue.json splits_hash vs files on disk) blocks everything."""
import os, sys, json, re, hashlib
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
def sha(p): return hashlib.sha256(open(p, "rb").read()).hexdigest()
def splits_ok(q):
    sp = json.load(open(HERE + "/SPLITS.json"))
    for n, f in sp["files"].items():
        if sha(os.path.join(ROOT, f["path"])) != f["sha256"]: return False, f"{n} differs from SPLITS.json"
    if q.get("splits_hash") and q["splits_hash"] != sp["split_hash"]: return False, "queue.json splits_hash != SPLITS.json"
    return True, sp["split_hash"]
def pred(s, q, status):
    if s is None: return False
    s = s.strip()
    if s == "now": return True
    m = re.fullmatch(r"(?:after |done\()(X\d+)\)?", s)
    if m: return status.get(m.group(1)) == "done"
    m = re.fullmatch(r"exists\((.+)\)", s)
    if m: return os.path.exists(os.path.join(ROOT, m.group(1)))
    return False   # free text = manual decision by the coordinator
def main():
    q = json.load(open(HERE + "/queue.json")); status = {e["id"]: e["status"] for e in q["experiments"]}
    ok, why = splits_ok(q); print("splits:", "OK" if ok else "BLOCKED", why)
    bp = HERE + "/BUDGET.json"; bud = json.load(open(bp)) if os.path.exists(bp) else None
    print("budget ledger:", "present" if bud else "MISSING (create before launching)")
    for e in q["experiments"]:
        if "--status" in sys.argv: print(f'{e["id"]:5} {e["status"]:10} {e["title"]}'); continue
        if e["status"] not in ("designed", "ready"): continue
        t = e.get("trigger", {}); ready = pred(t.get("start_when"), q, status) and not pred(t.get("cancel_when"), q, status) and ok
        print(("READY   " if ready else "waiting ") + e["id"], "|", t.get("start_when"), "| est $", e.get("est_usd_vast"), "|", e.get("spec"))
        if ready: print(f'   launch (coordinator approves spend): see spec {e["spec"]}; results -> {e["results_dir"]}')
if __name__ == "__main__": main()
