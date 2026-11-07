"""Shared helpers for the v3 generative harness. Pure stdlib (+numpy only in scorecard glue)."""
import hashlib, json, os, sys, datetime, importlib.util

HERE = os.path.dirname(os.path.abspath(__file__))
V2 = os.path.dirname(os.path.dirname(HERE))
T1 = os.path.join(V2, "tournament", "T1")
LABEL_FIELDS = ("gold", "gold_alts", "kind", "source")          # never visible to the candidate / base
VIEW_FIELDS = ("id", "query", "tools")                            # whole candidate view (whitelist, not blacklist)
FP16 = 2


def now(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha_bytes(b): return hashlib.sha256(b).hexdigest()


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for ch in iter(lambda: f.read(1 << 22), b""): h.update(ch)
    return h.hexdigest()


def tree_files(path):
    path = os.path.abspath(path); out = []
    for root, _, fs in os.walk(path):
        for f in fs:
            ap = os.path.join(root, f); out.append((os.path.relpath(ap, path), ap))
    return sorted(out)


def tree_sha256(path):
    h = hashlib.sha256()
    for rel, ap in tree_files(path): h.update(rel.encode() + b"\0" + sha_file(ap).encode() + b"\n")
    return h.hexdigest()


def tree_bytes(path): return sum(os.stat(ap).st_size for _, ap in tree_files(path))


def load_jsonl(p): return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
def dump_jsonl(rows, p):
    with open(p, "w", encoding="utf-8") as f:
        for r in rows: f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")


def rows_sha256(rows):
    h = hashlib.sha256()
    for r in rows: h.update((json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n").encode())
    return h.hexdigest()


def write_json(path, obj, ro=False):
    tmp = path + ".tmp"
    with open(tmp, "w") as f: json.dump(obj, f, indent=1, sort_keys=True, default=float)
    os.replace(tmp, path)
    if ro: os.chmod(path, 0o444)


def candidate_view(rows):
    """What the candidate and the base see. Whitelist only."""
    return [{k: r[k] for k in VIEW_FIELDS if k in r} for r in rows]


def get_scorer():
    spec = importlib.util.spec_from_file_location("t1_scorer", os.path.join(T1, "scorer.py"))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


def scorer_sha(): return sha_file(os.path.join(T1, "scorer.py"))


def locked_guard(exam_path, allow_locked=False):
    """Refuse to read/score the real locked exam (or anything with the locked hash) unless explicitly final.
    Checked on the file name AND on the sha256 recorded in tournament/T1/data/LOCK.json (hash computed on bytes; rows not parsed)."""
    p = os.path.abspath(exam_path)
    lk = json.load(open(os.path.join(T1, "data", "LOCK.json")))
    locked_sha = lk["splits"]["locked"]["file_sha256"]
    is_locked = os.path.basename(p) == "locked.jsonl" or sha_file(p) == locked_sha
    if is_locked and not (allow_locked and os.environ.get("TENPCT_FINAL") == "1"):
        raise PermissionError("REFUSING: this is the LOCKED exam. It is only runnable in the final proof "
                              "(--allow-locked AND env TENPCT_FINAL=1, with a harness lock).")
    return is_locked
