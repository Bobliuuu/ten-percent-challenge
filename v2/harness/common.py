"""Shared helpers for the v2 proof harness. Task-agnostic: an exam is a JSONL of {"id","input","label"}."""
import hashlib, json, os, datetime

HERE = os.path.dirname(os.path.abspath(__file__))
BUDGET_FRACTION = 0.10
HARNESS_FILES = ["common.py", "stats.py", "lock.py", "size_check.py", "exam_remote.py", "worker.py", "task_base.py", "standin_predict.py"]


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for ch in iter(lambda: f.read(1 << 22), b""):
            h.update(ch)
    return h.hexdigest()


def tree_files(path):
    path = os.path.abspath(path)
    if os.path.isfile(path):
        return [(os.path.basename(path), path)]
    out = []
    for root, _, fs in os.walk(path):
        for f in fs:
            ap = os.path.join(root, f)
            out.append((os.path.relpath(ap, path), ap))
    return sorted(out)


def tree_sha256(path):
    h = hashlib.sha256()
    for rel, ap in tree_files(path):
        h.update(rel.encode() + b"\0" + sha256_file(ap).encode() + b"\n")
    return h.hexdigest()


LETTERS = "ABCDEFGHIJKLMN"   # MMLU uses A-D, MDC-6 up to 14 classes


def load_exam(path):
    """MMLU-format rows {id,subject,question,choices[4],answer(0-3)} (schema of v2/base/*.jsonl)."""
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def exam_sha256(rows):
    """TASK.md hash_def: sha256 over rows in file order of f'{id}\\t{answer}\\t{question}\\t{choices joined by |}\\n'."""
    h = hashlib.sha256()
    for r in rows:
        if "question" not in r:   # MDC-6 style rows: canonical JSON of the whole row
            h.update((json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")); continue
        h.update(f"{r['id']}\t{r['answer']}\t{r['question']}\t{'|'.join(r['choices'])}\n".encode("utf-8"))
    return h.hexdigest()


def candidate_view(rows):
    """What the candidate sees: no answers."""
    return [{k: v for k, v in r.items() if k != "answer"} for r in rows]


def labels(rows):
    return [LETTERS[r["answer"]] for r in rows]


def harness_hashes():
    return {f: sha256_file(os.path.join(HERE, f)) for f in HARNESS_FILES if os.path.exists(os.path.join(HERE, f))}


def write_json_atomic(path, obj, ro=False):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
    os.replace(tmp, path)
    if ro:
        os.chmod(path, 0o444)


def base_fp16_bytes_from_hf(repo, revision=None):
    """FP16 size of the base = 2 bytes x total parameters (from the safetensors headers on the Hub; no weights downloaded)."""
    from huggingface_hub import get_safetensors_metadata
    m = get_safetensors_metadata(repo, revision=revision)
    n = sum(m.parameter_count.values())
    return 2 * n, n
