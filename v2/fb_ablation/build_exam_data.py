"""Materialise the splits L11 needs from the shared exam loader (verifies the MANIFEST first). Exam rows are included ONLY for the neighbour ids (read-only scoring of neighbour retention)."""
import sys, json, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "exam")); import loader
loader.verify(); os.makedirs("exam_data", exist_ok=True)
for T in ("SKILL7", "BANK6"):
    nid = set(loader.load(T, "neighbours")); ex = [r for r in loader.load(T, "exam") if r["id"] in nid]
    D = dict(labels=loader.labels(T), new=loader.new_label(T), ann_pool=loader.load(T, "ann_pool"), teach=loader.load(T, "teach"), neigh=ex, cal=loader.load(T, "cal_big" if T == "SKILL7" else "cal"), sel=loader.load(T, "sel"))
    json.dump(D, open(f"exam_data/{T}.json", "w")); print(T, {k: len(v) if isinstance(v, list) else v for k, v in D.items()})
