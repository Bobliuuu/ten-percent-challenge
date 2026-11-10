"""Annotation schema shared by L1/L2/L9b (see v2/lanes/ANNOTATION_SCHEMA.md). normalize() -> [{"user": full prompt, "answer": label}]."""
import json, os
def load_task(path=None):
    return json.load(open(path or os.path.join(os.path.dirname(os.path.abspath(__file__)), "skill_task.json")))
def normalize(rows, task):
    out = []
    for r in rows:
        lab = r.get("label", r.get("answer")) or (r["gold"][0]["name"] if r.get("gold") else None)
        if lab is None: raise ValueError(f"annotation row has no label: {list(r)}")
        q = r.get("query", r.get("user"))
        if q is None:
            if "text" not in r: raise ValueError(f"annotation row needs text or query: {list(r)}")
            q = task["prompt"].replace("{t}", r["text"])
        out.append({"user": q, "answer": lab})
    return out
def load_ann(path, task):
    return normalize([json.loads(l) for l in open(path) if l.strip()], task)
