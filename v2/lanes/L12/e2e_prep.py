"""Assemble the L12 artifact dir /r/art from the packed files (sha256-verified against MANIFEST.json) + config/tokenizer/chat template + shipped code. Prints the byte inventory."""
import sys, os, json, argparse, shutil, hashlib, glob
ap = argparse.ArgumentParser(); ap.add_argument("--out"); ap.add_argument("--root", default="/workspace/fin"); ap.add_argument("--art", default="/r/art"); ap.add_argument("--src", default="/r/pack1/packed"); ap.add_argument("--hub", default="/hf/hub"); a = ap.parse_args()
L = a.root + "/v2/lanes/L12"; M = json.load(open(L + "/MANIFEST.json")); os.makedirs(a.art, exist_ok=True)
snap = glob.glob(a.hub + "/models--Qwen--Qwen3.5-35B-A3B/snapshots/*")[0]
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""): h.update(b)
    return h.hexdigest()
for f, e in M["packed"]["files"].items():
    dst = f"{a.art}/{f}"
    if not (os.path.exists(dst) and os.path.getsize(dst) == e["bytes"]): shutil.copy(f"{a.src}/{f}", dst)
    assert sha(dst) == e["sha256"], f; print("ok", f, flush=True)
for f in ("config.json", "tokenizer.json", "tokenizer_config.json", "generation_config.json", "chat_template.jinja"): shutil.copy(f"{snap}/{f}", f"{a.art}/{f}")
CODE = ["l12_runtime.py", "answer.py", "feedback.py", "annot.py", "skill_task.json", "feedback_cal.jsonl"]
for f in CODE: shutil.copy(f"{L}/{f}", f"{a.art}/{f}")
inv = {f: os.path.getsize(f"{a.art}/{f}") for f in sorted(os.listdir(a.art)) if os.path.isfile(f"{a.art}/{f}")}
print("E2E PREP", json.dumps(dict(total=sum(inv.values()), n=len(inv), files={k: v for k, v in inv.items() if not k.startswith("layer")})))
json.dump(dict(inventory=inv, total=sum(inv.values())), open(a.out + "/prep.json", "w"), indent=1)
