"""Assemble the Bonsai harness artifact dir (flat, no subdirs): lossless pack + ship scripts + task-neutral calibration prompts + tensor_manifest.json; writes DIR.base_arch.json (+ fp16 base bytes = 2 x logical params).
  python make_bonsai_artifact.py --pack PACKDIR --ship SHIPDIR --calib calib_prompts.jsonl --hf-snapshot HFDIR --out DIR"""
import argparse, json, os, shutil, sys, math
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE + "/h3")
import checks
ap = argparse.ArgumentParser(); [ap.add_argument("--" + k, required=True) for k in ("pack", "ship", "calib", "hf-snapshot", "out")]; a = ap.parse_args()
os.makedirs(a.out)
for f in os.listdir(a.pack):
    if os.path.isfile(os.path.join(a.pack, f)): shutil.copy(os.path.join(a.pack, f), a.out)
for f in ("answer.py", "feedback.py", "common.py", "bonsai_loader.py", "news_task.json"): shutil.copy(os.path.join(a.ship, f), a.out)
shutil.copy(a.calib, a.out + "/feedback_calib.jsonl")
arch = checks.arch_of_dir(getattr(a, "hf_snapshot")); fp16 = sum(2 * math.prod(s) for _, s in arch.values())
json.dump(arch, open(a.out + "/tensor_manifest.json", "w"))
json.dump(dict(base="prism-ml/Bonsai-8B-unpacked", arch=arch, base_fp16_bytes=fp16, config=json.load(open(getattr(a, "hf_snapshot") + "/config.json"))), open(a.out.rstrip("/") + ".base_arch.json", "w"))
print("built", a.out, "base_fp16_bytes", fp16, "artifact bytes", sum(os.path.getsize(os.path.join(a.out, f)) for f in os.listdir(a.out)))
