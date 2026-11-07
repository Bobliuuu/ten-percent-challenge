#!/usr/bin/env python
"""Build the stand-in artifact: a verbatim copy of the base's model files + answer.py/feedback.py (gen_common inlined, so the artifact is self-contained).
  make_standin.py --base Qwen/Qwen3-0.6B --out DIR  [--src LOCAL_MODEL_DIR]
Also writes DIR/../<name>.base_arch.json (tensor names/shapes/dtypes of the base) used by the no-new-parts check."""
import argparse, json, os, shutil, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import checks
ap = argparse.ArgumentParser(); ap.add_argument("--base", required=True); ap.add_argument("--out", required=True); ap.add_argument("--src")
a = ap.parse_args()
src = a.src
if not src:
    from huggingface_hub import snapshot_download
    src = snapshot_download(a.base, allow_patterns=["*.safetensors", "*.json", "merges.txt", "vocab.json", "*.jinja"])
if os.path.exists(a.out): sys.exit(f"{a.out} exists")
os.makedirs(a.out)
for f in os.listdir(src):
    p = os.path.join(src, f)
    if os.path.isfile(p) and (f.endswith((".safetensors", ".json", ".txt", ".jinja"))): shutil.copy(os.path.realpath(p), os.path.join(a.out, f))
gc = open(os.path.join(HERE, "gen_common.py")).read()
for n in ("answer.py", "feedback.py"):
    t = open(os.path.join(HERE, "standin", n)).read().replace("#@@GEN_COMMON@@", "\n# ---- inlined gen_common ----\n" + gc + "\n# ---- end ----\n")
    open(os.path.join(a.out, n), "w").write(t)
arch = checks.arch_of_dir(src)
import math; fp16 = sum(2 * math.prod(s) for _, s in arch.values())
json.dump(dict(base=a.base, arch=arch, base_fp16_bytes=fp16, config=json.load(open(os.path.join(src, "config.json")))), open(a.out.rstrip("/") + ".base_arch.json", "w"))
print("standin built", a.out, "base_fp16_bytes", fp16)
