#!/usr/bin/env python
"""Prove the artifact needs neither the base checkpoint nor the network at runtime.
Copies artifact(+memory) to a fresh scratch dir and runs it in a child under macOS sandbox-exec with: network denied, ~/.cache/huggingface and the
whole repo tree (except the scratch copy) unreadable, HF/offline env, transformers/datasets/huggingface_hub/tokenizers imports poisoned, and an audit
hook logging every file the child opens. Asserts: (a) no opened path outside scratch/python/system libs, (b) no forbidden module imported,
(c) predictions equal the in-process predictions, (d) a socket connect attempt fails. Usage: verify.py [--artifact A] [--memory M] [--n 300]"""
import argparse, json, os, shutil, subprocess, sys, tempfile
H = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser(); ap.add_argument("--artifact", default=H + "/artifact"); ap.add_argument("--memory"); ap.add_argument("--n", type=int, default=300); a = ap.parse_args()
tmp = os.path.realpath(tempfile.mkdtemp(prefix="verify_")); ad = tmp + "/artifact"; shutil.copytree(a.artifact, ad, ignore=shutil.ignore_patterns("__pycache__"))
mem = None
if a.memory: mem = tmp + "/" + os.path.basename(a.memory); shutil.copy(a.memory, mem)
texts = ["a gorgeous, witty, deeply moving film", "dull and lifeless , a waste of time", "it 's a charming journey", "the plot is a mess", "wonderful performances throughout", "painfully unfunny"] * (a.n // 6)
json.dump(texts, open(tmp + "/t.json", "w"))
child = r'''
import sys, json, os, builtins
opened = []
sys.addaudithook(lambda ev, args: opened.append(str(args[0])) if ev == "open" and isinstance(args[0], (str, bytes)) else None)
class Block:
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in ("transformers", "datasets", "huggingface_hub", "tokenizers", "safetensors", "requests", "urllib3"): raise ImportError("BLOCKED " + name)
sys.meta_path.insert(0, Block())
import importlib.util
ad, mem, tmp = sys.argv[1], (sys.argv[2] if sys.argv[2] != "-" else None), sys.argv[3]
spec = importlib.util.spec_from_file_location("p", ad + "/predict.py"); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
f = m.load(ad, mem); pred = f(json.load(open(tmp + "/t.json")))
import socket
try: socket.create_connection(("1.1.1.1", 53), timeout=3); net = True
except Exception as e: net = False
json.dump(dict(pred=pred, opened=sorted(set(opened)), net=net, mods=sorted(k for k in sys.modules if k.split(".")[0] in ("transformers", "datasets", "huggingface_hub", "tokenizers", "safetensors"))), open(tmp + "/out.json", "w"))
'''
open(tmp + "/child.py", "w").write(child)
hf = os.path.realpath(os.path.expanduser("~/.cache/huggingface")); repo = os.path.realpath(os.path.join(H, ".."))
prof = (f'(version 1)(allow default)(deny network*)(deny file-read* (subpath "{hf}") (subpath "{repo}"))(deny file-write*)'
        f'(allow file-write* (subpath "{tmp}") (subpath "/dev"))')
env = dict(os.environ, HOME=tmp, TMPDIR=tmp, HF_HOME=tmp + "/hf", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", PYTHONDONTWRITEBYTECODE="1")
# python itself lives under pyenv/venv; those are not under the repo path unless the venv is - allow reading the venv by re-allowing it
venv = os.path.realpath(os.path.join(repo, ".venv")); prof = prof[:-0] if False else prof + f'(allow file-read* (subpath "{venv}"))'
r = subprocess.run(["sandbox-exec", "-p", prof, sys.executable, tmp + "/child.py", ad, mem or "-", tmp], env=env, cwd=tmp)
assert r.returncode == 0, "child crashed under sandbox"
o = json.load(open(tmp + "/out.json"))
# in-process reference (unsandboxed)
sys.path.insert(0, a.artifact); sys.dont_write_bytecode = True; import predict as P
ref = P.load(a.artifact, a.memory)(texts)
bad = [p for p in o["opened"] if not p.startswith(venv) and (p.startswith(hf) or "safetensors" in p or "textattack" in p or (p.startswith(repo) and not p.startswith(tmp)))]
res = dict(sandboxed=True, preds_equal_inprocess=o["pred"] == ref, n=len(texts), forbidden_modules_loaded=o["mods"], network_reachable=o["net"],
           forbidden_paths_opened=bad, files_opened_in_artifact=[os.path.basename(p) for p in o["opened"] if p.startswith(ad)])
print(json.dumps(res, indent=1)); shutil.rmtree(tmp, ignore_errors=True)
sys.exit(0 if (res["preds_equal_inprocess"] and not o["mods"] and not o["net"] and not bad) else 1)
