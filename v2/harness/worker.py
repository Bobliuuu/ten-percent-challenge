"""Candidate worker. Runs INSIDE the no-network sandbox. Sees only inputs (never labels, never base weights).
Contract: ARTIFACT/predict.py defines load(artifact_dir, memory_path_or_None) -> f(list[row dict: id,subject,question,choices]) -> list of "A".."D"."""
import argparse, importlib.util, json, os, socket, sys, time, glob
ap = argparse.ArgumentParser()
ap.add_argument("--artifact", required=True); ap.add_argument("--memory"); ap.add_argument("--inputs", required=True)
a = ap.parse_args()
ev = {}
try:   # evidence the network is off
    s = socket.create_connection(("1.1.1.1", 443), timeout=5); s.close(); ev["network"] = "REACHABLE"
except Exception as e:
    ev["network"] = f"blocked ({type(e).__name__})"
ev["hf_cache_files"] = len(glob.glob(os.path.expanduser("~/.cache/huggingface/**/*"), recursive=True))
inputs = json.load(open(a.inputs)); t0 = time.time()
sys.path.insert(0, a.artifact)
spec = importlib.util.spec_from_file_location("artifact_predict", os.path.join(a.artifact, "predict.py"))
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
f = mod.load(a.artifact, a.memory)
L = "ABCDEFGHIJKLMN"
def to_letter(x):   # accept a letter or a 0-based class index (int / digit string)
    s = str(x).strip()
    return s if s in tuple(L) else (L[int(s)] if s.isdigit() and int(s) < len(L) else "?")
pred = [to_letter(x) for x in f(inputs)]
assert len(pred) == len(inputs)
alt = None
import inspect
_p = inspect.signature(mod.load).parameters
if "use_bias" in _p and _p["use_bias"].default is True:   # only when the default is bias ON   # secondary, non-gating variant from the SAME run: bias off
    del f
    import gc; gc.collect()
    try:
        import torch; torch.cuda.empty_cache()
    except Exception: pass
    alt = [to_letter(x) for x in mod.load(a.artifact, a.memory, use_bias=False)(inputs)]
print("CAND_JSON:" + json.dumps(dict(pred=pred, pred_nobias=alt, seconds=time.time() - t0, evidence=ev)), flush=True)
