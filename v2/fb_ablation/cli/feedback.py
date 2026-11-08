"""feedback command: python feedback.py --model DIR --annotations ann.jsonl --calib calib_old.jsonl [--method '{...json...}']
ann.jsonl rows: {"text": request, "label": corrected skill word (may be a NEW word)}. Edits EXISTING tensors of DIR in place (safetensors shards rewritten); no new file, no side memory.
Prints wall time. Calibration texts (shipped, counted in bytes) are used only for the key covariance, never labels."""
import argparse, json, os, sys, time, glob, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from fblib import Env
from safetensors import safe_open
from safetensors.torch import save_file
ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True); ap.add_argument("--annotations", required=True); ap.add_argument("--calib", required=True)
ap.add_argument("--method", default='{"tag":"cli","carrier":"down","solver":"prot","layers":[20,21,22,23,24,25,26,27],"m":16,"lam":1000,"awt":10}'); ap.add_argument("--no-write", action="store_true"); a = ap.parse_args()
t0 = time.time(); os.makedirs("/tmp/fbcli", exist_ok=True)
for f in ("calib_old.jsonl",): os.system(f"cp {a.calib} /tmp/fbcli/{f}")
env = Env(a.model, "/tmp/fbcli"); t1 = time.time(); mt = json.loads(a.method)
items = []
for l in open(a.annotations):
    r = json.loads(l); items.append((dict(text=r["text"], pid=env.pid(r["text"])), r["label"]))
info = env.apply(mt, items); t2 = time.time()
print(json.dumps(dict(nsites=info["nsites"], load_and_calibration_s=round(t1 - t0, 1), edit_s=round(t2 - t1, 1), tensors=sorted(env.saved))))
if not a.no_write:
    idxf = os.path.join(a.model, "model.safetensors.index.json"); wm = json.load(open(idxf))["weight_map"] if os.path.exists(idxf) else None
    names = [n[2:] if n.startswith("p:") else n for n in env.saved]; by = {}
    for n in names: by.setdefault(wm[n] if wm else "model.safetensors", []).append(n)
    for shard, ns in by.items():
        p = os.path.join(a.model, shard); d = {}
        with safe_open(p, "pt") as f:
            meta = f.metadata()
            for k in f.keys(): d[k] = f.get_tensor(k)
        for n in ns: d[n] = env._named(n).data.detach().cpu().contiguous()
        save_file(d, p, metadata=meta or {"format": "pt"})
    print("wrote", {k: len(v) for k, v in by.items()}, "total_s", round(time.time() - t0, 1))
