"""L10 FEEDBACK COMMAND: python feedback.py --pack body.pack --annotations ann.jsonl --calib calib_old.jsonl --out body_fb.pack
ann rows {"text": request, "label": corrected label word, may be a NEW word}; at most 25. Closed-form protected edit (L11 recipe `o_prot_L8_m32`) of the EXISTING attention o_proj tensors of the last 8 layers,
which the pack keeps in bf16 for exactly this purpose (counted in the body). No head, adapter, lookup, prompt text or stored annotations; the output pack has the SAME size (tensors replaced in place).
text = the raw request (put into task.json's template); calib rows {"text": raw request, "gold": ...}: unlabeled old-skill requests, used only for the key covariance."""
import sys, os, json, argparse, tempfile, shutil
sys.dont_write_bytecode = True; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import packlib
a = argparse.ArgumentParser(); a.add_argument("--pack", required=True); a.add_argument("--annotations", required=True); a.add_argument("--calib", required=True); a.add_argument("--out", required=True); a.add_argument("--task", default=os.path.dirname(os.path.abspath(__file__)) + "/task.json")
a.add_argument("--method", default='{"tag":"o_prot_L8_m32","carrier":"o","solver":"prot","m":32,"layers":[20,21,22,23,24,25,26,27]}'); a = a.parse_args()
ann = [json.loads(l) for l in open(a.annotations) if l.strip()]; assert len(ann) <= 25, "at most 25 annotated items"
if not ann: shutil.copyfile(a.pack, a.out); print(json.dumps({"edited": []})); sys.exit()
td = tempfile.mkdtemp(); body = td + "/body"; packlib.write_hf_dir(a.pack, body); os.makedirs(td + "/data"); T = json.load(open(a.task))["prompt"]; wrap = lambda t: T.replace("{t}", t)
open(td + "/data/calib_old.jsonl", "w").write("".join(json.dumps(dict(text=wrap(json.loads(l)["text"]), gold=json.loads(l).get("gold"))) + "\n" for l in open(a.calib) if l.strip()))
from fblib import Env
env = Env(body, td + "/data", raw=True, ncal=min(300, sum(1 for _ in open(a.calib)))); mt = json.loads(a.method)
info = env.apply(mt, [(dict(text=r["text"], pid=env.pid(wrap(r["text"]))), r["label"]) for r in ann])
names = sorted(n[2:] if n.startswith("p:") else n for n in env.saved)
packlib.replace_raw(a.pack, a.out, {n: env._named(n).data for n in names}); print(json.dumps({"edited": names, "nsites": info["nsites"], "out_bytes": os.path.getsize(a.out)})); shutil.rmtree(td, ignore_errors=True)
