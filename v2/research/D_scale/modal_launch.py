"""launcher: python modal_launch.py variants.json  (app tenpct-dscale must be deployed: modal deploy modal_job.py)"""
import modal, json, sys
fn = modal.Function.from_name("tenpct-dscale", "run"); out = []
for v in json.load(open(sys.argv[1])): h = fn.spawn(v); out.append((v["name"], h.object_id)); print(out[-1], flush=True)
json.dump(out, open(sys.argv[1] + ".calls", "w"))
