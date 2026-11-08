"""spawn census jobs on the deployed app: python3 launch.py jobs.json (deploy first: modal deploy moe_probe.py)"""
import modal, json, sys
J = json.load(open(sys.argv[1])); out = []
for j in J:
    c = modal.Function.from_name("tenpct-moe-census", "run_" + j.get("size", "s")).spawn(j); out.append((j["model"], j.get("size"), c.object_id)); print(out[-1], flush=True)
json.dump(out, open(sys.argv[1] + ".calls", "w"))
