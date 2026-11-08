"""Crash-safe launcher: python3 launch_l9b.py job.json [s|m2]  (spawns on the deployed app tenpct-l9b)"""
import modal, json, sys
job = json.load(open(sys.argv[1])); sz = sys.argv[2] if len(sys.argv) > 2 else ("m2" if job["stage"] == "pack" else "s")
c = modal.Function.from_name("tenpct-l9b", "run_" + sz).spawn(job); print("spawned", sz, job.get("tag"), c.object_id, flush=True)
