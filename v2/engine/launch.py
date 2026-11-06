"""Crash-safe launcher: spawns on the DEPLOYED app (modal deploy sweep.py first); calls survive the client. Results land in volume hf-cache:/v2results and app logs.
usage: python3 launch.py cfgs.json"""
import modal, json, sys, time
f=modal.Function.from_name("tenpct-v2engine","run")
for c in json.load(open(sys.argv[1])):
    call=f.spawn(c); print(c.get("name"),call.object_id,flush=True); time.sleep(12)
