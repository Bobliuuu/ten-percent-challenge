"""usage: python3 launch.py cfgs.json   (after `modal deploy legal.py`); crash-safe spawn."""
import modal, json, sys, time
f = modal.Function.from_name("tenpct-fblegal", "run")
for c in json.load(open(sys.argv[1])):
    call = f.spawn(c); print(c["name"], call.object_id, flush=True); time.sleep(5)
