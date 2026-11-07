"""usage: python3 launch.py cfgs.json   (after `modal deploy gen_c2.py`); spawned calls survive the client."""
import modal, json, sys, time
f = modal.Function.from_name("tenpct-fixreal", "run")
for c in json.load(open(sys.argv[1])):
    call = f.spawn(c); print(c.get("name"), call.object_id, flush=True); time.sleep(5)
