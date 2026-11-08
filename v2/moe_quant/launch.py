import modal, json, sys
fn = modal.Function.from_name("tenpct-moe-qs", "run")
cfgs = json.load(open(sys.argv[1]))
for c in cfgs:
    h = fn.spawn([c]); print(c["name"], h.object_id, flush=True)
