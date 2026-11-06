import modal, json, sys, time
f = modal.Function.from_name("tenpct-alt-probe3", "run")
for c in json.load(open(sys.argv[1])):
    print(f.spawn(c).object_id, c, flush=True)
