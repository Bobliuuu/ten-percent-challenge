import modal, json, sys
f = modal.Function.from_name("tenpct-fbharness", "run")
print(f.spawn(json.loads(sys.argv[1])).object_id)
