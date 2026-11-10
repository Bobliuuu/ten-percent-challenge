import modal, sys, json
f = modal.Function.from_name("tenpct-l14", "run"); j = json.load(open(sys.argv[1])); c = f.spawn(j); print(j["tag"], c.object_id)
