import modal, json, sys
f = modal.Function.from_name("tenpct-l12", "run"); tag, cf = sys.argv[1], sys.argv[2]
c = f.spawn(tag, json.load(open(cf))); print(tag, c.object_id)
