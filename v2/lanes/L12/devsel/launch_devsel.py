import modal, json, sys
f = modal.Function.from_name("tenpct-l12-devsel", "run"); tag, gridf = sys.argv[1], sys.argv[2]; args = sys.argv[3:]
c = f.spawn(tag, args, json.load(open(gridf))); print(tag, c.object_id)
