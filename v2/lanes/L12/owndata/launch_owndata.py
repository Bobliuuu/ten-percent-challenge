import modal, sys
f = modal.Function.from_name("tenpct-l12-owndata", "run"); tag = sys.argv[1]; c = f.spawn(tag); print(tag, c.object_id)
