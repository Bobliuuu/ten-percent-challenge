import modal, sys
fn, tag, script = sys.argv[1], sys.argv[2], sys.argv[3]; f = modal.Function.from_name("tenpct-l12e", fn); c = f.spawn(tag, script, sys.argv[4:]); print(tag, c.object_id)
