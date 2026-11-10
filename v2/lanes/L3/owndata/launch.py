import modal
c = modal.Function.from_name("tenpct-l3-owndata", "run").spawn(); print("spawned", c.object_id, flush=True)
