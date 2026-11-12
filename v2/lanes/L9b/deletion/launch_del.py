import modal, json, sys
job = json.load(open(sys.argv[1])); c = modal.Function.from_name("tenpct-l9b-del", "run_s").spawn(job); print("spawned", job.get("tag"), c.object_id, flush=True)
