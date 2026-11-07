import modal, sys
for m in sys.argv[1:]:
    f = modal.Function.from_name("tenpct-tox-routing-probe", "run_big" if "32B" in m else "run_small"); c = f.spawn(m); print("SPAWNED", m, c.object_id, flush=True)
