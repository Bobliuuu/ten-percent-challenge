import modal, sys, time
f = modal.Function.from_name("tenpct-commercial-base", "run")
tasks=sys.argv[1]
for m in sys.argv[2:]:
    print(f.spawn(m, tasks).object_id, m, flush=True); time.sleep(15)
