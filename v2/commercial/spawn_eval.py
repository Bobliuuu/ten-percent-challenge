import modal, sys, time
f = modal.Function.from_name("tenpct-commercial-base", "run")
for m in sys.argv[1:]:
    print(f.spawn(m).object_id, m, flush=True); time.sleep(15)
