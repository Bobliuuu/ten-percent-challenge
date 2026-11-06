import modal, time
f = modal.Function.from_name("tenpct-commercial-bino", "run")
for s in ["1.5B","7B"]:
    print(f.spawn(s).object_id, s, flush=True); time.sleep(15)
