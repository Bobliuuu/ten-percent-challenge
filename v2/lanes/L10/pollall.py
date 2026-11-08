import modal
for tag,fid in [("fb","fc-01M6XZRDWSG7FZ5ZMK35A4HKJ1"),("exam","fc-01M6XZAAE64DEPX3HJ4P11XS8Z"),("omni3","fc-01M6XWMHHDFG0HMV7SWEB6KV17")]:
    fc=modal.FunctionCall.from_id(fid)
    try: print(tag,"DONE",str(fc.get(timeout=0))[:200])
    except TimeoutError: print(tag,"running")
    except Exception as e: print(tag,"ERR",repr(e)[:300])
