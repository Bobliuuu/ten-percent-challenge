import modal,sys
for tag,fid in [("fb","fc-01M6XZ1SXA2D18HYCB87F5V926"),("omni3","fc-01M6XWMHHDFG0HMV7SWEB6KV17")]:
    fc=modal.FunctionCall.from_id(fid)
    try: print(tag,"DONE",str(fc.get(timeout=0))[:200])
    except TimeoutError: print(tag,"running")
    except Exception as e: print(tag,"ERR",repr(e)[:300])
