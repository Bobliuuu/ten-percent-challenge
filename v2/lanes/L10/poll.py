import modal,sys,json
for tag,fid in [("llama","fc-01M6XS73MP9P4213XKN5EY10PB"),("omni7","fc-01M6XRS4AHZ9CQAQNA1JN56X3D"),("omni3","fc-01M6XRS4DS6EWJS8QYEN11NAVM")]:
    fc=modal.FunctionCall.from_id(fid)
    try: print(tag,"DONE",str(fc.get(timeout=0))[:300])
    except TimeoutError: print(tag,"running")
    except Exception as e: print(tag,"ERR",repr(e)[:300])
