import modal,sys,json
for tag,fid in [("llama","fc-01M6XS73MP9P4213XKN5EY10PB"),("omni7","fc-01M6XSZPKTYTJDVP07R2X0D2BE"),("omni3","fc-01M6XSZPQ1DT3QJS6XSQQRHYTY")]:
    fc=modal.FunctionCall.from_id(fid)
    try: print(tag,"DONE",str(fc.get(timeout=0))[:300])
    except TimeoutError: print(tag,"running")
    except Exception as e: print(tag,"ERR",repr(e)[:300])
