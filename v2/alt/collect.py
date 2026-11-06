import modal, json, sys
out=[]
for l in open(sys.argv[1]):
    i,n=l.split()
    try: r=modal.FunctionCall.from_id(i).get(timeout=0); out.append(dict(name=n,**r)); print(n,"acc",round(r["acc"],4),"base",round(r["base_acc"],4),"agree",round(r["agree"],3),"ratio",round(r["ratio_bf16"],4),"bpw",round(r["body_bpw"],3),r["per_task"])
    except TimeoutError: print(n,"pending")
    except Exception as e: print(n,"ERR",repr(e)[:200])
json.dump(out,open(sys.argv[2],"w"),indent=1)
