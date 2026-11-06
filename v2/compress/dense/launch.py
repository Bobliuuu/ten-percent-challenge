"""spawn jobs against the deployed app (durable). usage: python launch.py cfgs.json"""
import modal, json, sys, time
cs=json.load(open(sys.argv[1])); ids=[]
fns={k:modal.Function.from_name("tenpct-dense2","run_"+k) for k in "smlx"}
for c in cs:
    h=fns[c.get("size","s")].spawn(c); ids.append((c.get("model"),c.get("name"),h.object_id)); print(ids[-1],flush=True)
json.dump(ids,open(sys.argv[1]+".calls","w"))
