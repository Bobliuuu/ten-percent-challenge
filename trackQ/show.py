import json,sys
for f in sys.argv[1:]:
    r=json.load(open(f)); print(f,r["model"],r["params"],r["base_acc"])
    for c in r["cfgs"]:
        k=c["cfg"]; d=c["dev"]; print("  %-70s ent%.3f fix%.3f acc%.4f agree%.4f"%(json.dumps(k),c["ratio_ent"],c["ratio_fix"],d["acc"],d["agree"]))
