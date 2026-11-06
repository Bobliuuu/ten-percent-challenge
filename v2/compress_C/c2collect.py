import json,subprocess,glob,os
os.makedirs("c2res",exist_ok=True)
subprocess.run("modal volume ls tenpct-C-res / | grep c2_ | awk '{print $NF}' | sed 's#.*/##' | while read f; do modal volume get --force tenpct-C-res /$f c2res/$f; done",shell=True,capture_output=True)
for f in sorted(glob.glob("c2res/c2_*.json")):
    r=json.load(open(f)); print(r["cfg"]["name"],"L",r["cfg"]["L"],"dz",r["cfg"]["ec"]["dz"],"ratio %.4f"%r["ratio_bf16"],"base %.4f acc %.4f agree %.4f | bias: base %.4f acc_b %.4f"%(r["base_acc"],r["acc"],r["agree"],r["base_acc_b"],r.get("acc_b",-1)),r["per_task"])
