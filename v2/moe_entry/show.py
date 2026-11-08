"""pull result JSONs from volume tenpct-moe into v2/moe_entry/results and print a table"""
import json, glob, os, subprocess, sys
D = os.path.dirname(os.path.abspath(__file__)) + "/results"; os.makedirs(D, exist_ok=True)
subprocess.run("modal volume ls tenpct-moe res 2>/dev/null | grep -v 'outs_\\|\\.pt\\|base_' | sed 's#.*res/##' ", shell=True, capture_output=True)
names = subprocess.run("modal volume ls tenpct-moe res", shell=True, capture_output=True, text=True).stdout.split()
for n in names:
    f = n.split("/")[-1]
    if f.endswith(".json") and not f.startswith("outs_") and not f.startswith("base_") and not os.path.exists(f"{D}/{f}"):
        subprocess.run(["modal", "volume", "get", "tenpct-moe", f"res/{f}", f"{D}/{f}"], capture_output=True)
for p in sorted(glob.glob(D + "/*.json")):
    try: r = json.load(open(p))
    except Exception: continue
    ds = r.get("cfg", {}).get("ds", "news")
    for sn, s in r.get("sets", {}).items():
        print(ds, r["name"], sn, "ratio", round(r["bytes"]["ratio"], 4), "acc", round(s["acc"], 4), "ret", round(s.get("ret", 0), 4), "lo", round(s.get("ret_lo", 0), 4), "mincell", round(s.get("min_cell", 0), 3), "n_bc", s.get("n_bc"), r.get("secs"))
