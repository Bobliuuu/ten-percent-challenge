"""create/destroy one Vast box labelled tenpct-l1p (only touches its own label). usage: box.py create | destroy ID | status"""
import sys, os, json, time
sys.path.insert(0, os.path.expanduser("~/ten-percent/v2/research/V_vast")); os.environ.setdefault("VAST_CPU_RAM", "48")
import launch_vast as L
cmd = sys.argv[1]
if cmd == "create":
    offers = L.search(46, 100, 0.8, 120, 300); print(len(offers), "offers"); iid = None
    for o in offers[:10]:
        rc, out, err = L.vast("create", "instance", str(o["id"]), "--image", L.IMAGE, "--disk", "120", "--label", "tenpct-l1p", "--ssh", "--direct", "--cancel-unavail", "--raw")
        try: res = json.loads(out)
        except Exception: res = {}
        if res.get("success") and res.get("new_contract"):
            iid = int(res["new_contract"]); line = f"vast id {iid} label tenpct-l1p GPU {o['gpu_name']} ${o['dph_total']:.3f}/h started (L1+ experiments)"
            L.append("~/ten-percent/RUNS.md", "- " + time.strftime("%Y-%m-%dT%H:%MZ", time.gmtime()) + " " + line); L.append(L.HERE + "/INSTANCES.md", "- " + line); print(line); break
        print("refused", str(res)[:100])
    print("ID", iid)
    for _ in range(60):
        time.sleep(10); i = L.mine(iid)
        if i and i.get("actual_status") == "running" and i.get("ssh_host"): print("HOST", i["ssh_host"], i["ssh_port"]); break
    print(L.vast("attach", "ssh", str(iid), open(L.KEY + ".pub").read().strip())[1][:100])
elif cmd == "destroy": L.destroy(int(sys.argv[2]))
else: print([ (i["id"], i["label"], i["actual_status"]) for i in json.loads(L.vast("show", "instances", "--raw")[1])])
