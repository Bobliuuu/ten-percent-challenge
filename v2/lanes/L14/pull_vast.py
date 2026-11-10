"""Pull L14 results from all tenpct-l14 Vast boxes into results/ (hosts never printed). usage: python pull_vast.py"""
import json, subprocess, os
HERE = os.path.dirname(os.path.abspath(__file__)); KEY = os.path.abspath(HERE + "/../../research/V_vast/.vast_ssh_key")
ins = json.loads(subprocess.run(["vastai", "show", "instances", "--raw"], capture_output=True, text=True).stdout or "[]")
for i in ins:
    if i.get("label") != "tenpct-l14" or not i.get("ssh_host"): continue
    O = f"ssh -i {KEY} -p {i['ssh_port']} -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -o ConnectTimeout=15"
    r = subprocess.run(["rsync", "-az", "-e", O, f"root@{i['ssh_host']}:/workspace/bundle/results/", HERE + f"/results/vast_{i['id']}/"], capture_output=True, text=True)
    print(i["id"], i.get("actual_status"), "rc", r.returncode, sorted(os.listdir(HERE + f"/results/vast_{i['id']}")) if os.path.isdir(HERE + f"/results/vast_{i['id']}") else [])
