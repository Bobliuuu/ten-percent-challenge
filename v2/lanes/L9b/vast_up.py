"""Create a tenpct-l9b Vast box (1 GPU >=120GB, amd64), attach the repo ssh key, rsync the bundle, start run.sh detached. Prints only the instance id. Destroy with: vastai destroy instance ID -y"""
import json, subprocess, time, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); V = os.path.join(HERE, "..", "..", "research", "V_vast"); KEY = os.path.join(V, ".vast_ssh_key")
def vast(*a, t=90): r = subprocess.run(["vastai", *a], capture_output=True, text=True, timeout=t); return r.stdout
q = "num_gpus=1 gpu_ram>=120 gpu_ram<=200 reliability>0.97 inet_down>=800 disk_space>=400 rentable=true cuda_vers>=12.8"
offers = [o for o in json.loads(vast("search", "offers", q, "-o", "dph", "--raw")) if o.get("cpu_arch", "amd64") == "amd64" and "GB10" not in o["gpu_name"] and o["dph_total"] < 6]
iid = None
for o in offers[:8]:
    r = json.loads(vast("create", "instance", str(o["id"]), "--image", "pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel", "--disk", "400", "--label", "tenpct-l9b", "--ssh", "--direct", "--cancel-unavail", "--raw") or "{}")
    if r.get("success") and r.get("new_contract"): iid = int(r["new_contract"]); print("instance", iid, o["gpu_name"], o["dph_total"], flush=True); break
assert iid
for _ in range(60):
    time.sleep(10)
    for i in json.loads(vast("show", "instances", "--raw") or "[]"):
        if i["id"] == iid and i.get("actual_status") == "running" and i.get("ssh_host"): host, port = i["ssh_host"], i["ssh_port"]; break
    else: continue
    break
else: sys.exit("not running")
vast("attach", "ssh", str(iid), open(KEY + ".pub").read().strip())
O = ["-i", KEY, "-p", str(port), "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null", "-o", "LogLevel=ERROR", "-o", "ConnectTimeout=15"]
for _ in range(30):
    try:
        if subprocess.run(["ssh", *O, f"root@{host}", "echo ok"], capture_output=True, text=True, timeout=30).stdout.strip() == "ok": break
    except Exception: pass
    time.sleep(10)
subprocess.run(["ssh", *O, f"root@{host}", "mkdir -p /workspace/bundle"], timeout=60)
subprocess.run(["rsync", "-az", "-e", "ssh " + " ".join(O), os.path.join(V, "bundles", "l9b") + "/", f"root@{host}:/workspace/bundle/"], check=True, timeout=1800)
subprocess.Popen(["ssh", "-f", "-n", *O, f"root@{host}", "cd /workspace/bundle && (setsid nohup bash run.sh > /workspace/run.log 2>&1 < /dev/null &) ; exit 0"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).wait(timeout=60)
print("started", iid, flush=True)
