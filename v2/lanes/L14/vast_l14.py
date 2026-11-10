"""Launch ONE Vast H200 box for L14 jobs, rsync the bundle, run the given job JSONs sequentially, results to /workspace/res14 (pull with pull_vast.sh). Prints only the instance id. Destroy: vastai destroy instance ID -y
usage: python vast_l14.py jobs/a.json jobs/b.json ...   (key ~/.config/vastai/vast_api_key is read by the vastai CLI; never printed)"""
import json, subprocess, time, os, sys, shutil, tempfile
HERE = os.path.dirname(os.path.abspath(__file__)); L9 = os.path.abspath(HERE + "/../L9b"); V = os.path.abspath(HERE + "/../../research/V_vast"); KEY = V + "/.vast_ssh_key"
def vast(*a, t=90): return subprocess.run(["vastai", *a], capture_output=True, text=True, timeout=t).stdout
B = tempfile.mkdtemp(); Q = B + "/q"; os.makedirs(Q + "/data")
for f in ("l14_run.py", "qlib.py", "qlib_b.py", "_helpers.py", "skill_task.json"): shutil.copy(f"{HERE}/{f}", Q)
shutil.copy(f"{HERE}/data/cal_general.jsonl", Q + "/data"); shutil.copy(f"{L9}/core.py", Q); shutil.copy(f"{L9}/qpack.py", Q)
for f in ("skill7_cal.jsonl", "skill7_sel.jsonl"): shutil.copy(f"{L9}/data/{f}", Q + "/data")
os.makedirs(B + "/queue")
for j in sys.argv[1:]: shutil.copy(j, B + "/queue/")
open(B + "/run.sh", "w").write('''#!/bin/bash
cd /workspace/bundle; pip install -q torch==2.8.0 transformers==5.22.2 accelerate numpy huggingface_hub sentencepiece protobuf tiktoken safetensors kernels 2>&1 | tail -1
export HF_HOME=/workspace/hf HF_HUB_ENABLE_HF_TRANSFER=0 PYTHONDONTWRITEBYTECODE=1; mkdir -p /workspace/res14 results; rm -rf /root/q; ln -sfn /workspace/bundle/q /root/q
for J in queue/*.json; do echo RUN $J; python -c "
import sys,json; sys.path.insert(0,'/root/q'); import l14_run; l14_run.RES='/workspace/res14'; l14_run.main(json.load(open('$J')))" > results/run_$(basename $J).log 2>&1; cp /workspace/res14/*.json results/ 2>/dev/null; done; touch results/DONE
''')
q = "num_gpus=1 gpu_ram>=120 gpu_ram<=200 reliability>0.97 inet_down>=800 disk_space>=400 rentable=true cuda_vers>=12.8"
offers = [o for o in json.loads(vast("search", "offers", q, "-o", "dph", "--raw")) if o.get("cpu_arch", "amd64") == "amd64" and "GB10" not in o["gpu_name"] and o["dph_total"] < 6]
iid = None
for o in offers[:8]:
    r = json.loads(vast("create", "instance", str(o["id"]), "--image", "pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel", "--disk", "400", "--label", "tenpct-l14", "--ssh", "--direct", "--cancel-unavail", "--raw") or "{}")
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
subprocess.run(["rsync", "-az", "-e", "ssh " + " ".join(O), B + "/", f"root@{host}:/workspace/bundle/"], check=True, timeout=1800)
subprocess.Popen(["ssh", "-f", "-n", *O, f"root@{host}", "cd /workspace/bundle && (setsid nohup bash run.sh > /workspace/run.log 2>&1 < /dev/null &) ; exit 0"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).wait()
open(HERE + f"/vast_{iid}.state", "w").write(json.dumps(dict(iid=iid, bundle=B)))   # host/port deliberately not stored
print("started", iid, flush=True)
