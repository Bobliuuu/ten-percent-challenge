#!/usr/bin/env python3
"""Manual manager for tenpct-t3-* instances: ensure bundle+job started, pull results every 4 min, destroy when DONE. usage: t3_mgr.py <label> <jobidx> [--started]"""
import json, subprocess, sys, time, os, datetime as dt
HERE = os.path.dirname(os.path.abspath(__file__)); KEY = HERE + "/.vast_ssh_key"; B = HERE + "/bundles/t3"
label, job = sys.argv[1], sys.argv[2]; started = "--started" in sys.argv
RES = os.path.expanduser(f"~/ten-percent/v2/task_search/T3/results/{label}"); os.makedirs(RES, exist_ok=True)
def log(*a): print(dt.datetime.now(dt.timezone.utc).strftime("%H:%M:%SZ"), *a, flush=True)
def inst():
    for _ in range(25):
        r = subprocess.run(["vastai", "show", "instances", "--raw"], capture_output=True, text=True)
        try:
            for i in json.loads(r.stdout):
                if i.get("label") == "tenpct-" + label: return i
            return None
        except Exception: time.sleep(8)
    raise RuntimeError("vast list failed")
def ssh(i, cmd, t=120):
    return subprocess.run(["ssh", "-i", KEY, "-p", str(i["ssh_port"]), "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null", "-o", "ConnectTimeout=15", "-o", "ServerAliveInterval=20", "-o", "LogLevel=ERROR", f"root@{i['ssh_host']}", cmd], capture_output=True, text=True, timeout=t)
def rsync_to(i):
    e = f"ssh -i {KEY} -p {i['ssh_port']} -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR"
    return e
def pull(i):
    e = rsync_to(i)
    subprocess.run(f"rsync -az -e '{e}' root@{i['ssh_host']}:/workspace/bundle/results/ {RES}/ ; rsync -az -e '{e}' root@{i['ssh_host']}:/workspace/run.log {RES}/run.log", shell=True, timeout=900)
t0 = time.time()
try: subprocess.run(["vastai", "attach", "ssh", str(inst()["id"]), open(KEY + ".pub").read().strip()], capture_output=True)
except Exception as ex: log("attach skipped", type(ex).__name__)
while True:
    try: i = inst()
    except Exception: time.sleep(10); continue
    if i is None: log("instance gone"); break
    try:
        if ssh(i, "echo ok", 30).stdout.strip() == "ok": break
    except Exception: pass
    if time.time() - t0 > 2400: log("ssh never up; destroying"); subprocess.run(["vastai", "destroy", "instance", str(i["id"]), "-y"]); sys.exit(2)
    time.sleep(20)
if not started:
    ssh(i, "mkdir -p /workspace/bundle"); e = rsync_to(i)
    subprocess.run(f"rsync -az -e '{e}' {B}/ root@{i['ssh_host']}:/workspace/bundle/", shell=True, check=True, timeout=1800)
    subprocess.Popen(["ssh", "-i", KEY, "-p", str(i["ssh_port"]), "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null", "-o", "LogLevel=ERROR", f"root@{i['ssh_host']}", f"cd /workspace/bundle && setsid nohup bash run.sh {job} > /workspace/run.log 2>&1 < /dev/null &"], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    log("job started")
miss = 0
while True:
    time.sleep(240)
    try: i = inst()
    except Exception: continue
    if i is None: break
    try:
        rr = ssh(i, "echo SSHOK; pgrep -f 'runner.py' >/dev/null && echo ALIVE", 60); st = rr.stdout
        if "SSHOK" not in st: raise RuntimeError("ssh flake")
        pull(i)
    except Exception as ex: log("poll err", type(ex).__name__); continue
    log("poll", st.split())
    miss = (miss + 1) if "ALIVE" not in st else 0
    if miss >= 3: break
    if time.time() - t0 > 8 * 3600: log("max hours"); break
try:
    i = inst(); pull(i)
finally:
    i = inst()
    if i: 
        for _ in range(5):
            subprocess.run(["vastai", "destroy", "instance", str(i["id"]), "-y"]); time.sleep(5)
            if inst() is None: log("destroyed (verified)"); break
