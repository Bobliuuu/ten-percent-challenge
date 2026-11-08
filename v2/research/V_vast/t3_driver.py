#!/usr/bin/env python3
"""queue driver: python t3_driver.py <label> <maxconc> TASK:JOB ... ; starts the next task whenever fewer than maxconc runner.py are alive on the box."""
import json, subprocess, sys, time, os
HERE = os.path.dirname(os.path.abspath(__file__)); K = HERE + "/.vast_ssh_key"
label, maxc, queue = sys.argv[1], int(sys.argv[2]), sys.argv[3:]
def box():
    for _ in range(10):
        r = subprocess.run(["vastai", "show", "instances", "--raw"], capture_output=True, text=True)
        try:
            for i in json.loads(r.stdout):
                if i.get("label") == "tenpct-" + label: return i
            return None
        except Exception: time.sleep(10)
def ssh(i, c, t=60):
    r = subprocess.run(["ssh", "-i", K, "-p", str(i["ssh_port"]), "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null", "-o", "LogLevel=ERROR", "-o", "ConnectTimeout=20", f"root@{i['ssh_host']}", c], capture_output=True, timeout=t)
    return r.returncode, r.stdout.decode(errors="replace")
while queue:
    try:
        i = box()
        if i is None: print("box gone"); break
        rc, o = ssh(i, "pgrep -fc 'runner.p[y]'")
        if rc in (0, 1) and o.strip().isdigit() and int(o.strip()) < maxc:
            tj = queue.pop(0); t, j = tj.split(":")
            cmd = f"cd /workspace/bundle && (T3_ITEMS=items_p_{t}.jsonl T3_CALIB=calib_p_{t}.jsonl T3_TAG=_p_{t} setsid nohup python code/runner.py {j} > /workspace/run_{t}.log 2>&1 < /dev/null &) ; echo started"
            print(time.strftime("%H:%M"), label, "start", tj, ssh(i, cmd)[1].strip(), flush=True)
            time.sleep(150)   # let it load before the next check
    except Exception as e: print("err", type(e).__name__, flush=True)
    time.sleep(60)
print("queue empty")
