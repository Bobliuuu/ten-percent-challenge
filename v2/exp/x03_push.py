#!/usr/bin/env python3
"""Push cfgs + job names to a running tenpct-x03* box (queue worker).  x03_push.py LABEL job1 job2 ...   (cfgs read from v2/exp/cfgs/<job>.json)
Never prints host/port/keys."""
import json, os, subprocess, sys
HERE = os.path.dirname(os.path.abspath(__file__)); KEY = os.path.expanduser("~/ten-percent/v2/research/V_vast/.vast_ssh_key")
def box(label):
    out = subprocess.run(["vastai", "show", "instances", "--raw"], capture_output=True, text=True).stdout
    for i in json.loads(out):
        if i.get("label") == label and i.get("ssh_host"): return i["ssh_host"], i["ssh_port"]
    raise SystemExit("no such running box: " + label)
def main():
    label, jobs = sys.argv[1], sys.argv[2:]; assert label.startswith("tenpct-x03")
    host, port = box(label); ssh = ["ssh", "-i", KEY, "-p", str(port), "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null", "-o", "LogLevel=ERROR"]
    for j in jobs: subprocess.run(["scp", "-q", "-i", KEY, "-P", str(port), "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null", "-o", "LogLevel=ERROR", f"{HERE}/cfgs/{j}.json", f"root@{host}:/workspace/bundle/v2/exp/cfgs/{j}.json"], check=True)
    extra = [f for f in os.environ.get("X03_PUSH_FILES", "").split(",") if f]
    for f in extra: subprocess.run(["scp", "-q", "-i", KEY, "-P", str(port), "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null", "-o", "LogLevel=ERROR", f, f"root@{host}:/workspace/bundle/v2/exp/results/X03/{os.path.basename(f)}"], check=True)
    r = subprocess.run(ssh + [f"root@{host}", "cat >> /workspace/bundle/jobs.txt"], input="\n".join(jobs) + "\n", text=True, capture_output=True); print("pushed", jobs, r.returncode)
if __name__ == "__main__": main()
