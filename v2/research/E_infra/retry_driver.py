#!/usr/bin/env python3
"""Retry capacity every 5 min for 3 h; each slot = list of (region,type) alternatives; first success wins the slot."""
import json, subprocess, time, os, sys
from pathlib import Path
HERE = Path(__file__).parent; HOURS = 12; END = time.time() + 3*3600
LOG = HERE/"launched.jsonl"
slots = [[("us-east-1","p5.4xlarge")]]*2 + [[("us-east-1","p4d.24xlarge")],
  [("ap-south-1","g7e.12xlarge"),("ap-south-1","g6e.12xlarge")],
  [("ap-northeast-2","g7e.12xlarge"),("ap-northeast-2","g6e.12xlarge")],
  [("eu-north-1","g7e.4xlarge"),("eu-north-1","g7e.2xlarge")],
  [("eu-central-1","g7e.2xlarge"),("eu-central-1","g6e.2xlarge")],
  [("eu-west-2","g7e.2xlarge")],[("ap-northeast-1","g7e.2xlarge"),("ap-northeast-1","g6e.2xlarge")]]
done = [False]*len(slots)
if LOG.exists():
    pass
def nextname():
    n = 9
    while True:
        if f"tenpct-g{n}" not in (Path.home()/".ssh/config").read_text(): return f"tenpct-g{n}"
        n += 1
while time.time() < END and not all(done):
    for k, alts in enumerate(slots):
        if done[k]: continue
        for region, typ in alts:
            name = nextname()
            r = subprocess.run([sys.executable, str(HERE/"launch_box.py"), name, region, typ, str(HOURS)], capture_output=True, text=True)
            if r.returncode == 0:
                info = json.loads(r.stdout.strip().splitlines()[-1]); info["hours"] = HOURS; info["launched_at"] = time.strftime("%FT%TZ", time.gmtime())
                with (Path.home()/".ssh/config").open("a") as f:
                    f.write(f"Host {name}\n  HostName {info['instance_id']}\n  User ubuntu\n  IdentityFile ~/.ssh/id_ed25519\n  StrictHostKeyChecking accept-new\n  ServerAliveInterval 30\n  ProxyCommand env AWS_PROFILE=lab aws ssm start-session --region {region} --target %h --document-name AWS-StartSSHSession --parameters portNumber=%p\n")
                subprocess.Popen([str(HERE/"local_guard.sh"), region, info["instance_id"], str(HOURS*3600+1200)], start_new_session=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                with LOG.open("a") as f: f.write(json.dumps(info)+"\n")
                done[k] = True; break
            else:
                with (HERE/"retry_errors.log").open("a") as f: f.write(f"{time.strftime('%FT%TZ', time.gmtime())} {region} {typ}: {r.stderr.strip()[-200:]}\n")
    if not all(done): time.sleep(300)
with (HERE/"retry_errors.log").open("a") as f: f.write("driver finished\n")
