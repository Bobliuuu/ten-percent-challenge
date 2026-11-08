#!/usr/bin/env python3
"""Launch a tenpct bundle on Vast: search -> create (label tenpct-<job>) -> upload -> run -> poll -> pull results -> destroy.
Usage: launch_vast.py JOB --bundle bundles/JOB --results-dir DEST [--min-vram 46] [--max-dph 1.5] [--max-hours 3]
Never prints/logs instance_api_key or the Vast key. Only touches instances it created (label tenpct-*)."""
import argparse, json, os, subprocess, sys, time, datetime as dt
HERE = os.path.dirname(os.path.abspath(__file__))
KEY = os.path.join(HERE, ".vast_ssh_key")
IMAGE = "pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel"
def vast(*a, timeout=90):
    r = subprocess.run(["vastai", *a], capture_output=True, text=True, timeout=timeout)
    return r.returncode, r.stdout, r.stderr
def log(*a):
    print(dt.datetime.now(dt.timezone.utc).strftime("%H:%M:%SZ"), *a, flush=True)
def append(path, line):
    with open(os.path.expanduser(path), "a") as f: f.write(line + "\n")
def search(min_vram, max_vram, max_dph, disk, inet=300):
    q = f"num_gpus=1 gpu_ram>={min_vram} gpu_ram<={max_vram} reliability>0.97 inet_down>={inet} disk_space>={disk} rentable=true compute_cap>=800 cuda_vers>=12.8 cpu_ram>={os.environ.get('VAST_CPU_RAM',0)}"
    rc, out, _ = vast("search", "offers", q, "-o", "dph", "--raw")
    offers = [o for o in json.loads(out) if "CMP" not in o["gpu_name"] and o["dph_total"] <= max_dph]
    return offers
def mine(iid):
    rc, out, _ = vast("show", "instances", "--raw")
    for i in json.loads(out or "[]"):
        if i["id"] == iid:
            assert str(i.get("label", "")).startswith("tenpct-"); return i
def destroy(iid):
    i = mine(iid)
    if i is None: return
    for _ in range(5):
        vast("destroy", "instance", str(iid), "-y")
        time.sleep(4)
        if mine(iid) is None: log("destroyed (verified gone)", iid); return
    log("!!! DESTROY FAILED for", iid)
def ssh_args(host, port):
    return ["-i", KEY, "-p", str(port), "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null",
            "-o", "ConnectTimeout=15", "-o", "ServerAliveInterval=30", "-o", "LogLevel=ERROR"]
def sh(host, port, cmd, timeout=600):
    return subprocess.run(["ssh", *ssh_args(host, port), f"root@{host}", cmd], capture_output=True, text=True, timeout=timeout)
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("job"); ap.add_argument("--bundle"); ap.add_argument("--results-dir", required=True)
    ap.add_argument("--min-vram", type=int, default=46); ap.add_argument("--max-vram", type=int, default=100)
    ap.add_argument("--max-dph", type=float, default=1.5); ap.add_argument("--disk", type=int, default=150)
    ap.add_argument("--max-hours", type=float, default=3.0); ap.add_argument("--run-args", default="")
    ap.add_argument("--pull-every", type=int, default=300); ap.add_argument("--load-timeout", type=int, default=480); ap.add_argument("--min-inet", type=int, default=300)
    a = ap.parse_args()
    bundle = a.bundle or os.path.join(HERE, "bundles", a.job)
    label = f"tenpct-{a.job}"; os.makedirs(a.results_dir, exist_ok=True)
    offers = search(a.min_vram, a.max_vram, a.max_dph, a.disk, a.min_inet)
    if not offers: log("no offers"); sys.exit(2)
    iid = None; tried = set(); host = port = None
    for attempt in range(5):
        iid = None
        for o in [x for x in offers if x["machine_id"] not in tried][:8]:
            rc, out, err = vast("create", "instance", str(o["id"]), "--image", IMAGE, "--disk", str(a.disk), "--label", label,
                                "--ssh", "--direct", "--cancel-unavail", "--raw")
            try: res = json.loads(out)
            except Exception: res = {}
            if res.get("success") and res.get("new_contract"):
                iid = int(res["new_contract"]); break    # only the contract id is kept; instance_api_key is dropped
            log("offer", o["id"], "refused:", (str(res.get("msg") or res.get("error") or err))[:100])
        if iid is None: log("could not create"); sys.exit(3)
        tried.add(o["machine_id"])
        now = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%MZ")
        line = f"{now} vast id {iid} label {label} GPU {o['gpu_name']} {o['gpu_ram']/1024:.0f}GB ${o['dph_total']:.3f}/h (+5% guard) started"
        append("~/ten-percent/RUNS.md", "- " + line); append(os.path.join(HERE, "INSTANCES.md"), "- " + line); log(line)
        t1 = time.time()
        while time.time() - t1 < a.load_timeout:      # image pull on a slow host: give up, destroy, try another machine
            time.sleep(10); i = mine(iid)
            if i and i.get("actual_status") == "running" and i.get("ssh_host"): host, port = i["ssh_host"], i["ssh_port"]; break
        if host: break
        log("load timeout, destroying", iid); destroy(iid)
        append(os.path.join(HERE, "INSTANCES.md"), f"- id {iid} destroyed (load timeout {a.load_timeout}s, ~${(time.time()-t1)/3600*o['dph_total']:.2f})")
    if not host: log("no host came up"); sys.exit(4)
    t0 = time.time()
    try:
        rc, out, err = vast("attach", "ssh", str(iid), open(KEY + ".pub").read().strip())
        for _ in range(30):
            time.sleep(10)
            try:
                if sh(host, port, "echo ok", 30).stdout.strip() == "ok": break
            except Exception: pass
        else: raise RuntimeError("ssh never up")
        log("ssh up", host, port)
        import shlex; e = "-e " + shlex.quote("ssh " + " ".join(ssh_args(host, port)))
        sh(host, port, "mkdir -p /workspace/bundle")
        subprocess.run(f"rsync -az {e} {bundle}/ root@{host}:/workspace/bundle/", shell=True, check=True, timeout=1800)
        sh(host, port, f"cd /workspace/bundle && setsid nohup bash run.sh {a.run_args} > /workspace/run.log 2>&1 < /dev/null & echo started")
        log("started job")
        def pull():
            subprocess.run(f"rsync -az {e} root@{host}:/workspace/bundle/results/ {a.results_dir}/ ; rsync -az {e} root@{host}:/workspace/run.log {a.results_dir}/run.log",
                           shell=True, timeout=900)
        while True:
            time.sleep(a.pull_every)
            try:
                st = sh(host, port, "test -f /workspace/bundle/results/DONE && echo DONE; pgrep -f '[b]ash run.sh' >/dev/null && echo ALIVE", 60).stdout
            except Exception as ex: log("poll err", type(ex).__name__); continue
            pull()
            if "DONE" in st or "ALIVE" not in st: log("finished", st.split()); break
            if time.time() - t0 > a.max_hours * 3600: log("max-hours reached"); break
    finally:
        destroy(iid)
        hrs = (time.time() - t0) / 3600
        append(os.path.join(HERE, "INSTANCES.md"), f"- id {iid} {label} destroyed after {hrs:.2f} h (~${hrs*o['dph_total']:.2f})")
        append("~/ten-percent/RUNS.md", f"- {dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%dT%H:%MZ')} vast id {iid} {label} destroyed after {hrs:.2f} h (~${hrs*o['dph_total']:.2f})")
if __name__ == "__main__": main()
