"""Run a candidate command with the network off and the label directories hidden.
  mode 'netns' : `sudo unshare -mn` = new network namespace (no interfaces at all) + new mount namespace in which every `hide_dirs` entry is
                 covered by an empty tmpfs; the command then drops back to the calling uid. OS-level, needs passwordless sudo (Linux box).
  mode 'guard' : python-level netguard (sitecustomize) + HF_HUB_OFFLINE; weaker (a non-python or ctypes socket would pass). Dev/Mac only.
  mode 'auto'  : netns if usable else guard.  A REAL result requires evidence['network'] to start with 'blocked' AND mode == 'netns'.
Evidence is measured by a probe run under the SAME wrapper right before the command."""
import json, os, shlex, subprocess, sys, tempfile
import common3 as C

PROBE = r'''
import json, os, socket, sys
ev = {}
try:
    s = socket.create_connection(("1.1.1.1", 443), timeout=3); s.close(); ev["network"] = "REACHABLE"
except Exception as e:
    ev["network"] = "blocked (%s)" % type(e).__name__
ev["hidden"] = {d: (len(os.listdir(d)) if os.path.isdir(d) else "absent") for d in sys.argv[1:]}
print("EVIDENCE:" + json.dumps(ev))
'''


def netns_usable():
    try:
        r = subprocess.run(["sudo", "-n", "unshare", "-mn", "true"], capture_output=True, timeout=20); return r.returncode == 0
    except Exception: return False


def wrap(argv, mode, hide_dirs=(), env=None, netlog=None):
    env = dict(env or {}); env.setdefault("HF_HUB_OFFLINE", "1"); env.setdefault("TRANSFORMERS_OFFLINE", "1")
    if mode == "guard":
        env["PYTHONPATH"] = os.path.join(C.HERE, "netguard") + os.pathsep + env.get("PYTHONPATH", "")
        if netlog: env["NETGUARD_LOG"] = netlog
        return argv, env
    uid, gid = os.getuid(), os.getgid()
    mounts = " ".join(f"mount -t tmpfs none {shlex.quote(os.path.abspath(d))} &&" for d in hide_dirs)
    envs = " ".join(f"{k}={shlex.quote(v)}" for k, v in env.items())
    inner = f"{mounts} exec setpriv --reuid={uid} --regid={gid} --init-groups -- env {envs} " + " ".join(shlex.quote(a) for a in argv)
    return ["sudo", "-n", "unshare", "-mn", "sh", "-c", inner], {}


def resolve_mode(mode):
    if mode == "auto": return "netns" if netns_usable() else "guard"
    return mode


def probe(mode, hide_dirs=(), python=sys.executable):
    mode = resolve_mode(mode)
    with tempfile.TemporaryDirectory() as td:
        argv, env = wrap([python, "-c", PROBE, *map(os.path.abspath, hide_dirs)], mode, hide_dirs, netlog=os.path.join(td, "ng.log"))
        r = subprocess.run(argv, capture_output=True, text=True, env={**os.environ, **env}, timeout=60)
    line = [l for l in r.stdout.splitlines() if l.startswith("EVIDENCE:")]
    ev = json.loads(line[0][9:]) if line else {"network": "UNKNOWN", "stderr": r.stderr[-300:]}
    ev["mode"] = mode; return ev


def run(argv, mode="auto", hide_dirs=(), env=None, cwd=None, timeout=7200, python=sys.executable):
    """-> dict(rc, stdout, stderr, evidence, netlog)"""
    mode = resolve_mode(mode)
    ev = probe(mode, hide_dirs, python)
    with tempfile.TemporaryDirectory() as td:
        netlog = os.path.join(td, "ng.log")
        a, e = wrap(argv, mode, hide_dirs, env, netlog)
        r = subprocess.run(a, capture_output=True, text=True, env={**os.environ, **e}, cwd=cwd, timeout=timeout)
        nl = open(netlog).read().splitlines() if os.path.exists(netlog) else []
    ev["network_attempts_logged"] = nl
    return dict(rc=r.returncode, stdout=r.stdout, stderr=r.stderr, evidence=ev)
