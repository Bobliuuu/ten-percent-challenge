#!/usr/bin/env python3
"""HARD COST CAP for OUR Vast.ai instances only (user: $300 TOTAL on Vast, separate from the AWS cap).
THE VAST ACCOUNT IS SHARED with another project (qweninqwen fleet manager, its own cap): this guard ONLY accounts for and destroys instances
whose label starts with MINE (tenpct-); it must NEVER touch any other instance. Always create our instances with --label tenpct-<name>.
Every 60 s: list instances (vastai show instances --raw), accrue dph_total * dt (conservative: dph_total + 5% for storage/egress),
persist state, and at DESTROY_AT dollars DESTROY ALL instances and keep verifying. Needs `vastai` CLI with api key set by the user.
Run: nohup python3 vast_guard.py > vast_guard.log 2>&1 &     State: vast_guard_state.json
Also: any single instance older than MAX_HOURS is destroyed."""
import json, subprocess, time, datetime as dt, os
CAP = 50.0; DESTROY_AT = 47.0; MAX_HOURS = 6.0; MINE = "tenpct-"
STATE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vast_guard_state.json")
def vast(*a):
    try:
        r = subprocess.run(["vastai", *a], capture_output=True, text=True, timeout=60)
        return r.returncode, r.stdout, r.stderr
    except Exception as e:   # never let a timeout kill the guard
        return 1, "", type(e).__name__
def instances():
    rc, out, err = vast("show", "instances", "--raw")
    if rc != 0: return None
    try: allinst = json.loads(out) if out.strip() else []
    except Exception: return None
    return [i for i in allinst if str(i.get("label") or "").startswith(MINE)]   # ONLY ours; never touch others
def destroy(i):
    assert str(i.get("label") or "").startswith(MINE), "refusing to destroy a non-tenpct instance"
    rc, out, err = vast("destroy", "instance", str(i["id"]))
    print(dt.datetime.now(dt.UTC).isoformat(), "DESTROY", i["id"], rc, (out + err)[:160].replace("\n", " "), flush=True)
def main():
    st = json.load(open(STATE)) if os.path.exists(STATE) else {"accrued": {}, "last": None, "tripped": False}
    while True:
        now = time.time(); inst = instances()
        if inst is None: time.sleep(30); continue
        dtm = (now - st["last"]) if st["last"] else 0
        for i in inst:
            rate = float(i.get("dph_total") or i.get("dph_base") or 10.0) * 1.05
            iid = str(i["id"])
            if iid not in st["accrued"]:
                start = float(i.get("start_date") or now)
                st["accrued"][iid] = max(0.0, (now - start) / 3600) * rate
            else:
                st["accrued"][iid] += min(dtm, 300) / 3600 * rate
            age_h = (now - float(i.get("start_date") or now)) / 3600
            if age_h > MAX_HOURS: destroy(i)
        st["last"] = now; total = sum(st["accrued"].values()); burn = sum(float(i.get("dph_total") or 10.0) * 1.05 for i in inst)
        print(dt.datetime.now(dt.UTC).isoformat(), f"total=${total:.2f} burn=${burn:.2f}/h instances={len(inst)}", flush=True)
        if total >= DESTROY_AT or st["tripped"]:
            st["tripped"] = True
            for i in inst: destroy(i)
        json.dump(st, open(STATE, "w"), indent=1); time.sleep(60)
if __name__ == "__main__": main()
