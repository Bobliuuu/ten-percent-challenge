#!/usr/bin/env python3
"""HARD COST CAP for the tenpct AWS boxes (user: $300 TOTAL, then turn them off).
Every 60 s: list running/pending instances tagged Experiment=tenpct-g* in all regions, accrue cost (conservative list prices,
x1.15 outside us-east-1/2), persist state, and at TERMINATE_AT dollars terminate EVERY tenpct-g* instance and keep verifying.
Run: nohup python3 cost_guard.py > cost_guard.log 2>&1 &     State: cost_guard_state.json"""
import json, subprocess, time, datetime as dt, os, sys
PROFILE = "lab"; CAP = 300.0; TERMINATE_AT = 285.0
REGIONS = ["us-east-1", "us-east-2", "us-west-2", "eu-north-1", "eu-west-2", "eu-central-1", "ap-south-1", "ap-northeast-1", "ap-northeast-2"]
PRICE = {"g7e.2xlarge": 3.36312, "g7e.4xlarge": 3.99816, "g7e.8xlarge": 5.26824, "g7e.12xlarge": 8.0, "g6e.xlarge": 1.86, "g6e.2xlarge": 2.24, "g6e.4xlarge": 3.0,
         "g6e.12xlarge": 10.49, "g6e.24xlarge": 15.07, "g6e.48xlarge": 30.13, "p5.4xlarge": 6.9, "p5.48xlarge": 98.32, "p4d.24xlarge": 21.96, "g5.2xlarge": 1.21}
STATE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cost_guard_state.json")
def aws(*a):
    try:
        r = subprocess.run(["aws", "--profile", PROFILE, "--output", "json", "--no-cli-pager", *a], capture_output=True, text=True, timeout=60)
        return json.loads(r.stdout) if r.returncode == 0 and r.stdout.strip() else None
    except Exception as e:   # timeouts / bad json must never kill the guard
        print("aws call failed:", type(e).__name__, flush=True); return None
LAST = {}
def running():
    out = []
    for reg in REGIONS:
        d = aws("ec2", "describe-instances", "--region", reg, "--filters", "Name=tag:Experiment,Values=tenpct-g*", "Name=instance-state-name,Values=pending,running,stopping,stopped")
        if d is None:
            out += LAST.get(reg, []); continue            # failed call: reuse last known list for this region (never undercount silently)
        cur = []
        for rs in d.get("Reservations", []):
            for i in rs["Instances"]:
                cur.append(dict(id=i["InstanceId"], region=reg, type=i["InstanceType"], launch=i["LaunchTime"], state=i["State"]["Name"],
                                name=next((t["Value"] for t in i.get("Tags", []) if t["Key"] == "Experiment"), "?")))
        LAST[reg] = cur; out += cur
    return out
def rate(i): return PRICE.get(i["type"], 12.0) * (1.0 if i["region"] in ("us-east-1", "us-east-2") else 1.15)
def terminate_all(inst):
    for reg in sorted({i["region"] for i in inst}):
        ids = [i["id"] for i in inst if i["region"] == reg]
        r = subprocess.run(["aws", "--profile", PROFILE, "ec2", "terminate-instances", "--region", reg, "--instance-ids", *ids], capture_output=True, text=True)
        print(dt.datetime.utcnow().isoformat(), "TERMINATE", reg, ids, r.returncode, r.stderr[:200], flush=True)
def main():
    st = json.load(open(STATE)) if os.path.exists(STATE) else {"accrued": {}, "last": None, "tripped": False}
    while True:
        now = time.time(); inst = running()
        if inst is None: time.sleep(30); continue
        dtm = (now - st["last"]) if st["last"] else 0
        for i in inst:
            if i["state"] not in ("running", "pending"): continue
            lt = dt.datetime.fromisoformat(i["launch"].replace("Z", "+00:00")).timestamp()
            if i["id"] not in st["accrued"]: st["accrued"][i["id"]] = (now - lt) / 3600 * rate(i)   # first sight: cost since launch
            else: st["accrued"][i["id"]] += min(dtm, 3600) / 3600 * rate(i)
        st["last"] = now; total = sum(st["accrued"].values()); burn = sum(rate(i) for i in inst if i["state"] in ("running", "pending"))
        print(dt.datetime.utcnow().isoformat(), f"total=${total:.2f} burn=${burn:.2f}/h boxes={len(inst)} remaining_h={(TERMINATE_AT-total)/burn if burn else 0:.2f}", flush=True)
        if total >= TERMINATE_AT or st["tripped"]:
            st["tripped"] = True; live = [i for i in inst if i["state"] in ("pending", "running", "stopping", "stopped")]
            if live: terminate_all(live)
        json.dump(st, open(STATE, "w"), indent=1); time.sleep(60)
if __name__ == "__main__": main()
