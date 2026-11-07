#!/usr/bin/env python3
"""Regenerates the AUTO block of BOXES.md (batch 2 table + honest running total) from launched.jsonl and live EC2 state."""
import json, subprocess, os, datetime as dt, re
from pathlib import Path
H = Path(__file__).parent
PRICE = {"g7e.2xlarge":3.363,"g7e.4xlarge":3.998,"g7e.12xlarge":8.3,"g6e.2xlarge":2.24,"g6e.4xlarge":3.004,"g6e.12xlarge":10.493,"p5.4xlarge":6.88,"p4d.24xlarge":21.96}  # us-east-1 list, approx (g7e.12xl estimated)
# (name,region,type,launched,deadline) for batch 1 + extensions; deadlines as currently set
rows = [("tenpct-g1","us-east-2","g7e.2xlarge","2026-11-06T21:32","2026-11-07T05:32","REDACTED-HOST"),
 ("tenpct-g2","us-east-2","g7e.2xlarge","2026-11-06T21:32","2026-11-07T05:32","REDACTED-HOST"),
 ("tenpct-g3","eu-north-1","g7e.4xlarge","2026-11-07T00:07","2026-11-08T00:07 (EXTENDED +16h)","REDACTED-HOST"),
 ("tenpct-g4","eu-north-1","g7e.4xlarge","2026-11-07T00:10","2026-11-07T08:10","REDACTED-HOST"),
 ("tenpct-g5","us-east-1","g7e.4xlarge","2026-11-07T00:10","2026-11-08T00:10 (EXTENDED +16h)","REDACTED-HOST"),
 ("tenpct-g6","ap-south-1","g6e.4xlarge","2026-11-07T00:10","2026-11-07T08:10","REDACTED-HOST"),
 ("tenpct-g7","ap-northeast-2","g6e.4xlarge","2026-11-07T00:10","2026-11-07T08:10","REDACTED-HOST"),
 ("tenpct-g8","eu-north-1","g7e.4xlarge","2026-11-07T00:14","2026-11-07T08:14","REDACTED-HOST")]
if (H/"launched.jsonl").exists():
    for l in (H/"launched.jsonl").read_text().splitlines():
        j = json.loads(l); rows.append((j["name"],j["region"],j["type"],j["launched_at"][:16],j["deadline"][:16]+" (12h)",j["instance_id"]))
now = dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)
out = ["<!-- AUTO:BEGIN -->", f"## Full box list and running cost (auto, {now:%Y-%m-%d %H:%MZ})", "",
       "| alias | instance id | region | type | $/h | launched Z | deadline Z | state | est. $ so far |", "|---|---|---|---|---|---|---|---|---|"]
tot_rate = tot = 0.0
for n,r,t,l,d,i in rows:
    st = subprocess.run(["aws","ec2","describe-instances","--region",r,"--instance-ids",i,"--query","Reservations[0].Instances[0].State.Name","--output","text"],capture_output=True,text=True,env=dict(os.environ,AWS_PROFILE="lab")).stdout.strip() or "?"
    p = PRICE.get(t,0); hrs = (now-dt.datetime.fromisoformat(l)).total_seconds()/3600
    cost = p*max(hrs,0) if st in ("running","pending") else p*max(hrs,0)  # upper bound if ended early
    tot += cost; tot_rate += p if st=="running" else 0
    out.append(f"| {n} | {i} | {r} | {t} | {p:.3f} | {l} | {d} | {st} | {cost:.1f} |")
out += ["", f"Current burn: ${tot_rate:.1f}/h. Estimated total spent on tenpct-g1..g{len(rows)} so far: ${tot:.0f} (list prices, upper bound; excludes EBS/egress)."]
out.append("<!-- AUTO:END -->")
p = H/"BOXES.md"; t = p.read_text()
blk = "\n".join(out)
t = re.sub(r"<!-- AUTO:BEGIN -->.*<!-- AUTO:END -->", lambda m: blk, t, flags=re.S) if "AUTO:BEGIN" in t else t + "\n" + blk + "\n"
p.write_text(t)
