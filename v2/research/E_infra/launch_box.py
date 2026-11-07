#!/usr/bin/env python3
"""launch_box.py NAME REGION TYPE HOURS [ZONES] -- hardened non-g7e launch (same pattern as aws_g7e.py): no inbound SG,
SSM profile, 8h on-host poweroff guard, shutdown=terminate, user data = ten-percent g7e-bootstrap.sh. Prints JSON."""
import json, subprocess, sys, datetime as dt, os, hashlib
from pathlib import Path
name, region, itype, hours = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4])
zones_arg = sys.argv[5].split(",") if len(sys.argv) > 5 else None
env = dict(os.environ, AWS_PROFILE="lab")
def aws(*a, ok=False):
    r = subprocess.run(["aws", *a, "--region", region, "--output", "json"], capture_output=True, text=True, env=env)
    if r.returncode and not ok: raise RuntimeError(r.stderr.strip())
    return (json.loads(r.stdout) if r.stdout.strip() else {}) if not r.returncode else {"error": r.stderr.strip()}
ami = aws("ec2","describe-images","--owners","898082745236","--filters","Name=name,Values=Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 24.04) 20260925")["Images"][0]["ImageId"]
vpc = aws("ec2","describe-vpcs","--filters","Name=is-default,Values=true")["Vpcs"][0]["VpcId"]
subs = {s["AvailabilityZone"]: s["SubnetId"] for s in aws("ec2","describe-subnets","--filters",f"Name=vpc-id,Values={vpc}","Name=default-for-az,Values=true")["Subnets"]}
offered = [o["Location"] for o in aws("ec2","describe-instance-type-offerings","--location-type","availability-zone","--filters",f"Name=instance-type,Values={itype}")["InstanceTypeOfferings"]]
zones = [z for z in (zones_arg or sorted(offered)) if z in offered and z in subs]
sg = aws("ec2","describe-security-groups","--filters",f"Name=group-name,Values={name}",f"Name=vpc-id,Values={vpc}")["SecurityGroups"]
if sg: sg = sg[0]["GroupId"]
else:
    sg = aws("ec2","create-security-group","--group-name",name,"--description",name+" no inbound","--vpc-id",vpc)["GroupId"]
deadline = dt.datetime.now(dt.timezone.utc).replace(microsecond=0) + dt.timedelta(hours=hours)
ud = Path("~/ten-percent/v2/aws/g7e-bootstrap.sh").expanduser().read_text()
for k, v in {"@DEADLINE_ISO@": deadline.isoformat(), "@DEADLINE_CAL@": f"{deadline:%Y-%m-%d %H:%M:%S} UTC",
             "@PUBKEY@": Path("~/.ssh/id_ed25519.pub").expanduser().read_text().strip()}.items(): ud = ud.replace(k, v)
tags = [{"Key":"Name","Value":name},{"Key":"Project","Value":"ten-percent"},{"Key":"Experiment","Value":name},{"Key":"Deadline","Value":deadline.isoformat()}]
for z in zones:
    spec = {"ImageId":ami,"InstanceType":itype,"MinCount":1,"MaxCount":1,"IamInstanceProfile":{"Name":"inference-lab-machine"},
      "Placement":{"AvailabilityZone":z},"NetworkInterfaces":[{"DeviceIndex":0,"SubnetId":subs[z],"Groups":[sg],"AssociatePublicIpAddress":True,"DeleteOnTermination":True}],
      "BlockDeviceMappings":[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":500,"VolumeType":"gp3","Encrypted":True,"DeleteOnTermination":True}}],
      "MetadataOptions":{"HttpTokens":"required","HttpEndpoint":"enabled","HttpPutResponseHopLimit":1},
      "InstanceInitiatedShutdownBehavior":"terminate",
      "TagSpecifications":[{"ResourceType":k,"Tags":tags} for k in ("instance","volume")],"UserData":ud,
      "ClientToken":name+"-"+hashlib.sha256((z+deadline.isoformat()).encode()).hexdigest()[:10]}
    spec_file = f"/private/tmp/claude-501/{name}-{z}.json"; Path(spec_file).write_text(json.dumps(spec))
    r = subprocess.run(["aws","ec2","run-instances","--region",region,"--cli-input-json","file://"+spec_file,"--cli-read-timeout","60","--cli-connect-timeout","20","--output","json"],capture_output=True,text=True,env=env)
    if r.returncode == 0:
        i = json.loads(r.stdout)["Instances"][0]["InstanceId"]
        print(json.dumps({"name":name,"instance_id":i,"region":region,"zone":z,"type":itype,"deadline":deadline.isoformat(),"sg":sg})); sys.exit(0)
    print(z, r.stderr.strip()[:300], file=sys.stderr)
sys.exit(2)
