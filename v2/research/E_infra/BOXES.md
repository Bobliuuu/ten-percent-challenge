# Extra GPU boxes (Track E infra)  -- updated 2026-11-07 00:20Z

All: AWS acct 618283574049, profile `lab`, SSM-only (no inbound SG), `ssh tenpct-gN` via SSM ProxyCommand, key ~/.ssh/id_ed25519,
env at /opt/dlami/nvme/work/venv (vllm 0.11.0, transformers 4.57.1), repo clone in /opt/dlami/nvme/work/repo, `~/READY` marks env done.
Guards: on-host systemd poweroff (instance-initiated shutdown = terminate) at deadline + Mac backup `nohup local_guard.sh` (terminates at launch+8h20m;
dies if the Mac reboots, so the on-host guard is the primary). Hand terminate: `AWS_PROFILE=lab aws ec2 terminate-instances --region R --instance-ids ID`.

| alias | instance id | region/zone | type | GPU | vCPU/RAM | $/h (us-east-1 list; other regions slightly higher) | launched Z | deadline Z |
|---|---|---|---|---|---|---|---|---|
| tenpct-g1 | i-0f3e6bfab2f844fbf | us-east-2a | g7e.2xlarge | RTX PRO 6000 96GB | 8/64 | 3.363 | 21:32 (6 Nov) | 05:32 (7 Nov) |
| tenpct-g2 | i-0c68c077fe220f3ea | us-east-2a | g7e.2xlarge | RTX PRO 6000 96GB | 8/64 | 3.363 | 21:32 | 05:32 |
| tenpct-g3 | i-00fb643caf1cd2022 | eu-north-1a | g7e.4xlarge | RTX PRO 6000 96GB | 16/124 | 3.998 | 00:07 (7 Nov) | 08:07 |
| tenpct-g4 | i-01e334cc873378bde | eu-north-1a | g7e.4xlarge | RTX PRO 6000 96GB | 16/124 | 3.998 | 00:10 | 08:10 |
| tenpct-g5 | i-063a7c01f06d1b0df | us-east-1c | g7e.4xlarge | RTX PRO 6000 96GB | 16/124 | 3.998 | 00:10 | 08:10 |
| tenpct-g6 | i-0e27c8289574859c0 | ap-south-1a | g6e.4xlarge | L40S 48GB | 16/124 | ~3.004 | 00:10 | 08:10 |
| tenpct-g7 | i-039b94d6f9331007e | ap-northeast-2a | g6e.4xlarge | L40S 48GB | 16/124 | ~3.004 | 00:10 | 08:10 |
| tenpct-g8 | i-01892865a3ce7e505 | eu-north-1c | g7e.4xlarge | RTX PRO 6000 96GB | 16/124 | 3.998 | 00:14 | 08:14 |

g3-g7 verified over SSH at 00:20Z (GPU visible, READY present, deadline timer set). g8 launched later; verify with `ssh tenpct-g8`.

Hourly burn of the six new boxes (g3-g8) ~ $22.0/h (4x3.998 + 2x3.004; list prices) ; with g1/g2 total ~ $28.7/h.
Max exposure to deadline (8 h each): g3-g8 ~ $176, plus g1/g2 remaining. Running total spent at 00:20Z on g3-g8: ~ $2 (about 0.15 h each).

## Quota requests (step 1)
FILED: none. IAM user mmmzzz66g has no `servicequotas:RequestServiceQuotaIncrease` (AccessDenied for all 12 attempts: us-east-1, us-east-2, us-west-2 x L-DB2E81BA, L-417A185B, L-3819A6DF, L-7212CCBC; see quota_requests.txt).
Not worked around. Needs the user to either grant the permission to the user/profile or file the requests in the console (Service Quotas -> EC2).
Current quotas found (all regions scanned; vCPU): G on-demand L-DB2E81BA: us-east-1 16, us-east-2 16, eu-north-1 64, ap-south-1 64, ap-northeast-2 64, eu-central-1 8, eu-west-2 8, ap-northeast-1 8, others 0.
P on-demand L-417A185B: us-east-1 128 (the only one; fits p4d.24xlarge x1 (96) or p5.4xlarge x8 (16 each)); spot G us-east-1 16; spot P 0 everywhere.

## Capacity findings
- p5.4xlarge (1xH100 80GB) us-east-1: InsufficientInstanceCapacity in all 6 zones at 00:11Z. Retry worth doing (quota allows 8). p4d.24xlarge us-east-1 (96 vCPU) not tried yet.
- g7e.4xlarge: ok in eu-north-1 (a, c), us-east-1c; none in ap-south-1 a/b, ap-northeast-2 a/b, us-east-1b. g6e.4xlarge ok in ap-south-1a, ap-northeast-2a.
- Remaining quota headroom: eu-north-1 G 64-48=16 vCPU used up? (g3,g4,g8 = 48 vCPU, 16 left), ap-south-1 48 left, ap-northeast-2 48 left, us-east-1 G 0 left, us-east-2 G 0 left.
- Latency note: eu/ap boxes are far from the Mac but SSM works fine; HF downloads fast.

## Modal (step 4), probed 00:12Z with 9 concurrent spawns (workspace is shared with qwen-in-qwen)
Immediate (4 s): A10G, L40S. ~1 min: A100-80GB (56 s), A100-40GB (59 s), RTX-PRO-6000 (59 s), H200 (61 s), H100 (64 s). L4 119 s. B200 not scheduled in 150 s.
(Time includes image/container start; workspace GPU cap still applies to concurrent running jobs.)

## Not done / follow-ups
- Step 5 poll loop (2 h) NOT run by this agent invocation; to retry: `python3 launch_box.py NAME REGION TYPE 8` in this directory (prints JSON on success; add ssh Host + local_guard.sh manually).

## Batch 2 (retry driver, started 01:15Z, 3 h, every 5 min)
`retry_driver.py` (nohup, log retry_errors.log, successes in launched.jsonl; names tenpct-g9+; 12 h deadline; same hardened bootstrap, ssh alias + Mac guard added automatically).
Targets: 2x p5.4xlarge and 1x p4d.24xlarge (us-east-1, all zones), g7e.12xlarge/g6e.12xlarge (ap-south-1, ap-northeast-2), g7e.4xlarge/2xlarge (eu-north-1), g7e/g6e.2xlarge (eu-central-1, ap-northeast-1), g7e.2xlarge (eu-west-2).
Refresh the table below with `AWS_PROFILE=lab python3 update_boxes.py`. Driver ends itself after 3 h; a launched box always has its own on-host guard.
First result: tenpct-g9 = g6e.12xlarge (4x L40S, 192 GB) ap-south-1a launched 01:17Z.

## Deadline extensions (+16 h, done 01:2xZ)
tenpct-g3: on-host OnCalendar now 2026-11-08 00:07:35 UTC (verified `systemctl list-timers`); Mac guard restarted for 00:25Z 8 Nov. tenpct-g5: 2026-11-08 00:10:07 UTC; Mac guard 00:25Z 8 Nov. Extra cost: ~16 h x $4.0 = ~$64 each.
Extend another box: `ssh tenpct-gN "sudo sed -i 's|^OnCalendar=.*|OnCalendar=YYYY-MM-DD HH:MM:SS UTC|' /etc/systemd/system/g7e-deadline.timer && sudo systemctl daemon-reload && sudo systemctl restart g7e-deadline.timer"`, then `pkill -f "local_guard.sh REGION INSTANCEID"` and restart local_guard.sh with new seconds (Mac guards die if the Mac reboots).

## How to file the quota increases yourself (console)
1. Sign in to account 618283574049 as a user with Service Quotas rights (the `lab` CLI user cannot).
2. Open https://console.aws.amazon.com/servicequotas/home/services/ec2/quotas?region=us-east-1 (change `region=` for each region: us-east-2, us-west-2, eu-north-1, ap-south-1, ap-northeast-2).
3. Search `Running On-Demand G and VT instances` (L-DB2E81BA): Request increase at account level, value 256 (vCPU). Repeat for `Running On-Demand P instances` (L-417A185B), value 384 for p5 (p5.48xlarge = 192 vCPU). Optional spot: `All G and VT Spot Instance Requests` (L-3819A6DF), `All P Spot Instance Requests` (L-7212CCBC).
4. Direct URL pattern: https://console.aws.amazon.com/servicequotas/home/services/ec2/quotas/L-DB2E81BA?region=us-east-1 (swap code/region). Large asks may be routed to AWS Support; a case at https://support.console.aws.amazon.com/support/home#/case/create?issueType=service-limit-increase also works.
5. Or CLI with an authorized profile: `aws service-quotas request-service-quota-increase --service-code ec2 --quota-code L-DB2E81BA --desired-value 256 --region us-east-1`.
Status lists: https://console.aws.amazon.com/servicequotas/home/requests (per region).

## Local NVMe (instance store) by type (describe-instance-types, TB total; models up to ~810 GB need >1 TB)
>1 TB: g7e.2xlarge/4xlarge/8xlarge 1.9 TB; g7e.12xlarge 3.8; g7e.24xlarge 7.6; g7e.48xlarge 15.2; g6e.12xlarge 3.8; g6e.24xlarge 3.8; g6e.48xlarge 7.6; g5.12xlarge 3.8; g5.48xlarge 7.6; g6.48xlarge 7.5; p5.4xlarge 3.8 (1xH100 80GB, 256 GB RAM); p5.48xlarge 30.4; p4d/p4de.24xlarge 8.0.
<1 TB: g6e.2xlarge 0.45, g6e.4xlarge 0.6, g6e.8xlarge 0.9 (use the EBS root, 500 GB gp3 in launch_box.py, or resize).
Note: the g1-g8 boxes already see /opt/dlami/nvme at 1.9 TB (g7e) -- g6e.4xlarge (g6, g7) only 0.6 TB. GPU memory: g7e.12xlarge 2x96=192 GB; g6e.12xlarge 4x46=183 GB; p4de.24xlarge 8x80=640 GB.

<!-- AUTO:BEGIN -->
## Full box list and running cost (auto, 2026-11-07 01:19Z)

| alias | instance id | region | type | $/h | launched Z | deadline Z | state | est. $ so far |
|---|---|---|---|---|---|---|---|---|
| tenpct-g1 | i-0f3e6bfab2f844fbf | us-east-2 | g7e.2xlarge | 3.363 | 2026-11-06T21:32 | 2026-11-07T05:32 | running | 12.7 |
| tenpct-g2 | i-0c68c077fe220f3ea | us-east-2 | g7e.2xlarge | 3.363 | 2026-11-06T21:32 | 2026-11-07T05:32 | running | 12.7 |
| tenpct-g3 | i-00fb643caf1cd2022 | eu-north-1 | g7e.4xlarge | 3.998 | 2026-11-07T00:07 | 2026-11-08T00:07 (EXTENDED +16h) | running | 4.8 |
| tenpct-g4 | i-01e334cc873378bde | eu-north-1 | g7e.4xlarge | 3.998 | 2026-11-07T00:10 | 2026-11-07T08:10 | running | 4.6 |
| tenpct-g5 | i-063a7c01f06d1b0df | us-east-1 | g7e.4xlarge | 3.998 | 2026-11-07T00:10 | 2026-11-08T00:10 (EXTENDED +16h) | running | 4.6 |
| tenpct-g6 | i-0e27c8289574859c0 | ap-south-1 | g6e.4xlarge | 3.004 | 2026-11-07T00:10 | 2026-11-07T08:10 | running | 3.5 |
| tenpct-g7 | i-039b94d6f9331007e | ap-northeast-2 | g6e.4xlarge | 3.004 | 2026-11-07T00:10 | 2026-11-07T08:10 | running | 3.5 |
| tenpct-g8 | i-01892865a3ce7e505 | eu-north-1 | g7e.4xlarge | 3.998 | 2026-11-07T00:14 | 2026-11-07T08:14 | running | 4.3 |
| tenpct-g9 | i-01ef459d53edf4244 | ap-south-1 | g6e.12xlarge | 10.493 | 2026-11-07T01:17 | 2026-11-07T13:17 (12h) | running | 0.4 |
| tenpct-g10 | i-0de0bb143338dbb70 | ap-northeast-2 | g6e.12xlarge | 10.493 | 2026-11-07T01:17 | 2026-11-07T13:17 (12h) | running | 0.4 |
| tenpct-g11 | i-03e1be7d61ef20f37 | eu-north-1 | g7e.4xlarge | 3.998 | 2026-11-07T01:18 | 2026-11-07T13:17 (12h) | running | 0.1 |
| tenpct-g12 | i-0e2d05d688ad5c33a | eu-west-2 | g7e.2xlarge | 3.363 | 2026-11-07T01:18 | 2026-11-07T13:18 (12h) | running | 0.1 |
| tenpct-g13 | i-00aa3bff6a95f228c | ap-northeast-1 | g6e.2xlarge | 2.240 | 2026-11-07T01:19 | 2026-11-07T13:19 (12h) | running | 0.0 |

Current burn: $59.3/h. Estimated total spent on tenpct-g1..g13 so far: $52 (list prices, upper bound; excludes EBS/egress).
<!-- AUTO:END -->
