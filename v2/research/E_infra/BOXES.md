# Extra GPU boxes (Track E infra)  -- updated 2026-11-07 00:20Z

All: AWS acct 618283574049, profile `lab`, SSM-only (no inbound SG), `ssh tenpct-gN` via SSM ProxyCommand, key ~/.ssh/id_ed25519,
env at /opt/dlami/nvme/work/venv (vllm 0.11.0, transformers 4.57.1), repo clone in /opt/dlami/nvme/work/repo, `~/READY` marks env done.
Guards: on-host systemd poweroff (instance-initiated shutdown = terminate) at deadline + Mac backup `nohup local_guard.sh` (terminates at launch+8h20m;
dies if the Mac reboots, so the on-host guard is the primary). Hand terminate: `AWS_PROFILE=lab aws ec2 terminate-instances --region R --instance-ids ID`.

| alias | instance id | region/zone | type | GPU | vCPU/RAM | $/h (us-east-1 list; other regions slightly higher) | launched Z | deadline Z |
|---|---|---|---|---|---|---|---|---|
| tenpct-g1 | REDACTED-HOST | us-east-2a | g7e.2xlarge | RTX PRO 6000 96GB | 8/64 | 3.363 | 21:32 (6 Nov) | 05:32 (7 Nov) |
| tenpct-g2 | REDACTED-HOST | us-east-2a | g7e.2xlarge | RTX PRO 6000 96GB | 8/64 | 3.363 | 21:32 | 05:32 |
| tenpct-g3 | REDACTED-HOST | eu-north-1a | g7e.4xlarge | RTX PRO 6000 96GB | 16/124 | 3.998 | 00:07 (7 Nov) | 08:07 |
| tenpct-g4 | REDACTED-HOST | eu-north-1a | g7e.4xlarge | RTX PRO 6000 96GB | 16/124 | 3.998 | 00:10 | 08:10 |
| tenpct-g5 | REDACTED-HOST | us-east-1c | g7e.4xlarge | RTX PRO 6000 96GB | 16/124 | 3.998 | 00:10 | 08:10 |
| tenpct-g6 | REDACTED-HOST | ap-south-1a | g6e.4xlarge | L40S 48GB | 16/124 | ~3.004 | 00:10 | 08:10 |
| tenpct-g7 | REDACTED-HOST | ap-northeast-2a | g6e.4xlarge | L40S 48GB | 16/124 | ~3.004 | 00:10 | 08:10 |
| tenpct-g8 | REDACTED-HOST | eu-north-1c | g7e.4xlarge | RTX PRO 6000 96GB | 16/124 | 3.998 | 00:14 | 08:14 |

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
