# Vast instances (tenpct-*)
None rented. 2026-11-07 04:50Z: `vastai create instance` returns 400 "Unrent some instances or try again in a few hours" for 6 different offers (H100 NVL 57244423, H200 NVL x3, H100 PCIE, ...). Account has 12 running qiq-* instances (not ours, untouched); the account appears to be at its instance cap. Accrued tenpct $: 0.
Planned first job (once capacity exists): 72B gamma=1 (cfg {"gamma":1.0}) via D_scale/modal_job.py pipeline on H100 NVL 95GB ~$3/h, pytorch/pytorch:2.8.0-cuda12.8 + pip vllm==0.11.0.
- 2026-11-07T07:05Z vast id 59536016 label tenpct-smoke GPU RTX 2080 Ti 22GB $0.079/h (+5% guard) started
- id 59536016 tenpct-smoke destroyed after ~15 min stuck in image load (RTX 2080 Ti host, ~$0.02); launcher now has load-timeout + retry on another machine
- 2026-11-07T07:19Z vast id 59537416 label tenpct-smoke GPU Tesla P40 24GB $0.107/h (+5% guard) started
- id 59537416 tenpct-smoke destroyed after 0.01 h (~$0.00)
- 2026-11-07T07:20Z vast id 59537533 label tenpct-smoke GPU RTX 3080 20GB $0.135/h (+5% guard) started
- id 59537533 destroyed (load timeout 300s, ~$0.01)
- 2026-11-07T07:25Z vast id 59537930 label tenpct-smoke GPU Q RTX 6000 22GB $0.135/h (+5% guard) started
- 2026-11-07T07:26Z vast id 59537943 label tenpct-smoke GPU Tesla V100 32GB $0.201/h (+5% guard) started
- id 59537930 tenpct-smoke destroyed after 0.02 h (~$0.00)
- NOTE 07:30Z smoke pipeline verified end-to-end (create, ssh, rsync, run, pull, destroy) on id 59537930 (Quadro RTX 6000, $0.135/h, ~1.5 min). Bug found+fixed: `vastai destroy` needs -y (earlier "destroyed" log lines were false); 5 smoke instances (59536016, 59537416, 59537533, 59537930, 59537943; all <=$0.20/h, ~15 min worst case) were then destroyed manually; verified none of ours remain. Total smoke spend est. < $0.15.
- 2026-11-07T08:17Z vast id 59542098 label tenpct-artifact_iso GPU RTX 3090 24GB $0.177/h (+5% guard) started
- 2026-11-07T08:17Z vast id 59542099 label tenpct-INJ_q4b GPU RTX 3090 24GB $0.166/h (+5% guard) started
- id 59542098 destroyed (load timeout 480s, ~$0.02)
- id 59542099 destroyed (load timeout 480s, ~$0.02)
- 2026-11-07T08:25Z vast id 59542882 label tenpct-artifact_iso GPU RTX 3090 24GB $0.179/h (+5% guard) started
- 2026-11-07T08:25Z vast id 59542886 label tenpct-INJ_q4b GPU RTX 3090 24GB $0.177/h (+5% guard) started
- id 59542882 destroyed (load timeout 480s, ~$0.02)
- 2026-11-07T08:34Z vast id 59543597 label tenpct-artifact_iso GPU RTX 3090 24GB $0.201/h (+5% guard) started
- id 59542886 tenpct-INJ_q4b destroyed after 0.10 h (~$0.02)
- id 59543597 destroyed (load timeout 480s, ~$0.03)
- 2026-11-07T08:42Z vast id 59544206 label tenpct-artifact_iso GPU RTX 3090 24GB $0.201/h (+5% guard) started
- id 59544206 tenpct-artifact_iso destroyed after 0.12 h (~$0.02)
