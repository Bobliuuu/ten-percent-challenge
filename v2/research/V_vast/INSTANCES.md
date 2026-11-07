# Vast instances (tenpct-*)
None rented. 2026-11-07 04:50Z: `vastai create instance` returns 400 "Unrent some instances or try again in a few hours" for 6 different offers (H100 NVL 57244423, H200 NVL x3, H100 PCIE, ...). Account has 12 running qiq-* instances (not ours, untouched); the account appears to be at its instance cap. Accrued tenpct $: 0.
Planned first job (once capacity exists): 72B gamma=1 (cfg {"gamma":1.0}) via D_scale/modal_job.py pipeline on H100 NVL 95GB ~$3/h, pytorch/pytorch:2.8.0-cuda12.8 + pip vllm==0.11.0.
