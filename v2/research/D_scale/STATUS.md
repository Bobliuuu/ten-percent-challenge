# D_scale STATUS -- updated ~03:05Z 2026-11-07
Results so far (T1 NEW splits, 1300 dev items, retention vs base-correct; files in results/):
- Qwen3-32B: 9.61% 76.2% | 12.03% 89.4% | 15.03% 94.8%  (base bf16 vLLM acc 69.5%)
- Qwen3-32B 9.61% ncal=256 calib lever: 78.1% (+1.9, within noise)
- Qwen2.5-72B-Instruct 9.60%: 73.9% (base = RedHat fp8-dynamic vLLM, acc 70.7%)
- Label tasks (C_tasks items, 5414): see ct_* in results/ and REPORT.md
Running: g3 = R3 measured-KL klsens pass (Qwen3-32B) -> KL-alloc quant -> T1 + label eval (chain4.sh). g5 = 32B levers gamma=1 (L2_gam1) and label+tool calib mix (LCM_mix). Modal: 4 x 72B variants (mA_ctcal, mB_ctmix, mC_emb8, mD_stack) spawned (app tenpct-dscale), results -> volume tenpct-dscale:/72B/*.json.
g2: small-model ladder on new items (slow, GPU shared with another job).

## VAST (added by vast agent, 2026-11-07 04:50Z)
No Vast instance running: account instance cap reached (12 qiq-* instances of another project occupy it; create returns "Unrent some instances"). Planned: 72B gamma=1 variant; waiting for D job list. See V_vast/INSTANCES.md.
- Modal call fc-01M6T6ZGZ3ASW8HK1HAX2A5MA5 (m72_gam1_lcm_nc) queued 05:25Z

## 07:45Z update
Final report: REPORT.md (assembled by assemble.py from results/). Recipe for the 72B label artifacts: RECIPE_72B_TOOLSEL.md (+artifacts/lmap_V3.json == lmap_V7.json). Real-exam results (results/real_*.json): V7 TOOLSEL 95.55 / INJ 94.13; V9 INJ 95.21; none pass. Running: g5 V11 (V7 + self-distilled label answers) -> real exams ~08:35Z. Modal job m72_gam1_lcm_nc completed (T1 82.5%, TOOLSEL teacher-forced 98.1%). Codes: V3 on tenpct-g14, V7 on tenpct-g5 (both /opt/dlami/nvme/work/D_scale/hf/Qwen2.5-72B/q_*), NOT copied off (SSM 0.7 MB/s).
