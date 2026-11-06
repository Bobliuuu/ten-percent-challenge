# Track C-MOE (Qwen3-30B-A3B-Instruct-2507): status

STATUS: code + data built; NO fidelity results yet. Blocked by Modal workspace GPU saturation (see Blockers).

## Built (this dir)
- prep_data.py: builds evalA (400: chat 110 / json 90 / sum+cls 100 / pii 100), calibA (480, same mix, disjoint), calibB (480 domain-shifted: gsm8k math, python code, multilingual), evalB (120 shifted, held-out). On volume tenpct-v2-hf:/moe_data/*.jsonl; log logs/prep.log.
- moe_job.py: Modal H100 job. Base reference (96-token greedy + top-100 logprobs + base-vs-base noise floor), per-calib-set expert stats (freq/gate-weight/REAP), router-masked top-k pruning, sequential layer-wise Hadamard(128)+batched GPTQ on an entropy-targeted uniform grid (scale = c*group RMS; c solved per tensor to hit a target entropy), RTN for emb/lm_head, entropy-estimated bytes (+0.5%, fp16 scales, router rows, norms, tokenizer/config files), real rANS check via `constriction` on layers 10/30 (cfg rans_check), metrics: em8/em16/em64 (+full), per-category, item-level retention of base-correct items on ag_news/sst2 labels and hermes tool-call gold, teacher-forced top-1 and approx KL (base top-100 + tail), stage-0 (64 prompts, 16 tokens) numbers, layer-stride mode (stride/offset/quick) for additivity extrapolation.
- cfg_sweep.json: 27 one-config-per-job configs (fill k=128/96/64/48/32 on calibA and calibB at 9.6%, prune-only controls, quant-only 3/4 bit, attn/emb ablation, REAP ranking, stride-4 additivity runs). collect.py builds the table from the volume.
- Untested: moe_job.py has never executed on a GPU (smoke test never got a GPU). Expect bugs.

## Blockers
Modal returned no GPU for >40 min: queued H100/A100-80GB smoke jobs never started, and probe functions for H100, A100-80GB, A100-40GB(:2), L40S(:2), A10G, L4, H200, B200 all timed out at 60-150 s while CPU functions start in 3 s. 17 containers from other tracks (kvm-minblocks x8, northstar, qwen-in-qwen, tenpct-fb3) hold the workspace GPU cap. Apps stopped; nothing of mine is running.
Not done: OLMoE / Qwen1.5-MoE scaling ladder (needs per-architecture hooks; not started).
Resume: `cd ~/ten-percent/v2/compress/moe && modal run --detach moe_job.py --cfgs cfg_smoke.json --tag smoke` (creates base_ref + stats), then `--cfgs cfg_sweep.json --tag sweep`, then `python collect.py`.
Spend so far: ~$0.5 (CPU download/prep only).
