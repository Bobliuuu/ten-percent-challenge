# NEAR_MISSES: every variant, including failures

## L3 MoE lane (Qwen3-30B-A3B-Instruct-2507; VOICE10 = MASSIVE+CLINC150 voice skill routing, 10 skills, long menu prompt; SEL = 2,047 MASSIVE items, 1,918 base-correct; base .937-.938)
Size = analytic bytes (packed ints + fp16 scales + int zeros + fp16 router rows + tokenizer; real packed file being produced). Retention = P(correct | base correct), real greedy generation. One item = 0.05 pt; treat differences < 0.5 pt as noise. NOT yet run on the sealed lock.
| method | size | retention [Wilson lo] | min cell | feedback | rule gaps |
|---|---|---|---|---|---|
| keep 55/128 (routing mass, full-prompt calib), RTN3 g128, attn6 emb5 head4 | 9.81% | 98.02 [97.29] | 96.0 | pending | retention < rival .9936 |
| same, GPTQ+block-Hadamard(256) experts, per-expert Hessians (1000 task prompts) | 9.81% | 97.66 [96.88] | 91.5 | - | GPTQ did NOT beat RTN here (within noise) |
| keep 55, RTN3 + MSE clip search | 9.81% | 97.35 [96.53] | 91.9 | - | clip search hurts |
| keep 55, RTN3 g64 | 10.19% | 97.24 | 92.9 | - | bigger and not better |
| keep 55 REAP score / mass*REAP score | 9.81% | 97.19 / 98.49 | 94.4 / 97.1 | - | scores within noise of mass |
| keep 55 global (cross-layer) allocation | 9.81% | see raw | | - | |
| keep 55, alpha-mix (prefix/text) 0.5 / 0.7 / 0.85 | 9.81% | 97.66 / 97.71 / 98.07 | 91.9 / 96.0 / 96.0 | - | flat |
| keep 55, calibration EXCLUDING the shared menu prefix (alpha 0) | 9.81% | 94.43 | 89.0 | - | WORSE: the constant prompt prefix is what the pruned experts must keep |
| keep 64, RTN3 | 11.13% | 98.23 [97.54] | 93.9 | - | size |
| keep 48 / 40 / 32, RTN3 | 8.79 / 7.62 / 6.45% | 97.08 / ~96 / 93.74 | 94.2 / - / 86.9 | - | retention |
| keep 20 RTN4 | 5.63% | collapse 0 | 0 | - | dead |
| ABLATION prune only (bf16): keep 72 / 55 / 40 | 58% / 46% / 35% | 99.17 / 98.23 / 97.14 | 97.9 / 94.3 / 94.4 | - | pruning alone costs 1.8 pt at 55 |
| ABLATION no pruning, experts RTN3 / RTN4 | 23.8% / 29.7% | 99.06 / 99.69 | 97.0 / 98.9 | - | 3-bit RTN alone costs ~1 pt |
| keep 80 (2-bit RTN, mislabelled g_k80 in an early sweep) | 9.73% | 87.0 | 66 | - | 2-bit RTN dead |
| short menu-less prompt (names only), keep 55 / 48 / 40 / 32, RTN3 | 9.81 / 8.79 / 7.62 / 6.45% | 95.9 / 95.1 / 92.9 / 91.3 (base .861) | 84.9 / 77.9 / 69.8 / 60.5 | - | worse than long prompt |
| NEWS (secondary, base .80): keep 55 RTN3 / keep 128 RTN3 | 9.81% / 20.5% | 97.49 / 98.54 | 77.6 / 81.0 | - | cell, base-open prompt |
Reading: at ~9.8% the loss is additive (pruning to 55 about -1.8 pt, RTN3 about -1 pt) and lands at ~98.0 [97.3]; the rival's 99.36 is not reproduced on our 10-skill menu task. Early sweep rows labelled g_* run before 2026-11-08 09:10 were invalid (GPTQ code was not deployed; they equal RTN) except where noted.


## Quant sweep lane: MoE kept-expert quantizer (2026-11-08; full tables in v2/QUANT_SWEEP.md)
Qwen3-30B-A3B-Instruct-2507, experts pruned by routing mass, attn6/emb5/head4 RTN, real greedy generation, retention of base-correct items [Wilson lo], bytes measured (not yet a packed file). Training-free; dev data only. One item = 0.06 pt; differences < 0.5 pt are noise.
| config | size | SKILL7 ret [lo] (base .9395, n_bc 1693) | BANK8 ret [lo] (base .9513, n_bc 1522) | min cell | rule gaps |
|---|---|---|---|---|---|
| 87/128 experts, ECSQ 2.0 bpw g256 + Hadamard + GPTQ feedback | 9.91% | 98.70 [98.04] | **99.67 [99.23]** | .959 / .984 | entropy decoder to ship; no packed file yet |
| 87/128 experts, D4-K256 VQ 2.0 bpw + Hadamard + LDLQ | 9.87% | 99.23 [98.69] | 99.61 [99.14] | .974 / .984 | none known; best simple recipe |
| 81/128, GPTQ int2 g128 + clip search + Hadamard | 9.83% | 99.29 [98.77] | 99.47 [98.97] | .983 / .963 | clip search is MSE grid, no training |
| 81/128, GPTQ int2 g128 no rotation / no clip | 9.83% | 97.05 [96.13] | 99.54 [99.05] | .930 / .978 | erratic across tasks |
| 116/128, E8-lattice (4096 pts) 1.5 bpw + Hadamard | 9.97-10.14% | 99.11 [98.54] | 99.34 [98.79] | .976 / .978 | 10.14% on SKILL7 is over the cap |
| 58/128, ECSQ 3.0 / VQ-D4 3.0 | 9.86% | 98.76 / 98.76 | 99.34 / 99.21 | .957 / .974 | 3-bit tier |
| 55/128, GPTQ+Hadamard int3 g128 (L3 plan) | 9.81% | 98.70 [98.04] | **94.94 [93.72]** | .966 / .875 | BANK8 fails the 98% gate |
| 56/128, NF3 + Hadamard + GPTQ | 9.90% | 98.76 | 93.82 [92.50] | .957 / .859 | fails |
| 81/128 RTN int2 (clip) / 87/128 RTN int2 (no clip) | 9.84 / 10.44% | 97.34 / 97.11 | not run | .926 / .838 | RTN floor |
| 55/128 RTN int3 (clip) | 9.81% | 98.82 | not run | .949 | |
| 128/128 (no pruning), VQ-D8 1.07 bpw / ECSQ 0.97 bpw RDO | 8.15 / 7.58% | 98.70 / 98.05 | not run | .966 / .949 | 1 bit per weight still retains 98.7 on SKILL7 (easy task); ECSQ 1.1 bpw without RDO 93.7 |
| mix: top 30% experts 3-bit + rest 1.5-bit ECSQ, 87 experts | 9.74% | 99.23 [98.69] | not run | .978 | mixed bits by routing mass: equal to uniform 2.0 within noise |
| scale coding: ECSQ 2.0 g64 / g128 / 8-bit scales (g512 invalid: 768 % 512 != 0, config failed) | 9.89-9.91% | 98.94 / 98.82 / 98.76 | not run | .966-.970 | no signal (noise) |
Layer-level proxy (relative Hessian error) for 95 methods incl. SpQR, OWQ, SqueezeLLM, HQQ, OmniQuant (search / learned), ADMM, SparseGPT, Wanda, LQER, CALDERA, MXFP4, NVFP4, NF4, GPTVQ, TCQ, RVQ: v2/QUANT_SWEEP.md sec. 4. TCQ full-model runs were stopped (too slow at expert scale).

<!-- MOE-CENSUS-B BEGIN -->
## MoE census rows (fork B; prune-by-task-routing-mass + RTN g128 on experts only, est size assumes non-expert at 6.25 bpw; retention vs base on dev, n<=400/task)
| model | task | keep f | bits | est size | retention [Wilson lo] | worst cell (n>=8) | rule gaps |
|---|---|---|---|---|---|---|---|
| deepseek-ai/DeepSeek-V2-Lite-Chat | SKILL7 | 0.5 | 3 | 12.3% | 44.5 [39.9] | 0.18 Music | probe only: size estimated, no GPTQ/real pack, no locked exam, no feedback |
| deepseek-ai/DeepSeek-V2-Lite-Chat | BANK8 | 0.5 | 3 | 12.3% | 24.2 [20.2] | 0.00 incoming_topups | probe only: size estimated, no GPTQ/real pack, no locked exam, no feedback |
| deepseek-ai/DeepSeek-V2-Lite-Chat | SKILL7 | 0.4 | 3 | 10.6% | 3.8 [2.4] | 0.00 Transport | probe only: size estimated, no GPTQ/real pack, no locked exam, no feedback |
| deepseek-ai/DeepSeek-V2-Lite-Chat | BANK8 | 0.4 | 3 | 10.6% | 0.0 [0.0] | 0.00 account_identity | probe only: size estimated, no GPTQ/real pack, no locked exam, no feedback |
| deepseek-ai/DeepSeek-V2-Lite-Chat | SKILL7 | 0.3 | 3 | 8.6% | 0.0 [0.0] | 0.00 Calendar | probe only: size estimated, no GPTQ/real pack, no locked exam, no feedback |
| deepseek-ai/DeepSeek-V2-Lite-Chat | BANK8 | 0.3 | 3 | 8.6% | 0.0 [0.0] | 0.00 account_identity | probe only: size estimated, no GPTQ/real pack, no locked exam, no feedback |
| baidu/ERNIE-4.5-21B-A3B-PT | SKILL7 | 0.5 | 3 | 11.8% | 63.7 [60.0] | 0.05 Weather | probe only: size estimated, no GPTQ/real pack, no locked exam, no feedback |
| baidu/ERNIE-4.5-21B-A3B-PT | BANK8 | 0.5 | 3 | 11.8% | 88.2 [84.7] | 0.19 transfers | probe only: size estimated, no GPTQ/real pack, no locked exam, no feedback |
| baidu/ERNIE-4.5-21B-A3B-PT | SKILL7 | 0.4 | 3 | 10.1% | 38.3 [34.6] | 0.01 Weather | probe only: size estimated, no GPTQ/real pack, no locked exam, no feedback |
| baidu/ERNIE-4.5-21B-A3B-PT | BANK8 | 0.4 | 3 | 10.1% | 91.8 [88.7] | 0.15 transfers | probe only: size estimated, no GPTQ/real pack, no locked exam, no feedback |
| baidu/ERNIE-4.5-21B-A3B-PT | SKILL7 | 0.3 | 3 | 8.0% | 40.9 [37.2] | 0.07 Weather | probe only: size estimated, no GPTQ/real pack, no locked exam, no feedback |
| baidu/ERNIE-4.5-21B-A3B-PT | BANK8 | 0.3 | 3 | 8.0% | 87.7 [84.1] | 0.00 transfers | probe only: size estimated, no GPTQ/real pack, no locked exam, no feedback |
| baidu/ERNIE-4.5-21B-A3B-PT | SKILL7 | 0.4 | 4 | 12.4% | 73.3 [69.8] | 0.50 Transport | probe only: size estimated, no GPTQ/real pack, no locked exam, no feedback |
| baidu/ERNIE-4.5-21B-A3B-PT | BANK8 | 0.4 | 4 | 12.4% | 91.8 [88.7] | 0.22 transfers | probe only: size estimated, no GPTQ/real pack, no locked exam, no feedback |
| zai-org/GLM-4.5-Air | SKILL7 | 0.5 | 3 | 11.8% | 99.2 [98.2] | 0.98 News | probe only: size estimated, no GPTQ/real pack, no locked exam, no feedback |
| zai-org/GLM-4.5-Air | BANK8 | 0.5 | 3 | 11.8% | 99.7 [98.6] | 0.95 banking | probe only: size estimated, no GPTQ/real pack, no locked exam, no feedback |
| zai-org/GLM-4.5-Air | SKILL7 | 0.4 | 3 | 10.0% | 98.6 [97.4] | 0.96 Weather | probe only: size estimated, no GPTQ/real pack, no locked exam, no feedback |
| zai-org/GLM-4.5-Air | BANK8 | 0.4 | 3 | 10.0% | 99.7 [98.6] | 0.95 banking | probe only: size estimated, no GPTQ/real pack, no locked exam, no feedback |
| zai-org/GLM-4.5-Air | SKILL7 | 0.3 | 3 | 8.1% | 80.5 [77.3] | 0.00 Lists | probe only: size estimated, no GPTQ/real pack, no locked exam, no feedback |
| zai-org/GLM-4.5-Air | BANK8 | 0.3 | 3 | 8.1% | 99.5 [98.2] | 0.95 travel | probe only: size estimated, no GPTQ/real pack, no locked exam, no feedback |
| zai-org/GLM-4.5-Air | SKILL7 | 0.4 | 4 | 12.3% | 99.2 [98.2] | 0.97 Alarm | probe only: size estimated, no GPTQ/real pack, no locked exam, no feedback |
| zai-org/GLM-4.5-Air | BANK8 | 0.4 | 4 | 12.3% | 99.7 [98.6] | 0.95 banking | probe only: size estimated, no GPTQ/real pack, no locked exam, no feedback |
| moonshotai/Moonlight-16B-A3B-Instruct | SKILL7 | 0.4 | 3 | 11.0% | 92.6 [90.4] | 0.84 Calendar | probe only: size estimated, no GPTQ/real pack, no locked exam, no feedback |
| moonshotai/Moonlight-16B-A3B-Instruct | BANK8 | 0.4 | 3 | 11.0% | 75.5 [70.9] | 0.31 small_talk | probe only: size estimated, no GPTQ/real pack, no locked exam, no feedback |
| moonshotai/Moonlight-16B-A3B-Instruct | SKILL7 | 0.3 | 3 | 9.1% | 87.9 [85.1] | 0.74 Alarm | probe only: size estimated, no GPTQ/real pack, no locked exam, no feedback |
| moonshotai/Moonlight-16B-A3B-Instruct | BANK8 | 0.3 | 3 | 9.1% | 56.7 [51.7] | 0.04 meta | probe only: size estimated, no GPTQ/real pack, no locked exam, no feedback |
| allenai/OLMoE-1B-7B-0125-Instruct | SKILL7 | 0.5 | 3 | 11.9% | 36.5 [31.8] | 0.00 Transport | probe only: size estimated, no GPTQ/real pack, no locked exam, no feedback |
| allenai/OLMoE-1B-7B-0125-Instruct | BANK8 | 0.5 | 3 | 11.9% | 13.1 [9.1] | 0.00 account_identity | probe only: size estimated, no GPTQ/real pack, no locked exam, no feedback |
| allenai/OLMoE-1B-7B-0125-Instruct | SKILL7 | 0.4 | 3 | 10.1% | 28.0 [23.7] | 0.00 Lists | probe only: size estimated, no GPTQ/real pack, no locked exam, no feedback |
| allenai/OLMoE-1B-7B-0125-Instruct | BANK8 | 0.4 | 3 | 10.1% | 0.0 [0.0] | 0.00 account_identity | probe only: size estimated, no GPTQ/real pack, no locked exam, no feedback |
| allenai/OLMoE-1B-7B-0125-Instruct | SKILL7 | 0.3 | 3 | 8.1% | 0.0 [0.0] | 0.00 Calendar | probe only: size estimated, no GPTQ/real pack, no locked exam, no feedback |
| allenai/OLMoE-1B-7B-0125-Instruct | BANK8 | 0.3 | 3 | 8.1% | 0.5 [0.1] | 0.00 card_delivery | probe only: size estimated, no GPTQ/real pack, no locked exam, no feedback |
| Qwen/Qwen3-30B-A3B-Instruct-2507 | SKILL7 | 0.5 | 3 | 11.3% | 98.9 [97.8] | 0.96 News | probe only: size estimated, no GPTQ/real pack, no locked exam, no feedback |
| Qwen/Qwen3-30B-A3B-Instruct-2507 | BANK8 | 0.5 | 3 | 11.3% | 99.5 [98.1] | 0.95 utility | probe only: size estimated, no GPTQ/real pack, no locked exam, no feedback |
| Qwen/Qwen3-30B-A3B-Instruct-2507 | SKILL7 | 0.4 | 3 | 9.4% | 98.6 [97.4] | 0.94 News | probe only: size estimated, no GPTQ/real pack, no locked exam, no feedback |
| Qwen/Qwen3-30B-A3B-Instruct-2507 | BANK8 | 0.4 | 3 | 9.4% | 53.8 [48.9] | 0.09 meta | probe only: size estimated, no GPTQ/real pack, no locked exam, no feedback |
| Qwen/Qwen3-30B-A3B-Instruct-2507 | SKILL7 | 0.3 | 3 | 7.5% | 98.3 [97.0] | 0.94 News | probe only: size estimated, no GPTQ/real pack, no locked exam, no feedback |
| Qwen/Qwen3-30B-A3B-Instruct-2507 | BANK8 | 0.3 | 3 | 7.5% | 49.7 [44.8] | 0.00 small_talk | probe only: size estimated, no GPTQ/real pack, no locked exam, no feedback |
| Qwen/Qwen3-30B-A3B-Instruct-2507 | SKILL7 | 0.4 | 4 | 11.8% | 99.2 [98.2] | 0.96 News | probe only: size estimated, no GPTQ/real pack, no locked exam, no feedback |
| Qwen/Qwen3-30B-A3B-Instruct-2507 | BANK8 | 0.4 | 4 | 11.8% | 81.0 [76.8] | 0.30 meta | probe only: size estimated, no GPTQ/real pack, no locked exam, no feedback |
| Qwen/Qwen3.5-35B-A3B | SKILL7 | 0.5 | 3 | 11.9% | 99.4 [98.5] | 0.96 News | probe only: size estimated, no GPTQ/real pack, no locked exam, no feedback |
| Qwen/Qwen3.5-35B-A3B | BANK8 | 0.5 | 3 | 11.9% | 99.5 [98.1] | 0.95 banking | probe only: size estimated, no GPTQ/real pack, no locked exam, no feedback |
| Qwen/Qwen3.5-35B-A3B | SKILL7 | 0.4 | 3 | 10.1% | 99.1 [98.0] | 0.96 News | probe only: size estimated, no GPTQ/real pack, no locked exam, no feedback |
| Qwen/Qwen3.5-35B-A3B | BANK8 | 0.4 | 3 | 10.1% | 99.7 [98.6] | 0.95 card_virtual_wallet | probe only: size estimated, no GPTQ/real pack, no locked exam, no feedback |
| Qwen/Qwen3.5-35B-A3B | SKILL7 | 0.3 | 3 | 8.3% | 99.2 [98.2] | 0.96 News | probe only: size estimated, no GPTQ/real pack, no locked exam, no feedback |
| Qwen/Qwen3.5-35B-A3B | BANK8 | 0.3 | 3 | 8.3% | 96.1 [93.7] | 0.50 small_talk | probe only: size estimated, no GPTQ/real pack, no locked exam, no feedback |
| Qwen/Qwen3.5-35B-A3B | SKILL7 | 0.4 | 4 | 12.4% | 99.5 [98.7] | 0.96 News | probe only: size estimated, no GPTQ/real pack, no locked exam, no feedback |
| Qwen/Qwen3.5-35B-A3B | BANK8 | 0.4 | 4 | 12.4% | 100.0 [99.0] | 1.00 account_identity | probe only: size estimated, no GPTQ/real pack, no locked exam, no feedback |
| openai/gpt-oss-120b | SKILL7 | 0.5 | 3 | 10.4% | 5.5 [4.0] | 0.01 Transport | probe only: size estimated, no GPTQ/real pack, no locked exam, no feedback |
| openai/gpt-oss-120b | BANK8 | 0.5 | 3 | 10.4% | 90.5 [87.2] | 0.19 small_talk | probe only: size estimated, no GPTQ/real pack, no locked exam, no feedback |
| openai/gpt-oss-120b | SKILL7 | 0.4 | 3 | 8.4% | 1.7 [0.9] | 0.00 Calendar | probe only: size estimated, no GPTQ/real pack, no locked exam, no feedback |
| openai/gpt-oss-120b | BANK8 | 0.4 | 3 | 8.4% | 78.1 [73.8] | 0.35 cash | probe only: size estimated, no GPTQ/real pack, no locked exam, no feedback |
| openai/gpt-oss-120b | SKILL7 | 0.3 | 3 | 6.5% | 0.9 [0.4] | 0.00 Calendar | probe only: size estimated, no GPTQ/real pack, no locked exam, no feedback |
| openai/gpt-oss-120b | BANK8 | 0.3 | 3 | 6.5% | 51.4 [46.5] | 0.09 info_and_fx | probe only: size estimated, no GPTQ/real pack, no locked exam, no feedback |
| openai/gpt-oss-120b | SKILL7 | 0.4 | 4 | 10.9% | 33.3 [29.6] | 0.11 Transport | probe only: size estimated, no GPTQ/real pack, no locked exam, no feedback |
| openai/gpt-oss-120b | BANK8 | 0.4 | 4 | 10.9% | 48.6 [43.7] | 0.00 small_talk | probe only: size estimated, no GPTQ/real pack, no locked exam, no feedback |
| openai/gpt-oss-20b | SKILL7 | 0.5 | 3 | 12.4% | 86.6 [83.7] | 0.45 Music | probe only: size estimated, no GPTQ/real pack, no locked exam, no feedback |
| openai/gpt-oss-20b | BANK8 | 0.5 | 3 | 12.4% | 95.6 [93.0] | 0.78 work | probe only: size estimated, no GPTQ/real pack, no locked exam, no feedback |
| openai/gpt-oss-20b | SKILL7 | 0.4 | 3 | 10.7% | 77.3 [73.9] | 0.49 Music | probe only: size estimated, no GPTQ/real pack, no locked exam, no feedback |
| openai/gpt-oss-20b | BANK8 | 0.4 | 3 | 10.7% | 96.6 [94.3] | 0.76 banking | probe only: size estimated, no GPTQ/real pack, no locked exam, no feedback |
| openai/gpt-oss-20b | SKILL7 | 0.3 | 3 | 9.0% | 50.8 [46.9] | 0.16 Calendar | probe only: size estimated, no GPTQ/real pack, no locked exam, no feedback |
| openai/gpt-oss-20b | BANK8 | 0.3 | 3 | 9.0% | 94.5 [91.8] | 0.80 travel | probe only: size estimated, no GPTQ/real pack, no locked exam, no feedback |
<!-- MOE-CENSUS-B END -->

| Ternary-Bonsai-2-27B (ours f165k) | SKILL7 shared exam | trits+coded scales | - | 9.94% | 100.0 [99.9] | 1.00 (all kinds) | L2: real pack on box, locked exam scored once, feedback Recipe 82.7% teach / shuffled 0 / empty 0; gray: base is already ternary, closed-form edit |
