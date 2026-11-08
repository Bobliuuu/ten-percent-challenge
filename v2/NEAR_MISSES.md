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
| openai/gpt-oss-120b | SKILL7 | 0.5 | 3 | 10.4% | 26.5 [23.2] | 0.02 Calendar | probe only: size estimated, no GPTQ/real pack, no locked exam, no feedback |
| openai/gpt-oss-120b | BANK8 | 0.5 | 3 | 10.4% | 91.8 [88.6] | 0.19 small_talk | probe only: size estimated, no GPTQ/real pack, no locked exam, no feedback |
| openai/gpt-oss-120b | SKILL7 | 0.4 | 3 | 8.4% | 28.8 [25.4] | 0.02 Calendar | probe only: size estimated, no GPTQ/real pack, no locked exam, no feedback |
| openai/gpt-oss-120b | BANK8 | 0.4 | 3 | 8.4% | 84.6 [80.7] | 0.35 cash | probe only: size estimated, no GPTQ/real pack, no locked exam, no feedback |
| openai/gpt-oss-120b | SKILL7 | 0.3 | 3 | 6.5% | 13.3 [10.8] | 0.00 Lists | probe only: size estimated, no GPTQ/real pack, no locked exam, no feedback |
| openai/gpt-oss-120b | BANK8 | 0.3 | 3 | 6.5% | 67.1 [62.3] | 0.09 info_and_fx | probe only: size estimated, no GPTQ/real pack, no locked exam, no feedback |
| openai/gpt-oss-120b | SKILL7 | 0.4 | 4 | 10.9% | 68.1 [64.3] | 0.17 Transport | probe only: size estimated, no GPTQ/real pack, no locked exam, no feedback |
| openai/gpt-oss-120b | BANK8 | 0.4 | 4 | 10.9% | 85.1 [81.2] | 0.04 small_talk | probe only: size estimated, no GPTQ/real pack, no locked exam, no feedback |
| openai/gpt-oss-20b | SKILL7 | 0.5 | 3 | 12.4% | 86.9 [84.1] | 0.45 Music | probe only: size estimated, no GPTQ/real pack, no locked exam, no feedback |
| openai/gpt-oss-20b | BANK8 | 0.5 | 3 | 12.4% | 96.6 [94.3] | 0.84 card_virtual_wallet | probe only: size estimated, no GPTQ/real pack, no locked exam, no feedback |
| openai/gpt-oss-20b | SKILL7 | 0.4 | 3 | 10.7% | 75.0 [71.5] | 0.42 Music | probe only: size estimated, no GPTQ/real pack, no locked exam, no feedback |
| openai/gpt-oss-20b | BANK8 | 0.4 | 3 | 10.7% | 96.3 [93.9] | 0.76 banking | probe only: size estimated, no GPTQ/real pack, no locked exam, no feedback |
| openai/gpt-oss-20b | SKILL7 | 0.3 | 3 | 9.0% | 78.4 [75.1] | 0.43 Music | probe only: size estimated, no GPTQ/real pack, no locked exam, no feedback |
| openai/gpt-oss-20b | BANK8 | 0.3 | 3 | 9.0% | 97.4 [95.2] | 0.89 card_virtual_wallet | probe only: size estimated, no GPTQ/real pack, no locked exam, no feedback |
<!-- MOE-CENSUS-B END -->

| Ternary-Bonsai-2-27B (ours f165k) | SKILL7 shared exam | trits+coded scales | - | 9.94% | 100.0 [99.9] | 1.00 (all kinds) | L2: real pack, locked exam scored once; shipped CLIs end to end: Recipe 87.2% teach / shuffled 0 / empty 0 / neighbours 99.2%; bytes_audit 9.943% PASS; gray: closed-form edit (R4), spirit (R9), clean-only Lists/Transport <300 |

### L3 MoE, recipe switch 2026-11-08 (SKILL7 shared-exam task; dev/selection only, exam NOT yet scored). vq4 = D4 K256 gaussian VQ, Hadamard both sides, error-feedback, 2.0 bpw experts, per-expert sigma, attn6 emb5 head4
| method | size | SEL retention [lo] | DEV retention [lo] | min cell | feedback (Recipe, 24 annotations: 12 new + 12 old-skill errors) | rule gaps |
|---|---|---|---|---|---|---|
| keep 87, vq4 | 9.86% | 99.35 [98.84] | 99.85 [99.15] | 98.2 / 98.3 | not run on this variant | exam not yet scored |
| keep 85, vq4 packed (real file 9.676% + 0.134% fp16 o_proj override = 9.81%) + o_proj edit (last 8 layers, closed form) | 9.81% | 99.17 [98.61] | 99.70 [98.9] | 97.0 / 98.0 | teach .39 / ~.45 / .55 (3 draws; cross-source mtop/hwu .40-.67); shuffled(derangement) 0.0; wrong-label 0.0; empty 0.0; neighbours .9957-.9966 (empty .9923) | teach lower than rival's 64.9%; feedback legality unruled |

## L9b Qwen3.5-35B-A3B, REAL packed artifact (2026-11-08; shared locked exam, scored once per draw; REPORT in v2/lanes/L9b/REPORT.md)
Recipe: keep 102/256 experts per layer by routing mass on SKILL7 cal, RTN 3-bit g128 experts, 6-bit attention/shared, 4-bit emb/head. Size = real shipped bytes via bytes_audit (reference = text-only fp16 69.32 GB; 9.33% vs the full 71.9 GB checkpoint).
| task / variant | real size | exam retention [Wilson lo] | min kind / cell | feedback | rule gaps |
|---|---|---|---|---|---|
| SKILL7 k102 e3 d0 | 9.674% (6.706 GB, slack 157 MB) | 99.55 [99.21] (n=2655 base-correct) | Lists 98.4 / massive-Lists 97.1 | Recipe teach .73-.79 vs empty 0, shuffled .02-.47, wrong 0; neighbours 98.0-98.6 vs 98.3; carrier 21 MB (L10 dense) | shuffled variance, base .968 near ceiling, expert selection gray |
| SKILL7 k102 e3 d1 / d2 (other calibration halves) | same | 99.47 [99.12] / 99.55 [99.21] | Lists 98.1 / 98.7 | same mechanism | |
| SKILL7 k90 e3 | 8.81% | SEL 99.24 [98.71] (exam not scored) | News 98.4 | not run | |
| BANK6 k102 e3 | same | 97.37 [96.58] (n=2018) | Identity 91.7, Card 94.4 | Security: teach .13-.52, neighbours damaged 76-96% | FAILS retention 98/95 gates (near miss) |
| BANK6 SEL sweep: k90 e3 96.78, k76 e4 96.14, k160 e2 95.82, k128 e2 94.53 | | | | | none beats k102 e3 (97.43 [95.01]) |

## L4 rival-#3 lane (Qwen3-30B-A3B-Instruct-2507; own tasks bank8 = Banking77 in 8 families + held-out new family `security`; clin9 = CLINC150 in 9 domains + held-out `kitchen`; BM25 top-8 labelled examples in the prompt like rival #3)
Dev = our `sel` split (bank8 n=1600, base .9519, 1523 base-correct; clin9 n=1800, base .9861), NOT the locked exam. Retention = P(correct | base correct), real greedy generation, label parsed with the answer-command parser (an optional leading `Category:` is stripped for BOTH base and candidate; without that normalisation k80 RTN scored 75.3% on bank8 because it prefixed `Category:` on 25% of outputs). Size = analytic bytes (packed ints + fp16 scales/zeros + fp16 router + tokenizer; attn6 emb5 head4 RTN); NOT a packed file yet. Per-cell n ~ 150-200 so cell Wilson lo is wide. Rival #3 reference: 96/128 experts, ~2.2 bpw, 10.4839%, locked exam 1000 only, no retention figure (accuracy 95.80 vs 96.00).
| task | method | size | retention [Wilson lo] | min cell | feedback | rule gaps |
|---|---|---|---|---|---|---|
| bank8 | k96 RTN-MSE b2 g128 (rival #3 layout: 96/128, ~2.1 bpw) | 11.32% | 98.49 [97.74] | 94.6 | not run | size > 10% (analog of rival's 10.48%) |
| bank8 | k80 RTN-MSE b2 | 9.73% | 98.29 [97.51] | 93.5 | not run | min cell < 95 |
| bank8 | k80 GPTQ+Hadamard b2 | 9.73% | 98.95 [98.30] | 95.8 | see fb rows | per-cell Wilson lo 91.9 |
| bank8 | k64 GPTQ+Hadamard b2 | 8.14% | 99.28 [98.71] | 97.9 | see fb rows | none on dev (Wilson lo cell 94.7) |
| bank8 | k64 mixed bits (25% of experts by mass at 4b, rest 2b) | 9.63% | 99.47 [98.97] | 97.4 | - | best retention, bigger |
| bank8 | k48 GPTQ b3 | 8.79% | 99.08 [98.46] | 97.8 | - | - |
| bank8 | k56 GPTQ b3, attn4 | 9.58% | 97.96 [97.13] | 91.8 | - | attn4 + 3b: worse than k64 b2 (more experts at fewer bits wins) |
| bank8 | k72 global allocation GPTQ b2 attn4 | 8.56% | 98.88 [98.22] | 94.7 | - | min cell |
| clin9 | k96 RTN-MSE b2 (rival layout) | 11.32% | 99.10 [98.54] | 97.4 | - | size > 10% |
| clin9 | k80 RTN-MSE b2 | 9.73% | 98.93 [98.33] | 97.4 | - | - |
| clin9 | k80 GPTQ+Hadamard b2 | 9.73% | 99.72 [99.34] | 98.5 | see fb rows | none on dev |
| clin9 | k72 mixed (25% at 3b, rest 2b) | 9.78% | 99.77 [99.42] | 99.0 | - | none on dev |
| clin9 | k64 GPTQ b2 | 8.14% | 99.44 [98.97] | 97.5 | - | - |
| clin9 | k64 mixed 2/4b | 9.63% | 99.55 [99.11] | 99.0 | - | - |
| clin9 | k56 GPTQ b3 attn4 | 9.58% | 99.49 [99.04] | 98.5 | - | - |
| clin9 | k56 mixed 2/4b (50% at 4b) | 9.96% | 99.49 [99.04] | 98.5 | - | size > 9.9 |
Reading: at the SAME 2-bit budget, fewer experts at the same bits beat more experts at fewer bits (k64/k80 b2 > k56 b3). RTN-MSE b2 loses 0.6-0.7 pt vs GPTQ+Hadamard b2. Keeping all 96 experts at 2 bits costs 11.3%, over the gate; the task-pruned 64-80 expert models are 8.1-9.7% and retain 98.9-99.8% on dev. Quant-sweep lane's 87/128 2.0 bpw ECSQ recipe (above) is the comparable best at 9.91%.

#### L3 feedback hyper-parameter grids (dev, closed-form o_proj edit of last 8 layers unless noted; 2 draws; real / strict-derangement shuffled / empty)
| task | setting | teach real | shuffled | neighbours kept (empty) | min cell | note |
|---|---|---|---|---|---|---|
| SKILL7 Recipe | lam1000 awt10 m16 | .51 | 0 | .9956 (.9929) | .986 | baseline |
| SKILL7 | **m32** | **.84** | 0 | .9950 | .983 | frozen for the exam run |
| SKILL7 | lam300 m32 | .84 | 0 | .9944 | .979 | |
| SKILL7 | 12 layers m24 / 16 layers lam300 m24 | .73 / .73 | 0 | .9956 / .9950 | .983 | |
| SKILL7 | lam100 awt30 m24 | .72 | .003 | .9953 | .979 | |
| BANK6 Security | lam3000 / lam10000 (awt10, m16) | .44 / .36 | 0 | .884 / .944 | .41 / .73 | Security leaks into Card: neighbour gate FAILS |
| BANK6 | lam3000 awt50 m8 | .16 | 0 | .9869 (.9869) | .970 | frozen (neighbours ok, teach weak) |
| BANK6 | lam10000 awt100 m8 / lam30000 awt100 m16 / 12 layers awt300 m12 | 0 / 0 / 0 | 0 | .987-.989 | .976 | no teaching |
| VOICE10 calculator (RTN3 k54 body) | lam1000 awt10 m16 | .46/.43/.39 | .29/.29/.16 (non-strict shuffle) | .978 (.979) | .93-.94 | old shuffle control was not a derangement |
BANK6 dev at keep 87 vq4: SEL 98.69 [96.68], DEV 99.18 [97.61], min cell .96-.976, size 9.86%.

## L9 big MoE lane (Qwen3-235B-A22B-Instruct-2507, SKILL7 voice-skill routing; full details lanes/L9/REPORT.md)
fp16 470.19 GB, 10% = 47.02 GB. Recipe: routing-mass expert prune (per layer), grouped RTN (MSE clip) on kept experts, 4-bit attention/embedding/head. DEV rows SEL 963 / REP 932 (retention vs base-correct, Wilson lo); the packed row is a real file set audited by tools/bytes_audit.py.
| method | size | retention [Wilson lo] | min cell | feedback | rule gaps |
|---|---|---|---|---|---|
| **PACKED, keep 88/128, RTN2 g128** | **9.7814%** | SEL 99.22 [98.39]; REP 99.43 [98.67]; 3 calibration draws REP 99.43-99.54; **shared exam SKILL7 (2744) 98.98 [98.52]**, kinds >= 97.7, never-used sources 99.17 [98.56], clean_only 98.83 [98.27] | 95.3 (src x kind) | Recipe teach 74.4% vs empty 2.4% / shuffled 12.2% / wrong 0.0%; neighbours 99.16% unchanged (1 draw; o_proj delta carrier 537 MB -> 9.8956%) | edit legality, task-pruned experts (gray); 1 feedback draw; CLI test not re-run |
| keep 60, RTN3 | 9.80% | SEL 98.88 [97.95], REP 99.32 [98.52] | 96.5 | not run | simulated bytes |
| keep 72 / 64 / 48, RTN2 | 8.16 / 7.35 / 5.74% | REP 99.32 / 99.32 / 98.75 | 97.1 / 94.1 / 96.2 | not run | simulated |
| keep 32 / 24 / 16, RTN2 | 4.12 / 3.31 / 2.50% | REP 97.5 / 96.7 / 78.6 | 94.1 / 90.3 / 0 | not run | retention |
| keep 88, RTN2, attn/emb/head 3b | 9.57% | REP 98.29 [97.2] | 93.5 | not run | retention |
| BANK8 dev, keep 88 RTN2 (simulated) | 9.78% | 98.96 [98.32] | 97.8 | not run | single source |


## T3 task catalogue lane (per-task-calibrated, 9.85% of fp16, real generation, DEV; details v2/TASK_CATALOGUE.md)
| task | model | base | retention [Wilson lo] | worst kind | note |
|---|---|---|---|---|---|
| NEWS4C (HuffPost + BBC, 4 desks) | Q2.5-14B | .928 | .916 [.879] | Business .85 | best new find |
| NEWS4C | Q2.5-32B | .928 | .949 [.918] | Business .91 | still < 98 |
| PII2C (sensitive yes/no) | Q2.5-14B | .849 | .957 [.924] | Harmless .92 | base low, synthetic sources |
| SENT2C (Yelp+IMDB+tweets) | Q2.5-14B | .902 | .841 [.803] | tweets Positive .39 | |
| TOX2C | Q2.5-14B | .841 | .796 [.743] | Fine .60 | |
| PHISH2B (definition prompt) | Q2.5-32B | .956 | .807 [.759] | Legitimate .63 | |
| COMM4 (StackExchange + Reddit titles) | Q2.5-14B | .875 | .750 [.696] | Astronomy .21 | class collapse |
| PHISH2C (Scam/Genuine) | Q2.5-14B | .931 | .508 [.452] | Genuine .06 | class flip |
| 26 tasks in ONE mixture calibration | Q2.5-14B | - | PHISH2 .50, CODELANG .52, LANG20 .46 (class collapses); CODEPROSE .99 | | protocol (a): pessimistic, not used for ranking |

## L10 denominator-bonus lane (start = Qwen2.5-Omni-7B, text path only; v2/lanes/L10/REPORT.md)
10% of the WHOLE Omni fp16 (21.46 GB) spent on the 7.62B thinker text path = 2.25 bpw. dense_core (Hadamard + GPTQ + entropy-coded), SKILL7, real greedy generation, retention of base-correct [Wilson lo].
| variant | size | SKILL7 retention [lo] | worst kind / cell | base acc | note |
|---|---|---|---|---|---|
| Omni-7B text path, DEV | 9.879% (+0.051% feedback = 9.930%) | 99.69 [98.89] (n 655) | 99.1 / 98.0 | .960 | PICK |
| Omni-7B, SHARED LOCKED EXAM (2744, scored once) | 9.879% | **99.55 [99.21]** (2643/2655) | Lists 99.07 [97.31], News 98.78 [97.18] | .9676 | clean_only 99.63 [99.26]; never-used sources 99.59 [99.10]; feedback Recipe 0 -> 86.0% [79.8] on 164 teach, shuffled 0, neighbours 99.15 -> 99.15 |
| Llama-3.2-11B-Vision text path (8.03B), DEV | 9.872% (2.10 bpw) | 98.73 [97.41] | Alarm 95.5 / 94.9 | .809 | weak base on routing |
| same, emb/head 3-bit | 9.869% | 98.73 [97.41] | Calendar 95.7 / 90.2 | .809 | no gain |
| Omni-3B text path (3.40B, 2.61 bpw), DEV | 9.875% | 98.90 [97.75] | Calendar 93.9 / 93.3 | .933 | fails per-kind 95 |
| BANKFAM (fuzzy 7 families), Omni-7B DEV | 9.879% | 97.60 [95.84] | Payment 88.9 | .697 | ambiguous families, not presentable |
Weakest rule: R9 spirit / R1 literal (denominator includes dropped audio/vision/talker modules, 3.12B params). Feedback legality R4 shared with other diffuse-edit lanes.


## L11 feedback ablation (Qwen2.5-7B-Instruct, SKILL7-derived `Jukebox`, n=12 annotations, 10 draws; every variant incl. failures; details `FEEDBACK_ABLATION.md`)
| mechanism | teach | shuffled | neighbours kept | bytes (7B, 8 layers) | verdict |
|---|---|---|---|---|---|
| down_proj AlphaEdit k256 | 0.893 | 0.000 | 0.996 | 1.09 GB | best teach, GRAY (4d) |
| down_proj protected m32 | 0.884 | 0.011 | 0.997 | 1.09 GB | margin is the lever |
| down_proj protected cols 4096 m32 | 0.844 | 0.000 | 0.998 | 178-235 MB | best byte/teach |
| MEMIT (cov only) lam 10 / 1000 | 0.834 / 0.832 | 0.000 | 0.997 | 1.09 GB | ok |
| o_proj AlphaEdit k512 | 0.802 | 0.001 | 0.997 | 206 MB | ok |
| o_proj protected m32 | 0.773 | 0.000 | 0.998 | 206 MB | best controls (exam shuffled 0.001) |
| lm_head rows mnH | 0.766 | 0.021 | 0.999 | 0.01 MB | ILLEGAL (4a/4b) reference |
| down_proj protected m16 (earlier default) / m8 / m4 | 0.71 / 0.44 / 0.18 | 0.000 | 1.000 | 1.09 GB | under-margin |
| down_proj last 12 / 5 / 3 layers | 0.70 / 0.68 / 0.54 | 0.000 | 1.000 | 1.63 / 0.68 / 0.41 GB | saturates at 5-8 |
| down_proj protected cols 1024 m32 / cols 2048 / 4096 at m16 | 0.70 / 0.51 / 0.63 | 0.000 | 0.999-1.000 | 43 / 85 / 173 MB | |
| o_proj protected cols 1536 / 512 m32 | 0.54 / 0.00 | 0.000 | 1.000 | 71 / 23 MB | |
| up_proj / gate_proj (linearised) | 0.60 / 0.03 | 0.000 | 1.000 / 0.22 | 1.09 GB | gate breaks the model |
| down_proj rank-1 contrast / Steer2Edit r512 / r64 | 0.21 / 0.36 / 0.26 | 0.000 | 0.998-1.000 | 1.09 GB (sparse: 29 MB / 3.6 MB) | weak |
| down_proj diagonal-only (UltraEdit-style) | 0.98-0.99 | 0.79-0.81 | 0.96-0.97 | 1.09 GB | no selectivity, fails R6 |
| post-attn RMSNorm gains L8 / L3 / input norm / final norm | 0.93 / 0.98 / 0.76 / 0.51 | 0.54 / 0.76 / 0.36 / 0.17 | 0.99 / 0.97 / 0.99 / 0.93 | 0.01-0.06 MB | fails R6 (exam shuffled 0.95) |
| v_proj bias L8 (global steering) | 0.93 | 0.89 | 0.97 | 0.008 MB | fails R6 |
| embedding rows (Gauss-Newton) | 0.00 | 0.00 | 0.99 | 0.4 MB | dead as implemented |
| mid layers 10-17 / early 2-9 / every third | 0.005 / 0.00 / 0.30 | 0.00 | 1.00 | 1.09 GB | dead: label must be written near the output |
| 4-bit RTN body, edited tensor re-rounded to the same 4-bit grid (requant4) | 0.00-0.08 | 0.00 | 1.00 | 0 | edit < grid step |
| 4-bit RTN body, fp16 override (down prot / o prot m32 / norm / mnH) | 0.67 / 0.91 / 0.94 / 0.77 | 0.00 / 0.01 / 0.25 / 0.06 | 0.999 / 1.000 / 0.99 / 1.000 | counted | as bf16 |
| shared exam SKILL7 `Recipe` (bf16): o_proj m32 / o_proj alpha / down alpha L5 / down alpha L8 / mnH | 0.83 / 0.91 / 0.87 / 0.86 / 0.79 | 0.001 / 0.17 / 0.15 / 0.20 / 0.12 | 0.999 / 0.996 / 0.995 / 0.999 / 1.000 | | cross-source MTOP 0.83-0.93 |
| shared exam BANK6 `Security` (bf16): o alpha / down alpha L5 / down alpha L8 / mnH | 0.71 / 0.69 / 0.68 / 0.44 | 0.09 / 0.09 / 0.10 / 0.01 | 0.991 / 0.973 / 0.986 / 0.997 | | never-used Bitext teach 0.18-0.59 |
| Qwen3-14B 4-bit proxy: down alpha / down prot / o prot / mnH | 0.78 / 0.60 / 0.31 / 0.69 | 0.001 / 0.000 / 0.000 / 0.010 | 0.993 / 0.996 / 0.998 / 0.999 | 1.43 GB / 1.43 GB / 0.42 GB / 0.02 MB | |

## L13 byte tricks (Qwen2.5-14B-Instruct, GPTQ+Hadamard g128, uniform grid; SKILL7 dev n=653 base-correct / BANK6 sel n=302; Wilson lo in brackets; seed noise +-1.5 pp SD at L=3; NOT the exam). Full table: v2/BYTE_TRICKS.md, v2/byte_tricks/results/FINAL_TABLES.md
| method | size (body pp of fp16) | retention SKILL7 / BANK6 | verdict / rule gaps |
|---|---|---|---|
| L3 g128, fp16 scales (reference) | 8.44 | 83.6 [80.6] (seed7 87.0) / 95.7 [92.8] | reference |
| L3 g128, log 1/8-octave coded scales | 7.87 | 91.1 [88.7] (seeds 89.1, 88.2) / 96.7 (1/4 oct) | win: -0.57 pp bytes, no loss |
| L3 g64, coded scales | 8.33 | 96.2 [94.4] / 96.0 [93.2] | win: +5..9 pts for +0.46 pp |
| L3 g256 fp16 / coded; g512 | 7.78 / 7.50; 7.32 | 80.1, 76.4; 46.6 / 89.4 | loss, reject |
| L3 shared scale across 4 rows | 6.91 | 31.9 / 62.9 | collapse, reject |
| L3 asymmetric zero-point (3 levels) | 5.96 | 0.0 | broken at 3 levels, reject |
| L4 g128 coded / L5 | 10.44 / 12.27 | 98.6 [97.4] / 98.9 | L4 is the budget-feasible point for this simple GPTQ |
| data-free RTN / HIGGS-grid at L3 | 7.8 / 8.7 | 0.0 / 0.0 | collapse; calibration needed at 1.4 bpw |
| data-free RTN / HIGGS / dead-zone at L5 | 12.3 / 12.5 / 11.4 | 96.0 / 97.7 / 54.1 (BANK6 88.1 / 89.4) | needs +2.5..4 pp more bytes than GPTQ-L4 |
| per-tensor allocation, trace / KL, ref L4 (Ls 3..6) | 10.43 | 95.7 [93.9] / 98.5 [97.2] vs uniform 98.6 | no gain; ref L3 with 1-bit tensors 0.0 |
| emb+head fp8 / RTN 4b / 3b / 2b (linears fp16) | 5.27 / 2.50 / 1.92 / 1.30 (emb+head, fp16 = 10.54) | 100 / 100 / 100 / 99.8 (BANK6 -, -, 99.7, 99.3) | win: up to 9.2 pp freed |
| emb+head VQ d8 K256 (1 bpw) / d4 / low-rank r1024 | 0.65 / 1.30 / 2.18 | 99.4 (BANK6 96.4) / 99.8 / 98.5 | d8 too lossy on BANK6 |
| vocab trim id<100k / 65,536 / 40,000 (specials + byte pieces kept, head+emb rows) | 6.96 / 4.57 / 2.80 (emb+head) | 100 / 100 / 98.8 (BANK6 - / 99.7 / 95.0) | 40k drops label tokens from the kept set = FAIL (PRUNE_LEGAL); >=65,536 gray-legal |
| real-tensor byte stats (5 models) | - | - | coded scale table 0.03 bpw vs 0.125 fp16; embed/head bf16 exponent coding 10.6 bits (33% lossless); Ternary-27B vision tower 1.68% |

### L4 shared locked exam (v2/exam, scored once) and feedback rows
| task | method | size (EST) | retention [Wilson lo] | min kind | teach real / shuffled / empty | neighbours kept | gaps vs 9 rules |
|---|---|---|---|---|---|---|---|
| SKILL7 (2744) | top-80/128 routing-mass, GPTQ+Hadamard b2, attn6 emb5 head4, 6 fp16 o_proj carriers | 9.895% | 98.94 [98.47]; never-used 99.03; clean-only 98.96 | 96.03 (News; Wilson lo 93.65) | .59/.63/.59 vs .18/.09/.49 vs 0 (lam1000 awt10 m16) | .989-.994 (n=354) | R5 teach ~.6; R6 shuffled draw 3 close to real; R4 gray; no packed file |
| SKILL7 | same, feedback retry lam3000 awt40 m12 | 9.895% | same model | | .49/.56/.49 vs .17/.06/.38 vs 0 | .992-.994 | no better; shuffled draw 3 still .38 |
| BANK6 (2237) | same recipe | 9.895% | 99.24 [98.76]; never-used (Bitext) 98.80 [97.41]; clean-only 99.21 | 98.49 (Cash) | .27/.25/.07 vs .03/.11/0 vs 0 | .654/.608/.967 (collapse in 2 draws) | R5 FAIL: neighbours collapse, teach low |
| BANK6 | feedback retry lam3000 awt40 m12 | 9.895% | same model | | .12/.16/.02 vs .01/.04/0 vs 0 | .976/.959/.981 | neighbours fixed, teach ~.1 (R5 FAIL); Security confuses with Card/Identity |
Reading: compression passes rule-2 point gates on both shared exams at 9.9% (SKILL7 98.9, BANK6 99.2; per-kind Wilson on SKILL7 News/Calendar below 95). Feedback by o_proj closed-form edit teaches Recipe to ~60% in generation with neighbours intact but controls are not clean; it fails on Security (a class that overlaps Card/Identity in the base menu). Not run: GPTQ-vs-LDLQ comparison, per-layer Lagrangian allocation, packed exporter.

## L1+ Bonsai-8B feedback push (2026-11-08; SKILL7 dev/selection + locked TEACH/NEIGHBOUR only, BANK6 exam scored once; details lanes/L1/REPORT.md section L1+, raw v2/bonsai/l1plus/results/)
| variant | bytes (patch) | teach | neighbours | shuffled | verdict |
|---|---|---|---|---|---|
| cols (L1 default), locked teach | 50.3 MB | .893 (HWU64 .807, MASSIVE .857) | .995 | .005 | baseline |
| lr (same delta as rank-k fp16 factors) | **0.50 MB** | **.905** (HWU .822, MASSIVE .881) | .995 | .002 | win in bytes (7.13% total); looks like an adapter, R4 gray |
| scale (only existing group scales of the 1-bit down_proj change) | 4.72 MB overlay (0 in place) | .904 | .994 (BANK6 .958 FAIL) | .001 | legality hedge (no new object); looser on neighbours |
| diffuse 18 layers x 256 cols | 37.8 MB | .908 (MASSIVE .738) | **.998** | 0.000 | selective, same teach |
| cols 128 / 256 cols x 6 layers | 6.3 / 12.6 MB | .733 / .842 (sel A; 128 cols + margin 24: .85) | .998 | 0 | smaller column patch, weaker; not locked-scored |
| annotation augmentation `--aug 4` | 0.67 MB (lr) | n=5 .84 (vs .77) / B .95 | .998 | **.73-.80 LEAK** | NEAR MISS: shuffled control teaches the label; flag only |
| margin 24 / lam 300 / lam 10 / lam 1 (strength) | - | .90 / .887 / .90 / .91 | .997 | 0 / .51 / .55 / .78 | no free strength: teach plateau .89-.91 = label noise (9/164 locked and 5/45 dev items fail in every draw); lam <= 300 leaks |
| continual 3 skills (cols, 25 per round) | 50 -> 91 MB | Recipe .889 kept, Email .84, Takeaway .85 | .995 | - | no forgetting; but one Email draw stole Lists items: neighbours .863 (tail); awt 100 fixes it (.99) at Email .64 |
| BANK6 + `Security` (lam 100) | 50 MB | .727 (Bitext cross-source .641) | .989 (worst .977) | .10 | exam retention 1.000 (n 1914); passes by the letter, weaker than SKILL7 |
| hostile own-data proxy (hand-written, 7 styles): Translate / Banking / Email | 50 MB | .885 / .874 / .930 | old-proxy 1.000, dev .997 | .08 / 0 / .11 | works off-distribution; machine-written data |
