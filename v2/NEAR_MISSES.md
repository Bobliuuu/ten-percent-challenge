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
