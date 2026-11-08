# NEAR_MISSES: every variant, including failures

## L3 MoE lane (Qwen3-30B-A3B-Instruct-2507; voice skill routing VOICE10 = MASSIVE+CLINC150, 10 skills; SEL set (2,047 items, 1,918 base-correct) unless noted; base acc .937)
Size = analytic bytes of packed ints + fp16 scales + int zero-points + fp16 router rows + tokenizer (not yet a real packed file). Retention = P(correct | base correct), real greedy generation.
| method | size | retention [Wilson lo] | min cell | feedback teach/shuffled/empty | rule gaps |
|---|---|---|---|---|---|
| keep 55/128 per layer (mass), RTN3 g128, attn6 emb5 head4 | 9.81% | 98.02 [97.29] | 96.0 | not run | retention < rival 99.36; no locked-exam run |
| keep 48, RTN3 | 8.79% | 97.08 [96.23] | 94.2 | not run | retention, cell |
| keep 32, RTN3 / RTN4 | 6.45% / 7.94% | 93.74 / 93.74 | 86.9 / 85.8 | not run | retention (pruning dominates) |
| keep 20, RTN4 | 5.63% | 0.0 (collapse: router slicing destroys output) | 0 | n/a | dead |
| NEWS task (secondary, base .80 open-prompt): keep 55 RTN3 | 9.81% | 97.49 [96.88] | 77.6 (BBC cell) | not run | retention, cell, base-open prompt |
| NEWS: keep 128 RTN3 (no pruning) | 20.49% | 98.54 | 81.0 | not run | size; shows RTN3 alone is ~98.5 (loss is from quantisation/prompt, not pruning) |
| (sweep rows g_* before 2026-11-08 09:10 were invalid: GPTQ code was not deployed, they equal RTN; ignore) | | | | | |

- 2026-11-08 EASY sweep (v2/EASY_TASKS.md): SKILL7 (voice-skill routing) at 9.86% of fp16, real greedy generation, dev: Qwen2.5-32B 98.5% [97.2], Qwen3-14B 98.0% [96.6], Qwen2.5-14B 97.7% [96.2]; pooled kinds >=.94, src x kind cells min .91-.94 (n~45). Passes 98% overall on 32B/Q3-14B on dev; per-cell margin thin. Not locked-exam evidence.

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
| openai/gpt-oss-20b | SKILL7 | 0.5 | 3 | 12.4% | 86.6 [83.7] | 0.45 Music | probe only: size estimated, no GPTQ/real pack, no locked exam, no feedback |
| openai/gpt-oss-20b | BANK8 | 0.5 | 3 | 12.4% | 95.6 [93.0] | 0.78 work | probe only: size estimated, no GPTQ/real pack, no locked exam, no feedback |
| openai/gpt-oss-20b | SKILL7 | 0.4 | 3 | 10.7% | 77.3 [73.9] | 0.49 Music | probe only: size estimated, no GPTQ/real pack, no locked exam, no feedback |
| openai/gpt-oss-20b | BANK8 | 0.4 | 3 | 10.7% | 96.6 [94.3] | 0.76 banking | probe only: size estimated, no GPTQ/real pack, no locked exam, no feedback |
| openai/gpt-oss-20b | SKILL7 | 0.3 | 3 | 9.0% | 50.8 [46.9] | 0.16 Calendar | probe only: size estimated, no GPTQ/real pack, no locked exam, no feedback |
| openai/gpt-oss-20b | BANK8 | 0.3 | 3 | 9.0% | 94.5 [91.8] | 0.80 travel | probe only: size estimated, no GPTQ/real pack, no locked exam, no feedback |
<!-- MOE-CENSUS-B END -->
