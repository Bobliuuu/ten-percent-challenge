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
