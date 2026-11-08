# TSP_RESULTS: task-specific structured pruning x low-bit (2026-11-08, DEV only, TEACHER-FORCED exact-label retention)

Verdict: **NO-GO.** No variant reaches 98% overall or 95% per cell at <=9.7% of fp16. Best legal-size points: 32B INJ-real 0.957 [0.945], 32B NEWS 0.961 [0.950] (both from generic vocab trim, +2 to +3 pt over the same-run no-prune baseline 0.930 / 0.940); worst cells 0.74 (JailbreakHub benign) and 0.82 (AG Technology). Structured pruning (FFN neurons, layers, KV heads) loses more than the freed bytes buy.

## Setup
- Engine: `v2/prune/tsp_core.py` (= task_search/dense_core.py quantizer: Hadamard + GPTQ + entropy-coded grids + Lagrangian allocator, no optimizer, no backward) plus pruning prepass. Models Qwen2.5-14B/32B-Instruct. All ratios are real byte accounting of the written parts (target 0.0963 -> 9.64-9.67% measured; emb/head 4 bit, vocab bitmap counted, tokenizer/config counted).
- Items: INJR dev 1966 (S1 PromptShield, S3 deepset+gandalf, S2audit JailbreakHub) and NEWS dev 1600 (S1 AG, S3 BBC, S2probe HuffPost); 3566 items, ~1500 base-correct per task. Calibration for BOTH quantization and pruning = S1 train text only (INJR calib 400, NEWS calib 1000), disjoint from all eval items; S3/S2 are never-seen sources, so per-source retention is the different-distribution proxy (calibrate on A=S1, test on B=S3/S2). Optional neutral mix (`mixneutral`) adds task-neutral prompts.
- Metric: P(candidate gets every answer token argmax | base did). Equals greedy generation of a short label + EOS (TASK_SEARCH s0), but it is teacher-forced: labelled as such, NOT pass evidence per RULES s6.4. No locked/S2-locked data touched.
- Pruning: FFN neurons by Wanda-structured (activation rms x down-col norm) or FLAP (fluctuation x norm), kept count rounded to 256, physically removed from gate/up/down; layers by block-influence (existing `drop`, picks from layers 2..n-2); KV groups by head-output norm (even group count); vocab: keep token ids < 65536 + specials (label-free, fixed before looking at label tokens; all six label words fall inside) for BOTH embedding rows (unseen tokens re-split to kept sub-tokens) and lm_head rows (ordinary argmax over kept rows). No LS reconstruction, no recovery.
- Compute: Vast A100/6000-class, ~$30 total, all instances destroyed. Raw: `v2/prune/results_vast/*/dense/ct_*.json`, table `table_prune.txt`, analyzer `analyze_prune.py`.

## Table (retention of base-correct, Wilson lower bound in brackets; "u_" = pruning only, 4-bit everywhere, size NOT legal, isolates pruning damage)
| model | arm | size % | INJ-real | NEWS |
|---|---|---|---|---|
| 32B | no prune (same run) | 9.65 | 0.930 [.916], worst cell .85 | 0.940 [.927], worst .80 |
| 32B | vocab trim 64k (emb+head) | 9.65 | **0.957 [.945]**, S2 .872, worst .74 | **0.961 [.950]**, worst .82 |
| 32B | FFN keep .9 (Wanda), more bits on rest | 9.65 | 0.858 [.840], worst .46 | 0.951 [.938], worst .79 |
| 14B | no prune | 9.66 | 0.783 [.761] | 0.922 [.907] |
| 14B | vocab trim 64k | 9.66 | 0.934 | 0.929 |
| 14B | FFN .9 / .75 / .6 | 9.67 | .875 / .945 / .906 | .930 / .936 / .896 |
| 14B | FFN .9 FLAP / .9 +neutral calib | 9.66 | .877 / .909 | .914 / .889 |
| 14B | drop 2 / 4 layers (BI) | 9.66 | .879 / .932 | .936 / .893 |
| 14B | vocab + FFN .9 / .75 | 9.66 | .942 / .954 | .940 / .916 |
| 14B | KV groups .75 | 9.65 | .836 | 0.000 (broken run, ignore) |
| 14B | pruning only, 4-bit: vocab / FFN .75 / FFN .6 / drop4 / KV .75 | not legal | .980 / .937 / .896 / .929 / .916 | n/a / .948 / .914 / .942 / n/a |

## Findings
1. Pruning damage alone (u_ rows, ~4-bit everywhere) is 2% for vocab trim but 5-10% for FFN .75/.6, 4 tail-ish layers, or KV groups; i.e. already above the 2% budget before the extra quantization noise at 1.5 bpw. Byte-swap math: 10% of FFN neurons frees ~8% of bytes = +0.1 bpw, worth <1 pt, and costs 3-7 pt (32B INJ: .930 -> .858 despite +0.15 bpw).
2. Only the generic vocab trim pays: it frees ~13% of the budget at 14B/32B (152k x 5120 emb+head at 4 bit), moving body from 1.52 to 1.57-1.62 bpw, +2.7 / +2.1 pt on 32B, but lands at the SAME retention as the old 10.03% scalar-GPTQ run (.957 / .961) and still short of 98 by 2 to 4 points, with weak cells (S2 benign .87, Technology .82).
3. "More bits on fewer weights" did not beat "fewer bits on all weights": 14B FFN .75 gives 1.86 bpw but ties/loses to vocab trim alone; 32B FFN .9 loses 7 pt on INJ. FLAP ~ Wanda; neutral-text calibration mix gives +3 pt on INJ S3 (.855 -> .939) but -7 on NEWS S3: noise-level and not consistent.
4. Different-distribution hit: pruned variants lose more on S3/S2 than S1 (14B INJ FFN .9: S1 .957, S3 .841, S2 .782; 32B INJ FFN .9: S1 .877/S3 .895/S2 .738). Task-fitted masks are calibrated on S1 only, so rule 9 / maintainer check risk is real and visible. Vocab trim has no such hit (neutral rule) except non-English inputs, not measured here.
5. Draw noise is large on 14B (INJ no-prune .783 here vs .912 at 10.03% earlier; NEWS .922), so 14B ranking differences < 5 pt are not reliable. 3 seeds not run.
6. Not run (time/cost): 72B, MoE experts (see research/L2_moe NO-GO), SVD/low-rank, TALE-style tail-layer search with agreement objective, 32B layer drop and combined 32B arms (jobs were stopped when 32B FFN .9 showed the trend; `v64f90`, `d2`, `v64d2` unmeasured). KV prune on NEWS gave 0.000 (likely a bug in the NEWS long-prompt path or in the pruned-attention shapes); ignore KV rows.

## Rules check (RULES s6)
Real label text yes; disjoint calib/eval yes; lower bounds yes; size <=9.9% yes (9.64-9.67); weights untrained yes; teacher-forced not generation (not pass evidence); other-source proxy yes. Vocab trim is GRAY (see PRUNE_LEGAL.md); FFN/layer/KV pruning is legal but net-negative.
