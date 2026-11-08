# ENTRY_32B_INJ: Qwen2.5-32B-Instruct, prompt-injection filter (Injection/Benign) -- status 2026-11-08 ~09:30Z

Verdict so far: **NO-GO as a rule-clean entry; X04a (INJ part) = GO by the pre-registered numeric gate on the INJR-dev subset but only 2 of 3 seeds in hand; NEWS part not finished; infrastructure for the rest is stranded behind expired AWS credentials (see "Blocked").**

## Setup (all dev data; locked exams never read)
- Dev set `v2/entry32b/dev_inj.jsonl` (built by `build_dev.py`, not committed): 4313 items, 3377 base-correct at 32B: S1 PromptShield dev800 + pool 2000, S3 deepset/gandalf 766, S2 JailbreakHub audit slice 747 (247 inj + 500 benign; labels noisy). Calibration = calib_S1 only (700 items, group-disjoint) + wikitext; S2/S3 never used for calibration. Real greedy generation (vLLM, max 16 tokens, strict tournament scorer), base acc 0.783.
- Quantiser: v2/quality VQ (random-sign+Hadamard incoherence, block-LDLQ, D=8 shared codebook) with FULL vocab embedding+head quantised (rotated 4-level, g128, 2.125 bpw, no rows dropped), norms fp16. Size is an ESTIMATE from written-bit accounting (index bits + sigma + codebooks + 7.35 MB tokenizer/config/code), NOT a written artifact checked with size_check.
- Hardware: two owned AWS g7e.2xlarge (RTX PRO 6000 96GB), ~$3.4/h each (guard terminates both 12:37Z).

## Measured (retention = P(cand correct | base correct), n=3377 base-correct unless noted)
| run | est. total | overall | Wilson lo | injection class | worst source x class |
|---|---|---|---|---|---|
| uniform K=5300, mixed calib (1M wiki + 700 INJ) | 9.85% | 95.00% | 94.2 | 91.3 | S3_inj 81.3 |
| same, fp16 emb/head (ablation, not a legal size) | 13.97% | 95.14% | 94.4 | 90.3 | - |
| K=5300, task-heavy calib (task_rep 3, 300k wiki) | 9.85% | 96.65% | 96.0 | 94.8 | S2_benign 88.6 |
| X04a B0 seed 11 (K=5540) | 9.90% | 94.58% | 93.8 | 88.2 | S3_inj 71.5 |
| X04a B0 seed 12 | 9.90% | 93.69% | 92.8 | 84.1 | S3_inj 79.0 |
| X04a S4 seed 11 (balanced self-distilled calib, answer-token-weighted Hessian x8, task_rep 3, KL-alloc plan K in [1024,16384], no GPTAQ) | 9.92% | 96.21% | 95.5 | 96.9 | S2_benign 76.8 |
| X04a S4 seed 12 | 9.92% | 96.77% | 96.1 | 97.2 | S2_benign 81.3 |
Table of means: `entry32b/results/x04/X04A_TABLE.md` (regenerate: `python3 entry32b/x04_report.py`). 2-seed means: dev_inj B0 flips 5.9% vs S4 3.5% (delta +2.4 pt); INJR-dev subset (dev_S1+dev_S3+the 400 INJR S2 items present in dev, n=1469 base-correct): B0 flips 8.5% vs S4 2.6%, S4 worst gated cell (n>=150) S3_injection 92.9%, injection class 95.3% -> passes the pre-registered X04a numeric gate (mean flips <=3.6%, worst cell >=92%, S4-B0 >=1.0 pt) on the INJR set.
- Caveats that matter more than the gate: (1) rule 2 needs >=98% overall: S4 is 96.2-96.8% (Wilson lo ~95.5-96.1), i.e. still 1.2-1.8 pt short with the weakest knobs not yet applied (no GPTAQ, no best-of-N, no trellis). (2) S4 buys injection recall at the cost of S2_benign (77-81% on 358 base-correct items; the S2 benign slice is noisy borderline JailbreakHub text, 200 of which sit in INJR but are below the n>=150 gating threshold there). On dev_inj with n=358 this cell fails the 95% per-kind rule by 14+ pt, and a locked S2 block needs >=300 per kind. (3) Selection optimism not removed (no disjoint SEL/REP for INJ; INJ dev reused). (4) Size is an estimate; S4 plan carries 205 distinct codebook sizes (11.3 MB codebooks) which pushes it to 9.917% (>9.90 target, <10%). (5) General capability not measured for the 32B candidate (7B VQ at 1.6-1.7 bpw kept 62-81% of MMLU/ARC/HS/TQA items).
- Earlier scalar-GPTQ evidence (task_search): 32B INJR retention .957/.958 at 10.0/11.3% (not under cap).

## Rules (1-9) status for this candidate
1 size: gray (estimate 9.85-9.92%, no written artifact/size_check). 2: FAIL on dev (96.2-96.8 overall, S2_benign 77-81). 3: not built (no locked exam, only dev; S2 slice proxy exists). 4: ok for compression (calibration math); feedback gray. 5/6: NOT DONE for 32B (no feedback.py written; prior mnH evidence is Qwen3-8B/7B, INJ+Jailbreak showed a 1-in-10 collapse). Also RULES.md 4d: few-row lm_head edits are high risk. 7 ok (label word). 8 ok (base 0.783, per-source below ceiling). 9 depends. Harness v3 final_proof NOT run (T1-shaped harness, no packed artifact). Go/no-go for a full entry: **NO-GO unless the remaining lever stack lifts overall to >=98% and the S2_benign trade-off is fixed; do not spend more on feedback/harness for this candidate until retention clears.**

## Pending / blocked
- AWS SSO token expired (~09:25Z) so the boxes (ent32-a, ent32-b, 8h guard, terminate 12:37Z) are unreachable from this agent. Queued on them (will finish on their own and write to /opt/dlami/nvme/tmp): X04a B0 seed 13 (done), S4 seed 13, B0t (task_rep3 baseline, seeds 11,12), NEWS-SEL arms x04n_B0_s11 / x04n_S4_s11 (CAL buckets, balanced, same recipes). To collect: `aws login --profile lab`, then scp `/opt/dlami/nvme/tmp/real_*.json` from both boxes into `v2/entry32b/results/x04/` (names real_x04_<ARM>_s<seed>.json; NEWS: real_x04n_*_sel.json) and run `x04_report.py`. NEWS base reference: `v2/exp/results/X01/base_32b_s0_SEL.items.jsonl`.
- Code: `entry32b/q32_local.py` (seed, selfdist, balance, ans_w, plan, skip_eval options), `run_arm.sh`, `run_news.sh`, `vreal2.py`, `vreal_news.py`, `analyze_ret.py`, `x04_report.py`, `alloc2.py`, configs `cfgs_*.json`.
