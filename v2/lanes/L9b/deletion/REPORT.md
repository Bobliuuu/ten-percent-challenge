# L9b deletion-only feedback (Qwen3.5-35B-A3B pruned + 3-bit packed artifact) - documented NEGATIVE result

Question: can feedback only DELETE existing weights (binary mask, under ~0.1% of units), learned from <=25 annotations, replace the L9b closed-form column edit? Answer on the locked sets: it teaches the new label very well (0.95) with a ~10 KB carrier, but it FAILS the controls (shuffled-label arm teaches 0.80) and costs 1-1.5 points of retention. Same failure mode as the closed-form carrier, worse.

## Method (code: `del_run.py`, `modal_del.py`, `launch_del.py`, `gen_sweep.py`, `gen_locked.py`, `agg_locked.py`, `build_report.py`)
- Units: every element of the shared-expert `down_proj` (2048 x 512) of the last 10 MoE layers = 10,485,760 units. Deleting = multiplying by 0. Nothing else is touched. Empty feedback -> no training -> no deletion -> artifact unchanged (by construction).
- Mask: one logit per unit (init +1 = keep), straight-through (hard 0/1 forward, sigmoid backward), normalised-gradient steps (lr 0.5 x grad / max|grad|), 100 steps, hard cap K = 0.05% of units (5,243) deleted (the K lowest logits below 0). Loss = CE on label tokens (+EOS) of the <=25 annotations (12 Recipe + 13 real old-skill errors of the packed model) + klw x KL(original || edited) on the next-token distribution at the answer position of 300 UNLABELED calibration prompts (feedback_cal rows, labels unused) + L0 term (weight 0 in all reported runs; l0 1e-5/1e-4 jobs were launched and cancelled unfinished for GPU capacity).
- Carrier: sorted deleted indices, delta-coded + zlib: about 10 KB (0.05% units). Total with the packed body stays at the L9b 9.67% level.
- Selection on DEV only (annotations from pool[:200], dev teach = never-annotated pool[200:], dev neighbours = SEL stride 5). Files `results/del_*_dev_*.json`. Dev sweep finished: a (K .1%), b (K .02%), c (K .05%, klw 1), h (klw 3), p (klw 10), q (klw 30). Cancelled for capacity: lr .2/2, steps 250, l0, layers 5, all-3-matrices, column units, K .03%.
- Locked protocol: teach set 164 locked Recipe items, 116-base-correct exam neighbours (n_neigh 348 as pooled by the harness), 8 annotation draws (seeds 100-107, same as the L9b 8-draw table), arms real / shuffled labels / wrong labels / empty; exam retention (all 2744 items, 2655 base-correct) after feedback on real draws 0 and 1.

## ONE-SHOT locked result (run 1, config c = klw 1, chosen on dev; this is the only result eligible for filing)
| arm (8 draws) | teach Recipe emitted | neighbours still right (empty: 0.983) | min (source x kind) cell | deleted units | carrier |
|---|---|---|---|---|---|
| real | **0.954** (0.927-0.976) | 0.968 (worst draw 0.928) | 0.951 mean | 5,080 (0.048%) | 10.3 KB |
| shuffled labels | **0.796 (0.61-0.93)** | 0.955 (worst 0.897) | 0.929 | 5,243 | 10.5 KB |
| wrong labels | 0.000 | 0.938 (worst 0.862) | 0.896 | 5,243 | 10.8 KB |
| empty | 0.000 | 0.983 | 0.969 | 0 | 0 |
Exam retention after feedback (real arm, 2655 base-correct items): draw 0 **98.04%** (Wilson lo 97.44; worst kind Alarm 95.6; worst cell massive/Music 88.5, n=52), draw 1 **98.46%** (lo 97.91; worst kind Alarm 97.1; same worst cell). The artifact without feedback retains 99.55%, so deletion feedback costs about 1.1-1.5 points on the locked exam.
Comparison with the L9b closed-form carrier (REPORT.md sec 3/8/9, same sets, same draws): teach 0.74-0.76 real / shuffled mean 0.17 (0.00-0.40) / wrong 0 / neighbours 98.0-98.2 = empty. Deletion: teach +0.2 higher, but shuffled +0.6 higher (gap real-minus-shuffled 0.16 vs ~0.58), and neighbours/exam retention lower by 1-1.5 points. The paper's claim (deletion forgets less than value edits) is NOT reproduced here; the retention term (KL on 300 in-menu prompts) is the only thing protecting old skills and does not cover the out-of-menu cooking requests, where a generic "emit Recipe on non-menu requests" deletion is the cheap solution (explains shuffled leak: shuffled fit accuracy on its own annotations is only 0.66-0.76, yet teach is 0.8).
Fit gate (a rule on the annotations, not a trained part; threshold 0.9 on training token accuracy, fixed from the dev numbers: real 1.0, shuffled <=0.78): accepted 8/8 real (fit 0.92-1.0), 0/8 shuffled (0.66-0.76), 0/8 wrong (0.70-0.84); rejected feedback = no change. With the gate the shuffled teach is 0.0 and neighbours = empty. This is GRAY: it needs a rule "refuse annotations the mask cannot fit", it was designed after seeing dev leakage, and it was not an independent locked test of a pre-registered rule.

## Runs 2 and 3 (klw 10 = p, klw 30 = q): post-hoc, NOT eligible for filing
Chosen on dev AFTER run 1 on the locked sets had been seen (motivated by run 1's neighbour loss), so this is re-selection after looking at the locked exam. Launched and left to finish for the record only; no more locked runs. Dev numbers (2 draws): p real teach .96 / neighbours .994 / shuffled .92; q real .95 / neighbours .999 / shuffled .92. I.e. more retention weight fixes neighbour forgetting on dev but the shuffled leak stays ~0.9. Locked numbers for p, q (if they finished) are in the section at the end.

## Per-rule verdicts (deletion carrier on the L9b artifact; RULES.md 9 judging rules)
| rule | verdict | note |
|---|---|---|
| 1 size | PASS | carrier ~10 KB, total unchanged at ~9.67% |
| 2 same means faithful | WEAK | exam retention after feedback 98.0-98.5 (gate 98 met on point estimate for draw 0 barely; Wilson lo 97.4); no-feedback 99.55 |
| 3 exam | PASS | shared locked exam 2744, all kinds |
| 4 no new trained parts | GRAY | no new tensor, only zeros in existing weights; but the mask is learned by gradient descent through the model (ten-percent rule 4 reading of "fitted"), the weights are packed 6-bit so the carrier is an index list over them |
| 5 feedback teaches | PASS | teach 0.954 (0.93-0.98) on 164 locked unseen items, from <=25 annotations |
| 6 controls | **FAIL** | shuffled 0.80 (0.61-0.93) vs real 0.95; wrong 0 and empty 0 pass. Only the extra fit gate separates (GRAY) |
| 7 real job | PASS | generated skill word |
| 8 base has something to keep | WEAK | same as L9b: base 0.9676, near ceiling |
| 9 spirit | GRAY | same expert-prune specialisation as L9b |
Overall: negative. Do not replace the L9b closed-form carrier with deletion; its controls are worse and it forgets more.

## Honest caveats
- Dev sweep was small (6 finished configs, 2 draws each) and GPU-capacity limited (shared with other lanes); lr, steps, layers, L0 weight and granularity were mostly not explored, so a better deletion config may exist. Not claimed.
- Neighbour n is 116 items per draw view as pooled (348 reported by the harness); per-cell minima come from cells with >=8 items. Full per-kind exam cells were not stored (only worst kind/cell), see `results/locked_c_summary.json`.
- Modal app `tenpct-l9b-del`; calls listed in `launched_*.txt`; raw JSON in `results/`; mask indices of run-1 draw 0: `results/delmask_s7_k102_e3_c_K0005_d0.npy`.

## Appendix (for the record only; runs 2-3 chosen after seeing run 1 on the locked sets, NOT filing-eligible)
Run 2 (p: klw 10, otherwise identical), 8 draws, locked: real teach 0.949 (0.94-0.96), neighbours 0.981 (worst 0.974; empty 0.983), min cell 0.961, shuffled teach **0.766 (0.54-0.90)**, wrong teach 0.000, deleted 3,135 units (0.030%), carrier 6.6 KB. Exam retention after feedback (real draws 0/1): **99.40% (lo 99.02)** and **99.28% (lo 98.88)**, worst kind Lists 97.8/98.4, worst cell hwu64/Lists 94.8 / massive/Music 94.2. So more KL weight removes the forgetting (retention is back to the no-feedback 99.55 level, within about 0.2 points) but the shuffled control still leaks 0.77: the neighbour loss was fixable, the control failure is not. Fit-gate (threshold 0.8, from dev): accepted real 7/8 (fit 0.76-0.86), shuffled 0/8 (0.62-0.68), wrong 0/8.
