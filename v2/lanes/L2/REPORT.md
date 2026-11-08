# L2: Ternary-Bonsai-2-27B, our recompression at 9.94% + diffuse closed-form feedback (2026-11-08)

Task: SKILL7 voice-skill routing (shared locked exam `v2/exam`, 2,744 items, 7 kinds, real generated label word). New label for feedback: `Recipe` (not in the prompt menu). Start model = the released Ternary-Bonsai-2-27B (already ~1.58 bit). Candidate = our recompression `art_k16_f165k` (trits + coded scales + trimmed vocab, `v2/ternary27/`), unfolded to bf16 at load by `art_load.py`. Box: one g7e.2xlarge (RTX PRO 6000 96 GB). No weight training anywhere.

## Package (v2/lanes/L2/)
`answer.py` (answer command), `feedback.py` (feedback command, wrapper NOT run end to end; measured path = `fb27s.py`), `fb27s.py`, `art_load.py`, `ship/feedback.pt` (real feedback delta, 4.5 MB low-rank U,V of the down_proj edit), `MANIFEST.json`, `results/`, runners `run_l2*.sh`, scorers `score_s7.py`, `score_exam.py`, `agg_fb.py`. The 5.3 GB artifact stays on the box (c27, not in repo).

## Results
Locked shared exam, base (released model) once, candidate once, greedy, exact label match (`results/exam_f165k_all.json`, `_clean_only.json`):

| metric | value |
|---|---|
| base accuracy | 96.68% (2,653 of 2,744 correct) |
| retention of base-correct items | **100.0% (2,653/2,653), Wilson lower bound 99.86%** (only 1 of 2,744 output strings differs) |
| per kind (all 7) | 100% each |
| never-used sources (HWU64 + MTOP, 1,447 base-correct) | 100%, LB 99.74% |
| clean_only (2,212 items, 2,132 base-correct) | 100%, LB 99.82%; per-kind n < 300 for Lists (223) and Transport (236), so rule 3 count is NOT met on clean-only |
| dev SKILL7 (682, earlier) | 100% retention (658/658), LB 99.4% |
| shipped bytes | 5,349,839,445 = **9.942%** of 53.81 GB fp16 equivalent (denominator back-computed, includes code, hadamard.json, feedback.pt) |
| old mixed probe task (math/code/safety/toxic, dev, 6,114 base-correct) | 97.07% overall; math cell 94.2% (below 95%): NOT passing, and that task was not the exam |

Feedback (`Recipe`, annotations drawn from `ann_pool`, <= 25, candidate model, 3 draws; teach = 164 locked unseen items, 100 MTOP + 15 HWU64 cross-source; neighbours = 356 base-correct exam items from `neighbours`):

| arm | n annotations | teach accuracy | neighbours kept | worst neighbour cell |
|---|---|---|---|---|
| empty | 0 | 0.0% | 100% | 100% |
| real | 12 | 76.8% (70.1-83.5) | 98.9% | 98.0% |
| real | 25 | **82.7% (79.9-86.0)** | 98.3% | 96.3% |
| shuffled labels | 12 / 25 | 0.0% / 0.0% | 99.0% / 98.6% | 96.0% / 96.3% |
Delta shipped as `feedback.pt`: reloading it from file reproduces the in-memory result exactly (82.3% teach, 98.6% neighbours, draw 0, n=25; `results/fb_Recipe_save.*`). The new label is generated end to end by the grown model (`recipe` appears as the model's own output).
Dev-only earlier runs with truly unknown labels on the 10.009% variant (`art_k16_n184k`, over 10%): Jukebox n=25 teach 68.8%, Briefing n=25 73.0%, shuffled 0%, empty 0%, neighbours >= 99.1%. NEWS task (Entertainment), base model, n=25, 8 draws: teach 84.7%, neighbours 97.4%, shuffled 0.1%.

## 9-rule table
| # | rule | status | note |
|---|---|---|---|
| 1 | size counts everything | PASS (9.942%) but GRAY | denominator is the fp16 size of a model that is itself ternary; whether a low-bit start counts is unruled (RULES.md section 5) |
| 2 | same means faithful | PASS on the exam: 100% overall (LB 99.86%), every kind 100% | base is a strong 96.7%; easy task |
| 3 | exam unseen | PASS with caveats: 2,744 items, kinds 344-420, 1,489 items never-used source; 532 exam items are tier `seen_tox`; clean-only fails the 300/kind count | shared exam built by another agent, sealed by hash |
| 4 | no new trained parts | GRAY | no head/adapter/side table/prompt, but feedback.pt is a fitted closed-form edit of two existing MLP tensors (rank <= 75) |
| 5 | feedback teaches | PASS on measured numbers: 82.7% on 164 unseen teach items incl. cross-source, <= 25 annotations, neighbours 98.3% | neighbour loss up to 3.7% in worst cell; one candidate/seed family, no best-of-N on the exam |
| 6 | controls | PASS: shuffled 0%, empty 0% | |
| 7 | real job | PASS: real generated skill word | |
| 8 | something to keep | PASS: base 96.7%, above chance, below ceiling (borderline high) | |
| 9 | spirit | GRAY | easy 7-way routing task; start is already ternary |

**Weakest rule: R1/R9 base identity (starting from an already-ternary release) and R4 (closed-form edit legality).** Neither is ruled by the maintainer.

## Honest caveats
- feedback.py wrapper untested end to end; answer.py untested with feedback.pt via the HF path (the same math was verified inside fb27s).
- The exam was scored once for base and once for the candidate, but the same candidate was chosen on the old probe-mix task and bytes, not on the exam. Feedback hyperparameters (layers 62,63, m=8) were chosen on a NEWS dev sweep, not on the exam.
- Feedback arms evaluated on the exam's teach/neighbour sets (as the plan prescribes) after config freeze.
- Cost: one g7e.2xlarge for roughly 10 h (about $3-4/h, estimate).
