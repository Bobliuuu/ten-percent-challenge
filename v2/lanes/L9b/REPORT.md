# L9b: Qwen3.5-35B-A3B, REAL packed artifact, SKILL7 (2026-11-08)

Start model Qwen/Qwen3.5-35B-A3B (35.95B params, 256 experts/layer, top-8, 40 MoE layers; hybrid Gated-DeltaNet + attention). Recipe (no training, no gradient): routing-mass expert prune (keep 102 of 256 per layer, chosen on the unlabeled SKILL7 calibration prompts `v2/exam` cal_big) + grouped RTN 3-bit (g128, asym, fp16 scale, 3-bit zero) on the kept experts, 6-bit attention/linear-attention/shared-expert, 4-bit embedding and lm_head, bf16 router rows (kept experts only) and small tensors. The vision tower is not shipped (text-only; see rule 1). Code: `qpack.py` (format), `core.py` (pack/load/eval), `runjob.py`+`modal_l9b.py`/`vast_up.py` (runners), `answer.py`, `feedback.py`+`fbcore.py`. Config chosen on the SEL split only (102 vs 90 experts); the exam was scored once per draw artifact, config not re-picked.

## 1. Real artifact bytes (v2/tools/bytes_audit.py, every file)
| item | bytes |
|---|---|
| body.bin (packed ints + scales + zeros + raw tensors) | 6,685,765,376 |
| tokenizer.json + config + template + code + manifest | 20,133,949 |
| **TOTAL shipped** | **6,705,899,325** |
| fp16 reference, text-only (2 x 34.66B, the model the pipeline loads) | 69,321,221,376 -> **9.6737%** (PASS cap 9.9%, slack 156.9 MB) |
| fp16 reference, full checkpoint incl. vision tower (2 x 35.95B) | 71,903,645,408 -> 9.326% |
| feedback carrier (L10 dense, 10 layers x 2.1 MB) | 20,981,760 -> total with feedback 9.704% |
The 90-expert variant (`s7_k90_e3`) is 6,109,683,653 B = 8.81% (slack 753 MB), 99.24% [98.71] on SEL.

## 2. Retention on the shared LOCKED exam (v2/exam, SKILL7, 2744 items, scored once per artifact; real greedy generation, exact label)
Base accuracy on the exam 0.9676 (2655 base-correct). Retention = P(correct | base correct).
| draw (quantizer/calibration seed) | retention [Wilson lo] | worst kind | worst source x kind cell |
|---|---|---|---|
| d0 (all calibration chunks) | **99.55% [99.21]** | Lists 98.43 | massive/Lists 97.1 (n=104) |
| d1 (random half of chunks) | 99.47% [99.12] | Lists 98.12 | hwu64/Lists 96.9 |
| d2 (other random half) | 99.55% [99.21] | Lists 98.75 | hwu64/Alarm 97.8 |
d0 per kind: Alarm 99.3, Calendar 99.7, Lists 98.4, Music 99.8, News 99.8, Transport 100, Weather 99.8. Per source: clinc150 100, hwu64 99.2, massive 99.4, mtop 99.6. Never-used sources (HWU64+MTOP, 1448 base-correct): 99.45% [98.91]. clean-only tier (excluding the flagged seen_tox rows): 99.53% [99.14]. Gates: overall >=98 YES, every kind >=95 YES, rule-3 counts YES (loader.report). Files: `results/exam_report_skill7.json`, `results/outs_*_EXAM.json`.
SEL (selection split, 1802 items): k102 99.48% [99.01]; k90 99.24% [98.71]; draw variance on exam 0.08 pt.

## 3. Feedback (new skill `Recipe`, never in the prompt menu; <=25 annotations = 12 Recipe requests + 13 real old-skill errors of the packed model; teach set = 164 locked unseen Recipe items)
Mechanism: closed-form diffuse edit of the dense shared-expert down_proj of the last 10 MoE layers (keys = sigmoid(shared_expert_gate) x input), carrier = replaced bf16 columns in `fb.bin` (20.98 MB dense). Runs on the packed artifact; `feedback.py ART ann.jsonl` writes it, `answer.py` applies it. End-to-end check on the packed artifact (25 Recipe annotations): "how do i bake chocolate chip cookies" -> Recipe, "what is the best way to cook a turkey recipe" -> Recipe, alarm/music/weather requests unchanged (`results/e2e_feedback_answer.log`).
| condition (3 annotation draws, L10 dense carrier) | teach Recipe emitted | neighbours still right (n=348 exam neighbours) |
|---|---|---|
| empty (no feedback) | 0.0 | 98.28% |
| real | 0.762 / 0.793 / 0.726 | 98.56 / 97.99 / 98.28 |
| shuffled labels | 0.110 / 0.018 / 0.470 | 98.3 / 97.4 / 97.4 |
| wrong labels | 0.0 / 0.0 / 0.0 | - |
Weaker carriers: 10 layers x 128 columns (5.2 MB) teach 0.38-0.43; 20 layers x 256 columns teach 0.63-0.68. Min neighbour cell 95.8-97.9. Honest gap: shuffled draw 2 reaches 0.47 (the edit also generalises the bias towards the new word), so "clearly worse" is strong in 2 of 3 draws; leakage of Recipe on neighbours 0.0.

## 4. 9-rule table (SKILL7)
| # | rule | status |
|---|---|---|
| 1 | size counts everything | PASS 9.674% (text-only reference; 9.33% vs full checkpoint). Reading: the vision tower is dropped (text task) |
| 2 | same means faithful | PASS 99.55 [99.21] overall, every kind >=98.4 |
| 3 | exam | PASS counts (2744, kinds 344-420, 1448 from never-used sources); flagged tier `seen_tox` 532 rows; clean-only 99.53 |
| 4 | no new trained parts | prune+RTN are data-dependent but not fitted heads; feedback = closed-form edit of existing weights, stored as replacement columns (gray: carrier file) |
| 5 | feedback teaches | teach 0.73-0.79 on 164 locked items, neighbours kept (98.0-98.6 vs 98.3 empty) |
| 6 | controls | empty 0, wrong 0, shuffled 0.02-0.47 (one draw weak) |
| 7 | real job | PASS: generated skill word |
| 8 | base has something to keep | base 0.9676 (chance 0.14), 3.2% errors: close to ceiling (weak) |
| 9 | spirit | gray: 60% of experts pruned for the task (task specialisation), expert choice uses unlabeled task prompts |
Weakest: rule 6 variance, rule 8 headroom, rule 9/4 grayness of the expert selection and the carrier.

## 5. BANK6 (same recipe, `b6_*`, calibration = bank6 cal 698 rows)
SEL: k102 3-bit 97.43% [95.01] (worst Identity 89.3), k90 96.78, k76 4-bit 96.14, k128 2-bit 94.53, k160 2-bit 95.82. Locked exam BANK6 (2237, scored once, k102): base .902, **retention 97.37% [96.58]**, Card 94.4, Identity 91.7, others >=99 -> FAILS the 98/95 gates (near miss). Feedback `Security`: teach 0.13-0.52 and neighbours damaged (Card/Identity overlap): 76.2/96.1/85.4% vs 93.9% empty. Not presentable.

## 6. Hardware to run answer.py
One GPU with >=40 GB (loader dequantises the packed body into a 31 GB bf16 text model; KV small for short prompts); tested on a 141 GB H200 (load of packed body 6 s after the file is in page cache, 8 new tokens per request). A100/H100-80GB comfortable; 48 GB cards (L40S/A6000) possible; not CPU-practical. A fused 3-bit kernel was NOT written (dequantise-then-bf16), the slow PyTorch fallback of the Gated-DeltaNet layers was used (no flash-linear-attention installed) and throughput was not benchmarked.

## 7. Provenance
Runs: Vast H200 (one instance, destroyed and verified) via `vast_up.py` + queue runner; pack job `results/logs/run_job_pack2.log`; eval/fb JSON in `results/`. Routing-mass files `results/mass_*.pt` regenerate any artifact in minutes (pack is 8 s once the model is loaded). The artifact itself (6.7 GB) is not in git (not enough disk locally); it must be re-packed. Shared exam hashes: v2/exam/MANIFEST.json; my earlier private 2209-item exam (kinds < 300) was superseded and not scored.
