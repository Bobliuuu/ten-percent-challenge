# L9b: Qwen3.5-35B-A3B, REAL packed artifact, SKILL7 (2026-11-08)

Start model Qwen/Qwen3.5-35B-A3B (35.95B params, 256 experts/layer, top-8, 40 MoE layers; hybrid Gated-DeltaNet + attention). Recipe (no training, no gradient): routing-mass expert prune (keep 102 of 256 per layer, chosen on the unlabeled SKILL7 calibration prompts `v2/exam` cal_big) + grouped RTN 3-bit (g128, asym, fp16 scale, 3-bit zero) on the kept experts, 6-bit attention/linear-attention/shared-expert, 4-bit embedding and lm_head, bf16 router rows (kept experts only) and small tensors. The vision tower is not shipped (text-only; see rule 1). Code: `qpack.py` (format), `core.py` (pack/load/eval), `runjob.py`+`modal_l9b.py`/`vast_up.py` (runners), `answer.py`, `feedback.py`+`fbcore.py`. Config chosen on the SEL split only (102 vs 90 experts); the exam was scored once per draw artifact, config not re-picked.

## 1. Real artifact bytes (v2/tools/bytes_audit.py, every file)
| item | bytes |
|---|---|
| body.bin (packed ints + scales + zeros + raw tensors) | 6,685,765,376 |
| tokenizer.json + config + template + code + manifest | 20,133,949 (code here = core.py, qpack.py, answer.py; `fbcore.py`, `feedback.py`, `annot.py`, `skill_task.json`, `feedback_cal.jsonl` are counted in the feedback line below, see MANIFEST) |
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


## Update: red-team fixes (2026-11-08)
- Shipped `feedback_cal.jsonl` (the 300 calibration rows used by the measured feedback runs = first 300 rows of `data/skill7_cal.jsonl`, 101,819 B, disjoint from the exam by exact text) and added `fbcore.py`, `feedback.py`, `annot.py`, `skill_task.json`, `feedback_cal.jsonl` to `MANIFEST.json` byte accounting (total with feedback and code, see MANIFEST `total_with_feedback_and_code`).
- `feedback.py` takes the shared annotation format `{text,label}` (`v2/lanes/ANNOTATION_SCHEMA.md`); `answer.py` takes `--task`. Old positional form kept.
- The artifact itself is still not in the repo (destroyed box). `MANIFEST.json` lists per-file sha256 for everything that is in the repo; `body.bin` sha256 is NOT available (needs a rebuild on a GPU box; hash must then be recorded and, if it differs from 6,685,765,376 B or the measured outputs, the exam re-run). Calibration rows are labelled (`answer` column, used for the answer-token covariance); an unlabeled variant would be cleaner and has not been measured.


## 8. Rebuild, shuffled-control rework, BANK6 side experiment (2026-11-08, later)
**Rebuild.** `body.bin` re-packed on Modal (2xA100-80GB, `job_rebuild.json`, mass file `results/mass_skill7.pt`): **6,685,765,376 B, sha256 `e0a1266db5eb7c4f877dfaeb107bf41baa5b5240ff945eb4d42a52f400efe88b`** (also in MANIFEST.json, per-file hashes in `results/rebuild_pack.json`). The rebuilt `manifest.json` (kept expert ids, every tensor offset and size) is byte-identical to the original box's (`results/manifest_s7_k102_e3.json`, sha256 f4b1eb7c...459a). No hash of the original body ever existed, so byte identity of the body content is argued (identical layout, deterministic RTN on the same bf16 weights and kept ids), not proven. Behavioural check: exam-neighbour retention with no feedback is 98.3% (n=348) before and after; the feedback numbers shift slightly (see below), so GPU-batching noise or a small body difference cannot be separated. Because bytes/layout are identical, the sealed SKILL7 exam was NOT re-scored. Durable copy: private Modal volume `tenpct-l9b`, path `l9b/art_s7_k102_e3/` (no public hosting; no S3 copy was made).
**Feedback controls, 8 draws on the locked teach (164) and neighbour (348) sets, reporting only** (`results/fb_locked8.json`; draws 0-2 reuse the earlier annotation seeds; config chosen on dev data only: annotations from pool[:200], dev-teach = never-annotated pool[200:], dev neighbours = SEL; dev sweeps `results/fb_dev1..5.json`):
| carrier | real teach (mean, min-max) | shuffled teach (mean, min-max) | wrong | empty | neighbours real |
|---|---|---|---|---|---|
| old: shared down_proj L10, 512 cols, m16, lam1000, awt10 (shipped) | 0.739 (0.60-0.82) | 0.177 (0.01-0.37) | 0.000 | 0.0 | 98.0 (97.1-98.9) |
| new: same carrier, m32, lam2000, awt30 (dev pick) | 0.758 (0.64-0.83) | 0.160 (0.00-0.40) | 0.000 | 0.0 | 98.2 (97.7-98.6) |
Result: the weak shuffled control is NOT fixed. Over 8 draws both configs have shuffled at 0.00-0.40 (old draws 0-2 re-measured here .29/.01/.37 vs .11/.02/.47 in section 3; same real teach within 0.03), so shuffled leakage is a draw-dependent property, mean about 0.17, with a real-minus-shuffled gap of about 0.58-0.60 on average. State it that way.
What did not transfer from L11 (dev, 2 draws unless noted): attention-output carrier on this hybrid. Editing `linear_attn.out_proj` (Gated-DeltaNet layers) with m32 over 8-16 layers destroys the model (neighbours 0.0-0.76); with lam >= 1e4 nothing happens (teach 0). True `o_proj` of the 10 full-attention layers works but is no better: 4 layers dense m16 teach .77 / shuffled .17-.19; 10 layers cols 1536 m16 .80 / .11-.14 (63 MB, 3x the shared carrier); m32 leaks (shuffled .4-.7); lam2e3 awt30 m32 o_proj gives teach 0. Margin is not a free lever here: on the shared carrier m24/m32/m48 raise shuffled together with teach (default lam: .55-.90). The only knob that separated them was stronger protection: lam2e3 awt30 m32 on dev (4 new draws) teach .83-.88, shuffled 0-.15, but on the locked sets shuffled came back at up to .40. Dev-best not locked: lam2e3 awt50 m32 (dev teach .65-.75, shuffled .00-.01, 4 draws) and cols 256 (teach .55-.67, shuffled 0); untested on locked. Shipped defaults (`fbcore.py`) are unchanged, so MANIFEST hashes of code are unchanged.
**BANK6 side experiment** (`bank6q/RESULTS.md`, fake-quant on SEL only, locked BANK6 exam not touched): Hadamard + error-feedback (GPTQ) quantisation of the kept experts clears the 98/95 gates on SEL at <=9.9% modelled bytes where RTN failed: vq4_had 2-bit k160 99.68% [98.20] at 9.66% (worst kind Exchange 97.6), vq4 k128 99.36% at 8.2%, gptq_had 3-bit k102 99.36%, ecsq_had 2-bit 98.71%; RTN k102 re-simulated 96.78% (Identity 87.5). Most of the gain is Hadamard+error feedback, not 2-bit-with-more-experts; differences between good variants are 1-4 of 311 items. No real packer/loader exists and bytes are modelled; BANK6 stays not presentable until one is built and the exam is scored once by its owner.
