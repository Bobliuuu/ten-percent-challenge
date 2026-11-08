# FEEDBACK_ABLATION: which closed-form edit of EXISTING tensors teaches a new label best (lane L11, 2026-11-08)

Code, configs, raw results: `v2/fb_ablation/` (`fblib.py` = all mechanisms, `run_abl.py` = ablation runner, `run_exam.py` = runner on the shared locked-exam splits, `cli/answer.py` + `cli/feedback.py` = end-to-end commands, `cfgs/`, `results/modal/*.json`, `analyze.py`, `gen_tables.py`; Modal apps `tenpct-fbabl`, `tenpct-fbcli`; run ids in `RUNS.md`). Dev data only (SKILL7-derived, hash buckets [0,70)) for the ablation; the final ranking (T7-T9) uses the shared exam splits `v2/exam` (ann_pool, teach, neighbour ids) with configs fixed BEFORE that run. No sealed/locked file of any lane was read, nothing posted. Companion docs: `FEEDBACK_EXPERIMENTS.md`, `FEEDBACK_LEGAL.md` (earlier single-mechanism work), `RULES.md` 4a-4d.

## 0. Protocol (identical for every mechanism)
- Model: Qwen2.5-7B-Instruct, full model, real forward pass (no cached tails), edited tensors stored back as bf16 (what a shipped bf16 body holds). Second arm: 4-bit RTN g128 body (all q/k/v/o/gate/up/down of the 28 layers). Third arm: Qwen3-14B (4-bit RTN body and bf16), the model family of the coordinator's 9.86% SKILL7 body.
- Task: SKILL7-derived routing, 5-skill menu (Alarm, Weather, Calendar, Lists, Transport). NEW label `Jukebox` = the held-out Music class (never in a prompt menu; the base cannot emit it, base teach 0.000; the label is 4 tokens `J|uke|box|<|im_end|>`). Second label `Briefing` (held-out News) for the continual test. Sources MASSIVE + CLINC150; buckets: calibration [0,40), annotation pool [40,55), teach [55,70) (96 unseen Music items, 46 massive + 50 clinc) plus 150 more unseen items from calibration-bucket Music (`teach2`), neighbours = 714 old-skill dev items of which 690 base-correct (cells source x skill, min 20 items).
- Feedback: n annotated REAL mistakes (n = 12 headline; 5 and 25 for the core methods) drawn from the annotation pool (every Music item is a mistake of the base because it cannot say Jukebox; base answers Lists/Alarm/None...), 10 draws (8 for Qwen3-14B and the exam runs, 6 or 4 for the slow Gauss-Newton family). Controls per draw: real, shuffled (same n labels `Jukebox` moved onto unrelated ordinary old-skill requests), wrong (real Music requests with a wrong old label), empty (= base, byte-identical, teach 0.000). Cross-source: annotations from MASSIVE only evaluated on the CLINC half of the teach set, and the reverse. Paraphrase: base-model paraphrases of the annotated items. Neighbour retention is on base-correct items (aggregate, worst draw, worst cell), plus agreement of the model's own greedy answers on 135 held-out calibration-format prompts (`cal agree`).
- Evaluation = teacher-forced greedy match of the whole label string (4 tokens incl. end-of-turn) in one forward pass; validated against REAL greedy generation (T6: agreement 0.98-1.00).
- All edits are closed form from the annotated sites (teach-forced corrected answer; sites = positions where the base argmax differs). Target: the final-residual change `dx = rms * (gap + m) * u/(u.u)`, `u = gamma*(w_new - w_competitor)` (margin m logits), split over the carrier layers sequentially (layer l delivers 1/(layers left) of the CURRENT remaining dx, recomputed through the real forward). The only gradient use is the Jacobian rows of the Gauss-Newton family (one linearised min-norm step, <= 3 iterations), no optimiser anywhere.
- Calibration (key covariance, never labels): 300 task-format prompts of the old skills + the model's own greedy answers (ships as text, counted in bytes).

## 1. Mechanisms tested (tag prefix -> what it is)
| tag | mechanism |
|---|---|
| `down_ridge` | MEMIT closed form on `down_proj`: dW = R K^T (lam C + K K^T)^-1, C = key covariance of calibration tokens (no answer-position protection) |
| `down_prot` | our whitened/protected version: C + awt * C_ans (key covariance at the old-label/EOS positions), Woodbury solve; `lam`, `awt`, margin `m` swept |
| `down_alpha` | AlphaEdit-style null-space projection: keys projected off the top-k eigenspace (k = 256/1024/4096) of the calibration key covariance before the ridge solve |
| `down_ultra` | UltraEdit-style: no covariance matrix, diagonal (running-rms) key normalisation only, ridge solve ("memory-free, lifelong") |
| `down_contrast` | rank-1 whitened diff-of-means (annotated keys vs calibration answer keys), LDA-like |
| `down_steer2` | Steer2Edit-style: steering direction from the targets distributed over the top-r contrast neurons (rank-1, r = 64/512) |
| `o_prot` / `o_alpha` | same solvers on attention `o_proj` (3584-d keys, writes into the residual) |
| `up_prot` / `gate_prot` | MLP up / gate edits: target through a damped min-norm pseudo-inverse of down_proj and the silu*up linearisation |
| `norm_post`, `norm_in`, `norm_final` | RMSNorm gain edits (post-attention norm of last k layers, input norm, final norm): Gauss-Newton one-step min-norm, metric = calibration Fisher diagonal |
| `vbias` | existing v_proj bias of the last 8 layers (a Steer2Edit-style global steering vector through the existing bias), Gauss-Newton |
| `embed_gn` | input-side embedding rows of the annotated request words, Gauss-Newton, metric = token frequency in calibration prompts |
| `lm_mnH` | lm_head row edit (the earlier mnH): UNSHIPPABLE reference (4a/4b caught it) |
| `L3/L5/L8/L12/mid8/early8/spread8/all28` | layer set of the carrier (last 3/5/8/12 layers, layers 10-17, 2-9, every third, all 28) |
| `m4/m8/m16/m32`, `autocal` | edit margin; `autocal` = the feedback command picks the smallest m (4,8,16,32,64) with annotated accuracy >= 0.95 and calibration agreement >= 0.985, using ONLY annotations + calibration prompts |
| `cols*` | column-restricted carrier: only the top-r input columns of each edited tensor (by key energy at the annotated sites) are rewritten (r columns x rows x 2 bytes) |
| `requant4` | q4 body: edited tensor re-rounded onto the SAME 4-bit grid with the original scales (0 extra bytes) vs `fp16ovr` (edited tensors kept bf16/fp16 = counted bytes) |

## 2. Headline findings (all numbers in the tables of section 5; bootstrap CIs over draws there)
1. **Real feedback teaches, controls do not, for the covariance-protected down_proj / o_proj family.** On Qwen2.5-7B (dev, n=12, 10 draws) the best selective carriers reach teach 0.77-0.89 on 96 unseen items with shuffled <= 0.01-0.02, wrong-label exactly 0.000, empty 0.000, neighbours kept 0.996-0.999 (every cell >= 0.98), paraphrases 0.93-0.99, cross-source 0.77-0.92. Real greedy generation agrees with the teacher-forced numbers (T6: 0.98-1.00). The new label is emitted through the unmodified answer path (T10, answer.py).
2. **Ranking on dev (teach - shuffled, then neighbours, then bytes), Qwen2.5-7B bf16 n=12:** `down_alpha_k256_L8` 0.893 (nb .996, 1.09 GB dense) > `down_prot_m32` 0.873 (nb .997) > `down_prot_cols4096_m32` 0.844 (nb .998, column-restricted, 178 MB at 99% energy / 235 MB all selected columns) > `down_ridge_L8_lam10` 0.834 > `o_alpha_L8` 0.801 (nb .997, 206 MB) > `o_prot_L8_m32` 0.773 (nb .998, 206 MB) > `lm_mnH` 0.745 (UNSHIPPABLE) > `down_prot_L8` m16 0.71 > `down_prot_cols1024_m32` 0.704 (nb .999, 43 MB at 99% energy / 59 MB all selected columns) > `o_prot_cols1536_m32` 0.536 (71 MB). The edit MARGIN m is the biggest lever of the whole ablation (down_prot at m = 4/8/16/32: 0.18/0.44/0.71/0.88, shuffled stays <= 0.01, neighbours 1.000 -> 0.997): our earlier default m=16 was under-tuned. The self-chosen margin (`autocal`: smallest m that fixes the annotated items and keeps calibration agreement) picks too small an m (0.66) because the annotated items are fixed long before generalisation saturates.
3. **Protection is what makes the edit selective.** Without a key-covariance (UltraEdit-style diagonal-only `down_ultra`) teach is 0.98-0.99 but shuffled is 0.79-0.81 and neighbours 0.96 (it teaches "say Jukebox" to anything); MEMIT with covariance (`down_ridge`) is selective (shuffled 0.00); answer-position protection (`awt` 10 -> 100) trades teach for nothing (0.71 -> 0.36, neighbours already ~1.0); `lam` 100 helps a little (0.75), `lam` 10000 kills teach (0.17). AlphaEdit null-space projection with a SMALL protected subspace (k=256) is the best solver (0.893); a larger protected space (k=4096) loses teach (0.74) for no neighbour gain (0.999).
4. **Rank-1 whitened contrast (`down_contrast`) and Steer2Edit-style sparse neuron steering (`down_steer2`) are weak (0.21-0.36, wide CIs)**: a rank-1 write cannot move an unseen label's multi-token string; the closed-form ridge over all annotated keys is needed.
5. **Carrier choice.** `down_proj` (MLP) > `o_proj` per layer-set at m=16 (0.71 vs 0.41), but `o_proj` at m=32 closes the gap (0.77) with 5x fewer bytes (206 MB vs 1.09 GB at 8 layers of a 7B) and the best neighbour retention (0.998-1.000). MLP `up_proj` 0.60 and `gate_proj` 0.03 (gate edit breaks the model: neighbours 0.22): the linearised up/gate targets are worse than editing the output projection directly. Layer set: last 3 / 5 / 8 / 12 layers of down_proj: 0.54 / 0.68 / 0.71 / 0.70 (saturates at 5-8); mid (10-17), early (2-9) and every-third layers give 0.00-0.30 (the label must be written close to the output). All 28 o_proj layers 0.49 vs last 8 0.41 (m16): more layers help o_proj a bit.
6. **Norm gains, the existing v bias and a global steering vector are NOT selective (disqualifying under rule 6)**: `norm_post_L8` teach 0.93 but shuffled 0.54 (dev) and 0.95-0.97 on the exam splits (real - shuffled = -0.07 to -0.11); `norm_in`, `norm_final`, `vbias_L8` (0.93 / shuffled 0.89), `norm_post_L3` the same. Only 0.006-0.06 MB, so they are the cheapest carrier, but they fail control (d). Same result on the 4-bit body and on Qwen3-14B (shuffled 0.7-0.83). Also they break the first feedback when a second label is added (T5: Jukebox 0.94 -> 0.21).
7. **Embedding-row (input side) Gauss-Newton edit: dead as implemented** (annotated items never reach the margin, teach 0.00, two settings). We did not debug it further (the row change is far below bf16 resolution of the embedding values and the 3-iteration linearisation does not cross a 20+ logit gap); do not use.
8. **Quantised body (4-bit RTN g128).** Edits solved on the quantised model's own hidden states work as well as on bf16 (T3, T7 q4: down_prot 0.67-0.86, o_prot_m32 0.91 with shuffled 0.013). BUT only if the edited tensors stay higher precision: re-rounding the edited tensor onto the SAME 4-bit grid with the original scales (`requant4`, 0 extra bytes) loses the edit completely (teach 0.00-0.08): the edit is smaller than a grid step. The carrier bytes are therefore real extra bytes on a quantised body (fp16/bf16 rewrite of the touched tensors: 206 MB per 8 o_proj layers of a 7B, 1.09 GB per 8 down_proj layers, 420 MB / 1.43 GB for Qwen3-14B); the only way to shrink them is the column-restricted carrier (finding 9).
9. **Column-restricted carrier (bytes).** Rewriting only the top-r input columns of each edited tensor (selected by key energy at the annotated sites; solve restricted to those columns): at m=32 down_proj cols 1024 / 4096 give 0.70 / 0.84 with shuffled 0.000 and neighbours 0.999 / 0.998 for 43 / 178 MB (7B, 8 layers, counted at 99% of the edit energy; storing every selected column costs r x 3584 x 2 B x 8 = 59 / 235 MB); at m=16 cols 2048/4096 give 0.51/0.63, cols 256 gives 0.00. o_proj cols 1536 at m=32 0.54 (71 MB), cols 512 0.00. On the 4-bit body cols4096 (m16) 0.55, cols1024 0.13. Unconstrained solutions are NOT sparse (70% of entries change after bf16 rounding; 99%-energy column count is ~75% of all columns: `MB sparse99` column), so sparsity must be imposed in the solve.
10. **Continual (feedback applied twice, T5, T9, T10).** For every protected solver the first feedback survives the second: first-round annotated items stay 1.000, Jukebox teach after the 2nd round unchanged (0.71 -> 0.71 for prot, 0.83 -> 0.84 alpha, 0.77 -> 0.77 lm_mnH), the second (new) label `Briefing` is taught at 0.69-0.85 with neighbours 0.995-1.000 (down_prot, alpha, lm_mnH). Same on the shared-exam splits (SKILL7: alpha 0.880 -> 0.880, annotated-1 kept 1.000). `down_ultra` second round leaves neighbours at 0.89; `norm_post` loses the first label (0.94 -> 0.21); `o_prot` m16 first-round annotated kept only 0.72-0.75 (under-fit). The CLI test (answer.py / feedback.py as separate processes, the second run starting from the already-edited shards and WITHOUT the in-memory key memory of the first run) also kept all six first-label probes after the 2nd feedback.
11. **End to end (T10).** `cli/feedback.py` (load 8 s + closed-form edit 11.5 s + shard rewrite = about 21-25 s wall on one A100 for 8 down_proj tensors; edits `model-0000X-of-00004.safetensors` in place, no new file) then `cli/answer.py` (plain generate with the 5-skill prompt): before feedback music requests answer `None/Lists`, after feedback 6/6 sampled music requests answer `Jukebox`, after the second feedback 4 of 6 sampled news requests answer `Briefing` (one hallucinated `Web`, one `Weather`) and all 6 music requests still answer `Jukebox`, non-related requests unchanged except one leak ("celeb news" -> Jukebox after the first round, corrected by the news feedback).
12. **Shared locked-exam splits (T7-T9; configs fixed before the run; neighbour retention on the 379 / 804 shared neighbour ids; teach on the 164 / 200 locked teach items, of which MTOP/HWU64 (SKILL7) and Bitext (BANK6) are sources never used for building).** SKILL7 `Recipe`, bf16: `o_prot_L8_m32` teach 0.831, shuffled 0.001, wrong 0.000, neighbours 0.999 (min cell 0.946); `o_alpha_L8_m32` 0.909 / 0.171 / 0.996; `down_alpha_k256_L5` 0.869 / 0.153 / 0.995; `down_alpha_k256_L8` 0.863 / 0.198 / 0.999; `lm_mnH` 0.788 / 0.123 / 1.000. Shuffled is higher here than on dev (0.15-0.37 for ridge/alpha/prot) because Recipe is lexically close to the Lists skill and the shuffled pool contains Lists-like requests: the clean separation of the dev run is not guaranteed, o_proj with the plain protected solver is the only carrier with shuffled ~0 on both. BANK6 `Security` is harder (annotation pool is 150 Banking77 items, all of them mistakes the base turns into Identity/Card; teach on Banking77 0.58-0.86, on never-used Bitext 0.18-0.59): best `o_alpha_L8_m32` 0.709 / 0.089 / neigh 0.991 and `down_alpha_k256_L5` 0.694 / 0.091 / 0.973 (q4: min cell 0.81); the norm-gain edit again has teach 0.88 / shuffled 0.95 and ruins neighbours (0.79 bf16, 0.64 q4). On the 4-bit body the same ranking holds (T7 q4: o_prot_m32 0.909 / 0.013 / 1.000; o_alpha 0.957 / 0.203 / 1.000).
13. **Qwen3-14B (T4, the family of the coordinator's 9.86% SKILL7 body), 4-bit RTN proxy body (RTN 4-bit is NOT the 9.86% GPTQ+ECSQ body: it is a numerics proxy only; no real 9.86% artifact was available here).** down_alpha k256 L8 fp16 override 0.776 / shuffled 0.001 / neighbours 0.993 (min cell 0.970); down_prot L8 0.595 (bf16 body 0.583: the quantised body costs nothing); o_prot L8 m16 0.314; norm gains again non-selective (shuffled 0.7-0.83); `requant4` 0.00. Dense carriers for this model: down_proj 8 layers 1.43 GB, o_proj 8 layers 420 MB.

## 3. Byte accounting against the 9.86% body and the recommended carriers
fp16 size of Qwen3-14B = about 29.5 GB, so 9.90% leaves 0.04% = about 12 MB over the 9.86% body (0.14% = about 41 MB up to 10.0%). Carrier bytes (edited tensors rewritten at 16 bit; indices negligible):
| carrier (Qwen3-14B, 8 layers) | bytes | selective (shuffled) | teach (7B dev, n=12) | fits 9.9% | fits 10.0% |
|---|---|---|---|---|---|
| lm_head rows (mnH) | 0.02-0.03 MB | yes | 0.77 | yes | yes (ILLEGAL by 4a/4b, the maintainer's own words) |
| RMSNorm gains / v bias | 0.01-0.08 MB | NO (0.5-1.0) | 0.93 | yes | yes (but fails rule 6) |
| down_proj cols 1024, m32 | 84 MB all selected columns (7B: 43-59 MB) | yes (0.000) | 0.70 | no | no |
| down_proj cols 4096, m32 | 335 MB (7B: 178-235 MB) | yes | 0.84 | no | no |
| o_proj dense L8 m32 | 420 MB (7B: 206 MB) | yes (0.001-0.013) | 0.77-0.83 (exam .83) | no | no |
| down_proj dense L8 (alpha k256) | 1.43 GB (7B: 1.09 GB) | yes on dev (0.00), 0.15-0.2 on exam | 0.89 | no | no |
So with a 9.86%-sized compressed body NO legal selective carrier of this ablation fits under 9.9% (the cheapest selective one, column-restricted down_proj at about 40 MB per 7B-sized model, needs 84 MB at Qwen3-14B width, i.e. a body of about 9.6% to stay <= 9.9%); bodies at <= 9.6% (or a body that is quantised slightly harder, e.g. 9.5%) can host `down_proj cols 1024 m32` or `o_proj` columns; the 4-bit carrier cannot be re-quantised onto the body's own grid (finding 8).

## 4. Recommended carrier per lane (guess; each = weakest reading named)
| lane | recommended carrier (this ablation) | why / bytes | weakest point |
|---|---|---|---|
| L1 Bonsai-8B (7.1% body, 1-bit) | column-restricted down_proj, last 8 layers, protected solver, m=32, cols 1024-4096 (fp16 patch about 67-270 MB at 8B width for 1024-4096 columns x 8 layers) | selective (shuffled 0.000), neighbours >= 0.998, fits (7.1% + 0.3-1.1%) | counted patch bytes; base-identity reading |
| L2 Ternary-Bonsai-27B (9.94% incl. 4.5 MB delta) | its own 4.5 MB column patch is the largest that fits; per this ablation cols < 1000 give < 0.25 teach at m16 (0.7 at m32 needs ~40-80 MB for a 27B) | no slack: 0.06% = 16 MB | teach will be partial; consider the margin m=32 before more bytes |
| L3 / L4 / L9 MoE (task-pruned experts, 2-bit) | attention `o_proj` of the last 8 layers (dense, protected solver, m=32, optionally column-restricted to ~1536 columns, 71 MB at 7B width) | the o_proj of a MoE is one dense tensor per layer; expert down_proj would be 100+ tensors; best neighbour retention (0.998-1.000), survives a 4-bit body when kept 16-bit | m=32 needed; o_proj m16 under-fits (0.41); 200 MB-class bytes on a 9.9% artifact |
| L5 dense 32B+ (scalar body 9.75-9.86%) | o_proj columns (cols 1536 m32) or accept the dense o_proj last 4-5 layers (about 130 MB at 32B-width) | 0.14% = 90 MB at 64 GB fp16 | selective but teach ~0.5-0.8; no room for the MLP carrier |
| L10 multimodal bf16/quantised text path | `down_proj` alpha (k=256) last 5 layers or `o_proj` m32 | 0.68 GB / 0.2 GB if the body leaves room | exam shuffled 0.15-0.2 for alpha; o_proj protected has shuffled ~0 |
| L6/L8/L12 other bases | same as L3 (o_proj m32); if bytes are free: `down_alpha_k256_L8` (best teach 0.86-0.89) | | |
| any lane, reference / NEVER shipped | lm_head rows (mnH) | 0.01 MB, teach 0.77-0.79 | caught by 4a/4b |
| any lane, do NOT use | norm gains, v bias, UltraEdit-style diagonal solve, up/gate, embedding rows, rank-1 contrast/steer2, mid/early layers, `requant4` | shuffled ~ real or teach 0 | rule 6 |

## 5. Legality read, mechanism by mechanism (RULES 4d tests: (a) existing tensors only, (b) new behaviour through the model's own diffuse forward pass, (c) new class generated end to end in real greedy generation, (d) real vs shuffled / wrong / empty clearly different; plus neighbours)
| mechanism | (a) | (b) | (c) | (d) | neighbours | verdict |
|---|---|---|---|---|---|---|
| down_proj protected (prot / ridge / alpha), last 5-8 layers | yes (bytes counted if the body is quantised) | yes, 5-8 layers (an associative memory in the MLP: the weakest reading) | yes (T6, T10) | dev yes (shuffled 0.00-0.01, wrong 0.000, empty 0.000); exam: shuffled 0.15-0.37 for ridge/alpha/prot m16 | 0.995-1.000 | GRAY (closed-form weight fitting unruled); best teach |
| o_proj protected (prot / alpha), m=32 | yes | yes (attention output of the last 8 layers: a routing-like change) | yes | yes (shuffled 0.001-0.013 on dev and exam, wrong 0.000-0.008) | 0.998-1.000 | GRAY; BEST CONTROLS and neighbours |
| column-restricted variants of the above | yes (fp16 patch columns of existing tensors) | yes | yes | yes (0.000) | >= 0.998 | GRAY plus "extra stored bytes only because of feedback" |
| MLP up / gate | yes | yes | yes | yes | gate breaks the model | weak / dead |
| RMSNorm gains, v bias | yes | not selective (a global per-channel scale) | yes | NO (shuffled 0.54-1.00) | 0.79-0.99 | FAILS rule 6 |
| embedding rows | yes | n/a | no (never flips the annotated items) | n/a | | dead |
| lm_head rows (mnH) | yes | NO (a 1-3 row readout = "the same head as before", 4b) | yes | yes | 0.997-1.000 | ILLEGAL by the maintainer's words; reference only |
| UltraEdit-style diagonal solve | yes | yes | yes | NO (shuffled 0.8) | 0.89-0.96 | fails (d) |
Plain statement: none of the selective mechanisms is a head, classifier, centroid, adapter, side file, prompt or lookup; all of them are closed-form fits of existing weights to <= 25 annotations (a covariance estimated from shipped calibration TEXT at feedback time; no stored statistics). Whether "no retraining of the weights" (4c) covers a closed-form edit is not ruled. If it is read as retraining, every row above is caught and no feedback mechanism survives.

## 6. Honest limits
- One base (Qwen2.5-7B; the Qwen3-14B arm is a 4-bit RTN proxy, plus one bf16 row), two tasks (SKILL7-derived with Jukebox/Briefing; SKILL7 Recipe and BANK6 Security on the shared exam splits). Teach is partial (0.7-0.9), not 1.0; BANK6 teach on the never-used Bitext source is only 0.18-0.59.
- The exam-split neighbour sets are used for ranking and not for choosing hyper-parameters (all configs were fixed from the dev ablation before that run), but they are real locked-exam rows: do not treat the exam-split numbers as an untouched score.
- `autocal` (self-chosen margin) was weaker than the best fixed margin (0.66 vs 0.88): the margin rule in `fblib.apply_any` is a heuristic and should be replaced by margin 32 plus a calibration-agreement guard.
- Not tested: attention q/k/v edits, the embedding edit beyond two settings, magnitude beyond m=32 (m=32 was still improving: 0.71 -> 0.88 from m=16, neighbours 1.000 -> 0.997), a 3-bit or GPTQ body, the real 9.86% Qwen3-14B body, cross-source teach for the News/Briefing label (single source), rows of Qwen3-14B for column carriers, more than 2 sequential feedbacks.
- Process notes: Modal GPU queue was saturated by other lanes, so the exam-split and column-carrier runs were run on Vast A100s (boxes destroyed, verified); runs A1b/F crashed once (CUDA OOM from an unpruned Cholesky cache; fixed, A1c reran the remainder) and F2 (Qwen3-14B bf16 rows other than down_prot) was cancelled for GPU availability. Raw JSONs are in `results/`.

## 7. Tables (auto-generated by `gen_tables.py` from `results/modal/*.json`)

### T1. Qwen2.5-7B-Instruct bf16 body, SKILL7-derived, new label Jukebox, n=12 annotations (ranked by teach - shuffled, then neighbours)

| model | method | draws | teach [CI] | shuffled | wrong | teach-shuf | neigh kept | worst draw | min cell | cal agree | paraphrase | teach2 | xsrc massive->clinc | xsrc clinc->massive | MB dense | MB sparse99 | s/edit |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen2.5-7B-Instruct | down_alpha_k256 | 10 | 0.893 [0.85,0.92] | 0.000 | 0.000 | 0.893 | 0.996 | 0.993 | 0.980 | 0.985 | 0.983 | 0.873 | 0.924 | 0.876 | 1086.325 | 826.554 | 1 |
| Qwen2.5-7B-Instruct | down_prot_m32 | 10 | 0.884 [0.85,0.91] | 0.011 | 0.000 | 0.873 | 0.997 | 0.996 | 0.977 | 0.995 | 0.983 | 0.870 | 0.902 | 0.854 | 1086.325 | 842.404 | 1 |
| Qwen2.5-7B-Instruct | down_prot_cols4096_L8_m32 | 8 | 0.844 [0.81,0.87] | 0.000 | 0.000 | 0.844 | 0.998 | 0.996 | 0.986 | 0.999 | 0.990 | 0.845 | 0.857 | 0.834 | 1086.325 | 177.526 | 1 |
| Qwen2.5-7B-Instruct | down_ridge_L8_lam10 | 10 | 0.834 [0.80,0.87] | 0.000 | 0.000 | 0.834 | 0.997 | 0.996 | 0.982 | 0.988 | 0.983 | 0.837 | 0.858 | 0.791 | 1086.325 | 866.337 | 1 |
| Qwen2.5-7B-Instruct | down_ridge_L8 | 10 | 0.832 [0.79,0.87] | 0.000 | 0.000 | 0.832 | 0.997 | 0.994 | 0.981 | 0.994 | 0.983 | 0.834 | 0.856 | 0.791 | 1086.325 | 856.961 | 1 |
| Qwen2.5-7B-Instruct | down_alpha_k1024 | 10 | 0.827 [0.79,0.86] | 0.000 | 0.000 | 0.827 | 0.996 | 0.991 | 0.984 | 0.993 | 0.975 | 0.829 | 0.840 | 0.765 | 1086.325 | 871.886 | 1 |
| Qwen2.5-7B-Instruct | o_alpha_L8 | 10 | 0.802 [0.77,0.84] | 0.001 | 0.000 | 0.801 | 0.997 | 0.996 | 0.987 | 0.996 | 0.967 | 0.813 | 0.794 | 0.680 | 205.521 | 173.448 | 1 |
| Qwen2.5-7B-Instruct | o_prot_L8_m32 | 10 | 0.773 [0.75,0.80] | 0.000 | 0.000 | 0.773 | 0.998 | 0.997 | 0.987 | 0.999 | 0.925 | 0.786 | 0.772 | 0.730 | 205.521 | 165.393 | 1 |
| Qwen2.5-7B-Instruct | down_prot_lam100 | 10 | 0.754 [0.72,0.78] | 0.002 | 0.000 | 0.752 | 0.998 | 0.996 | 0.994 | 0.999 | 0.975 | 0.767 | 0.700 | 0.687 | 1086.325 | 861.073 | 1 |
| Qwen2.5-7B-Instruct | lm_mnH | 10 | 0.766 [0.73,0.80] | 0.021 | 0.000 | 0.745 | 0.999 | 0.999 | 0.994 | 0.999 | 0.967 | 0.778 | 0.680 | 0.680 | 0.012 | 0.012 | 0 |
| Qwen2.5-7B-Instruct | down_alpha_k4096 | 10 | 0.742 [0.71,0.78] | 0.008 | 0.000 | 0.733 | 0.999 | 0.997 | 0.995 | 0.999 | 0.967 | 0.755 | 0.674 | 0.674 | 1086.325 | 936.367 | 4 |
| Qwen2.5-7B-Instruct | down_prot_L8 | 6 | 0.712 [0.66,0.75] | 0.000 | 0.000 | 0.712 | 1.000 | 0.999 | 0.998 | 1.000 | 0.958 | 0.721 | 0.630 | 0.641 | 1086.325 | 808.227 | 1 |
| Qwen2.5-7B-Instruct | down_prot_cols1024_L8_m32 | 8 | 0.704 [0.68,0.72] | 0.000 | 0.000 | 0.704 | 0.999 | 0.999 | 0.994 | 1.000 | 0.906 | 0.743 | 0.680 | 0.666 | 1086.325 | 42.798 | 1 |
| Qwen2.5-7B-Instruct | down_prot_L12 | 10 | 0.699 [0.67,0.72] | 0.000 | 0.000 | 0.699 | 0.999 | 0.997 | 0.994 | 1.000 | 0.950 | 0.721 | 0.600 | 0.622 | 1629.487 | 1252.783 | 3 |
| Qwen2.5-7B-Instruct | down_prot_L5 | 10 | 0.681 [0.64,0.72] | 0.000 | 0.000 | 0.681 | 1.000 | 0.999 | 0.998 | 1.000 | 0.867 | 0.715 | 0.590 | 0.635 | 678.953 | 482.692 | 1 |
| Qwen2.5-7B-Instruct | down_prot_autocal | 6 | 0.661 [0.52,0.75] | 0.012 | 0.000 | 0.649 | 1.000 | 0.999 | 0.998 | 1.000 | 0.847 | 0.672 | - | - | 1086.325 | 808.227 | 9 |
| Qwen2.5-7B-Instruct | down_prot_cols4096_L8 | 8 | 0.625 [0.59,0.65] | 0.000 | 0.000 | 0.625 | 1.000 | 1.000 | 1.000 | 1.000 | 0.875 | 0.676 | 0.453 | 0.568 | 1086.325 | 172.508 | 1 |
| Qwen2.5-7B-Instruct | up_prot_L8 | 10 | 0.603 [0.59,0.62] | 0.000 | 0.000 | 0.603 | 1.000 | 0.999 | 0.998 | 1.000 | 0.883 | 0.653 | 0.460 | 0.539 | 1086.325 | 949.561 | 1 |
| Qwen2.5-7B-Instruct | down_prot_L3 | 10 | 0.542 [0.52,0.56] | 0.000 | 0.000 | 0.542 | 1.000 | 0.997 | 0.998 | 1.000 | 0.583 | 0.561 | 0.376 | 0.491 | 407.372 | 263.547 | 1 |
| Qwen2.5-7B-Instruct | o_prot_cols1536_L8_m32 | 8 | 0.536 [0.51,0.56] | 0.000 | 0.000 | 0.536 | 1.000 | 0.999 | 0.998 | 1.000 | 0.781 | 0.575 | 0.455 | 0.435 | 205.521 | 70.955 | 1 |
| Qwen2.5-7B-Instruct | down_prot_cols2048_L8 | 8 | 0.512 [0.47,0.54] | 0.000 | 0.000 | 0.512 | 1.000 | 1.000 | 1.000 | 1.000 | 0.719 | 0.569 | 0.328 | 0.459 | 1086.325 | 84.986 | 1 |
| Qwen2.5-7B-Instruct | o_prot_all28 | 10 | 0.487 [0.46,0.52] | 0.000 | 0.000 | 0.487 | 1.000 | 0.999 | 0.999 | 0.999 | 0.767 | 0.530 | 0.388 | 0.359 | 719.323 | 555.108 | 3 |
| Qwen2.5-7B-Instruct | o_prot_L12 | 10 | 0.470 [0.44,0.49] | 0.000 | 0.000 | 0.470 | 1.000 | 0.999 | 0.999 | 1.000 | 0.742 | 0.516 | 0.382 | 0.348 | 308.281 | 241.814 | 1 |
| Qwen2.5-7B-Instruct | down_prot_m8 | 10 | 0.438 [0.40,0.47] | 0.000 | 0.000 | 0.438 | 1.000 | 0.999 | 0.999 | 1.000 | 0.575 | 0.493 | 0.274 | 0.409 | 1086.325 | 773.464 | 1 |
| Qwen2.5-7B-Instruct | o_prot_L8 | 10 | 0.411 [0.38,0.44] | 0.000 | 0.000 | 0.411 | 1.000 | 1.000 | 1.000 | 1.000 | 0.633 | 0.457 | 0.296 | 0.293 | 205.521 | 164.036 | 1 |
| Qwen2.5-7B-Instruct | norm_in_L8 | 6 | 0.764 [0.62,0.89] | 0.356 | 0.000 | 0.408 | 0.991 | 0.987 | 0.941 | 0.990 | 0.750 | 0.733 | 0.900 | 0.732 | 0.057 | 0.057 | 1 |
| Qwen2.5-7B-Instruct | norm_post_L8 | 6 | 0.931 [0.90,0.96] | 0.540 | 0.000 | 0.391 | 0.991 | 0.987 | 0.939 | 0.989 | 0.986 | 0.902 | 0.947 | 0.967 | 0.057 | 0.057 | 1 |
| Qwen2.5-7B-Instruct | down_steer2_r512 | 10 | 0.358 [0.11,0.61] | 0.000 | 0.000 | 0.358 | 0.997 | 0.994 | 0.976 | 0.997 | 0.358 | 0.336 | 0.202 | 0.578 | 1086.325 | 28.506 | 1 |
| Qwen2.5-7B-Instruct | down_prot_awt100 | 10 | 0.357 [0.33,0.39] | 0.000 | 0.000 | 0.357 | 1.000 | 1.000 | 1.000 | 1.000 | 0.508 | 0.407 | 0.204 | 0.324 | 1086.325 | 702.562 | 2 |
| Qwen2.5-7B-Instruct | norm_final | 6 | 0.510 [0.26,0.74] | 0.167 | 0.000 | 0.344 | 0.932 | 0.703 | 0.823 | 0.920 | 0.514 | 0.493 | 0.640 | 0.638 | 0.007 | 0.007 | 1 |
| Qwen2.5-7B-Instruct | down_prot_spread8 | 10 | 0.303 [0.26,0.35] | 0.000 | 0.000 | 0.303 | 0.999 | 0.999 | 0.993 | 0.997 | 0.450 | 0.325 | 0.242 | 0.259 | 1086.325 | 846.064 | 2 |
| Qwen2.5-7B-Instruct | down_steer2_r64 | 10 | 0.255 [0.08,0.51] | 0.000 | 0.000 | 0.255 | 0.998 | 0.996 | 0.978 | 0.999 | 0.275 | 0.247 | 0.082 | 0.513 | 1086.325 | 3.584 | 1 |
| Qwen2.5-7B-Instruct | down_prot_cols1024_L8 | 8 | 0.249 [0.21,0.28] | 0.000 | 0.000 | 0.249 | 1.000 | 1.000 | 1.000 | 1.000 | 0.344 | 0.280 | 0.113 | 0.253 | 1086.325 | 42.317 | 1 |
| Qwen2.5-7B-Instruct | norm_post_L3 | 6 | 0.977 [0.96,1.00] | 0.755 | 0.000 | 0.222 | 0.973 | 0.948 | 0.890 | 0.953 | 0.986 | 0.948 | 0.997 | 0.989 | 0.022 | 0.022 | 1 |
| Qwen2.5-7B-Instruct | down_contrast | 10 | 0.208 [0.07,0.42] | 0.000 | 0.000 | 0.208 | 1.000 | 0.999 | 0.996 | 1.000 | 0.258 | 0.216 | 0.058 | 0.400 | 1086.325 | 608.795 | 1 |
| Qwen2.5-7B-Instruct | down_ultra_lam100 | 10 | 0.990 [0.98,1.00] | 0.790 | 0.000 | 0.200 | 0.969 | 0.886 | 0.844 | 0.956 | 0.983 | 0.957 | 0.994 | 0.980 | 1086.325 | 769.006 | 1 |
| Qwen2.5-7B-Instruct | down_prot_m4 | 10 | 0.176 [0.15,0.21] | 0.000 | 0.000 | 0.176 | 1.000 | 0.997 | 0.998 | 1.000 | 0.108 | 0.195 | 0.080 | 0.185 | 1086.325 | 764.091 | 2 |
| Qwen2.5-7B-Instruct | down_ultra | 10 | 0.983 [0.96,1.00] | 0.807 | 0.000 | 0.176 | 0.963 | 0.910 | 0.841 | 0.948 | 0.983 | 0.950 | 0.994 | 0.972 | 1086.325 | 763.041 | 1 |
| Qwen2.5-7B-Instruct | down_prot_lam10000 | 10 | 0.171 [0.14,0.21] | 0.000 | 0.000 | 0.171 | 1.000 | 1.000 | 1.000 | 1.000 | 0.183 | 0.204 | 0.066 | 0.174 | 1086.325 | 701.826 | 1 |
| Qwen2.5-7B-Instruct | vbias_L8 | 6 | 0.934 [0.90,0.97] | 0.885 | 0.000 | 0.049 | 0.974 | 0.949 | 0.922 | 0.957 | 0.986 | 0.920 | 0.967 | 0.967 | 0.008 | 0.008 | 1 |
| Qwen2.5-7B-Instruct | gate_prot_L8 | 10 | 0.028 [0.00,0.08] | 0.000 | 0.000 | 0.028 | 0.224 | 0.000 | 0.110 | 0.211 | 0.042 | 0.028 | 0.000 | 0.000 | 1086.325 | 982.767 | 1 |
| Qwen2.5-7B-Instruct | down_prot_mid8 | 10 | 0.005 [0.00,0.01] | 0.002 | 0.000 | 0.003 | 0.998 | 0.996 | 0.991 | 0.998 | 0.008 | 0.003 | 0.008 | 0.004 | 1086.325 | 905.338 | 2 |
| Qwen2.5-7B-Instruct | down_prot_cols256_L8 | 8 | 0.000 [0.00,0.00] | 0.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1086.325 | 11.121 | 2 |
| Qwen2.5-7B-Instruct | o_prot_cols512_L8_m32 | 8 | 0.000 [0.00,0.00] | 0.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 | 205.521 | 23.228 | 1 |
| Qwen2.5-7B-Instruct | embed_gn | 4 | 0.000 [0.00,0.00] | 0.000 | 0.000 | 0.000 | 0.997 | 0.996 | 0.990 | 0.996 | 0.000 | 0.000 | 0.000 | 0.000 | 0.387 | 0.387 | 3 |
| Qwen2.5-7B-Instruct | down_prot_early8 | 10 | 0.000 [0.00,0.00] | 0.000 | 0.000 | 0.000 | 0.995 | 0.993 | 0.984 | 0.990 | 0.000 | 0.000 | 0.000 | 0.000 | 1086.325 | 769.168 | 2 |
| Qwen2.5-7B-Instruct | embed_gn_lam5 | 4 | 0.000 [0.00,0.00] | 0.000 | 0.000 | 0.000 | 0.991 | 0.977 | 0.977 | 0.993 | 0.000 | 0.000 | 0.000 | 0.000 | 0.387 | 0.387 | 5 |
| Qwen2.5-7B-Instruct | norm_post_L8_m16 | 6 | 0.990 [0.98,1.00] | 1.000 | 0.000 | -0.010 | 0.970 | 0.957 | 0.895 | 0.956 | 0.986 | 0.962 | 0.997 | 0.978 | 0.057 | 0.057 | 1 |
| Qwen2.5-7B-Instruct | norm_post_all28 | 4 | 0.318 [0.00,0.70] | 0.669 | 0.000 | -0.352 | 0.979 | 0.967 | 0.927 | 0.965 | 0.396 | 0.301 | 0.240 | 0.000 | 0.201 | 0.201 | 3 |


### T2. n sweep (5 / 25 annotations) for the core methods

| method | n | teach | shuffled | neigh | min cell | MB dense |
|---|---|---|---|---|---|---|
| down_prot_L8 | 5 | 0.531 | 0.000 | 1.000 | 1.000 | 1086.325 |
| down_prot_L8 | 25 | 0.835 | 0.007 | 0.998 | 0.987 | 1086.325 |


### T3. 4-bit RTN g128 body (Qwen2.5-7B-Instruct), edited tensors as fp16 override vs re-quantised onto the same 4-bit grid (0 extra bytes)

| model | method | draws | teach [CI] | shuffled | wrong | teach-shuf | neigh kept | worst draw | min cell | cal agree | paraphrase | teach2 | xsrc massive->clinc | xsrc clinc->massive | MB dense | MB sparse99 | s/edit |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen2.5-7B-Instruct-q4 | q4_lm_mnH | 10 | 0.771 [0.75,0.79] | 0.055 | 0.000 | 0.716 | 1.000 | 0.999 | 0.996 | 0.999 | 0.933 | 0.772 | 0.648 | 0.728 | 0.014 | 0.014 | 0 |
| Qwen2.5-7B-Instruct-q4 | q4_norm_post_L8 | 10 | 0.944 [0.91,0.97] | 0.248 | 0.000 | 0.696 | 0.992 | 0.983 | 0.955 | 0.990 | 0.942 | 0.911 | 0.940 | 0.937 | 0.057 | 0.057 | 1 |
| Qwen2.5-7B-Instruct-q4 | q4_down_prot_L8_fp16ovr | 10 | 0.673 [0.63,0.71] | 0.000 | 0.000 | 0.673 | 0.999 | 0.999 | 0.991 | 1.000 | 0.817 | 0.684 | 0.514 | 0.600 | 1086.325 | 853.105 | 2 |
| Qwen2.5-7B-Instruct-q4 | q4_down_prot_cols4096_L8 | 8 | 0.553 [0.50,0.60] | 0.000 | 0.000 | 0.553 | 0.999 | 0.999 | 0.987 | 1.000 | 0.729 | 0.584 | 0.347 | 0.503 | 1086.325 | 184.758 | 1 |
| Qwen2.5-7B-Instruct-q4 | q4_down_prot_cols2048_L8 | 8 | 0.393 [0.34,0.44] | 0.000 | 0.000 | 0.393 | 0.999 | 0.999 | 0.989 | 1.000 | 0.552 | 0.422 | 0.205 | 0.389 | 1086.325 | 90.054 | 1 |
| Qwen2.5-7B-Instruct-q4 | q4_down_prot_cols1024_L8 | 8 | 0.134 [0.11,0.17] | 0.000 | 0.000 | 0.134 | 0.999 | 0.999 | 0.991 | 1.000 | 0.177 | 0.151 | 0.060 | 0.179 | 1086.325 | 44.878 | 1 |
| Qwen2.5-7B-Instruct-q4 | q4_down_alpha_requant4 | 10 | 0.084 [0.02,0.17] | 0.000 | 0.000 | 0.084 | 0.995 | 0.984 | 0.984 | 0.997 | 0.042 | 0.068 | 0.018 | 0.011 | 1086.325 | 21.165 | 2 |
| Qwen2.5-7B-Instruct-q4 | q4_down_prot_cols256_L8 | 8 | 0.000 [0.00,0.00] | 0.000 | 0.000 | 0.000 | 1.000 | 0.999 | 0.998 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1086.325 | 11.461 | 2 |
| Qwen2.5-7B-Instruct-q4 | q4_o_prot_L8_requant4 | 10 | 0.000 [0.00,0.00] | 0.000 | 0.000 | 0.000 | 1.000 | 0.999 | 0.996 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 | 205.521 | 40.144 | 1 |
| Qwen2.5-7B-Instruct-q4 | q4_down_prot_L8_requant4 | 10 | 0.000 [0.00,0.00] | 0.000 | 0.000 | 0.000 | 0.999 | 0.999 | 0.992 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1086.325 | 16.916 | 1 |


### T4. Qwen3-14B (E = 4-bit RTN body, F = bf16 body)

| model | method | draws | teach [CI] | shuffled | wrong | teach-shuf | neigh kept | worst draw | min cell | cal agree | paraphrase | teach2 | xsrc massive->clinc | xsrc clinc->massive | MB dense | MB sparse99 | s/edit |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen3-14B-q4 | q4_down_alpha_L8_fp16ovr | 8 | 0.776 [0.74,0.81] | 0.001 | 0.000 | 0.775 | 0.993 | 0.988 | 0.970 | 0.991 | 0.979 | 0.767 | 0.843 | 0.652 | 1426.063 | 1028.945 | 2 |
| Qwen3-14B-q4 | q4_lm_mnH | 8 | 0.694 [0.66,0.73] | 0.010 | 0.000 | 0.684 | 0.999 | 0.997 | 0.994 | 0.993 | 0.938 | 0.704 | 0.762 | 0.579 | 0.020 | 0.020 | 1 |
| Qwen3-14B-q4 | q4_down_prot_L8_fp16ovr | 8 | 0.595 [0.55,0.64] | 0.000 | 0.000 | 0.595 | 0.996 | 0.991 | 0.966 | 0.992 | 0.885 | 0.608 | 0.685 | 0.505 | 1426.063 | 955.994 | 2 |
| Qwen3-14B | bf_down_prot_L8_fp16ovr | 8 | 0.583 [0.54,0.64] | 0.000 | 0.000 | 0.583 | 0.998 | 0.994 | 0.983 | 1.000 | 0.844 | 0.615 | 0.640 | 0.497 | 1426.063 | 979.081 | 2 |
| Qwen3-14B-q4 | q4_o_prot_L8 | 8 | 0.314 [0.27,0.36] | 0.000 | 0.000 | 0.314 | 0.998 | 0.997 | 0.983 | 0.993 | 0.448 | 0.330 | 0.270 | 0.231 | 419.430 | 297.971 | 2 |
| Qwen3-14B-q4 | q4_norm_post_L3 | 6 | 0.920 [0.90,0.94] | 0.705 | 0.000 | 0.215 | 0.944 | 0.845 | 0.772 | 0.940 | 0.958 | 0.916 | 0.963 | 0.837 | 0.031 | 0.031 | 1 |
| Qwen3-14B-q4 | q4_norm_post_L8 | 6 | 0.925 [0.91,0.94] | 0.825 | 0.000 | 0.101 | 0.980 | 0.935 | 0.903 | 0.974 | 0.986 | 0.922 | 0.993 | 0.848 | 0.082 | 0.082 | 2 |
| Qwen3-14B-q4 | q4_norm_final | 6 | 0.903 [0.88,0.92] | 0.826 | 0.000 | 0.076 | 0.924 | 0.812 | 0.772 | 0.935 | 0.944 | 0.918 | 0.920 | 0.841 | 0.010 | 0.010 | 1 |
| Qwen3-14B-q4 | q4_down_prot_L8_requant4 | 8 | 0.000 [0.00,0.00] | 0.000 | 0.000 | 0.000 | 0.998 | 0.996 | 0.976 | 0.989 | 0.000 | 0.000 | 0.000 | 0.000 | 1426.063 | 23.851 | 1 |


### T5. Continual: feedback applied twice (5 draws x 12 annotations each)

| method | 2nd feedback | after 1st: Jukebox teach | 1st-round annotated kept | after 2nd: Jukebox teach | 1st-round annotated kept | 2nd label (or same) annotated | Briefing teach (new label) | neighbours kept |
|---|---|---|---|---|---|---|---|---|
| down_alpha_k1024 | new label Briefing | 0.827 | 1.000 | 0.840 | 1.000 | 0.917 | 0.714 | 0.995 |
| down_alpha_k1024 | more Jukebox | 0.827 | 1.000 | 0.900 | 1.000 | 1.000 | 0.000 | 0.995 |
| down_prot_L8 | new label Briefing | 0.704 | 1.000 | 0.708 | 1.000 | 0.967 | 0.686 | 1.000 |
| down_prot_L8 | more Jukebox | 0.704 | 1.000 | 0.835 | 1.000 | 1.000 | 0.000 | 0.998 |
| down_ultra | new label Briefing | 0.998 | 1.000 | 0.992 | 1.000 | 0.917 | 0.848 | 0.888 |
| down_ultra | more Jukebox | 0.998 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.932 |
| lm_mnH | new label Briefing | 0.765 | 1.000 | 0.765 | 1.000 | 1.000 | 0.741 | 0.999 |
| lm_mnH | more Jukebox | 0.765 | 1.000 | 0.902 | 1.000 | 1.000 | 0.000 | 0.997 |
| norm_post_L8 | new label Briefing | 0.944 | 1.000 | 0.212 | 0.233 | 1.000 | 0.897 | 0.927 |
| norm_post_L8 | more Jukebox | 0.944 | 1.000 | 0.992 | 1.000 | 1.000 | 0.000 | 0.985 |
| o_prot_L8 | new label Briefing | 0.400 | 0.717 | 0.425 | 0.750 | 0.733 | 0.548 | 1.000 |
| o_prot_L8 | more Jukebox | 0.400 | 0.717 | 0.742 | 0.900 | 0.883 | 0.000 | 0.999 |


### T6. Real greedy generation vs teacher-forced evaluation (2 draws x 96-100 items)

| run | method | set | teacher-forced acc | real greedy generation acc | agreement |
|---|---|---|---|---|---|
| A2 | down_alpha_k1024 | teach | 0.823 | 0.823 | 1.000 |
| A2 | down_alpha_k1024 | neigh | 0.990 | 0.990 | 1.000 |
| A2 | down_alpha_k1024 | teach | 0.927 | 0.927 | 1.000 |
| A2 | down_alpha_k1024 | neigh | 0.990 | 0.990 | 1.000 |
| B1 | o_prot_L8 | teach | 0.427 | 0.448 | 0.979 |
| B1 | o_prot_L8 | neigh | 1.000 | 1.000 | 1.000 |
| B1 | o_prot_L8 | teach | 0.448 | 0.448 | 1.000 |
| B1 | o_prot_L8 | neigh | 1.000 | 1.000 | 1.000 |
| B2 | norm_post_L8 | teach | 0.979 | 0.979 | 1.000 |
| B2 | norm_post_L8 | neigh | 0.990 | 0.990 | 1.000 |
| B2 | norm_post_L8 | teach | 1.000 | 1.000 | 1.000 |
| B2 | norm_post_L8 | neigh | 0.990 | 0.990 | 1.000 |
| C | q4_down_prot_L8_fp16ovr | teach | 0.760 | 0.750 | 0.990 |
| C | q4_down_prot_L8_fp16ovr | neigh | 0.990 | 0.990 | 1.000 |
| C | q4_down_prot_L8_fp16ovr | teach | 0.729 | 0.719 | 0.990 |
| C | q4_down_prot_L8_fp16ovr | neigh | 0.990 | 0.990 | 1.000 |

### T7. SHARED LOCKED-EXAM SPLITS: SKILL7 new label Recipe, Qwen2.5-7B-Instruct bf16

base teach 0.000; neighbour set 379 exam items (356 base-correct); annotations from ann_pool only ({'None': 159, 'Lists': 134, 'Weather': 2, 'News': 2, 'Alarm': 2, 'Music': 1}). 'xsrc' columns are not used here (annotations are tuned-on sources, teach includes never-used sources; per-source teach below).

| model | method | draws | teach [CI] | shuffled | wrong | teach-shuf | neigh kept | worst draw | min cell | cal agree | paraphrase | teach2 | xsrc massive->clinc | xsrc clinc->massive | MB dense | MB sparse99 | s/edit |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen2.5-7B-Instruct | o_prot_L8_m32 | 8 | 0.831 [0.81,0.85] | 0.001 | 0.000 | 0.830 | 0.999 | 0.997 | 0.946 | 1.000 | - | - | - | - | 205.521 | 164.213 | 1 |
| Qwen2.5-7B-Instruct | o_alpha_L8_m32 | 8 | 0.909 [0.90,0.93] | 0.171 | 0.000 | 0.739 | 0.996 | 0.992 | 0.893 | 0.999 | - | - | - | - | 205.521 | 168.705 | 1 |
| Qwen2.5-7B-Instruct | down_alpha_k256_L5 | 8 | 0.869 [0.84,0.90] | 0.153 | 0.000 | 0.716 | 0.995 | 0.992 | 0.911 | 0.999 | - | - | - | - | 678.953 | 486.973 | 1 |
| Qwen2.5-7B-Instruct | lm_mnH | 8 | 0.788 [0.76,0.82] | 0.123 | 0.000 | 0.665 | 1.000 | 1.000 | 1.000 | 1.000 | - | - | - | - | 0.010 | 0.010 | 0 |
| Qwen2.5-7B-Instruct | down_alpha_k256_L8 | 8 | 0.863 [0.83,0.89] | 0.198 | 0.000 | 0.665 | 0.999 | 0.997 | 0.929 | 0.999 | - | - | - | - | 1086.325 | 795.348 | 2 |
| Qwen2.5-7B-Instruct | down_ridge_L8_lam10 | 8 | 0.840 [0.81,0.87] | 0.329 | 0.000 | 0.511 | 0.998 | 0.994 | 0.964 | 0.999 | - | - | - | - | 1086.325 | 837.257 | 2 |
| Qwen2.5-7B-Instruct | down_prot_L8 | 8 | 0.777 [0.75,0.81] | 0.373 | 0.000 | 0.405 | 1.000 | 0.997 | 0.982 | 1.000 | - | - | - | - | 1086.325 | 794.318 | 1 |
| Qwen2.5-7B-Instruct | norm_post_L8 | 8 | 0.834 [0.80,0.88] | 0.947 | 0.000 | -0.114 | 0.988 | 0.980 | 0.893 | 0.999 | - | - | - | - | 0.057 | 0.057 | 1 |


### T8. per-source teach (SKILL7 bf16)

| method | teach clinc150 | teach hwu64 | teach massive | teach mtop |
|---|---|---|---|---|
| down_alpha_k256_L5 | 0.905 | 0.725 | 0.643 | 0.891 |
| down_alpha_k256_L8 | 0.908 | 0.742 | 0.625 | 0.879 |
| down_prot_L8 | 0.848 | 0.708 | 0.589 | 0.771 |
| down_ridge_L8_lam10 | 0.899 | 0.742 | 0.625 | 0.845 |
| lm_mnH | 0.851 | 0.725 | 0.571 | 0.786 |
| norm_post_L8 | 0.923 | 0.758 | 0.696 | 0.817 |
| o_alpha_L8_m32 | 0.938 | 0.775 | 0.768 | 0.927 |
| o_prot_L8_m32 | 0.902 | 0.742 | 0.625 | 0.829 |


### T9. continual on shared exam splits (SKILL7 bf16)

| method | teach after 1st | annotated-1 kept after 2nd | teach after 2nd | neigh after 2nd |
|---|---|---|---|---|
| down_alpha_k256_L8 | 0.880 | 1.000 | 0.880 | 0.998 |
| down_ridge_L8_lam10 | 0.867 | 1.000 | 0.863 | 0.999 |


### T7. SHARED LOCKED-EXAM SPLITS: SKILL7 new label Recipe, Qwen2.5-7B-Instruct q4

base teach 0.000; neighbour set 379 exam items (350 base-correct); annotations from ann_pool only ({'None': 181, 'Lists': 77, 'Weather': 2, 'News': 37, 'Recipe': 1, 'Alarm': 2}). 'xsrc' columns are not used here (annotations are tuned-on sources, teach includes never-used sources; per-source teach below).

| model | method | draws | teach [CI] | shuffled | wrong | teach-shuf | neigh kept | worst draw | min cell | cal agree | paraphrase | teach2 | xsrc massive->clinc | xsrc clinc->massive | MB dense | MB sparse99 | s/edit |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen2.5-7B-Instruct-q4 | o_prot_L8_m32 | 8 | 0.909 [0.90,0.92] | 0.013 | 0.008 | 0.896 | 1.000 | 1.000 | 1.000 | 1.000 | - | - | - | - | 205.521 | 166.487 | 1 |
| Qwen2.5-7B-Instruct-q4 | o_alpha_L8_m32 | 8 | 0.957 [0.95,0.96] | 0.203 | 0.000 | 0.755 | 1.000 | 0.997 | 0.999 | 1.000 | - | - | - | - | 205.521 | 170.966 | 1 |
| Qwen2.5-7B-Instruct-q4 | down_alpha_k256_L8 | 8 | 0.918 [0.91,0.92] | 0.239 | 0.002 | 0.679 | 1.000 | 1.000 | 1.000 | 1.000 | - | - | - | - | 1086.325 | 820.411 | 2 |
| Qwen2.5-7B-Instruct-q4 | lm_mnH | 8 | 0.862 [0.85,0.88] | 0.183 | 0.000 | 0.679 | 1.000 | 1.000 | 1.000 | 1.000 | - | - | - | - | 0.008 | 0.008 | 0 |
| Qwen2.5-7B-Instruct-q4 | down_alpha_k256_L5 | 8 | 0.914 [0.91,0.92] | 0.288 | 0.001 | 0.626 | 1.000 | 1.000 | 1.000 | 1.000 | - | - | - | - | 678.953 | 493.645 | 1 |
| Qwen2.5-7B-Instruct-q4 | down_ridge_L8_lam10 | 8 | 0.916 [0.91,0.93] | 0.361 | 0.002 | 0.556 | 1.000 | 1.000 | 1.000 | 1.000 | - | - | - | - | 1086.325 | 861.485 | 2 |
| Qwen2.5-7B-Instruct-q4 | down_prot_L8 | 8 | 0.858 [0.85,0.87] | 0.415 | 0.003 | 0.444 | 1.000 | 1.000 | 1.000 | 1.000 | - | - | - | - | 1086.325 | 811.137 | 1 |
| Qwen2.5-7B-Instruct-q4 | norm_post_L8 | 8 | 0.890 [0.86,0.91] | 0.971 | 0.014 | -0.081 | 0.997 | 0.986 | 0.996 | 0.994 | - | - | - | - | 0.057 | 0.057 | 1 |


### T8. per-source teach (SKILL7 q4)

| method | teach clinc150 | teach hwu64 | teach massive | teach mtop |
|---|---|---|---|---|
| down_alpha_k256_L5 | 0.926 | 0.875 | 0.750 | 0.926 |
| down_alpha_k256_L8 | 0.920 | 0.875 | 0.732 | 0.936 |
| down_prot_L8 | 0.920 | 0.867 | 0.714 | 0.841 |
| down_ridge_L8_lam10 | 0.929 | 0.867 | 0.732 | 0.931 |
| lm_mnH | 0.905 | 0.858 | 0.643 | 0.860 |
| norm_post_L8 | 0.935 | 0.867 | 0.911 | 0.874 |
| o_alpha_L8_m32 | 0.940 | 0.933 | 1.000 | 0.965 |
| o_prot_L8_m32 | 0.929 | 0.892 | 0.768 | 0.912 |


### T7. SHARED LOCKED-EXAM SPLITS: BANK6 new label Security, Qwen2.5-7B-Instruct bf16

base teach 0.000; neighbour set 804 exam items (707 base-correct); annotations from ann_pool only ({'Card': 24, 'Identity': 123, 'Cash': 3}). 'xsrc' columns are not used here (annotations are tuned-on sources, teach includes never-used sources; per-source teach below).

| model | method | draws | teach [CI] | shuffled | wrong | teach-shuf | neigh kept | worst draw | min cell | cal agree | paraphrase | teach2 | xsrc massive->clinc | xsrc clinc->massive | MB dense | MB sparse99 | s/edit |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen2.5-7B-Instruct | o_alpha_L8_m32 | 8 | 0.709 [0.65,0.76] | 0.089 | 0.000 | 0.620 | 0.991 | 0.987 | 0.984 | 0.991 | - | - | - | - | 205.521 | 165.341 | 1 |
| Qwen2.5-7B-Instruct | down_alpha_k256_L5 | 8 | 0.694 [0.63,0.76] | 0.091 | 0.000 | 0.603 | 0.973 | 0.917 | 0.944 | 0.978 | - | - | - | - | 678.953 | 463.420 | 1 |
| Qwen2.5-7B-Instruct | down_alpha_k256_L8 | 8 | 0.676 [0.61,0.73] | 0.103 | 0.000 | 0.573 | 0.986 | 0.973 | 0.975 | 0.991 | - | - | - | - | 1086.325 | 755.634 | 2 |
| Qwen2.5-7B-Instruct | down_ridge_L8_lam10 | 8 | 0.582 [0.52,0.64] | 0.057 | 0.000 | 0.525 | 0.992 | 0.990 | 0.987 | 0.998 | - | - | - | - | 1086.325 | 810.912 | 2 |
| Qwen2.5-7B-Instruct | lm_mnH | 8 | 0.438 [0.35,0.52] | 0.009 | 0.000 | 0.429 | 0.997 | 0.996 | 0.993 | 0.999 | - | - | - | - | 0.012 | 0.012 | 1 |
| Qwen2.5-7B-Instruct | down_prot_L8 | 8 | 0.394 [0.34,0.44] | 0.013 | 0.000 | 0.382 | 0.998 | 0.997 | 0.997 | 0.999 | - | - | - | - | 1086.325 | 781.725 | 1 |
| Qwen2.5-7B-Instruct | o_prot_L8_m32 | 8 | 0.381 [0.33,0.43] | 0.000 | 0.000 | 0.381 | 0.999 | 0.997 | 0.998 | 1.000 | - | - | - | - | 205.521 | 157.402 | 1 |
| Qwen2.5-7B-Instruct | norm_post_L8 | 8 | 0.878 [0.80,0.96] | 0.953 | 0.000 | -0.074 | 0.794 | 0.539 | 0.752 | 0.928 | - | - | - | - | 0.057 | 0.057 | 2 |


### T8. per-source teach (BANK6 bf16)

| method | teach banking77 | teach bitext_bank |
|---|---|---|
| down_alpha_k256_L5 | 0.859 | 0.529 |
| down_alpha_k256_L8 | 0.844 | 0.507 |
| down_prot_L8 | 0.607 | 0.181 |
| down_ridge_L8_lam10 | 0.775 | 0.389 |
| lm_mnH | 0.634 | 0.243 |
| norm_post_L8 | 0.930 | 0.826 |
| o_alpha_L8_m32 | 0.833 | 0.585 |
| o_prot_L8_m32 | 0.581 | 0.180 |


### T9. continual on shared exam splits (BANK6 bf16)

| method | teach after 1st | annotated-1 kept after 2nd | teach after 2nd | neigh after 2nd |
|---|---|---|---|---|
| down_alpha_k256_L8 | 0.627 | 1.000 | 0.689 | 0.973 |
| down_ridge_L8_lam10 | 0.477 | 1.000 | 0.570 | 0.989 |


### T7. SHARED LOCKED-EXAM SPLITS: BANK6 new label Security, Qwen2.5-7B-Instruct q4

base teach 0.000; neighbour set 804 exam items (713 base-correct); annotations from ann_pool only ({'Card': 36, 'Identity': 112, 'Cash': 2}). 'xsrc' columns are not used here (annotations are tuned-on sources, teach includes never-used sources; per-source teach below).

| model | method | draws | teach [CI] | shuffled | wrong | teach-shuf | neigh kept | worst draw | min cell | cal agree | paraphrase | teach2 | xsrc massive->clinc | xsrc clinc->massive | MB dense | MB sparse99 | s/edit |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen2.5-7B-Instruct-q4 | down_alpha_k256_L5 | 8 | 0.738 [0.64,0.82] | 0.073 | 0.000 | 0.665 | 0.954 | 0.874 | 0.812 | 0.973 | - | - | - | - | 678.953 | 462.698 | 2 |
| Qwen2.5-7B-Instruct-q4 | down_alpha_k256_L8 | 8 | 0.722 [0.64,0.79] | 0.121 | 0.000 | 0.601 | 0.965 | 0.923 | 0.938 | 0.983 | - | - | - | - | 1086.325 | 756.028 | 3 |
| Qwen2.5-7B-Instruct-q4 | o_alpha_L8_m32 | 8 | 0.728 [0.68,0.77] | 0.139 | 0.000 | 0.589 | 0.979 | 0.965 | 0.966 | 0.990 | - | - | - | - | 205.521 | 159.616 | 1 |
| Qwen2.5-7B-Instruct-q4 | down_ridge_L8_lam10 | 8 | 0.619 [0.53,0.70] | 0.079 | 0.000 | 0.540 | 0.980 | 0.959 | 0.972 | 0.996 | - | - | - | - | 1086.325 | 806.213 | 3 |
| Qwen2.5-7B-Instruct-q4 | lm_mnH | 8 | 0.481 [0.40,0.57] | 0.036 | 0.000 | 0.446 | 0.997 | 0.996 | 0.994 | 1.000 | - | - | - | - | 0.007 | 0.007 | 1 |
| Qwen2.5-7B-Instruct-q4 | down_prot_L8 | 8 | 0.451 [0.37,0.53] | 0.029 | 0.000 | 0.422 | 0.994 | 0.992 | 0.986 | 1.000 | - | - | - | - | 1086.325 | 782.361 | 1 |
| Qwen2.5-7B-Instruct-q4 | o_prot_L8_m32 | 8 | 0.397 [0.32,0.45] | 0.000 | 0.002 | 0.397 | 0.997 | 0.996 | 0.994 | 1.000 | - | - | - | - | 205.521 | 152.531 | 1 |
| Qwen2.5-7B-Instruct-q4 | norm_post_L8 | 8 | 0.931 [0.87,0.98] | 0.978 | 0.000 | -0.047 | 0.637 | 0.261 | 0.466 | 0.862 | - | - | - | - | 0.057 | 0.057 | 2 |


### T8. per-source teach (BANK6 q4)

| method | teach banking77 | teach bitext_bank |
|---|---|---|
| down_alpha_k256_L5 | 0.865 | 0.611 |
| down_alpha_k256_L8 | 0.854 | 0.590 |
| down_prot_L8 | 0.587 | 0.314 |
| down_ridge_L8_lam10 | 0.779 | 0.460 |
| lm_mnH | 0.640 | 0.323 |
| norm_post_L8 | 0.973 | 0.890 |
| o_alpha_L8_m32 | 0.866 | 0.589 |
| o_prot_L8_m32 | 0.581 | 0.212 |


### T10. End-to-end CLI test (answer.py / feedback.py, safetensors shards rewritten in place, feedback applied twice)

- `python cli/answer.py --model /tmp/model --text "play so...` 11.6 s rc=0
```
'play some music for me' -> None
'commence playing music' -> None
'start playing some music' -> None
'play my post-rock playlist' -> Lists
'play the song called guerrilla radio' -> None
'can you play me that, "baby shark, doo doo," song' -> Lists
'go to nytimes dot com' -> Lists
'read me the newest article on the new solar system' -> None
'tell me c. n. n. news for me right now' -> Weather
'celeb news' -> None
'notify me with news updates on nigeria' -> Weather
'can you give me a flash briefing' -> Weather
'are there any meetings for today on my calendar' -> Calendar
'please put cleaning on my list of things to do' -> Lists
'what is the weather like in toronto' -> Weather
'i want to set a timer' -> Alarm
'tell me when ninety minutes have passed' -> Alarm
"what's the distance between home and the closest starbucks" -> Transport

```
- `python cli/feedback.py --model /tmp/model --annotations...` 24.8 s rc=0
```
label tokens {'Alarm': ['Alarm', '<|im_end|>'], 'Weather': ['Weather', '<|im_end|>'], 'Calendar': ['Calendar', '<|im_end|>'], 'Lists': ['Lists', '<|im_end|>'], 'Transport': ['Transport', '<|im_end|>'], 'Jukebox': ['J', 'uke', 'box', '<|im_end|>'], 'Briefing': ['Brief', 'ing', '<|im_end|>']}
cal answers [('Calendar', 95), ('Lists', 71), ('Transport', 51), ('Alarm', 41), ('Weather', 41), ('None', 1)]
{"nsites": 14, "load_and_calibration_s": 8.0, "edit_s": 11.5, "tensors": ["model.layers.20.mlp.down_proj.weight", "model.layers.21.mlp.down_proj.weight", "model.layers.22.mlp.down_proj.weight", "model.layers.23.mlp.down_proj.weight", "model.layers.24.mlp.down_proj.weight", "model.layers.25.mlp.down_proj.weight", "model.layers.26.mlp.down_proj.weight", "model.layers.27.mlp.down_proj.weight"]}
wrote {'model-00003-of-00004.safetensors': 2, 'model-00004-of-00004.safetensors': 6} total_s 21.6

```
- `python cli/answer.py --model /tmp/model --text "play so...` 9.7 s rc=0
```
'play some music for me' -> Jukebox
'commence playing music' -> Jukebox
'start playing some music' -> Jukebox
'play my post-rock playlist' -> Jukebox
'play the song called guerrilla radio' -> Jukebox
'can you play me that, "baby shark, doo doo," song' -> Jukebox
'go to nytimes dot com' -> Web
'read me the newest article on the new solar system' -> Weather
'tell me c. n. n. news for me right now' -> Weather
'celeb news' -> Jukebox
'notify me with news updates on nigeria' -> Weather
'can you give me a flash briefing' -> Weather
'are there any meetings for today on my calendar' -> Calendar
'please put cleaning on my list of things to do' -> Lists
'what is the weather like in toronto' -> Weather
'i want to set a timer' -> Alarm
'tell me when ninety minutes have passed' -> Alarm
"what's the distance between home and the closest starbucks" -> Transport

```
- `python cli/feedback.py --model /tmp/model --annotations...` 24.2 s rc=0
```
label tokens {'Alarm': ['Alarm', '<|im_end|>'], 'Weather': ['Weather', '<|im_end|>'], 'Calendar': ['Calendar', '<|im_end|>'], 'Lists': ['Lists', '<|im_end|>'], 'Transport': ['Transport', '<|im_end|>'], 'Jukebox': ['J', 'uke', 'box', '<|im_end|>'], 'Briefing': ['Brief', 'ing', '<|im_end|>']}
cal answers [('Calendar', 98), ('Lists', 69), ('Transport', 52), ('Alarm', 41), ('Weather', 39), ('Jukebox', 1)]
{"nsites": 25, "load_and_calibration_s": 7.5, "edit_s": 11.5, "tensors": ["model.layers.20.mlp.down_proj.weight", "model.layers.21.mlp.down_proj.weight", "model.layers.22.mlp.down_proj.weight", "model.layers.23.mlp.down_proj.weight", "model.layers.24.mlp.down_proj.weight", "model.layers.25.mlp.down_proj.weight", "model.layers.26.mlp.down_proj.weight", "model.layers.27.mlp.down_proj.weight"]}
wrote {'model-00003-of-00004.safetensors': 2, 'model-00004-of-00004.safetensors': 6} total_s 21.0

```
- `python cli/answer.py --model /tmp/model --text "play so...` 9.9 s rc=0
```
'play some music for me' -> Jukebox
'commence playing music' -> Jukebox
'start playing some music' -> Jukebox
'play my post-rock playlist' -> Jukebox
'play the song called guerrilla radio' -> Jukebox
'can you play me that, "baby shark, doo doo," song' -> Jukebox
'go to nytimes dot com' -> Web
'read me the newest article on the new solar system' -> Briefing
'tell me c. n. n. news for me right now' -> Briefing
'celeb news' -> Briefing
'notify me with news updates on nigeria' -> Briefing
'can you give me a flash briefing' -> Weather
'are there any meetings for today on my calendar' -> Calendar
'please put cleaning on my list of things to do' -> Lists
'what is the weather like in toronto' -> Weather
'i want to set a timer' -> Alarm
'tell me when ninety minutes have passed' -> Alarm
"what's the distance between home and the closest starbucks" -> Transport

```

