# L2: Ternary-Bonsai-2-27B, our recompression at 9.94% + diffuse closed-form feedback (2026-11-08)

Task: SKILL7 voice-skill routing (shared locked exam `v2/exam`, 2,744 items, 7 kinds, real generated label word). New label for feedback: `Recipe` (not in the prompt menu). Start model = the released Ternary-Bonsai-2-27B (already ~1.58 bit). Candidate = our recompression `art_k16_f165k` (trits + coded scales + trimmed vocab, `v2/ternary27/`), unfolded to bf16 at load by `art_load.py`. Box: one g7e.2xlarge (RTX PRO 6000 96 GB). No weight training anywhere.

## Package (v2/lanes/L2/)
`answer.py` (answer command), `feedback.py` (feedback command; run end to end, see the update below), `fb27s.py`, `art_load.py`, (`ship/feedback.pt` removed 2026-11-09, see update at the end), `MANIFEST.json`, `results/`, runners `run_l2*.sh`, scorers `score_s7.py`, `score_exam.py`, `agg_fb.py`. The 5.3 GB artifact stays on the box (c27, not in repo).

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
| shuffled labels (see caveat: random OLD labels on the same Recipe inputs, weak control) | 12 / 25 | 0.0% / 0.0% | 99.0% / 98.6% | 96.0% / 96.3% |
Delta shipped as `feedback.pt`: reloading it from file reproduces the in-memory result exactly (82.3% teach, 98.6% neighbours, draw 0, n=25; `results/fb_Recipe_save.*`). The new label is generated end to end by the grown model (`recipe` appears as the model's own output).
Dev-only earlier runs with truly unknown labels on the 10.009% variant (`art_k16_n184k`, over 10%): Jukebox n=25 teach 68.8%, Briefing n=25 73.0%, shuffled 0%, empty 0%, neighbours >= 99.1%. NEWS task (Entertainment), base model, n=25, 8 draws: teach 84.7%, neighbours 97.4%, shuffled 0.1%.

## 9-rule table
| # | rule | status | note |
|---|---|---|---|
| 1 | size counts everything | PASS: 9.943% by bytes_audit on the exact shipped tree (incl. code, feedback.pt, cal/neigh data) | starting from a released pre-compressed model is allowed (RULES.md 4f); denominator = the start model's own 53.81 GB fp16 reference |
| 2 | same means faithful | PASS on the exam: 100% overall (LB 99.86%), every kind 100% | base is a strong 96.7%; easy task |
| 3 | exam unseen | PASS with caveats: 2,744 items, kinds 344-420, 1,489 items never-used source; 532 exam items are tier `seen_tox`; clean-only fails the 300/kind count | shared exam built by another agent, sealed by hash |
| 4 | no new trained parts | GRAY (weakest) | no head/adapter/side table/prompt, but feedback.pt is a fitted closed-form edit of two existing MLP tensors (rank <= 75) |
| 5 | feedback teaches | PASS on measured numbers: 82.7% on 164 unseen teach items incl. cross-source, <= 25 annotations, neighbours 98.3% | neighbour loss up to 3.7% in worst cell; one candidate/seed family, no best-of-N on the exam |
| 6 | controls | GRAY: empty 0% is clean; the 'shuffled' arm gives the same Recipe inputs random OLD labels, so the new word is never in the labels and 0% is nearly guaranteed (red team). A proper arm set (shuffle among annotation labels incl. the new word, wrong-label, unrelated inputs) is now run: see the 2026-11-09 update (the new word DOES appear on shuffled labels, teach 0.30-0.76, so controls are clearly lower than real but not 0) | |
| 7 | real job | PASS: real generated skill word | |
| 8 | something to keep | PASS: base 96.7%, above chance, below ceiling (borderline high) | |
| 9 | spirit | GRAY | easy 7-way routing task with a strong base (96.7%); compression of an already-ternary release |

**Weakest rules: R4 (legality of a closed-form fitted edit of existing tensors; unruled), R9 spirit (gray), and rule 3 on clean-only (Lists 223 / Transport 236 < 300).**

## Honest caveats
- feedback.py and answer.py were run end to end (`results/cli/`, see update; an earlier 'not run end to end' note was stale). Remaining untested: batch size/other prompts beyond the exam, a second new label through the CLIs.
- The exam was scored once for base and once for the candidate, but the same candidate was chosen on the old probe-mix task and bytes, not on the exam. Feedback hyperparameters (layers 62,63, m=8) were chosen on a NEWS dev sweep, not on the exam.
- Feedback arms evaluated on the exam's teach/neighbour sets (as the plan prescribes) after config freeze.
- Cost: one g7e.2xlarge for roughly 10 h (about $3-4/h, estimate).

## Update: shipped CLIs end to end, bytes audit, durable copy, lineage (2026-11-08, later)
- **CLIs run end to end** from the artifact dir through `answer.py` (art_load unfold, trimmed-vocab config remap from vocabmap.json) and `feedback.py` (25 annotations from `ann_pool`, seed 7, `data/ann25.jsonl`; writes `feedback.pt` 4.5 MB; shuffled-label control via `--shuffle-labels`; empty = no feedback.pt). `answer.py --batch` then answered the 164 teach + 379 neighbour items (`results/cli/`, `results/cli_summary.json`):

| arm | teach (Recipe, 164 unseen) | neighbours kept vs start model (357 base-correct) | worst neighbour kind |
|---|---|---|---|
| empty | 0.0% | 100% (LB 98.9%) | 100% |
| shuffled labels | 0.0% | 98.3% (LB 96.4%) | Lists 98.2% |
| real, 25 annotations | **87.2%** | **99.2%** (354/357, LB 97.6%) | Lists 99.1% |
  With the empty arm the CLI path reproduces the start model on all 357 neighbours, which also confirms the shipped unfold path matches the dev loader. (The earlier 82.7% in-memory figure is the mean of 3 draws; this is one draw through the CLIs.)
- **bytes_audit on the exact shipped tree:** 5,350,214,867 B = 9.9431% of 53,808,408,928 (PASS, slack 30.6 MB). Tree = artifact (13 files) + answer.py, feedback.py, fb27s.py, art_load.py + feedback.pt (4.5 MB) + data/exam/cal.jsonl and neigh.jsonl (needed by feedback.py). `results/cli/bytes_audit.txt`.
- **Durable copy of the 5.3 GB artifact:** a private object-store copy exists (`s3://<PRIVATE_BUCKET>/art_k16_f165k/`, bucket name and account id removed from the repo; the maintainer cannot reach it). Integrity list: `SHA256SUMS` in this folder (13 artifact files). Not uploaded anywhere public; publishing target (HF repo or release) is to be chosen by the owner and must reproduce these hashes. Feedback delta, code and data are in the repo.
- **Lineage (informational only, not a rule row):** the ancestor Qwen3.8-27B scores 96.90% on the exam (2,659 correct). Candidate retention against the ancestor's correct items: 99.32% (2641/2659, Wilson LB 98.93%), per kind min 96.1% (Calendar), never-used sources 99.59%; the ternary release itself retains the same 99.32% of the ancestor, so our recompression adds essentially no loss on top (`results/exam_vs_parent_*.json`). Neighbours kept vs ancestor after feedback: 99.2% (351/354).


## Update: red-team fixes (2026-11-08)
- Hardcoded absolute build-box paths replaced (`fb27s.py` uses `$HF_HOME`/`FBDIR` relative defaults; `run_l2*.sh` use `$WORK`); account id and bucket name removed from `MANIFEST.json`, this report and `results/cli/*` logs. `SHA256SUMS` lists the artifact files.
- `feedback.py` now takes the shared annotation format `{text,label}` (`v2/lanes/ANNOTATION_SCHEMA.md`) with `--artifact/--ann/--task`; legacy positional form kept. `answer.py` accepts `--task skill_task.json`.
- Known and NOT fixed here: shuffled arm is weak (see rule 6 row); `ship/feedback.pt` is a post-feedback state sitting in the tree (the clean artifact is the 13 files in `SHA256SUMS` without it); the covariance file `data/exam/cal.jsonl` is labelled; feedback.py also runs teach/neighbour evaluation (extra generation, edit unaffected).


## Update 2026-11-09: proper controls, o_proj carriers, second label, continual feedback (reporting only; start-model lineage is still MAYBE-VALID, RULES.md 4f: unruled, not a clean pass)
Setup: rented one RTX PRO 6000 (97 GB) box; **artifact rebuilt from the public release with `rebuild_artifact.sh` (art_build.py + keep list + hadamard_min.json) and all 13 files match `SHA256SUMS`** (artifact bytes are unchanged, so the sealed exam was NOT re-scored). Box destroyed. The earlier private object-store copy could not be refreshed (cloud login expired); the rebuild script is the durable route. No `/home/ubuntu` paths remain in the new code (`art_build.py`, `build_keep.py` use `$HF_HOME`). New code: `fb_l2x.py` (experiment harness + CLI backend), `mk_data.py`, `agg_x.py`, `score_cli2.py`, `jobs_*.json`; raw results in `results/x/*.jsonl` and `results/cli2/`. Kernel note: this box ran HF's torch fallback for the linear-attention layers, so single-item outputs can differ from the earlier box in rare cases (empty arm: 356/357 neighbours identical to the earlier start-model outputs).

**Selection was on DEV only** (Jukebox and Briefing unseen labels, 5-label menus; `results/x/sweep_dev1-4.jsonl`, 2-3 draws, n=25). Teach / neighbours kept (dev): down_proj L62-63 m8 (old default) 0.60 / 0.993; m32 0.80 / 0.986; 4-16 down layers 0.51-0.78 (more layers = more neighbour safety, lower teach); o_proj of 4 full-attention layers (63,59,55,51): m32 0.70, m48 0.87, **m64 0.91 / 0.994 (Briefing 0.92 / 0.997)**, m96 0.70, m128 0.90; 8 layers 0.72-0.78; 16 layers breaks the model (neighbours 0.08). `lam` has no effect. Frozen default: o_proj x4 layers, m=64, exact low-rank fp16 factors (fp16 rounding costs nothing measurable). L11's idea of column-restricted patches is not needed: the edit is exactly low rank (rank <= 25 + 5 per layer), so the factors are 0.68 MB, far below even the old 4.5 MB.

Locked Recipe sets (164 teach, 357 base-correct neighbours, 3-4 draws, real greedy generation, reporting only; "min cell" = worst source x kind cell with >= 15 items). New arms: `mreal` = 13 Recipe + 12 old-label annotations; `mshuf` = same inputs, labels permuted among them (Recipe label appears, on mostly wrong inputs); `mwrong` = Recipe inputs given random old labels; `unrel` = 25 old-skill inputs labelled Recipe; `real12` = 12 Recipe annotations.

| config | arm | Recipe teach | neighbours kept | min cell | Recipe word on neighbours |
|---|---|---|---|---|---|
| shipped before (down_proj L62-63, m8) | real25 | 0.841 (0.81-0.87) | 0.989 | 0.981 | 0 |
| | shuffled among labels | 0.295 (0.24-0.40) | 0.897 | 0.814 | 0.004 |
| | wrong-label | 0.000 | 0.991 | 0.981 | 0 |
| o_proj x4, m64 (new default) | empty | 0.000 | 0.997 (batch-noise floor) | 0.991 | 0 |
| | real25 (4 draws) | **0.973 (0.97-0.98)** | 0.971 | 0.952 | 0.0007 |
| | real12 | 0.954 | 0.976 | 0.963 | 0 |
| | mixed real (13 new + 12 old) | 0.951 | 0.979 | 0.964 | 0 |
| | **shuffled among labels** | **0.756 (0.71-0.84)** | 0.945 | 0.920 | 0.025 |
| | wrong-label | 0.000 | 0.973 | 0.956 | 0 |
| | Recipe on unrelated inputs | 0.003 | 0.884 | 0.813 | 0.109 |
| o_proj x4, m32 | real / shuffled / wrong | 0.888 / 0.423 / 0.000 | 0.987 / 0.985 / 0.988 | 0.979 | |
| o_proj x4, m16 | real / shuffled / wrong | 0.913 / 0.352 / 0.000 | 0.991 / 0.996 / 0.994 | 0.984 | |

Reading: (1) a proper shuffled arm does emit the new word (0.30-0.76), because the Recipe label is still an annotation target and the closed-form edit lifts that word near the menu; real is clearly higher (0.97 vs 0.76, 0.91 vs 0.35 at m16) and wrong-label is exactly 0, but the old claim "shuffled = 0%" was an artefact of never showing the word. The control gap shrinks as the margin grows. (2) Teach rose from 0.84 to 0.97 but neighbours fell from 0.989 to 0.971 (min cell 0.952); m16 is the better balanced setting on the locked set (0.913 teach, 0.991 neighbours, min cell 0.984, clearer control gap) but was NOT the dev-selected one and is listed only as sensitivity; `feedback.py --margin 16` selects it. (3) Unrelated-input labelling damages neighbours (0.88) as expected, so the edit is not a free global bias.

**CLI end to end** (`feedback.py` then `answer.py --batch` on the 543 teach+neighbour prompts; `results/cli2/`, `score_cli2.py`): empty 0.0% teach / 356 of 357 neighbours (batch noise); real (25 annotations, default m64) **teach 97.6% (160/164)**, neighbours 96.9% (346/357, Wilson LB 94.6%), worst kind Lists 96.6%; wrong-label 0.0% teach, neighbours 96.4%, worst kind Alarm 83% (random old labels hurt). `feedback.pt` = 678,311 B (ranks o:63 25, o:59 5).

**Second label, `Reminder`** (CLINC150 reminder + reminder_update, not mapped in the exam, not on the menu, close to Calendar/Alarm/Lists; ann from train, 100 teach from validation+test; `mk_data.py`; same 357 exam neighbours; I did not use "Timer" because the exam maps CLINC timer to Alarm, which would conflict with locked neighbours): real25 0.797 teach (0.73-0.85), neighbours 0.968, min cell 0.957; mixed real 0.690; shuffled 0.273; wrong-label 0.000; unrelated 0.013 (neighbours 0.982). Same pattern, lower teach.

**Continual 2-round feedback** (round 1 Recipe 25 annotations, then round 2 Reminder 25 annotations solved on the already edited model with fresh covariances; factors stacked, 1.3 MB total): after round 1 Recipe 0.978 / neighbours 0.967; after round 2 **Recipe 0.970 (no forgetting), Reminder 0.797 (0.74-0.90)**, neighbours 0.955 (min cell 0.933), Recipe-word leak on neighbours 0.003. With a replay of 12 round-1 Recipe annotations in round 2: Recipe 0.974, Reminder 0.713, neighbours 0.960. Neighbour loss accumulates over rounds (0.967 -> 0.955).

**Rule notes.** Rule 4: `ship/feedback.pt` was NOT a fitted head or side table: it was the low-rank factors (U, V) of the closed-form change to two existing `down_proj` tensors, produced by `feedback.py` from the 25 annotations. It was still a post-feedback state sitting in the shipped tree (and was gitignored), so it is **removed**; the shipped tree has no feedback file (empty feedback = the start model) and the feedback command writes `ARTIFACT/feedback.pt` (now 0.68 MB, counted in the audit). Closed-form fits of existing tensors stay GRAY (unruled). Bytes: artifact + code + `cal.jsonl` + 678 KB feedback = 5,346,244,844 B = 9.9357% of 53,808,408,928 (<= 9.99%, slack 34.6 MB; `results/cli2/bytes_audit.txt`). Rule 6 is now backed by real shuffled/wrong/unrelated arms (with the caveat above). Rule 5 teach is higher, neighbour retention lower (and m16 is the safer setting). The covariance file `data/exam/cal.jsonl` is still a labelled calibration set, and exam-based numbers above are reporting only (config frozen on dev, no exam use for selection except the m16/m32 sensitivity rows).
