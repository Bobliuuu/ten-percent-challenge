# L2: Ternary-Bonsai-2-27B, our recompression at 9.94% + diffuse closed-form feedback (2026-11-08)

Task: SKILL7 voice-skill routing (shared locked exam `v2/exam`, 2,744 items, 7 kinds, real generated label word). New label for feedback: `Recipe` (not in the prompt menu). Start model = the released Ternary-Bonsai-2-27B (already ~1.58 bit). Candidate = our recompression `art_k16_f165k` (trits + coded scales + trimmed vocab, `v2/ternary27/`), unfolded to bf16 at load by `art_load.py`. Box: one g7e.2xlarge (RTX PRO 6000 96 GB). No weight training anywhere.

## Package (v2/lanes/L2/)
`answer.py` (answer command), `feedback.py` (feedback command; run end to end, see the update below), `fb27s.py`, `art_load.py`, `ship/feedback.pt` (real feedback delta, 4.5 MB low-rank U,V of the down_proj edit), `MANIFEST.json`, `results/`, runners `run_l2*.sh`, scorers `score_s7.py`, `score_exam.py`, `agg_fb.py`. The 5.3 GB artifact stays on the box (c27, not in repo).

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
| 6 | controls | GRAY: empty 0% is clean; the 'shuffled' arm gives the same Recipe inputs random OLD labels, so the new word is never in the labels and 0% is nearly guaranteed (red team). A proper arm (Recipe on unrelated inputs + wrong-label) is not run here | |
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
