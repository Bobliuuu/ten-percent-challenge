# L1: Bonsai-8B (1-bit lossless repack) + column-sparse diffuse feedback. Task: SKILL7 (primary), NEWS (secondary)

Status 2026-11-08, DEV data only (no sealed/locked file read or scored, nothing posted). Real greedy generation everywhere (exact label string), 10 draws. Code: `v2/bonsai/ship/` (copied here), experiments `v2/bonsai/fbx/`, raw results `results/` here and `v2/bonsai/fbx/results/`.

## What ships
- Weights: the 1-bit Bonsai-8B-unpacked pack (bits.bin + scales.bin + small.pt + tokenizer/config/chat_template/LICENSE), lossless (round trip bit-exact, gate 1 earlier: 2000/2000 identical greedy sequences vs the reference fp16 model).
- `answer.py` (+ `bonsai_loader.py`, `common.py`, `skill_task.json`): plain HF generate, greedy, free-text label, no constrained decoding. It emits any word the model generates, so a label outside the original menu (`Recipe`) comes out of the system (see e2e below).
- `feedback.py`: <= 25 annotated items `{text, label}`. Closed-form edit of EXISTING tensors, inside the model's own forward pass: for each annotated answer position where the original argmax != annotated token, the final-residual target (gap + margin 16 logits along gamma*(W_U[t]-W_U[c])) is delivered over the last 6 layers (each layer 1/(layers left) of the remaining target) through the 1024 highest-scoring COLUMNS of that layer's MLP `down_proj` (12288 -> 4096). Solve per layer: `dW = R K^T (lam (C + awt C_ans + ridge I) + K K^T)^-1` (K = MLP activations at the annotated sites on the chosen columns; C = second moment of the same activations over the shipped UNLABELED calibration prompts + the model's own greedy answers; C_ans = same on answer/EOS positions = protection of the old labels). lam 1000, awt 10, ridge 0.05, margin 16, 6 layers, 1024 columns (chosen on the selection split, below). No gradient/optimiser, no head/adapter/new tensor/lookup, nothing in the prompt, annotations not stored. The edited columns are stored fp16 in `patch.pt` (a 1-bit weight cannot take a small step); `bonsai_loader` overlays them on the existing down_proj tensors. Empty annotations -> patch removed -> tree byte-identical (tested, tree sha equal).
- `feedback_calib.jsonl`: 182 UNLABELED SKILL7 prompts (45.5 KB, counted).

## Bytes (exact, `MANIFEST.json`, `results/bytes_skill.json`)
Base tree 1,168,094,685 B = 7.1306% of Qwen3-8B fp16 (16,381,470,720 B; = 7.1320% of Bonsai's own fp16 16,377,096,192 B). After 25 annotations: +50,347,483 B (patch.pt) = **1,218,442,168 B = 7.4379% (Qwen3-8B baseline) / 7.4399% (Bonsai baseline)**; code 15,939 B, calibration 45,503 B. Cap 9.9% = 1,621,765,601 B, slack 403.3 MB; cap 10% slack 419.7 MB.

## Task SKILL7 (voice-skill routing, MASSIVE + CLINC150, 7 skills) base health (Bonsai, real generation, 682 dev items)
Accuracy 0.896 [Wilson 0.871, 0.917], chance 0.143; 9 off-menu outputs (all `Timer`). Cells (acc, n): CLINC Alarm .52 (54, timers), Calendar 1.0, Lists .97, Music .97, Transport 1.0, Weather .96 (24); MASSIVE Alarm .92, Calendar .83, Lists .83, Music .94, News .94, Transport 1.0, Weather .77 (52 each). Several cells are at/near ceiling, one at .52: rule 8 per cell is only partly satisfied (overall is fine).

## Feedback results (new skill `Recipe`, never in the prompt menu; base teach 0.000 on every teach set)
Source A = MASSIVE `cooking` (annotations and teach set A = 45 unseen items), source B = CLINC150 cooking intents (teach set B = 444 items, another dataset). Neighbours = base-correct SKILL7 dev items (REP split 317; selection split 294 was used for tuning only). 10 draws of 25 annotations (n=12, n=5: 5 draws). Mean [bootstrap 95% CI over draws].
| arm | n | teach A | teach B (other source) | neighbours kept (REP) | worst draw |
|---|---|---|---|---|---|
| **real** | 25 | **0.876** [.862,.889] | **0.867** [.838,.894] | **0.999** [.998,1.0] | 0.9968 |
| shuffled (Recipe label on 25 unrelated inputs) | 25 | 0.002 | 0.013 | - | - |
| wrong (annotated inputs, a wrong old skill) | 25 | 0.000 | 0.000 | 0.996 | 0.9905 |
| mixed (half wrong) | 25 | 0.673 | 0.418 | 0.997 | 0.9968 |
| empty | 0 | 0.000 | 0.000 | 1.000 (byte-identical) | - |
| real | 12 | 0.849 | 0.758 | 0.997 | |
| real | 5 | 0.769 | 0.566 | 0.999 | |
Real - shuffled on A: +0.873 [.860,.884]. Every per-cell neighbour retention is 1.0 except MASSIVE News 0.986 (280 draws-items), per source MASSIVE .9978, CLINC 1.0. Selection sweep (2 draws, n=25, `results/`): awt/lam/margin/layers/columns all give neighbours >= .997 and teach .83-.93 on SKILL7; 1024 columns (50 MB) already has shuffled 0.00 where 2048 columns or L8 leaked (shuffled .15-.70), so the smallest patch is also the cleanest.
End to end through the SHIPPED commands (`e2e_skill.py`, copy of the artifact, `feedback.py` CLI -> `patch.pt` -> `answer.py` CLI real generation, `results/e2e_skill.json`): real teach A 0.844, teach B 0.80, neighbours 0.9935 (306 base-correct), shuffled 0.00 / 0.09, empty = 0 and tree sha unchanged. A first e2e run with the OLD default of 2048 columns leaked on its shuffled arm (teach 0.84, `results/e2e_skill_cols2048.json`): the 2048-column setting is not reliably selective; the shipped default is 1024.

## Old-label annotation policy (<= 25 annotated REAL mistakes)
An annotation is accepted iff the base model's own answer differs from the user's label (a real mistake), for old or new labels, at most 25 in total. Tested (5 draws, `results/sk_policy.json`): 25 new-label: teach .880, neighbours .998; 13 new + 12 old-label: teach .853, neighbours .994, held-out old errors fixed 18% (base 0); 20 + 5: .867 / .997 / 7%; 25 old-label only: neighbours .995, held-out old errors fixed 38% (base 0). Old-label corrections do not damage neighbours on SKILL7 (the NEWS BBC-World row damage seen with the lm_head-row mechanism does not appear), they teach partially (generalisation of the correction, 38% at 25 items).

## Maintainer-own-data proxy (calibrate on A, annotate A, test on B; and the reverse)
Calibration file restricted to ONE source (5 draws, `results/sk_proxy_*.json`): calibrate only on MASSIVE prompts -> teach on CLINC (other source) 0.88 (draws .82-.94), neighbours MASSIVE .998 / CLINC 1.0; calibrate only on CLINC -> teach CLINC .90, MASSIVE teach .88, neighbours .997-1.0. Annotating with source B (25 CLINC cooking items) -> teach B .962 [.959,.966], teach A .844, neighbours .997. Foreign label sets of the maintainer's data are not tested beyond Recipe; a new word that is far from the old menu is the easy case.

## 9-rule status (dev-level; the locked exam is NOT built for this lane)
| rule | status | evidence / risk |
|---|---|---|
| 1 size | PASS 7.44% incl. patch, calibration, code | exact bytes above; GRAY only on "fp16 size of the model you started from" (Bonsai fp16 = Qwen3-8B fp16 within 0.03%) |
| 2 same = faithful | PASS vs Bonsai (lossless, identity); **not vs the parent Qwen3-8B (88.9% earlier)** | WEAKEST: base identity is not ruled |
| 3 exam | NOT BUILT. Feasibility problem: SKILL7 locked candidates (hash bucket 70-100) are Alarm 184, Music 219, Weather 200 in total (< 300 per kind) and CLINC Alarm 60 / Weather 31 | needs an enlarged locked bucket or a different grouping (owner of EASY tasks) |
| 4 no new trained parts | GRAY | no head/adapter/tensor/prompt/lookup; but fp16 columns of existing tensors fitted in closed form to the annotations are stored, and unlabeled calibration prompts ship |
| 5 feedback teaches | PASS on dev: teach 0.876 (base 0), cross-source 0.867, neighbours .999, new label generated end to end | unseen locked teach items (>= 25) still to be locked; only 45 same-source items exist in dev, 444 cross-source |
| 6 shuffled/empty worse | PASS: shuffled .002, wrong .000, empty .000, real .876 | |
| 7 real job | PASS | the skill word is generated; strict string match; no constrained decoding |
| 8 something to keep | PARTIAL: .896 overall (chance .143) fine; per-cell: several cells 1.0, CLINC Alarm .52 | |
| 9 spirit | GRAY | no compression by us; the 1-bit model is the authors' release |
Weakest rule: base identity (2/9, unruled); next: closed-form-edit legality (4) and the unbuildable per-kind exam (3).

## Cost
Vast L40 boxes (2, about 3 h each at $0.456/h = under $3 plus a third of box 1 still running for NEWS), all destroyed at the end (see STATUS.md); Modal 0.
