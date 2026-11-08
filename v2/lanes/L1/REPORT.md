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

## LOCKED EXAM RUN (shared exam `v2/exam`, SKILL7, 2744 items; loader `python3 -I v2/exam/loader.py` passed before the run; run `run_locked.py`, raw `results/locked_L1.json`)
Start model for the rules = Bonsai-8B itself (RULES.md 4f). Base = reference HF fp16 Bonsai-8B-unpacked, 3 runs (batch 8/4/16): accuracy 0.9074 / 0.9074 / 0.9078 (pairwise agreement .9993-.9996). Candidate = the L1 artifact through the shipped loader, scored ONCE: accuracy 0.9078, outputs identical to the bs-16 base run on 100% of the 2744 items.
| gate | value |
|---|---|
| retention of base-correct (intersection of 3 runs, n=2489) | **1.000** (2489/2489), Wilson lower .9985 |
| retention per kind (Alarm, Calendar, Lists, Music, News, Transport, Weather; n 303-408) | all 1.000 |
| per source (CLINC, HWU64, MASSIVE, MTOP) | all 1.000 |
| never-used sources (HWU64+MTOP, n=1327 base-correct) | 1.000, lower .9971 |
| majority-of-3 base reference | 1.000 (2491/2491) |
| `clean_only` (exam without the seen_tox tier, n=1997) | retention 1.000, lower .9981; per-kind base-correct counts Alarm 253, Calendar 323, Lists 206, Music 319, News 372, Transport 233, Weather 291 (the per-kind >= 300 count gate is NOT met on clean_only) |
| rule 3 counts (all tiers) | 2744 items >= 2000, every kind >= 344, never-used 1489 >= 500: all true |
| base accuracy | 0.907 (chance 0.143): well above chance, below ceiling |
Feedback protocol on the LOCKED teach set (164 unseen `Recipe` items: MTOP 100, CLINC 42, HWU64 15, MASSIVE 7; <= 25 annotations from `ann_pool` only; in-process, same code as the shipped command; 10 draws): base teach 0.000 (the base says Lists 135, Transport 99, Music 22, ... for the annotation pool).
| arm | teach (locked) [CI over draws] | neighbours kept (352 base-correct of the 379 locked neighbours) |
|---|---|---|
| real | **0.893** [0.881, 0.905]; by source MTOP .909, CLINC .893, MASSIVE .857, HWU64 .807 | **0.9952** [.994, .996], worst draw .9915 |
| shuffled (Recipe on 25 unrelated ordinary inputs) | 0.005 [.001, .009] | - |
| wrong (annotated inputs, a wrong old skill) | 0.000 | 0.9943 |
| empty | 0.000 (no patch, tree byte-identical) | 1.000 |
Exam after feedback (draw 0, real arm, n=2489 base-correct): retention 0.9988 (2486/2489), lower .9965, per kind >= .9967, per source >= .998. Cross-source (MTOP/HWU64 never used for annotations or tuning) teach is .91 / .81, i.e. the feedback taught on a source it never saw: the maintainer-style own-data proxy. Generated label example: `Recipe` is emitted by real greedy generation (149 of 164 in draw 0).
Through the SHIPPED commands (`e2e_locked.py`: feedback.py CLI on a copy, then answer.py CLI): real teach 0.884 (MTOP .89, CLINC .905, HWU64 .867, MASSIVE .714), neighbours 0.9972; shuffled teach 0.0, neighbours 1.0; wrong teach 0.0, neighbours .9972; empty: tree byte-identical to the base artifact.
Bytes incl. `tensor_manifest.json` (item 4 done) and the patch: 1,218,467,357 B = **7.4381%** of Qwen3-8B fp16 (7.4401% of Bonsai's fp16); base tree 1,168,119,874 B (7.1307%); slack to 9.9% = 403.3 MB.

## 9-rule status (locked-exam numbers, start model = Bonsai-8B per RULES.md 4f)
| rule | status | evidence |
|---|---|---|
| 1 size | PASS | 7.44% of the fp16 size incl. patch, calibration, code, manifest (Bonsai fp16 and Qwen3-8B fp16 differ by 0.03%) |
| 2 same = faithful | **PASS by construction + measured** (start model = Bonsai) | retention 1.000 overall / per kind / per source on the locked exam, outputs identical; the 88.9% vs Qwen3-8B is informational only |
| 3 exam | PASS on counts (2744, every kind >= 344, 1489 never-used); `clean_only` sensitivity row: retention 1.000 but 3 kinds have < 300 base-correct items (Lists 206, Transport 233, Alarm 253) | exam by the shared build; scored once, not used to choose anything |
| 4 no new trained parts | GRAY | no head/adapter/new tensor/prompt/lookup; but a closed-form edit stores fitted fp16 columns of existing tensors (50.3 MB) and unlabeled calibration prompts ship |
| 5 feedback teaches | PASS | locked teach .893 (base 0), neighbours .995 (>= 98%), exam-after retention .9988, new label emitted in real generation, cross-source teach .81-.91 |
| 6 shuffled/empty worse | PASS | shuffled .005, wrong .000, empty .000 vs real .893 |
| 7 real job | PASS | the skill word is generated, exact string match, no constrained decoding |
| 8 something to keep | PASS overall (.907 vs chance .143, ceiling not reached); some cells near 1.0 | |
| 9 spirit | GRAY | the entrant does no compression of its own; 4f accepts the lane |
Weakest: R4 (closed-form-edit legality, unruled) and R9 (spirit); R3 clean-only per-kind counts.

## Secondary task: NEWS 4-label router + new label `Entertainment` (dev, same mechanism, config chosen on the selection split: awt 100, margin 24, 1024 columns, 6 layers, 50.3 MB)
Base acc .772. Only 5 draws were completed (box destroyed to stop spend; `results/news_final.json`). Teach A (HuffPost entertainment, unseen) 0.667 [.651,.683]; teach B (BBC entertainment, another source) 0.552 [.528,.566]; neighbours kept (REP split) 0.983 [.973,.991], worst draw 0.9625; shuffled 0.104, wrong 0.003, mixed 0.293, empty 0. Per-cell retention: min HuffPost Sports .933, BBC World .941, AG Sports .969, BBC Sports .963, the rest .99-1.0; per source AG .988, BBC .977, HuffPost .984. So NEWS passes the 98% aggregate only on average with a lower teach rate than SKILL7 (.67 vs .88), and fails the per-cell >= .95 bar in two cells; the earlier harsh 400-item HuffPost check (96.4% with the lm_head row) is not comparable. Policy arms on NEWS (13 new + 12 old: neighbours .923, teach .51 in the one draw finished) are worse than on SKILL7. Selection sweep for NEWS: lower awt/lam gives neighbours .82-.97 with shuffled .3-.9 (leak); only awt >= 100 with margin >= 24 holds neighbours >= .98 at teach .7. **SKILL7 is the better task and the one this lane ships.**

## Cost
Locked-exam run: one more A40 box (about 1.5 h, $0.27/h), destroyed and verified gone.
Two Vast L40 boxes (about 4 h and 2.5 h at $0.456/h, about $3 in total), both destroyed and verified gone (no instance with this lane's id remains); Modal 0; AWS 0.
