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

---

# L1+ (2026-11-08): push the Bonsai entry (stronger feedback, own-data proxy, continual, carriers, BANK6, robustness, usability)

Code: `v2/bonsai/ship/` (copied to this folder: `answer.py`, `feedback.py`, `common.py`, `bonsai_loader.py`, `README.md`), experiments `v2/bonsai/l1plus/` (`run_l1p.py`, `run_bank6.py`, `e2e_l1p.py`, `dnorm.py`, `cfgs_*.json`), raw results `v2/bonsai/l1plus/results/`. Hostile proxy data: `v2/bonsai/l1plus/proxy/`. Everything was built and chosen on DEV / SELECTION data (the same splits as L1: A = MASSIVE cooking, B = CLINC cooking, neighbours = base-correct dev items, sel half for choice, rep half for reporting). The sealed exam was NOT scored with any new candidate: the locked TEACH set and the locked NEIGHBOUR ids were used only for reporting (configs were fixed on dev before), the exam itself only for the BANK6 base x3 + candidate x1 run below (new task, authorised). Compute: 2 Vast boxes (L40S, RTX 4090 48 GB), about 9 h in total, about 7 USD; both destroyed and verified gone. A `lanes/L1` caveat first: the default flags still write a bit-identical patch to the sealed L1 code (`results/e2e_l1p.json`: `regression_patch_identical` true, max abs diff 0.0), so every L1 number above stands.

## L1+.1 Feedback strength: what moves it and what does not (dev, selection split, 25 annotations, 2 draws, `results/sw*.json`)
| lever | teach (sel A) | shuffled | neighbours | note |
|---|---|---|---|---|
| default (6 layers, 1024 cols, lam 1000, awt 10, margin 16) | .887 | 0.00 | .998 | L1 setting |
| margin 24 / 32 / 48 | .90 / - / - | 0.00 / - / - | .997 | margin saturates: the failures are not weak edits |
| lam 300 | .887 | .51 | .997 | weaker protection leaks into the shuffled control |
| annotation augmentation (`--aug 2 / 4`: lowercase, "please", "hey", upper variants of each annotated text) | .887 / .912 | .41 / .68 | .997 | better generalisation, but the edit becomes less selective: shuffled-label control teaches Recipe |
| 12 layers x 512 cols / 18 x 256 | .887 / .875 | 0.00 / 0.00 | .998 / 1.00 | diffuse, same teach |
| scale carrier lam 100 / 10 / 1 | .887 / .90 / .912 | 0.00 / .55 / .78 | .997 / .998 / .993 | lam <= 10 leaks |
Plateau: on dev A the ceiling is 0.889 (40/45) and it is label noise, not strength: the 5 items that fail in every draw are MASSIVE "cooking" rows that are not cooking ("freezing", "aroma", "pleasant", "weird things people want to know", "when is asparagus in season"); on the 40 clean items the real arm is about 1.0. On the LOCKED teach set the same reading holds (below): 9 of 164 items fail in every draw (e.g. "can i modify the apples for grapes", "What is the temperature for soft boil", "What should the internal temperature of fish be", "access recipe book please"): ambiguous or Weather/Lists-shaped; the ceiling for a clean selective edit is about 0.945, we get 0.905. **The 0.95 target is not reached by a selective edit**; the only knobs that go above it on some sets (aug) pay with the shuffled control. Honest status: teach 0.89 -> 0.905 (locked), HWU64 .807 -> .82, MASSIVE .857 -> .88 (lr carrier; scale .81, diffuse .74).

## L1+.2 Fewer annotations (dev, 10-draw-equivalent protocol, `results/fin1.json`, `fin2.json`; A = same-source, B = CLINC other source)
| carrier / setting | n | teach A | teach B | neighbours | shuffled (A / B) |
|---|---|---|---|---|---|
| cols (L1 default), 5 draws | 25 / 12 / 5 | .880 / .844 / .778 | .856 / .739 / .584 | .999 / 1.00 / .997 | .00 / .001 |
| lr (same delta as rank-k factors) | 25 / 12 / 5 | .880 / .844 / .767 | .857 / .739 / .584 | .999 / 1.00 / .997 | .00 / .001 |
| scale (existing group scales only) | 25 / 12 / 5 | .880 / .833 / .644 | .853 / .687 / .372 | .997 / .998 / 1.00 | .00 / .00 |
| diffuse 18 layers x 256 cols | 25 / 12 | .871 / .867 | .859 / .849 | .999 / 1.00 | .00 / .00 |
| lr + aug 4 | 25 / 12 / 5 | .880 / .856 / .844 | .949 / .860 / .747 | .998 / .997 / .997 | **.80 / .73** (leaks) |
So n = 12 gives .84-.87 on A, n = 5 .77 on A (.58 on B) with the selective setting. Augmentation lifts n = 5 to .84 (A) / .75 (B) but at a shuffled-control leak of .7-.8: do not ship it as default; it is a flag (`--aug`).

## L1+.3 Locked teach and neighbours for the new carriers (`results/locked.json`, 6 draws x 25 annotations from `ann_pool`; NOT the exam)
| carrier | bytes | teach (164 locked) | MTOP / CLINC / MASSIVE / HWU64 | neighbours kept (352 base-correct) | shuffled |
|---|---|---|---|---|---|
| cols default (L1, 10 draws) | 50.3 MB | .893 | .909 / .893 / .857 / .807 | .9952 | .005 |
| **lr** | **0.50 MB** | **.905** (draws .896-.921) | .923 / .897 / .881 / .822 | .9948 (worst .9915) | .002 |
| scale | 4.72 MB | .904 (.890-.927) | .928 / .897 / .809 / .811 | .9938 (worst .9915) | .001 |
| diffuse 18 x 256 | 37.8 MB | .908 (.890-.933) | .940 / .893 / .738 / .811 | **.9981** (worst .9972) | 0.000 |
(An earlier run of this file used 25 near-neighbour keyword items (food/shopping) with the Recipe label as the "shuffled" arm: it taught Recipe (.66), which is correct behaviour for semantically related items; that file is kept as `results/locked_bad_shuffled_pool.json` and is NOT the control. The control here is the L1 one: ordinary selection items.)

## L1+.4 Carrier variants (legality hedge, RULES 4d). Same 25 annotations, `results/dnorm.json`
| carrier | what is stored | bytes | weights changed (of 8.2 B) | rel. norm of delta in touched blocks | max per-weight rel. change | teach (locked) | neighbours |
|---|---|---|---|---|---|---|---|
| cols (L1) | 1024 fp16 columns x 6 layers | 50.3 MB | 0.30% | 0.0072 | 4.4 | .893 | .995 |
| **lr** | the same delta as rank-<=50 factors (A@B) | **0.50 MB** | 0.30% | 0.0072 | 4.4 | .905 | .995 |
| **scale** | NEW per-128-group scales of the existing 1-bit down_proj (weight = sign x scale); no new tensor, could be written back into `scales.bin` in place (+0 B) | 4.72 MB as an overlay | 3.6% | 0.042 | 1.6 | .904 | .994 |
| diffuse 18 layers x 256 cols | columns | 37.8 MB | 0.22% | 0.0032 | 2.1 | .908 | .998 |
| cols 128 cols x 6 layers | columns | 6.3 MB | 0.04% | 0.0031 | 2.2 | .733 (sel A, 3 draws; margin 24: .85; 256 cols 12.6 MB: .84; shuffled 0.00, neighbours .997-.999; `swC.json`, not run on the locked set) | .998 |
The `scale` carrier is the "delta of existing parameters" hedge: the edited object is a value the model already has (a group scale), the stored object is not a new tensor and the byte cost can be exactly zero. It is NOT smaller in effect (it touches every row of 6 down_proj matrices, relative norm 6x larger) and it is looser on neighbours (continual e2e .89-.97 vs cols .86-1.0 on the same annotations). `lr` is the cheapest (0.50 MB, 7.13% total) but literally a rank-k delta: by the wording of rule 4 it looks MORE like an adapter although it is merged into an existing tensor before use. No carrier changes the verdict of RULES 4d: each is a closed-form fit to the annotations; they only change how it looks. Totals incl. code, README, calibration, manifest: cols 7.438%, lr 7.134%, scale 7.160% (overlay), diffuse 7.361% of Qwen3-8B fp16 (`MANIFEST.json` key `l1plus`).

## L1+.5 Maintainer own-data proxy (hand-written hostile data, never seen, `results/proxy.json`, 3 draws)
Data (`v2/bonsai/l1plus/proxy/`, written by an agent without running any model, no dataset text copied): 315 old-skill utterances (7 skills x 45) in 7 styles (UK/Indian/Singlish/Aus phrasing, ASR-noise lowercase, typos/texting, 25-60-word LLM-style requests, indirect, Hinglish/Spanglish, keywords), plus three new skills never in the menu: `Translate`, `Banking`, `Email` (130 rows each, same styles). Base: old-skill accuracy .914 (5 off-menu), the new-skill rows are answered Transport/Calendar/News/Lists (teach 0). Annotation = 25 real mistakes from the first 40 rows of the new skill; teach = the other 90; old retention = base-correct old rows (half, used as unlabeled calibration in the `own_unlabeled` arm, other half as test; all 7 styles) plus the dev neighbours.
| new skill | calibration | teach (hostile, unseen) | old-proxy retention (per style 7/7) | dev neighbours | shuffled |
|---|---|---|---|---|---|
| Translate | shipped | .885 | 1.000 (all 7 styles) | .997 | .078 |
| Translate | the maintainer's own unlabeled prompts | .907 | 1.000 | .994 | |
| Banking | shipped / own | .874 / .881 | 1.000 | .998 / .995 | .000 |
| Email | shipped / own | .930 / .933 | 1.000 | .998 / 1.000 | .111 |
| (scale carrier) Translate / Banking / Email | shipped | .874 / .830 / .933 | 1.000 (shuffled .987-.994) | .996 / .994 / .998 | .056 / .022 / .011 |
So the system keeps working and keeps learning on a different distribution, locale, noise level and a different new skill with <= 25 annotations, old skills untouched. Limits: the proxy is machine-written (not real user data), the old-skill labels are the agent's, and a one-label new skill is the easy case (no old-label correction was tested on it).

## L1+.6 Continual feedback (`results/cont.json`, 3 draws; Recipe -> Email -> Takeaway, MASSIVE dev scenarios, held-out teach sets; `--append`)
| setting | after round 1 (Recipe) | round 2 (+Email) | round 3 (+Takeaway) | neighbours (rep) | bytes |
|---|---|---|---|---|---|
| cols, 25 per round | Recipe .859 | Recipe **.889**, Email .830 | Recipe **.889**, Email .837, Takeaway .852 | .997 / .999 / .995 | 50 / 74 / 91 MB (union of columns) |
| cols, 8 per round (24 total) | .822 | .852, Email .738 | .867, .747, Takeaway .708 | .997 / .997 / .994 | 50 / 75 / 92 MB |
| scale, 25 per round | .859 | .889, Email .812 | .889, .817, .826 | .997 / .995 / .995 | 4.7 MB (constant) |
No forgetting of the first skill (it even rises: the later edits also fire for earlier-skill phrasing), no loss on the old menu (min draw .994). The second and third skill are learned at .81-.85 (25 per round) or .71-.75 (8 per round). **Tail risk (found by the shipped-command test):** with one unlucky annotation draw for `Email`, 26 of 204 `Lists` neighbours ("what are all my lists", "delete last entry") were answered `Email` (neighbours .863 after round 2, .77 after round 3 in `results/e2e_l1p.json`): a new skill whose words overlap an old one ("list the subject lines") steals it. Protection knob on exactly those two rounds (`results/dbg3.log`): awt 30 -> neighbours .951 / Email teach .88; awt 100 or lam 3000 -> .99 / .64; scale carrier .89; scale + awt 100 .995 / .46. So strong protection costs the new skill; the default is NOT safe for a new skill adjacent to an old one, and the three-draw numbers above underestimate that tail. Mitigation shipped as flags only (`--awt`, `--lam`).

## L1+.7 Robustness (`results/robust.json`, patch from 25 annotations, 2 draws, dev; variants of the prompt that the patch never saw)
| prompt variant | base acc on neighbours | patched neighbours | patched teach (Recipe, base 0 except where noted) | off-menu (patched, 80 non-menu banking inputs) |
|---|---|---|---|---|
| original | 1.000 | .997 | .867 | 3 (base 1) |
| menu reordered | .980 | .977 | .767 | 3 |
| paraphrased instruction | .987 | .980 | .822 | 4 |
| markdown wrapper | .987 | .987 | .833 | 28 (base 21) |
| lowercase text | 1.000 | .997 | .867 | 4 |
| +600 chars of filler | .973 | .973 | .844 | 9 (base 7) |
| +3000 chars of filler | .907 | .913 | .800 | 9 (base 7) |
| no menu in the prompt | .487 | .507 | **.911** (base .20) | 80 (the model invents words: Bank, Balance; base 80) |
The patched model is exactly as robust as the base on every variant (neighbour accuracy within +-0.02 of the base under the same variant), and the new label transfers to prompts the edit never saw (.77-.91). Long inputs degrade the BASE (0.907 at 3000 chars), not the patch. Off-menu rate on an old-menu prompt: 1-4 of 80 for non-menu inputs (3 vs base 1), 0 on in-menu neighbours.

## L1+.8 Second task entry: BANK6 + new label `Security` (Bonsai-8B, shared BANK6 exam, `results/bank6_locked.json`, `run_bank6.py`)
Candidate = the same lossless pack with a BANK6 task json and the BANK6 `cal` prompts as calibration (shipped code unchanged). Config chosen on dev (annotation pool split, `results/b6dev_*.json`, `bank6_dev_default.json`): cols, lam 100 (the L1 lam 1000 gave .46; lam 100 .80 with shuffled .004; margin 32 .89 but shuffled .23).
| item | value |
|---|---|
| exam (2237 items, kinds 330-400, 600 Bitext never-used) | base HF fp16 run 3x (bs 8/4/16): accuracy .8556 / .8556 / .8556 (pairwise agreement 1.0); candidate once: .8556, outputs identical to the base on 100% |
| retention of base-correct (n=1914) | **1.000** overall (Wilson lower .9980), every kind 1.000, banking77 1.000, never-used Bitext (n=474) 1.000 (lower .9920); `clean_only` (without `seen_l5dev` rows, n=1581) 1.000, but Exchange 142 / Identity 187 base-correct < 300 per kind |
| rule 3 counts | 2237 >= 2000, every kind >= 330, 600 never-used >= 500 |
| rule 8 | base .856, chance .167, below ceiling |
| feedback: new label Security (200 locked teach: Banking77 100, Bitext 100), 10 draws x 25 annotations from `ann_pool` | real teach **.727** (Banking77 .814, Bitext cross-source .641), neighbours kept **.989** (709 base-correct, worst draw .977, Identity .985, Card .990), shuffled .101 (draws .01-.49), wrong .000, empty .000 |
| other configs (locked teach/neighbours only, `results/b6fb_*.json`) | lam 30: teach .749 / neigh .986 / shuffled .26; 12 layers x 512, lam 100: .75 / .988 / .19; scale lam 100: .57 / .958 / 0.00 (fails neighbours) |
| bytes | same pack: 7.44% with the 50 MB patch (lr: 7.13%) |
The exam was scored once for the candidate (no exam-after-feedback run). The task generalises, but BANK6 is a weaker second entry than SKILL7: `Security` sits inside the old Card/Identity classes (pin blocked, lost phone), so teach is .73 not .90, the shuffled control leaks .10 (one draw .49), one draw is below 98% neighbours (.977) and the cross-source (Bitext, synthetic) teach is .64. By the rule text it passes (clear improvement over 0, neighbours >= .98 on average, controls far below), but it is the grayer proof.

## L1+.9 Usability (shipped folder)
`README.md` (this folder): files, one command to answer, one to give feedback, flags. New: `answer.py --feedback ann.jsonl [--save-patch]` (feedback applied in the same process; without `--save-patch` nothing is written), `feedback.py --append` (continual), `--carrier cols|lr|scale`, `--aug N`. Verified through the CLIs on dev (`results/e2e_l1p.json`): regression (new code, default flags == sealed L1 code, identical patch tensors), in-process feedback == feedback.py + answer.py outputs (identical on 249 items), `--save-patch` persists and reproduces, empty feedback removes the patch, the 3-round `--append` chain works end to end (rounds above), the scale carrier works through the CLI (teach .867, neighbours .966, 4.72 MB).

## L1+ rule table delta (start model = Bonsai-8B per 4f)
R4 stays GRAY (every carrier is a closed-form fit; `scale` is the least "new object" and +0 bytes if written in place; `lr` is the smallest and looks most like an adapter). R5 SKILL7 PASS (locked teach .905, neighbours .995, shuffled .002, new label from generation, own-data proxy .87-.93 on three hostile new skills). R5 on BANK6 PASS-on-the-letter, weaker. Open risks that L1+ exposed: new skill adjacent to an old one (Email vs Lists) can steal old items in continual mode (tail .86); teach ceiling is bounded by ambiguous items (about .945), not by edit strength.


## Update: red-team fixes (2026-11-08)
- Retention 1.000 vs the start model is true BY CONSTRUCTION (lossless bit-exact repack of Bonsai-8B-unpacked, identical outputs); it measures nothing about our own compression. Against Qwen3-8B fp16 the retention is 88.9%.
- The SKILL7 `feedback_calib.jsonl` (182 unlabeled prompts, sha256 4d1c8d09...) is now in this folder; `v2/bonsai/fbx/setup_box.sh` copies it (it used to copy the NEWS calibration file). `build/pack1.py` is the pack step; `build/SHA256SUMS.build` lists reference hashes of a clean rebuild from the public snapshot.
- Annotation schema is documented once for L1/L2/L9b in `v2/lanes/ANNOTATION_SCHEMA.md`.


## L1 shipping update 2026-11-09: scale carrier default, private artifact, fresh-clone dry run
- **Default carrier is now `scale`** (existing per-128-group scales of the 1-bit down_proj of the last 6 layers; no new tensor; lam 100 for this carrier as measured, `--carrier cols|lr` kept). Parity check, shipped CLIs on the locked SKILL7 sets with the new default (`results/e2e_locked_scale_default.json`, one 25-annotation draw, same seed as the cols run in `results/e2e_locked.json`): real teach **.909** (cols .884; MTOP .92, CLINC .905, HWU64 .867, MASSIVE .857), neighbours kept **.9972** (352, same as cols), shuffled teach 0.000 / neigh .9972, wrong 0.000 / .9943, empty 0.000 / 1.000 with the tree byte-identical. Earlier 6-draw locked numbers agree (scale .904, neigh .9938, shuffled .001). Caveats unchanged: scale is weaker than cols at very few annotations (n=5: A .644 vs .778) and looser in continual mode (see L1+.2/L1+.6); its patch.pt is 4,720,692 B.
- **Bytes**: the scale patch is counted as a 4,720,692 B overlay file; total tree after feedback 1,172,852,247 B = 7.160% of Qwen3-8B fp16 (7.162% of Bonsai fp16) vs 7.438% with cols. Writing the new scales into `scales.bin` in place would add 0 B; not done so that empty feedback restores the original tree byte for byte. `MANIFEST.json` keys `scale_default`, `private_artifact`.
- **Private artifact**: weights rebuilt on Modal from the public snapshot with `build/modal_publish.py`, every hash matched `SHA256SUMS`, stored in the private Modal Volume `tenpct-l1-art` (dir `/art`), manifest `build/SHA256SUMS.artifact`. No public hosting; the old build note in the MANIFEST is kept as provenance.
- **Fresh-clone dry run** (`build/modal_dry_run.py`, `results/dry_run.json`): clean L40S container, shallow `git clone` of the repo (commit 1c0e7646), weights from the volume (0 sha256 mismatches), shipped answer.py -> feedback.py (8 Recipe annotations, default carrier, patch kind `scale`, 4,720,692 B) -> answer.py: 4/4 unseen recipe prompts now answer `Recipe` (base: Transport / off-menu / Transport / Lists), 7/7 neighbours unchanged; empty feedback removes patch.pt, tree identical to base, answers equal base.

## Update 2026-11-09 (late): fresh-clone re-run at HEAD
`build/modal_dry_run.py` re-run against repo commit e4249844: 0 sha256 mismatches, answer -> feedback (scale carrier, 4,720,692 B) -> answer gives 4/4 Recipe on unseen prompts, 7/7 neighbours unchanged, empty feedback restores the identical tree and base answers (`results/dry_run.json`). L1 locked-exam verification stays as above (retention 1.000 vs Bonsai-8B by construction).

## L1+.10 Own-data dry run (ENTRY_DECISION sec 3 item 1; same protocol and data builders as L9b REPORT sec 11; `results/owndata_summary.json`, raw outputs `results/owndata_full.json`, job `owndata/modal_owndata.py`, Modal L40S, 2026-11-09)
Base = released Bonsai-8B (artifact without `patch.pt`; bit-exact repack, so "no feedback" is identical to base by construction, re-generation determinism checked). Patched = the shipped feedback path in-process (`feedback.compute_patch`, default `scale` carrier, patch undone afterwards and base outputs re-verified identical). Greedy, no constrained decoding.
| set | n | base | patched (12 Recipe annotations) | patched (12 Timer annotations, draw 0) |
|---|---|---|---|---|
| Banking77 test, 77-label menu, exact label | 200 | acc 45.5% (91 correct) | acc 45.5%; retention **100% [Wilson lo 95.9]**; identical output 100% | retention 100% [95.9]; identical output 99.5% |
| same queries under shipped 7-skill router prompt (agreement only) | 100 | base answers mostly transport 69 | same label **94%** | same label 100% |
| generic prompts (arithmetic, capitals, translation, code, haiku, emails...) 24 tokens | 100 | - | identical continuation **96%**, same first line 98% | 98% / 99% |
Note the base is weak at Banking77 (45.5%) because the 77-label menu is hard for an 8B 1-bit model; retention is measured on its 91 correct items. Divergent generic items under the Recipe patch are small wording drifts (e.g. haiku and email wording), not wrong answers.
**Unseen label `Timer`** (12 Timer annotations from a 20-template pool, 28 held-out Timer requests, 60 Alarm + 60 other SKILL7 neighbours, 3 draws, scale carrier): the base ALREADY emits `Timer` on 42.9% of the held-out requests (Bonsai writes a free word) and sends 8.3% of Alarm requests to Timer; others 96.7% correct. Real feedback: Timer emitted **.964 / .964 / .964**, **Alarm -> Timer flip 21.7%** (Alarm kept 70.0%; base 71.7% kept, 8.3% flipped), others 96.7% (unchanged), others -> Timer 0. Controls: wrong-label (Weather/Music/News/Lists) Timer .11 / .25 / .18, Alarm -> Timer .03-.05; shuffled (half of labels set to Alarm) Timer .04 / .00 / .00, Alarm -> Timer 0-.02. So the flip is caused by the real labels (net +13 points over the base's own 8.3%), the same size as L9b's 21.7% but starting from a base that confuses the two already. The three real draws coincide to the last digit: the annotations come from 8 templates and the edit saturates.
Reading: unlike L9b (packed model diverges: retention 79.3%, generic identical 11%), L1 outside its task is the base model with a narrow edit: retention 100%, router agreement 94-100%, generic identical 96-98%. This answers the "narrow distribution" attack for L1; it does not change the exam numbers.
