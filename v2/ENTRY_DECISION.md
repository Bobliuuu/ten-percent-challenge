# ENTRY_DECISION: which entry to file, and what to do if the rulings go against us

Written 2026-11-08 (about 20:45 UTC). Deadline 2026-11-12 17:05 UTC; global freeze 2026-11-11 08:00 UTC. Numbers are from [`CANDIDATES.md`](CANDIDATES.md), the lane reports and [`RED_TEAM.md`](RED_TEAM.md); nothing was re-run. Nothing here has been posted anywhere, and nothing is filed without the user's explicit go-ahead. Rules: [`RULES.md`](RULES.md).

## 0. Ground rules for this decision

1. **Never claim a failing rule as a pass.** Every entry is filed with its exposures stated. A rule that is gray is written "gray", not "pass".
2. **Closed-form edits of existing tensors are gray (unruled).** The maintainer rejected a fitted head (issue #2) and a nearest-centroid row overwrite (issue #3, "the same head as before, just kept inside the model file"). He has not ruled on a closed-form edit that spreads over existing weights. Every feedback carrier we have (L1, L2, L3, L4, L9, L9b, L10) is such an edit, so R4 is gray for every candidate, and R5/R6 depend on it.
3. **Starting from an already-compressed release is "maybe valid, unruled"** (RULES 4f). Reading for: rule 1 says any model above 0.5B is fine "before or after you cut it", and "the original" can mean the model we start from. Reading against: rule 9 spirit (the compression is someone else's) or "original" = the full-precision ancestor. It is not settled in either direction. L1 and L2 are filed, if at all, as gray on this point, with retention reported against the start model as the primary number and against the ancestor as an informational row (L1 88.9%, L2 99.32%).
4. No entry has a second, broader task. The 7-way voice-skill routing task is commercial but small-label-set (R1/R9 "narrow distribution" exposure) and the bases sit near their ceiling (R8 headroom). This applies to every candidate equally.

## 1. Recommendation

| role | entry | why |
|---|---|---|
| **Primary** | **L9b**: Qwen3.5-35B-A3B, our own compression (routing-mass expert prune to 102 of 256 + 3-bit RTN), 9.674% (9.704% with the feedback carrier) | The only candidate that is (a) our own compression of an unmodified, publicly released full-precision model (no start-model-identity ruling needed), (b) clearly under 10%, (c) strong on R2 on our sealed exam (99.55% [99.21], 3 draws 99.47-99.55, every kind at least 98.4), and (d) runnable on one 40 GB GPU. Depends on rulings A (closed-form edit), C (task-calibrated expert pruning) and G (carrier columns). |
| Alternate 1 | **L1**: Bonsai-8B (released 1-bit) + 50 MB column patch, 7.44% | Smallest, runs on a 24 GB card, cleanest R5/R6 evidence (teach .893, shuffled .005, wrong 0, empty 0, neighbours .995, 10 draws). Retention 1.000 holds by construction against the released model only. Depends on ruling B (pre-compressed start) and R9 (we did no compression), plus A and G. |
| Alternate 2 | **L2**: Ternary-Bonsai-2-27B recompressed by us, 9.943% | Our compression is real, retention 100.0% against the released model and 99.32% against the fp16 ancestor, so it passes R2 under either reading of "original". Depends on B only partly, plus A and G. Cost: 30.6 MB of byte slack, 96 GB GPU plus a large cache to run, weak shuffled control, artifact only in private storage. |

Note on the red-team view: `RED_TEAM.md` ranked L2 first on R2/R9 and called L9b the weakest on R6/R8/R9 margins and own-data risk. The primary is L9b by the project owner's decision; the table above states L9b's weaknesses so the choice is made with them in view. If L9b's open items in section 3 cannot be closed before the freeze, switch to L2 (strongest on R2 under both readings) or L1 (smallest and simplest to ship), in that order of preference for a "most defensible" entry, and say so in the filing.

Considered and not chosen: **L3** (Qwen3-30B-A3B-Instruct-2507, 9.896%, LOCKED SKILL7 99.32% [98.92], teach .799 with shuffled 0, empty 0) has a cleaner R6 than L9b, but its feedback carrier is 134 MB of fp16 `o_proj` (9.896% total, 0.10 pt slack), its neighbour retention after the final carrier was not measured on the exam, and its second task (BANK6) fails R5 (teach .16). It is the first replacement if L9b's R6 variance cannot be fixed. L4, L9 (needs about 330 GB of GPU memory, impractical for the maintainer's check), L10 (R1/R9 denominator question, no packed file) and L5 (exam not scored) are not candidates for filing. L12 is a research lane.

## 2. L9b against the nine rules (most honest reading)

| rule | status | evidence |
|---|---|---|
| R1 size | PASS | 9.674% of the text-only fp16 reference (9.326% vs the full checkpoint including the vision tower we do not ship); 9.704% with the carrier; slack 157 MB. Feedback code and calibration file must be added to the byte manifest (RED_TEAM). |
| R2 faithful | PASS | LOCKED SKILL7, 2,744 items: 99.55% [99.21], worst kind Lists 98.4, worst source-by-kind cell 97.1; 3 quantizer draws 99.47-99.55. Single base run. |
| R3 exam | PASS on counts, caveats | 2,744 items, 344-420 per kind, 1,448 base-correct from never-used sources. 532 rows are tier `seen_tox`; on `clean_only` the per-kind count is below 300 for Lists and Transport (ruling E). The exam is ours, not the maintainer's. |
| R4 no new trained parts | **GRAY** | Expert pruning and RTN are data-dependent selection, not fitted heads; no head, adapter, lookup or prompt. The feedback carrier is a closed-form fit of existing dense shared-expert `down_proj` columns stored as replacement columns (rulings A and G unruled). |
| R5 feedback teaches | PASS (moderate), conditional on A | teach 0 -> .73-.79 on 164 locked items (cross-source MTOP/HWU64 included), neighbours 98.0-98.6 vs 98.3 empty, worst neighbour cell 95.8-97.9, 3 draws. |
| R6 controls | **PASS in 2 of 3 draws, GRAY in 1** | shuffled .110 / .018 / **.470** (real .762 / .793 / .726), wrong 0, empty 0. Draw 2 is a gap of only .26. Must be stated openly. |
| R7 real job | PASS | the model generates the skill word; exact match, no constrained decoding. |
| R8 something to keep | **weak** | base .9676 (chance .14), 3.2% errors, close to the ceiling. |
| R9 spirit | **GRAY** | 60% of experts pruned using the task's own unlabeled prompts; the packed model is a task specialist. BANK6 with the same recipe fails R2 (97.37% [96.58], Card 94.4), which shows how far it generalises. |
| MC (maintainer own data) | **untested, highest exposure of our entries** | the 7-skill prompt is hardcoded in `answer.py`; routing mass shifts on other text; annotation format differs from L1/L2. |

## 3. What must happen before L9b can be filed (ranked, effort S <= 1 h, M = a few hours, L = GPU half day)

1. [M] Own-data dry run: base vs packed on about 300 unrelated prompts (for example Banking77) and one unseen new label close to an old one (`Timer` vs Alarm). If outputs diverge badly, say so in the filing and consider L2 or L1 as primary.
2. [M] Re-run feedback with independent annotation sets for at least 6 draws (3 more), report the shuffled draw that reaches .47 openly, and try the L11 findings (margin 32, `o_proj`-style protected solver) to cut shuffled leakage. Do not re-run the exam artifact unless the bytes change.
3. [S] Ship the missing pieces and count them: `fbcore.py`, `feedback.py`, an unlabeled `feedback_cal.jsonl`; make `answer.py` take a task JSON; align the annotation format with L1/L2; remove private paths.
4. [S] Publish the 6.7 GB artifact with sha256 in `MANIFEST.json` (it is not in git; it must be re-packed from the public snapshot with `core.py`/`qpack.py` today). A fresh clone has to run `answer` and `feedback` end to end. RED_TEAM found that a fresh clone could not run any of L1, L2, L9b as committed.
5. [S] Fix report text where stale; state the base is a single run and that the exam is ours.
6. [L] A second, broader task for the chosen entry would answer the "narrow distribution" attack; no entry has one and none is expected before the freeze.

## 4. Fallbacks

- **If the maintainer rules that a closed-form edit of existing tensors counts as training/retraining or as a side memory (ruling A against)**: then R4/R5/R6 fail for every candidate we have, and no entry we have passes the nine rules. **File nothing as a pass.** The compression results (L1, L2, L9b, L3, L9, L10) stay documented in the repo as measured facts; if anything is posted, it is labelled "does not claim to pass; feedback step not satisfied under this ruling", never a pass. Nothing is posted without the user's explicit go-ahead.
- **If no ruling arrives by the freeze (2026-11-11 08:00 UTC)**: file L9b (or the fallback chosen in section 1) with every gray cell named, as an entry that passes the measurable rules on our own exam and carries the unruled questions, not as a clean pass. The user decides at that point; nothing is filed automatically.
- **If ruling C goes against task-calibrated expert pruning**: L9b and L3 fall to gray-to-fail on R9; L2 or L1 become the primary.
- **If ruling B (pre-compressed start) goes against us**: L1 and L2 are out; L9b stays.
- **If L9b's own-data dry run or the extra feedback draws show R6 or MC failing**: switch to L2, then L1.
- **If the denominator ruling D favours multimodal text paths**: L10 becomes interesting (99.55% retention at 9.879%), but it needs a packed file and carries the R1/R9 spirit risk; it is not planned.

## 5. Open rulings, summarised (none has been put to the maintainer)

A: closed-form weight edit legal for feedback. B: already-compressed start model. C: task-calibrated expert pruning. D: denominator for multimodal start models. E: `seen_tox` tier and pooled per-kind counts. F: how much worse "clearly worse" must be for shuffled labels. G: fp16 replacement columns or low-rank delta as "new part" vs the model's own weights. H: is 7-way routing commercial and not narrow. I: what "best" means among passing entries (if size, L1 wins; if own compression and retention, L9b). Entrant #4's issue (a row edit on a pruned MoE that does run the new label end to end) has no maintainer reply that we know of; its verdict is the best available signal for A and C and should be checked read-only.
