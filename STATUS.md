# STATUS: 10% challenge

**Last updated: 2026-11-08 about 20:45 UTC.** Deadline 2026-11-12 17:05 UTC (about 92 h); freeze 2026-11-11 08:00 UTC. The previous long status page is archived at [`v2/archive/STATUS_2026-11-08_0624.md`](v2/archive/STATUS_2026-11-08_0624.md).

Numbers are copied from committed result and report files; nothing was re-run for this page. "LOCKED" means scored once on our own sealed exam (`v2/exam`, SKILL7: 2,744 items), not the maintainer's exam. Other entrants are "entrant #3" / "entrant #4"; their numbers are their own claims, unverified by us.

Companions: [`v2/WHAT_WE_TRIED.md`](v2/WHAT_WE_TRIED.md) (every lane, method, task, dead end), [`v2/ENTRY_DECISION.md`](v2/ENTRY_DECISION.md) (which entry to file), [`v2/CANDIDATES.md`](v2/CANDIDATES.md) (per-entry detail), [`v2/RULES.md`](v2/RULES.md) (rules and rulings), [`v2/SOLUTIONS.md`](v2/SOLUTIONS.md) (registry).

## 1. Where we stand

- The **filed entry** (issue #2, Qwen2.5-7B tool selection, 9.404%) does **not** pass: 93.9% retention (rule 2 needs 98%), letter answers (rule 7), feedback through a fitted head (rule 4, ruled by the maintainer). It is not refiled.
- Since then, on an easy 7-way voice-skill routing task with a shared sealed exam, **compression is no longer the blocker**: several entries keep 99%+ of base-correct items at under 10% of fp16, with a real generated label word.
- **Feedback is the wall.** The maintainer rejected a fitted head (#2) and a nearest-centroid row overwrite (#3). Every feedback carrier we have is a closed-form edit of existing tensors (no head, adapter, lookup or prompt; new label emitted by the model's own forward pass). Whether that counts is **unruled**, so rule 4 is gray for every candidate, and rules 5-6 depend on it.
- Starting from an already-compressed release (L1, L2) is **maybe valid and unruled**; it is not settled.
- No entry is a clean pass. Recommended primary: **L9b**; alternates L1, L2. If the closed-form-edit ruling goes against us we file nothing as a pass ([`ENTRY_DECISION.md`](v2/ENTRY_DECISION.md)).

## 2. Candidate table (LOCKED SKILL7 unless marked; feedback = new label `Recipe`, 25 annotations, 164 locked teach items)

| id | start model | our compression? | bytes (measured unless marked) | retention [Wilson lo]; worst kind | teach / shuffled / empty; neighbours | weakest rules and rulings |
|---|---|---|---|---|---|---|
| **L9b** (primary) | Qwen3.5-35B-A3B | yes: 102/256 experts + 3-bit RTN | 9.674% (9.704% with carrier) | 99.55 [99.21]; Lists 98.4; 3 draws 99.47-99.55 | .73-.79 / .02-**.47** / 0; 98.0-98.6 vs 98.3 empty | R6 variance (1 of 3 draws), R8 headroom (base .9676), R9 (60% experts pruned for the task), own-data risk; rulings A, C, G |
| **L2** | Ternary-Bonsai-2-27B (released ternary) | yes: scale coding + vocab truncation | 9.943% (slack 30.6 MB) | 100.0 [99.86] vs released model; 99.32 [98.93] vs fp16 ancestor; all kinds 100 | .827 (CLI .872) / 0 / 0; 98.3-99.2 | R4 gray, R9, tiny slack, weak shuffled arm, 96 GB to run; rulings A, B, G |
| **L1** | Bonsai-8B (released 1-bit) | no (lossless repack) | 7.44% (7.13% + 50 MB patch) | 1.000 [.9985] vs released model (by construction); **88.9% vs Qwen3-8B ancestor** | .893 / .005 / 0; .995 | R9 (no compression by us), R4/G (50 MB patch); rulings A, B |
| L3 | Qwen3-30B-A3B-Instruct-2507 | yes: 85/128 experts + 2-bit VQ | 9.896% (packed 9.676% + 134 MB `o_proj` edit) | 99.32 [98.92]; Calendar 96.69 | .799 / 0 / 0; SKILL7 neighbours not measured on final carrier. BANK6: retention 99.24, teach **.16** (R5 fail) | BANK6 R5; per-cell minima 96.7; rulings A, C |
| L4 | Qwen3-30B-A3B-Instruct-2507 | yes: 80/128 experts + GPTQ 2-bit | 9.895% (estimate, no packed file) | 98.94 [98.47]; News 96.03 | .59-.63 / .09-**.49** / 0; .989-.994. BANK6 teach .07-.27, neighbours collapse in 2 of 3 draws | R5, R6; examples in prompt (R4/R9); no packed file |
| L9 | Qwen3-235B-A22B-Instruct-2507 | yes: 88/128 experts + 2-bit RTN | 9.7814% (9.896% with 537 MB carrier) | 98.98 [98.52]; Calendar 97.7 | .744 / .122 / .024; 99.16 | needs about 330 GB GPU memory; one draw; impractical to ship |
| L10 | Qwen2.5-Omni-7B text path | yes | 9.879% of the whole multimodal fp16 (9.930% with feedback; accounting only, no packed file) | 99.55 [99.21]; News 98.78 | .860 / 0 / 0; 99.15 unchanged (one draw) | R1 literal vs R9 spirit (unused towers inflate denominator); ruling D |
| L5 | Qwen2.5-32B dense | yes: scalar GPTQ + rANS | 9.697% (9.889% with carrier; loader not counted) | **pending**: dev .9873 [.9840], worst dev cell .961 | teacher-forced .671 / 0 / 0; 1.000 | exam not scored |
| L12 | Qwen3-30B-A3B, Qwen3.5-35B-A3B | research ablation, not an entry | pre-quant fractions | pending (`run_final.py` not run) | pending | not an entry |

Failed lanes (dev, kept for the record): L6 Qwen1.5-MoE 85.9% (R2), L7 BitNet-2B 95.8% (R2), L8 depth/width about 0%, X03 14B 95.44% unsealed once (R2), X04a 32B 96.5% (killed), dense 72B/32B/405B 90-96.8% (R2). Detail in [`v2/WHAT_WE_TRIED.md`](v2/WHAT_WE_TRIED.md).

## 3. What the maintainer has said (short)

| where | statement | consequence |
|---|---|---|
| issue #1 | size on disk against fp16-equivalent is fine; no weight retraining; no RAG or context editing; examples: agentic/tool use, PII redaction, AI-vs-human detection | size basis; feedback must not be in the prompt |
| issue #2 (ours) | steps 1-4 fine; step 5 fails: feedback only works through a fitted head, "without any external memory/parameters" is the objective | a head or side memory is out |
| issue #3 (entrant #3) | the 88.5% is a nearest-centroid classifier in numpy, the grown model was never run on the new items, the overwritten rows are "the same head as before, just kept inside the model file"; the compression side is "genuinely nice"; welcome to retry before the deadline | the new class must be generated end to end by the shipped system; few-row output edits are high risk; resubmission is allowed |
| issue #4 (entrant #4) | no reply known to us | its verdict is the best signal for closed-form edits and pruning |

## 4. Open rulings that decide the outcome (none put to the maintainer)

A closed-form edit of existing tensors as feedback; B already-compressed start model; C task-calibrated expert pruning; D denominator for multimodal start models; E the `seen_tox` tier and pooled per-kind counts; F how much worse "clearly worse" must be; G fp16 patch or low-rank delta as "new part"; H is 7-way routing commercial and not narrow; I what "best" means. Detail in [`v2/ENTRY_DECISION.md`](v2/ENTRY_DECISION.md) section 5.

## 5. Next steps

1. L9b: own-data dry run, more feedback draws, ship the missing files, publish the artifact with hashes, fresh-clone test ([`ENTRY_DECISION.md`](v2/ENTRY_DECISION.md) section 3).
2. L5 and L12: land the exam score and the final run, or drop them.
3. Re-check entrant #4's issue read-only for a ruling.
4. Freeze 2026-11-11 08:00 UTC; the user decides what, if anything, is filed. Nothing is posted without explicit go-ahead.

## 6. Honest estimate

No candidate is a clean pass today. If the closed-form-edit question is answered yes and the maintainer's own-data check is kind, L9b, L2 or L1 pass the measurable rules on our exam, with R6, R8 and R9 margins thin (L9b) or start-model questions open (L1, L2). If it is answered no, nothing we have passes rule 4/5, and we file nothing as a pass. Our earlier planning documents put the chance that any entry passes all nine rules as the maintainer reads them at roughly 5%; the lane results since then raise the compression side but not the feedback ruling, so we do not revise that number upward without a ruling.
