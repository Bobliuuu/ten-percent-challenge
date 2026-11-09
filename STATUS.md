# STATUS: 10% challenge

**Last updated: 2026-11-09.** Deadline 2026-11-12 17:05 UTC; freeze 2026-11-11 08:00 UTC. The previous long status page is archived at [`v2/archive/STATUS_2026-11-08_0624.md`](v2/archive/STATUS_2026-11-08_0624.md).

Numbers are copied from committed result and report files; nothing was re-run for this page. "LOCKED" means scored once on our own sealed exam (`v2/exam`, SKILL7: 2,744 items), not the maintainer's exam. Other entrants are "entrant #3" / "entrant #4"; their numbers are their own claims, unverified by us.

Companions: [`v2/WHAT_WE_TRIED.md`](v2/WHAT_WE_TRIED.md) (every lane, method, task, dead end), [`v2/ENTRY_DECISION.md`](v2/ENTRY_DECISION.md) (which entry to file), [`v2/CANDIDATES.md`](v2/CANDIDATES.md) (per-entry detail), [`v2/RULES.md`](v2/RULES.md) (rules and rulings), [`v2/SOLUTIONS.md`](v2/SOLUTIONS.md) (registry).

## 1. Where we stand

- The **filed entry** (issue #2, Qwen2.5-7B tool selection, 9.404%) does **not** pass: 93.9% retention (rule 2 needs 98%), letter answers (rule 7), feedback through a fitted head (rule 4, ruled by the maintainer). It is not refiled.
- Since then, on an easy 7-way voice-skill routing task with a shared sealed exam, **compression is no longer the blocker**: several entries keep 99%+ of base-correct items at under 10% of fp16, with a real generated label word.
- **Feedback is the wall.** The maintainer rejected a fitted head (#2) and a nearest-centroid row overwrite (#3). Every feedback carrier we have is a closed-form edit of existing tensors (no head, adapter, lookup or prompt; new label emitted by the model's own forward pass). Whether that counts is **unruled**, so rule 4 is gray for every candidate, and rules 5-6 depend on it.
- Starting from an already-compressed release (L1, L2) is **maybe valid and unruled**; it is not settled.
- No entry is a clean pass. Recommended primary: **L9b** (re-evaluated 2026-11-09, still first); alternates L1, then L5 (new: dense own compression with clean controls, independent of rulings B and C), then L2 (demoted: its proper shuffled control leaks). If the closed-form-edit ruling goes against us we file nothing as a pass ([`ENTRY_DECISION.md`](v2/ENTRY_DECISION.md)).

## 2. Candidate table (LOCKED SKILL7 unless marked; feedback = new label `Recipe`, 25 annotations, 164 locked teach items)

| id | start model | our compression? | bytes (measured unless marked) | retention [Wilson lo]; worst kind | teach / shuffled / empty; neighbours | weakest rules and rulings |
|---|---|---|---|---|---|---|
| **L9b** (primary) | Qwen3.5-35B-A3B | yes: 102/256 experts + 3-bit RTN | 9.674% (9.704% with carrier); body rebuilt, sha256 in MANIFEST | 99.55 [99.21]; Lists 98.4; 3 draws 99.47-99.55 | .73-.79 (8-draw mean .74-.76) / shuffled mean .16-.18, up to .40-.47 on bad draws, gap about .58 / 0; 98.0-98.6 vs 98.3 empty | R6 variance, R8 headroom (base .9676), R9 (60% of experts pruned for the task), own-data risk; BANK6 fails with RTN (97.37, Card 94.4); rulings A, C, G |
| **L1** | Bonsai-8B (released 1-bit) | no (lossless repack) | 7.44% (7.13% + 50 MB patch; L1+ low-rank patch 0.5 MB) | 1.000 [.9985] vs released model (by construction); **88.9% vs Qwen3-8B ancestor** | .893 (L1+ .905) / .005 / 0; .995. BANK6+`Security`: exam retention 1.000, teach .727, shuffled .10 (up to .49), neighbours .989 | R9 (no compression by us), R4/G (patch); BANK6 teach weak; rulings A, B |
| **L5** | Qwen2.5-32B dense | yes: scalar GPTQ+Hadamard+rANS, no pruning | packed body 9.797%; **9.926%** with 83.9 MB carrier and code; no single-file loader | **98.27 [97.70]**; News 96.4; hwu64 x News cell 89.1 (n 64), hwu64 x Weather 91.9 (n 37) | .532 (real gen .52-.54) / **0 / 0 / 0**; neighbours 1.000. BANK6 FAILS: .9696, Card .871 | thin R2 margin, modest teach, 0.03 pt over the internal 9.9% target, R4 gray, no loader; ruling A only (independent of B, C) |
| **L2** | Ternary-Bonsai-2-27B (released ternary) | yes: scale coding + vocab truncation | 9.943% (slack 30.6 MB) | 100.0 [99.86] vs released model; 99.32 [98.93] vs fp16 ancestor; all kinds 100 | default m64: .973 / **shuffled-among-labels .756** / 0; neighbours .971. m16: .913 / .352 / .991 neighbours | R6 (proper shuffled arm leaks), R4 gray, R9, tiny slack, 96 GB to run; rulings A, B, G |
| L3 | Qwen3-30B-A3B-Instruct-2507 | yes: 85/128 experts + 2-bit VQ | 9.896% (packed 9.676% + 134 MB `o_proj` edit) | 99.32 [98.92]; Calendar 96.69. BANK6 99.24 [98.76] | .799 / 0 / 0; exam retention after feedback 99.20; SKILL7 neighbours not measured on final carrier. BANK6 teach **.16** (R5 fail) | BANK6 R5; per-cell minima 96.7; rulings A, C |
| L4 | Qwen3-30B-A3B-Instruct-2507 | yes: 80/128 experts + GPTQ 2-bit | 9.895% (estimate, no packed file) | 98.94 [98.47]; News 96.03. BANK6 99.24 | .585-.634 / .085-**.494** / 0; .989-.994. BANK6 teach .07-.27, neighbours collapse in 2 of 3 draws | R5, R6; examples in prompt (R4/R9); no packed file |
| L9 | Qwen3-235B-A22B-Instruct-2507 | yes: 88/128 experts + 2-bit RTN | 9.7814% (9.896% with 537 MB carrier) | 98.98 [98.52]; Calendar 97.7 | .744 / .122 / .024; 99.16 (one draw) | needs about 330 GB GPU memory; one draw; impractical to ship |
| L10 | Qwen2.5-Omni-7B text path | yes | 9.879% of the whole multimodal fp16 (9.930% with feedback; accounting only, no packed file) | 99.55 [99.21]; News 98.78 | .860 / 0 / 0; 99.15 unchanged (one draw) | R1 literal vs R9 spirit; ruling D |
| L12 | Qwen3.5-35B-A3B (and Qwen3-30B-A3B) | research ablation: mass k25 of 256 experts + RTN4 everything | 7.61% analytic (fake-quant, no packer) | 99.17 [98.75]; min kind .972 | **0 / 0 / 0** (routed-expert `down_proj` edit does not teach on Qwen3.5); neighbours .963-.977; o_proj rerun not landed | feedback ineffective as built; no packed file; not an entry |

Failed lanes (dev, kept for the record): L6 Qwen1.5-MoE 85.9% (R2), L7 BitNet-2B 95.8% (R2), L8 depth/width about 0%, X03 14B 95.44% unsealed once (R2), X04a 32B 96.5% (killed), dense 72B/32B/405B 90-96.8% (R2). Detail in [`v2/WHAT_WE_TRIED.md`](v2/WHAT_WE_TRIED.md).

## 3. What the maintainer has said (short)

| where | statement | consequence |
|---|---|---|
| issue #1 | size on disk against fp16-equivalent is fine; no weight retraining; no RAG or context editing; examples: agentic/tool use, PII redaction, AI-vs-human detection | size basis; feedback must not be in the prompt |
| issue #2 (ours) | steps 1-4 fine; step 5 fails: feedback only works through a fitted head, "without any external memory/parameters" is the objective | a head or side memory is out |
| issue #3 (entrant #3) | the 88.5% is a nearest-centroid classifier in numpy, the grown model was never run on the new items, the overwritten rows are "the same head as before, just kept inside the model file"; the compression side is "genuinely nice"; welcome to retry before the deadline | the new class must be generated end to end by the shipped system; few-row output edits are high risk; resubmission is allowed |
| issue #4 (entrant #4) | no reply known to us | its verdict is the best signal for closed-form edits and pruning |

## 4. Unruled readings that decide the outcome (guessed readings; each is our best-evidence reading, not settled)

A closed-form edit of existing tensors as feedback; B already-compressed start model; C task-calibrated expert pruning; D denominator for multimodal start models; E the `seen_tox` tier and pooled per-kind counts; F how much worse "clearly worse" must be; G fp16 patch or low-rank delta as "new part"; H is 7-way routing commercial and not narrow; I what "best" means. Detail in [`v2/ENTRY_DECISION.md`](v2/ENTRY_DECISION.md) section 5.

## 5. Next steps

1. L9b: own-data dry run, publish a durable copy of the rebuilt artifact (hash recorded, content identity argued), fresh-clone test ([`ENTRY_DECISION.md`](v2/ENTRY_DECISION.md) section 3). More feedback draws are done (8) and did not fix the shuffled leak.
2. L5: a single-file packed loader and a standalone feedback command if it is to be a fallback. L12: only if a packer and a feedback carrier that teaches (o_proj rerun, or the L9b shared-expert carrier) land before the freeze; otherwise drop. L9b BANK6: needs a real packer for the Hadamard + error-feedback recipe.
3. Re-check entrant #4's issue read-only for a ruling.
4. Ops: the AWS `lab` session has expired and the AWS box sweep could not run; the user must run `aws login --profile lab`, then re-sweep eu-north-1, us-east-1, us-east-2 ([`BOX_SWEEP_2026-11-08.md`](v2/research/E_infra/BOX_SWEEP_2026-11-08.md)). No tenpct compute is running on Vast or Modal except the `tenpct-l9b` Modal app and one stopped Vast box.
5. Freeze 2026-11-11 08:00 UTC; the user decides what, if anything, is filed. Nothing is posted without explicit go-ahead.

## 6. Honest estimate

No candidate is a clean pass today. If the closed-form-edit question is answered yes and the maintainer's own-data check is kind, L9b, L2 or L1 pass the measurable rules on our exam, with R6, R8 and R9 margins thin (L9b) or start-model questions open (L1, L2). If it is answered no, nothing we have passes rule 4/5, and we file nothing as a pass. Byte tricks (emb/head 2-4-bit, rANS indices, coded scales, g64 with the freed bytes; [`v2/BYTE_TRICKS.md`](v2/BYTE_TRICKS.md)) free several points of budget and are partly used by L5, L9b and L2. Closed-form edits stay GRAY (R4) and pre-compressed starts stay MAYBE-VALID/unruled; nothing above claims a failing rule as a pass. Our earlier planning documents put the chance that any entry passes all nine rules as the maintainer reads them at roughly 5%; the lane results since then raise the compression side but not the feedback ruling, so we do not revise that number upward without a ruling.
