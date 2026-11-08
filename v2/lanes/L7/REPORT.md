# L7: BitNet-b1.58-2B-4T (released low-bit start) on SKILL7 voice-skill routing, 2026-11-08 (DEV only)
Start model = released microsoft/bitnet-b1.58-2B-4T; fp16 reference 4,825,641,380 B (2 x params). Package: `l7lib.py` (compression + edit), `answer.py`, `feedback.py`, `run_l7.py`, `MANIFEST.json`, results in `results/` and `v2/near_miss/e_native/results/bitnet2b/`.

## Numbers (dev; real greedy generation; vs the released model itself)
- Size 9.899% (MANIFEST; ternary body counted as ideal-coder bytes, 10.396% if written 5 trits/byte). Config: vocab prefix 74,167 (+specials + tokens of the fixed task prompt) + 3-bit RTN embedding.
- SKILL7 base accuracy .890 (dev). Retention of base-correct items at this config: 97.7% [96.2] (n=607); worst cell 91.5% (massive Music, n=47). Overall (T1/TOX/INJ/SKILL7 pooled) 95.8% [94.7].
- Neighbouring configs are brittle (4-bit/58k vocab: SKILL7 retention 10.9%), see v2/near_miss/e_native/ROWS.md.

## Feedback experiment (new skill Cooking; 25 annotations = 13 cooking + 12 old-label mistakes; 63 unseen teach items; 511 neighbours; compressed model, untied output rows edited)
| condition | teach acc (Cooking) | neighbour acc | neighbour retention vs empty |
|---|---|---|---|
| empty (no edit; label NAME in the task prompt) | **88.9%** | 89.0% | 100% |
| real closed-form head-row edit | 12.7% | 82.0% | 91.0% |
| shuffled labels (3 seeds) | 3.2 / 7.9 / 25.4% | 64.4 / 72.4 / 86.3% | 70.5 / 73.0 / 89.0% |
Finding (negative, honest): merely listing the new label name in the task definition already yields 88.9% on Cooking, so the teach set cannot show feedback learning (rule 5/6 FAIL: real is not better than empty; the edit as built is harmful). Real beats shuffled but both are far below empty. A fair test needs a new class whose name alone does not solve it. The edit (minimum-norm, margin 2) over-shoots; margins 1 and 4 give 19% / 9.5% teach.

## 9 rules (dev evidence)
R1 size PASS 9.899% (ideal-coder ternary; 10.40% with simple packing) | R2 FAIL (95.8% pooled, per-task SKILL7 97.7% but cells <98) | R3 UNTESTED (shared locked exam not scored; config not frozen worth scoring) | R4 GRAY (few-row output edit, same class as rulings 4a/4b) | R5 FAIL | R6 FAIL (real not clearly above empty) | R7 PASS (real label word generated) | R8 PASS for SKILL7/INJ, marginal T1 (.32) / TOX (.34 vs .20 chance) | R9 GRAY (low-bit released model as start).
**Weakest rule: R2 (retention under 98% with a brittle vocab trim), then R5/R6.**

## Cost / blockers
Modal A10G x ~3 (about $4-6) + ~$0.1 Vast; Vast hit its instance cap (could not launch boxes). Pending at writing: Falcon-E-3B-Instruct and the task-vocab BitNet re-run (Modal apps tenpct-near-miss-e). Shared locked exam NOT scored.
