# L14: broad-calibration expert pruning + Hadamard/GPTQ-VQ experts (Qwen3.5-35B-A3B) (2026-11-10)

Verdict: **gate 2 (own-data dry run) FAILS. Stopped there: no locked exam, no feedback carrier, no real packing.** Nothing was posted. All numbers are fake-quant (dequantised in place, stock expert forward), base = unmodified Qwen3.5-35B-A3B bf16; code `l14_run.py` (fork of `L9b/bank6q/qrun.py`), data `build_cal.py`, jobs `jobs/`, raw `results/` (`SUMMARY.txt`, per-variant `eval_*.json` incl. all outputs), plan with pre-registered gates `PLAN.md`.

Recipe: routing mass (experts kept) AND GPTQ Hessians from broad unlabeled calibration = SKILL7 task prompts (2031) + dolly chat/QA 696 + gsm8k 400 + CodeAlpaca 400 + wikitext 400 (prompt + response tokens); Hadamard + error-feedback D4 VQ experts; attention 6-bit, embedding/head 4-bit; bytes from the shipped-format model of bank6q (real packer not run).

Pre-registered gates: (1) SEL retention >= 98; (2) Banking77-200 retention >= 95, router agreement on banking >= 85, generic prompts identical (24 tokens) >= 70, (Timer only if those pass); (3) locked exam once.

## Table (SEL = SKILL7 dev selection split, 1802 items; own-data = same prompts as L1/L9b/L12 dry runs)
| variant | keep | expert bpw | bytes % | SEL ret | bank77 ret | router agree | generic ident |
|---|---|---|---|---|---|---|---|
| reference L1 (released 1-bit start) | - | - | 7.44 | - | 100 | 94 | 96 |
| reference L9b k102 RTN3 task-only | 40% | 3.15 | 9.67 | 99.48 | 79.3 | 46 | 11 |
| **t0** task-only cal, VQ2 had | 62.5% | 2.0 | 9.66 | 99.19 | 89.9 | 69 | 28 |
| **b0** broad cal, VQ2 had | 62.5% | 2.0 | 9.66 | 99.30 | 88.4 | 67 | 37 |
| b0 broad, GPTQ-had 3-bit | 40% | 3.15 | 9.67 | 99.30 | 80.4 | 60 | 19 |
| b1 broad, VQ D4K128 | 71.9% | 1.75 | 9.71 | 99.65 | 89.9 | 65 | 37 |
| b1 broad, VQ D4K64 | 80% | 1.5 | 9.39 | 99.48 | 88.4 | 59 | 36 |
| l0 broad VQ2 + head 6-bit | 62.5% | 2.0 | 9.85 | 99.36 | 90.6 | 71 | 39 |
| l0 broad VQ2 k150, head 6, attn 8 | 58.6% | 2.0 | 9.90 | 99.07 | 89.1 | 66 | 36 |
Diagnostics (bf16 experts, bytes NOT valid, isolate the cause):
| diagnostic | bank77 ret | router agree | generic ident |
|---|---|---|---|
| prune only, broad cal, k160 (62.5%) | 93.5 | 72 | 42 |
| prune only, task-only cal, k160 | 93.5 | 83 | 33 |
| prune only broad k184 / k205 / k230 (72/80/90%) | 91.3 / 96.4 / 100 | 85 / 86 / 93 | 51 / 57 / 75 |
| NO pruning, only attn6/emb4/head4 quantised | 97.8 | 80 | 71 |

## Findings
1. Gate 1 passes everywhere (99.1-99.65), so SKILL7 dev retention says nothing about general behaviour.
2. Gate 2 fails for every byte-valid variant: best is 90.6 / 71 / 39 (L1: 100/94/96). It is much better than L9b (79/46/11), but the gain comes from the Hadamard + error-feedback VQ quantiser (task-only t0 already 89.9/69/28), not from the broad calibration. Broad calibration moves generic-identical 28 -> 37 and router agreement/banking retention by nothing (within noise: 200 items, one calibration draw). Its effect on the pruning alone is mixed (router agree 83 -> 72, generic 33 -> 42).
3. The cause is the pruning fraction, and the byte budget does not allow keeping more: prune-only own-data approaches the base only at ~90% of experts (100/93/75), which needs about 1.4 bpw at 10%; 1.5 bpw VQ at 80% keep (b1) did not improve (88/59/36), i.e. quantisation noise then dominates. Even with zero pruning and expert weights untouched, 6/4/4-bit non-experts alone give 97.8/80/71, so the pre-registered generic >= 70 / router >= 85 gates are at the noise floor of ANY lossy compression of this model; only a start model that is bit-identical outside its task (L1's no-feedback arm) can score 96-100. Our own compression to 10% cannot be base-like on this metric on this model. Our best-evidence reading: the divergences are mostly paraphrases (French translations, 2541 vs 2551 style number drifts, wording), with some wrong arithmetic; a judge testing generic prompts will see small differences, not garbage like L9b.
4. Not tried: Qwen3-30B-A3B (same pruning-fraction budget: L3 keeps 66% at 2 bpw, so the same wall), real packer, Timer arm, feedback (moot after the gate 2 fail).

## Ops
Modal H200 queued >15 min for the later jobs, so the d1/d4/d5 diagnostics ran on two Vast H200 boxes (destroyed, verified none left with label tenpct-l14); Modal app `tenpct-l14` stopped; an earlier d0 job's k256 variant crashed (GPU memory), fixed by keeping original experts on CPU. Note: concurrent jobs each recomputed the base outputs (shared file); base SEL acc 0.954 throughout. Launches in RUNS.md.

## Recommendation
Does not change ENTRY_DECISION: L1 stays primary on own-data. If L9b/own-compression is filed, the L14 recipe (VQ2 Hadamard, 62.5% experts) is a strict improvement on L9b's own-data numbers (89.9/69/28 vs 79/46/11) at the same bytes, with SEL 99.2-99.3, but still fails the strict own-data bar; it would need a real packer (bank6q `vqpack.py` exists) and an exam scoring.
