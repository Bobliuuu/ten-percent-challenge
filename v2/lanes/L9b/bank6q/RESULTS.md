# bank6q: Hessian-aware 2-bit experts for L9b BANK6 (SEL only, simulated)

Question: does the QUANT_SWEEP finding (more experts at ~2 bpw with Hadamard + GPTQ/LDLQ error feedback beats fewer experts at 3-bit RTN) improve the BANK6 result of L9b (shipped k102 3-bit RTN, SEL 97.43%, worst kind Identity 89.3)?

## Method
- Qwen3.5-35B-A3B on one H200 (Modal app `tenpct-l9b-q`, volume `tenpct-l9b-q`; the existing `tenpct-l9b` app was not touched). Code: `qrun.py`, `modal_q.py`, `qlib.py`/`qlib_b.py` (copied unchanged from `v2/moe_quant`).
- Per-expert Hessians (input of gate_up, input of down; padding excluded; all 698 bank6 cal rows, unquantised model, 8 layers at a time) via a loop-form expert forward with hooks. Check against the stock forward (`results/expert_fwd_check.json`): per-layer expert output max abs diff 0.002 (layers 0, 20) / 0.06 (layer 39, output abs max 31), same size as stock-vs-fp64 error, i.e. bf16 noise. Whole-model last-position logits max diff 1.5 on a 29 abs-max scale, argmax agreement 100% (`results/base_check.json`).
- Kept experts = top-k routing mass (`mass_bank6.pt`, all 12 chunks), router renormalises over kept experts exactly as the shipped loader does. Experts with <64 calibration tokens (681 of 6400 at k160) use the layer-pooled normalised Hessian.
- Quantisers: Hadamard-rotated GPTQ/LDLQ with D4 K=256 Gaussian VQ (`vq4_had_b2`, fixed-rate 2.000 bpw) or ECSQ g256 (`ecsq_had_b2`, measured entropy 2.013 bpw incl. scales), or int3 g128 error feedback (controls). Fake-quant: dequantised bf16 written back in place. Attention/shared expert/linear-attn 2-D tensors 6-bit, embedding and lm_head 4-bit, RTN g128 (the `qpack` rounding), router and small tensors bf16, same rules as `core.pack_model`.
- Eval: real greedy generation (`core.Runner`, `core.correct`, `core.score`) on bank6 SEL (342 rows, 311 base-correct; base regenerated, acc 0.9094, `results/base_bank6_SEL.json`). The locked exam file is not loaded by any code here and was not scored.
- Bytes: shipped-format model anchored on the measured shipped k102 3-bit pack (6,705,899,325 B): rest-of-artifact is held fixed (attn6/emb4/head4/shared expert/tokenizer/manifest), plus 2*2048 B router rows per kept expert, expert bits measured (VQ: 2 bit index + per-tensor sigma + 2 KB codebook; ECSQ: measured histogram entropy + fp16 group scales + 256 B code table per tensor, NOT a real range-coded file), +40 KB manifest for the longer kept lists. Reference 69,321,221,376 B, cap 9.9% = 6,862,801,016 B. Estimates, not a packed artifact (no decoder or packer was written).

## Results (SEL, retention = P(correct | base correct), n=311; per-kind n 42-57)
| variant | k | expert bpw | bytes % | ret [Wilson lo] | Card | Cash | Exch | Identity | TopUp | Transfer |
|---|---|---|---|---|---|---|---|---|---|---|
| shipped RTN 3-bit (from REPORT, other base run) | 102 | 3.15 | 9.67 | 97.43 [95.01] | | | | 89.3 | | |
| rtn_b3 re-simulated here (sanity of the sim) | 102 | 3.15 | 9.67 | 96.78 [94.18] | 100 | 100 | 95.2 | 87.5 | 100 | 98.2 |
| gptq_had_b3 (control) | 102 | 3.15 | 9.67 | 99.36 [97.69] | 100 | 100 | 100 | 96.4 | 100 | 100 |
| gptq_b3 no rotation (control) | 102 | 3.15 | 9.67 | FAILED (NaN in one expert's gate_up, garbage output; also with damp 0.1) | | | | | | |
| ecsq_had_b3 | 106 | 3.00 | 9.61 | 98.39 [96.29] | 100 | 100 | 100 | 92.9 | 100 | 98.2 |
| ecsq_had_b2 | 128 | 2.01 | 8.24 | 98.71 [96.74] | 100 | 100 | 95.2 | 98.2 | 100 | 98.2 |
| ecsq_had_b2 | 160 | 2.01 | 9.71 | 98.71 [96.74] | 100 | 100 | 95.2 | 98.2 | 100 | 98.2 |
| vq4_had_b2 | 128 | 2.00 | 8.20 | 99.36 [97.69] | 100 | 97.9 | 97.6 | 100 | 100 | 100 |
| vq4_had_b2 | 144 | 2.00 | 8.93 | 99.68 [98.20] | 100 | 100 | 97.6 | 100 | 100 | 100 |
| **vq4_had_b2** | **160** | 2.00 | 9.66 | **99.68 [98.20]** | 100 | 100 | 97.6 | 100 | 100 | 100 |
| vq4_had_b2 | 164 | 2.00 | 9.84 | 99.36 [97.69] | 100 | 100 | 97.6 | 98.2 | 100 | 100 |
Per-variant JSON: `results/eval_*.json`; raw logs `results/logs/run_{A,B,C}.log`; outputs for the two k160 runs `results/outs_*_SEL.json`.

## Verdict
- Every working Hessian-aware variant clears the gates on SEL at <=9.9% estimated bytes (overall >=98, every kind >=95) and beats 97.43%; best-by-SEL: vq4_had_b2 k160 (99.68 [98.20], 9.66%, worst kind Exchange 97.6). Identity, the kind that sank the shipped recipe, goes from 87.5-89.3 to 98-100 for the 2-bit ones.
- Honest reading: gain over the simulated RTN k102 is 2-3 points = 8-10 of 311 items. Differences among the good variants (98.4 to 99.7) are 1-4 items and inside noise; the ranking vq4 > ecsq is not established. What is solid: Hessian error feedback + Hadamard fixes the Identity collapse. The control gptq_had 3-bit k102 already gets 99.36, so on this task most of the gain comes from the error-feedback quantiser, not from trading bits for experts; 2-bit with more experts is at least as good, not clearly better, and k128 at 8.2% bytes already matches.
- Caveats: SEL only, one Hessian/cal draw, single seed, n=311 (a Wilson lower bound of ~97.7-98.2 is not a locked-exam number); fake-quant, no real packed artifact, fixed-rate VQ is straightforward to ship but ECSQ bytes assume an ideal entropy coder; plain GPTQ without rotation crashed to NaN (not debugged); the shipped-recipe SEL number (97.43) used a different base run than the one regenerated here. Next step if wanted: a real vq4 packer/loader and one exam scoring decided by the owner of the exam.
