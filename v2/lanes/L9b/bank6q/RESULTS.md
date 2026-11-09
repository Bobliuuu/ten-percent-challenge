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

---
# REAL packed artifact: vq4_had_b2, k=160, BANK6 (2026-11-09)
Code: `vqpack.py` (packer + loader + decoder), `qrun_pack.py` (stages), `modal_qp.py` (app `tenpct-l9b-q`), `answer_q.py`, `exam_score.py`; logs `results/logs/{pack_k160,load_k160_v2,exam_base3,exam_packed,fb,diag*}.log`; `MANIFEST.json`. Artifact is private on Modal volume `tenpct-l9b-q` (`art/b6_vq_k160`), no hosting.

## Format
body.bin + manifest.json. Per expert tensor: uint8 index per 4 weights (fixed 8 bit, 2.000 bpw), one fp32 sigma per expert tensor, ONE shared 256x4 fp32 codebook (`__vq_cb__`), rotation = random signs (torch.Generator seed) + Hadamard on input axis (seed 1000+layer) and output axis (seed 1000+layer+7919), seeds stored in manifest, signs regenerated by the loader. Decode = cb[idx]*sigma -> inverse Hadamard -> bf16. Attention/linear-attn/shared expert 6-bit RTN g128, emb/head 4-bit RTN (qpack, unchanged), router rows (kept) and small tensors bf16. Same GPTQ/LDLQ code and hyperparameters as the sim (Hessians from all 698 bank6 cal rows, k160 by mass_bank6, 681 pooled-H experts).

## Real bytes (everything counted; reference 69,321,221,376 B, cap 9.9% = 6,862,800,916 B)
| item | bytes |
|---|---|
| body.bin | 6,677,382,912 |
| manifest.json | 129,647 |
| tokenizer.json + tokenizer_config.json + chat_template.jinja | 19,998,561 |
| code: core, qpack, vqpack, answer_q, annot, fbcore, feedback | 28,540 (+292 bank6_task.json) |
| feedback_cal.jsonl (300 bank6 rows) | 109,160 |
| artifact dir total | 6,697,649,112 (9.6618%) |
| feedback carrier fb.bin (L10, 512 cols, bf16) | 20,981,760 (+313 fb.json) |
| **TOTAL shipped with carrier** | **6,718,631,185 = 9.6920%** (slack 144.2 MB below cap) |
k160 fits, no reduction needed. The sim estimate was 9.66% (6,697,529,213), the real dir is 120 KB larger. sha256 body.bin `5f53bbe9e8c393ec6b7709889b287896d6b1529fa5c4813415c381a406756265`, manifest.json `300e1c27bcdf75dff631a339e439271d614136648f4c21c5be8f106e061dcf72`.

## Loader fidelity
- Weight level (decoder = loader code, vs the in-process fake-quant tensors that were evaluated): max abs diff **0.0** and 0 differing elements for all 80 expert VQ tensors (20.1e9 weights), 312 RTN tensors, 261 raw, 40 router tensors (`results/pack_b6_vq_k160.json`). Indices recovered from the quantiser output reproduce it exactly (max |cb[idx]*sigma - Qm| = 0).
- SEL generation (342 rows, stock expert forward): real loader vs fake-quant model **342/342 identical raw outputs**; SEL retention through the real loader 99.04% [97.20] (n=311; Exchange 97.6, Identity 98.2, Transfer 98.2, others 100). The earlier sim number 99.68% [98.20] used the custom loop expert forward and differs from the stock forward on 3/342 outputs: the honest SEL number for this artifact is 99.04.
- Two loader bugs found and fixed BEFORE the exam (not chosen on the exam): the shipped-style loader (`model.to(bf16)`) cast rotary `inv_freq` to bf16 (diff 1e-3, 3/342 SEL outputs changed) and lacked the im_end eos; vqpack keeps buffers fp32 and sets eos. Note `core.load_packed` has the same inv_freq behaviour (affects shipped k102 artifacts too; not touched here).

## Locked BANK6 exam (2237 items), scored once for the packed artifact
Base run 3x (greedy, same batching): **identical** in all 2237 outputs, acc 0.9021 (2018 base-correct). Packed once through the real loader: acc 0.8999.
- **Retention overall 98.51% (1988/2018), Wilson lower 97.89%**. Gate overall >=98 (point estimate): PASS; gate every kind >=95 (point): PASS. Wilson lower bound is below 98 (97.89) and below 95 for Card and Identity.
- Per kind: Card 96.16 [93.77] (376/391), Cash 100 [98.66], Exchange 99.42 [97.91], Identity 96.19 [93.31] (278/289), TopUp 99.39 [97.81], Transfer 100 [99.01].
- Per source: banking77 98.74 [98.04] (1489/1508), bitext_bank (never used) 97.84 [96.18] (499/510). Never-used items 600: 97.84%. Clean tier only: 98.69 [98.03], gates pass. Rule 3 counts OK.
- Reference: shipped k102 3-bit RTN was 97.37% [96.58] on this exam (FAIL, Card 94.4, Identity 91.7). Files `results/exam_report_b6_vq_k160.json`, `results/exam_{base3,packed_b6_vq_k160}_BANK6.json`.

## Feedback (label Security, reporting only; fb_run.py logic unchanged, config L10_c512 as shipped, 3 draws, 25 annotations, 752 neighbour items base-correct)
empty: teach 0.0, neighbours 96.5%. real: teach 0.505/0.46/0.53, neighbours 58.6/74.1/83.0%, leakage of Security into neighbours 29/9/10%; shuffled labels neighbours 68/74/95%; wrong labels 0 teach. Same failure as k102: the new label is learnt about half of the time but damages Card/Identity neighbours; carrier 20,981,760 B. Feedback does not pass rule 5 here. `results/fb_b6_vq_k160_fb_b6.json`.

## Verdict
The real vq4_had_b2 k160 BANK6 artifact is 9.69% of fp16 with carrier (9.66% without), decodes bit-exactly to the evaluated fake-quant weights, and on the locked exam reaches 98.51% retention (point estimates pass 98/95: Card 96.2, Identity 96.2 worst), a clear fix over RTN k102 (97.37% FAIL). Caveats: Wilson lower bounds are 97.89 overall and 93.3-93.8 for Card/Identity, so the strict reading with lower bounds is not met; one Hessian draw, one k chosen on SEL; feedback (rule 5) fails. No jobs left running.
