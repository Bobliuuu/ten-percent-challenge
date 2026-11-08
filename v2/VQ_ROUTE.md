# VQ_ROUTE: training-free trellis / vector quantisation vs scalar GPTQ (2026-11-08). DEV DATA ONLY, no locked/S2 row touched.

Question: does a better training-free quantiser (QTIP-style bitshift trellis, QuIP#/AQLM-style vector codebooks, layer-wise calibration only) beat the entropy-coded scalar GPTQ by >=2 points at <=9.9% of fp16 and reach >=98% overall / >=95% per cell?
**Answer: NO-GO.** Best measured on Qwen2.5-32B (real generation, 4313 INJ-real dev items + 3500 NEWS dev items): TCQ 1.5 bpw (9.57%) INJ 0.968 [0.9615], NEWS 0.923 [0.913]. Scalar GPTQ on the same items at 9.86%: INJ 0.9615 [0.9545], NEWS 0.944 [0.935]. Differences are +0.6 (INJ) and -2.1 (NEWS), inside the +-2 pt draw noise; weakest cells (S3_injection 0.86, Technology 0.67) are 10-30 points below the 95% gate for every scheme.

## 1. What was built (own code, pure PyTorch, no third-party repo run)
- `v2/vq/tcq.py`: bitshift trellis-coded quantiser. State s in [0,2^L), step emits kb bits, next = ((s<<kb)|u) mod 2^L, V-dim code C[s] per state; rate kb/V bpw exactly. Viterbi via reshape+min (predecessors (ns>>kb)+j*2^(L-kb)); tail-biting by two passes (free start fixes the overlap bits). Packed-bit round trip (`pack_bits`/`unpack`) verified exact. Codebook C (2^L x V fp16 = 16 KB for L=12,V=2) is Lloyd-trained through the trellis on synthetic N(0,1) data only (no model weights; shipped as 16 KB, counted). Mixed rates kb in {2,3,4} with V=2 give 1.0/1.5/2.0 bpw per tensor.
- `v2/vq/tcq_run.py` (32B/7B pipeline; `vq8_run.py` = existing QuIP#-style 8-d k-means VQ re-run on the same items): random-sign + Kronecker/Hadamard incoherence both sides, sequential block-LDLQ (16x16 tiles = 256-weight trellis sequences, Hessian from quantised-prefix activations, no gradients), full-vocab embedding/head 2.125 bpw, all-bytes accounting (indices, sigma, codebook, embedding, head, norms, tokenizer/config 7.35 MB, code ~150 KB). Options tested: per-tile log scales (`gscale`), answer-token-weighted Hessian (`ans_w`), asymmetric fp-target (`tgt`, GPTAQ-like), task-only vs task+wikitext calibration, KL-sensitivity mixed-rate plan (`alloc_tcq.py`, from the measured 32B per-group KL table).
- Eval: `vreal_both.py` (vLLM greedy, strict INJ scorer; NEWS exact label string), `analyze.py` (retention of base-correct items per cell, Wilson 95%). Dev data: INJ 4313 items (`entry32b/dev_inj.jsonl`, 3377 base-correct), NEWS 3500 items (`task_search/NEWS/items_dev_ext.jsonl` = the original 1600 + 1900 from spare buckets AG[70,80), HuffPost[10,20); never locked/pool/calib buckets; 3029 base-correct). Calibration `calib_mix.jsonl` (700 INJ + 1000 NEWS calib items, from disjoint calib files) + 1M wikitext tokens.
- Compute: Vast H200 (1 box, destroyed after; 2nd box destroyed earlier), ~$70 total. Modal queue was blocked by the 10-GPU cap.

## 2. Gaussian-source sanity (`results/gauss_tcq.json`, `ecsq_ref.py`)
At 1.5 bpw (N(0,1) source, RD bound 9.03 dB): TCQ L=12,V=2 8.39 dB; L=14 8.41; L=10 8.25; L=8 8.01; V=4 8.30; L=16 8.37 (under-trained). Entropy-coded uniform scalar at the same entropy (1.51 bit): 7.55 dB. So TCQ is +0.85 dB (about 18% less noise power, ~0.14 bit) better than ECSQ on an ideal Gaussian source, 0.65 dB from the bound. This is the entire theoretical headroom (<=1.5 dB); the measured end-to-end gain is correspondingly small.

## 3. Results, Qwen2.5-32B-Instruct, real generation, retention of base-correct items (Wilson lower bound)
Base acc: INJ .783 (3377 correct), NEWS .865 (3029 correct). All 32B rows use embedding/head 2.125 bpw full vocab.

| scheme | size (all bytes, est) | INJ ret [lo] | INJ worst cells | NEWS ret [lo] | NEWS Technology |
|---|---|---|---|---|---|
| scalar ECSQ GPTQ g256 (dense_core, task-only calib, TF-argmax = greedy) | 9.86% | 0.9615 [0.9545] | S2_benign .77, S3_injection .90 | 0.944 [0.935] | .81 |
| VQ-D8 K=5300 (1.546 bpw, QuIP#-style, mixed calib) | 9.85% | 0.9597 [0.9526] | S3_inj .82, S2_inj .89 | 0.921 [0.911] | .74 |
| **TCQ L12 V2 kb3 (1.5 bpw, mixed calib)** | 9.57% | **0.9680 [0.9615]** | S3_inj .86, S2_inj .89 | 0.923 [0.913] | .67 |
| TCQ + asymmetric fp-target (tgt) + task-only calib | ~9.6% | 0.855 (collapse: S3_inj .08) | | 0.553 (World .03) | | 

On the original 1600 NEWS items only (task_search protocol): scalar 0.961 [0.950] (task_search, 10.03%) / 0.944 (this lane's re-run, 9.86%) vs VQ-D8 0.930, TCQ 0.924.
Not run to completion: the 32B TCQ task-only-calibration run died saving the model (box disk full), and the KL-sensitivity mixed-rate TCQ plan (`plan_tcq_1.55.json`, 1.548 bpw: 133 groups at 2.0, 103 at 1.5, 20 at 1.0 bpw; model-predicted KL -20% vs uniform 1.5) was stopped at layer 1/64 to save budget once the uniform result showed no >=2 pt gain over scalar. It is the one remaining untested lever from this lane (expected +0.5 to +1 pt, cannot close a 4-30 pt per-cell gap).

Qwen2.5-7B (same items, 7B base acc INJ .8166 / NEWS .8306; 9.99-10.0%, TCQ kb3): retention INJ / NEWS: baseline calib 0.940 / 0.931; per-tile scales (gscale 4, +0.003 bpw) 0.946 / 0.902; task-only calib 0.928 / 0.923; answer-token-weighted Hessian (ans_w 50) 0.868 / 0.937. Class-bias flips (benign<->injection, Technology->Business) of 10-30 points dominate every variant; the +-2 pt overall differences are not resolvable.

## 4. Findings
1. TCQ is a real but small quantiser gain: Hessian-weighted proxy error 0.0112 vs 0.0137 (VQ-D8) final, +0.8 dB on Gaussian, INJ +0.8 pt vs VQ-D8 at 0.3 pt fewer bytes. It does not reach the >=2 pt over scalar bar that was set for continuing: INJ +0.65 (inside noise), NEWS -2.1.
2. The failures are bias flips of a whole class (S3_injection recall, S2_benign, Technology), not random noise: base-correct items lose their small label-logit margin to a systematic shift of a few logits. Calibration choice (task-only vs mixed), per-tile scales and answer-weighted Hessians move these class biases by +-10-30 pts without moving the average, so best-of-N selection on a selection set would mostly pick class-bias luck (selection must be on data disjoint from the report set; not done, no config clears even 97%).
3. Asymmetric calibration to the fp target (`tgt`, GPTAQ-like) with task-only calibration collapsed (INJ .855, NEWS .553): either an implementation instability (the cross-Hessian X_fp^T X_q solve in the code path was written for the 7B quality push) or an unstable fit; it is not evidence against GPTAQ in general. Not pursued.
4. The 98% / 95%-per-cell gate needs roughly 2x lower noise power than any variant reaches; TCQ's theoretical ceiling (+1.5 dB total, +0.85 dB realised on Gaussian) cannot supply it. Rate cost of +1 point of retention is about 5-7% of bytes (coordinator note), so 9.9% -> 98% needs about 25% more bytes than allowed.
5. Compliance notes (RULES.md section 6): no weight training; GPTQ/LDLQ are calibration math; the 16 KB trellis codebook is Lloyd-fitted on synthetic Gaussian noise (a codec constant, not data-fitted; gray only if someone reads "fitted computation" as including it); no learned rounding. Items used for configuration choices in section 3 are the same dev items reported (selection bias <= noise, not used to select a winner).

## 5. Files
`v2/vq/{tcq.py,tcq_run.py,vq8_run.py,vreal_both.py,analyze.py,analyze_ct.py,alloc_tcq.py,box_run.py,modal_stub.py,run_eval.sh,run_ct_scalar.py,run_ct_inj.py,gauss_modal.py,ecsq_ref.py,cfg_*.json,plan_tcq_1.55.json}`, results in `v2/vq/results/` (`ret_*.txt`, per-item `eval_*.json`, `gauss_tcq.json`), NEWS ext dev `v2/task_search/NEWS/{build_news_extra.py,items_dev_ext.jsonl}`.
