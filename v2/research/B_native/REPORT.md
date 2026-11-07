# Track B: native low-bit starting models (2026-11-07). Source: subagent hand-back (model output; measurements on T1 dev probe, 600 items, no locked/S2 data)
Raw data: ./results/, scripts: ./scripts/. Box tenpct-g7.

## Verdict: NO-GO as a primary entry under the strict reading
No native low-bit candidate is a clean pass on rules 1, 2 and 9 together.
- Best candidate, 1-bit Bonsai-8B (PrismML, Apache-2.0): lossless pack 1,154,719,808 B = 7.05% of its own published fp16 checkpoint (16,377,141,728 B); all 254 matrices incl. embedding and untied lm_head are exactly +-scale per 128-group (bit-exact round trip verified). Base tool-calling on T1 dev: S1 .823, S3 .583.
- Hole 1 (hard fail rule 2): PrismML NOTICE says Bonsai is "built from Qwen3-8B". Against Qwen3-8B as base, Bonsai keeps only 88.9% of Qwen3-8B-correct items (S3 79.6%, multi 84.7%); ternary Bonsai 91.0%.
- Hole 2 (gray rule 9): entrant does no compression; the 1.15 GB form is the authors' own release.
- Every model needing a lossy embedding step fails rule 2: ternary Bonsai with embed/head at 1 bit 94.6-97.2%; BitNet-2B4T at 2-bit embedding 57.8%, 1.58-bit 45.6%, 3-4 bit 90.6%.
- Feedback (rules 5, 6, maintainer's check) untouched by packing: OPEN for every candidate.

## Survey (lossless packed bytes incl. embedding, lm_head, tokenizer, ~6 KB code, as % of fp16 checkpoint)
Bonsai-8B 1-bit 7.05% (built, verified) | Ternary Bonsai-8B 10.19% ideal entropy coding (31.3 MB over cap; 2-bit packing 12.5%; maker's Q2_0 13.3%) | Bonsai 4B/1.7B ~7% / ~10.2% (not built) | Ternary-Bonsai-2-27B best file 11.05% (untested) | BitNet b1.58 2B4T: ternary body ~8.2% leaves only 2.1 bpw for the 328M-param tied embedding, which must then be lossy; bf16 master vs packed release are different forwards (ambiguous base) | Falcon-E ~9.97-10.09% (not re-verified, 401) | Falcon3-1.58 / TriLM / Llama3-8B-1.58 / OLMo-Bitnet: not computed or derived-model problem | Maple-Preview (20B MoE) 12.72% (fail) | gpt-oss/Kimi INT4/Gemma-Llama QAT/NVFP4 >= 25-31% (fail) | no 2-bit QAT open-weight releases found.

## Tool-calling accuracy (T1 dev probe, greedy)
Qwen3-8B reference .763 all | Bonsai-8B 1-bit .703 (S1 .823, S3 .583) | Ternary Bonsai .715 | BitNet 2B4T bf16 master .267 (33-38% unparseable), packed .300. Both Bonsai models pass rule 8 (well above chance, below ceiling).
Retention vs Qwen3-8B: 1-bit Bonsai 88.9% (S1 96.4, S3 79.6); ternary 91.0% (S1 96.0, S3 85.0).

## Packed forward vs reference (Bonsai 1-bit)
Weights torch.equal for all 254 matrices and norms; embed, rotary, 36 layers and final norm bit-identical to HF fp16; lm_head output differs by 1 fp16 ulp for sliced inputs (cuBLAS stride-dependent). Greedy generation 39/40 identical (one free-text refusal differs, both correct). Claim that holds: same weights, same computation; not same logit bits across kernels. Prove on the full exam with the deployed kernel.

## Embedding re-encode retention (alone)
Ternary Bonsai-8B (429 base-correct): embed->1bit .972, head->1bit .956, both .946 (all miss 98% overall; two miss 95% on a kind/source). BitNet 2B4T packed (180 base-correct, +-3-4 pts noise): 4-bit .906, 3-bit .906, 2-bit .578, ternary .456, 1-bit .472. BitNet totals as % of fp16 by embedding encoding: bf16 21.8, 4-bit 12.0, 3-bit 11.2, 2-bit 10.3, ternary 9.65, 1-bit 9.15.

## Rule table summary (strict)
A (1-bit Bonsai, base := Bonsai-unpacked): size CLEAN-numerically but GRAY on "fp16 size of the model you started from"; faithful CLEAN by construction; rule 9 GRAY-to-risky. A' (base := Qwen3-8B): faithful FAIL (88.9%). B (ternary lossless): size FAIL 10.19%. B' (lossy embed/head or vocab trim): faithful FAIL measured. C (BitNet lossy embedding): FAIL. E (Maple): size FAIL 12.7%. Rules 3, 4, 7 CLEAN for all tested; rules 5, 6 OPEN.

## Draft question for the maintainer (NOT POSTED; needs user OK)
Setup: model M (>0.5B) released by its authors with an fp16 checkpoint F whose weights are exactly low-bit valued (every weight of every matrix, embeddings and lm_head included, is +-s or {-s,0,+s} per group of 128, so a lossless pack of F is bit-exact). M was trained by its authors (e.g. PrismML's Bonsai-8B, derived from Qwen3-8B), not by me. I would ship a lossless re-encoding of F at 7% of F's size, no extra parts, same forward pass.
1. Is the size baseline for "at most 10% of its fp16 size" the fp16 checkpoint of M itself (F), even when M's authors also publish a packed form already below 10%? yes/no
2. Is "the model you started from" (and the model whose score I must match) M, even though M was derived by its authors from another model (Qwen3-8B)? yes/no
3. Does a lossless repack of a native low-bit model count as valid compression under the spirit rule, or must the entrant reduce a model that was not already <=10% in released form? yes (valid) / no (entrant must compress)
If 1 and 2 are yes but 3 is no, do not submit this route.

## Recommendation
No-go as primary. Conditional cheap go: post the question; if all three answers are yes, Bonsai-8B 1-bit is the safest entry assemblable in ~5 days (7.05%, lossless, T1 dev ~.70); remaining work = full locked exam + feedback. If any answer is no, drop this route and put effort into the dense lossy track.
Limits: 600 dev items (SE ~1.8 pts); BitNet used prompted rather than template-native tools; packed-forward check is 40 generation items / 24 logit prompts; not run: Falcon-E, Bonsai 4B/1.7B, Bonsai-2-27B, Maple behaviour, vocab trimming; ternary rANS size is a Shannon estimate.
