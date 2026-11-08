# CODING: more quality per byte through coding and structure (training-free)

Research-only lane, 2026-11-08, $0 compute, nothing run. Byte accounting computed from the repo's own result JSONs (`v2/research/D_scale/results/meta{32,72}_*.json`, accounting code `v2/task_search/dense_core.py` lines 277-370) and `v2/entropy/RESULTS.md`. Literature numbers are from the abstracts/tables I fetched; items marked [unverified] come from search snippets only.

## 0. Bottom line

1. About 88-91% of our bytes are the quantization index stream (rANS, ~1.40-1.47 bits/weight). Scales are 2.7% (32B, g256) to 5.4% (72B, g128). Embedding + lm_head are 4.4% (72B, ~2 bpw) to 6.1% (32B), and 14% to 19% on untied 14B/7B. Everything else (norms, tokenizer, config, code) is about 0.2%.
2. Lossless "context modelling" of the indices is a dead end at our point. We already measured it (`v2/entropy/RESULTS.md`, 14B GPTQ, real range coder, byte-exact decode): 0.5% of index bits when Hadamard-rotated, 1.8% when unrotated. Literature that reports bigger numbers (DFloat11 30%, ZipNN, ZipLLM 54%) is about bf16 or cross-model dedup, not about already-quantized, rotated, entropy-coded weights.
3. The real, cheap byte reservoir is **scale coding (1.5-3.2% of total bytes)**, then a smaller **embedding/head redesign (1-3%)**. Together about 3-6% of bytes. At the repo's measured slope (about 1.5-2 retention points per +1 point of size ratio, i.e. per ~10% more bytes; see section 5) that is **+0.5 to +1.0 retention point**. Not enough alone for the 2-3 point gap, so spend it together with a better quantizer (VQ/trellis, which the repo already shows is worth +20 points over scalar at 7B, C11).
4. Low-rank/sparse hybrids (CALDERA, LQER, SpQR) are better than plain 2-bit at equal bits in the literature, but (a) the repo rules list "adapters of any kind incl. training-free (EoRA)" under "not allowed / too risky" (`research/A_sota/REPORT.md` line 39) while `RULES.md` line 40 allows low-rank as compression, so a W = Q + LR decomposition is GRAY, and (b) at 1.5 bpw the factor bytes (0.06-0.10 bpw for rank 64) are 4-7% of body, comparable to what mixed precision buys. Medium risk, moderate upside.
5. Layer merging/cross-layer sharing (Basis Sharing, LaCo, ShortGPT) loses to quantization per byte by a wide margin and is known to collapse free-form generation. Rank last.

## 1. Where the bytes go at our operating point (9.6% target)

Computed from real run metadata. `body_bpw` in the JSONs already includes scales (`GB/g` with GB = 11 bits). Index bpw = body_bpw - 11/g.

| run (artifact est.) | total MB | indices | scales (11 b/group) | embedding | lm_head | norms/tok/code |
|---|---|---|---|---|---|---|
| Qwen2.5-32B, g256, t=0.096 (`meta32_C32R3g1`) | 6,301 | 5,735 MB, 91.0% (1.470 bpw) | 168 MB, 2.7% (0.043 bpw) | 195 MB, 3.1% | 191 MB, 3.0% | 13 MB, 0.2% |
| Qwen2.5-72B, g128, t=0.096 (`meta72_C72g1`) | 13,974 | 12,585 MB, 90.1% (1.434 bpw) | 754 MB, 5.4% (0.086 bpw) | 311 MB, 2.2% | 312 MB, 2.2% | 11 MB, 0.1% |
| Qwen2.5-72B, emb/head 8 levels (`meta72_V11`) | 13,972 | 12,313 MB, 88.1% (1.403 bpw) | 754 MB, 5.4% | 446 MB, 3.2% | 448 MB, 3.2% | 11 MB |
| Qwen2.5-14B (analytic, 152,064 x 5,120 untied, 2 bpw each) | 2,836 | ~83% | ~3-5% | 389 MB for emb+head = 13.7% | | |
| Qwen2.5-7B (analytic, 152,064 x 3,584 untied, 2 bpw each) | 1,462 | ~75% | ~3-5% | 272 MB for emb+head = 18.6% | | |

Notes. (a) Embedding and head sit at about 2.0 bpw (L=4..8 grid entropy) in the 72B/32B runs already. (b) The index stream has had per-tensor histogram entropy coding (H0 x 1.005) plus Lagrangian per-module level allocation (`dense_core.py` lines 296-322, 13 levels). (c) Entropy-coded scales exist only as a flat 11-bit field; nothing is context-coded.

## 2. Entropy and context-model coding of the indices

What the literature says and what it means for us:

| technique | reported number | transfer to us |
|---|---|---|
| Plain histogram + rANS on quantized ints (EntroLLM, [arXiv 2505.02380](https://arxiv.org/abs/2505.02380)) | up to 30% saving vs uint8, 65% vs uint4 (7-11x entropy compressibility), per-tensor codes | We already do this. It is why 3-bit grids store at ~2.7 bpw. No new gain. |
| EntQuant / "Float8@2bits" ([arXiv 2601.22787](https://arxiv.org/html/2601.22787)) | rANS over an entropy-penalised Float8 grid, 2.1 bpw effective; Llama-2-70B C4 6.47 vs QuIP# 6.12 (fine-tuned); one bf16 scale per output channel (<1% of memory); i.i.d. global frequency table, no context model | Confirms that i.i.d. entropy coding with an L1 entropy surrogate is competitive with VQ at 2 bpw. Says nothing about context. Its per-channel (not per-group) scale design is the interesting part (section 3). |
| Our measured context mixing (`v2/entropy/RESULTS.md`) | left/above neighbours, running column mean, group-scale bin, previous layer, logistic mixing: rotated 0.5% of index bits (0.007-0.018 bpw), unrotated 1.8% (0.03-0.06 bpw); mixing no better than best single context; previous-layer context ~0 | Hard number: at our rotated operating point context modelling is worth ~0.007 bpw = **~0.5% of index bytes = ~0.45% of total**. Not worth decoder code. |
| Neural Weight Compression ([arXiv 2510.11234](https://arxiv.org/html/2510.11234)) | learned nonlinear transforms + factorized entropy model; Llama-3-8B WikiText2 8.92 @2.3 b, 7.50 @2.7 b, 6.75 @3.2 b; transforms cut ppl 27.8 -> 14.5 at 2.12 b vs identity transform; column norms 0.004 bpw | Trained codec (11 GPU-hours on Llama-3 weights): violates R4 (no new trained parts) in spirit and is Llama-specific. Skip. |
| ECSQ vs rate-distortion bound | [general theory, not from a fetched paper] uniform-threshold ECSQ is ~0.25 b above the Gaussian bound at high rate and roughly 0.1 b at 1-2 b | The remaining headroom at 1.5 bpw is ~0.1 bpw (~7% of index bytes) and only vector/trellis quantizers capture it. This is the real "coding" gain and it lives in the quantizer, not the entropy coder (section 6). |
| Rate-aware rounding: CERWU ([arXiv 2505.18758](https://arxiv.org/abs/2505.18758)), Radio ([arXiv 2505.03031](https://arxiv.org/pdf/2505.03031)), HyperQuant ([arXiv 2606.23406](https://arxiv.org/abs/2606.23406)) | CERWU: 20-40% lower bitrate than NNCodec on vision nets, per-bit cost added to OBS/OPTQ rounding; Radio: Llama-2-7B 9.47 ppl @2.1 b vs OWQ 39.56, 6.21 @2.8 b vs 10.24; HyperQuant: RHT + E8/D4 lattices + Rice-coded indices, beats HIGGS 3-5 bpw, 3.9x at 4 bpw | We already use GPTQ onto an ECSQ grid with dead zone (`SOLUTIONS` C02 "dead zone 0.15"), so the rate term is partly in. Putting the true code length (not just a dead zone) into the GPTQ rounding decision is the cheap upgrade (experiment 3). |
| DFloat11 ([arXiv 2504.11651](https://arxiv.org/html/2504.11651v2)), Cloudflare Unweight ([blog](https://blog.cloudflare.com/unweight-tensor-compression)), ZipNN, ZipLLM ([arXiv 2505.06252](https://arxiv.org/html/2505.06252v3)) | bf16 exponent entropy ~2.6-3 bits so ~30% (DFloat11) or 15-22% (Unweight) lossless; ZipLLM 54.1% across 3,048 models via tensor dedup + BitX XOR-delta | Applies only to anything we keep in bf16/fp16 (norms, biases: 0.1% of bytes) and to cross-model deltas. We have one base model per entry. Nothing to gain. |

Conclusion for part (1): the repo's lossless stack is already within ~1% of what context modelling can add. Do not spend a day on a CM coder.

## 3. Scales (2.7-5.4% of bytes) and low-rank/sparse hybrids

### 3a. Scale cost and how to cut it
Currently 11 bits per group scale, flat. After a block-128 random-sign Hadamard the group RMS values are nearly concentrated (relative std about 1/sqrt(2g) = 4.4% at g=128 by chi-square concentration, plus a slow per-tensor/per-row drift). So the information in a scale given a per-tensor (or per-row) context is low. Estimate [my arithmetic, not a measured result]: quantizing log2(scale) with a 1.5% step and coding the residual to a per-row running mean needs ~4-5 bits; the quantization error from the step is 0.4% RMS of the scale, negligible against ECSQ noise. The entropy dir measured lossless fp16 scales at 10.4-10.9 b with only the exponent context-coded, and says lossy 8-bit log scales save another ~0.03 bpp (not accuracy tested).

| change | saving | where |
|---|---|---|
| 11 b -> ~4.5 b log-delta scales, same g | 72B g128: ~0.05 bpw = 440 MB = **3.2%**; 32B g256: ~0.025 bpw = 100 MB = **1.6%** | free bytes to spend on a smaller group or a higher L |
| Per-row/per-channel scale, EntQuant style (one scale per output channel, <1% of memory) instead of per-group | 72B: up to ~5% of bytes, but EntQuant has no Hadamard-group concentration; per-channel scale loses local adaptation | accuracy unknown; test against g256 with coded scales |
| Shared scale across a pair/quad of groups (g256 at same fidelity) | same as halving the scale table: 72B ~2.7%, 32B ~1.4% | cheapest to try; the repo already ran g128 vs g256 |
| LoRDS / low-rank scale matrix [unverified, not found by search] | scale matrix S ~ rank-r outer product instead of per-group | not checked; skip |

### 3b. Low-rank + quantized, sparse outliers
| method | numbers | fit |
|---|---|---|
| CALDERA ([arXiv 2405.18886](https://arxiv.org/html/2405.18886v1)) W = Q + LR, Q at 2 b E8, L/R at 4 b | Llama-2-7B, rank 64, 2.1 avg bits: Wiki2 7.37 (QuIP# no-FT at 2 b: 8.23); rank 256 at 2.4 b: 6.19. Llama-3-8B 2.4 b: 8.22 vs QuIP# 13.8. Llama-2-70B rank 64 16-bit factors 2.2 b: 4.50 vs 5.37 | Training-free path exists (calibration-weighted alternating Q / LR, no gradients). Cost at our shapes: 4-bit rank 64 on a 5120 x 5120 matrix = 64 x 10240 x 4 / 26.2M = **0.10 bpw (7% of a 1.5-bpw body)**; on 8192-wide 72B layers ~0.06 bpw. The CALDERA gain per extra 0.1 bit (8.23 -> 7.37 on 7B) is far larger than 0.1 bit of scalar-quant precision, but the gain shrinks at 70B (5.37 -> 5.07) and our bits are 0.5 lower. |
| LQER / ASVD / SVD-LLM V2 / ResSVD | SVD-LLM V2 + 2-bit beats BiLLM (1-bit) by ~69% ppl on LLaMA-7B ([arXiv 2503.12340](https://arxiv.org/abs/2503.12340)); LQER, ResSVD, LoRDS: [unverified], no numbers retrieved | Same bytes-vs-gain trade as CALDERA; low-rank-only (no Q) compression is much worse per byte than quantization (SVD-LLM V2 + GPTQ-4bit LLaMA-7B 9.97 ppl at 2.8 GB). Use only as a residual on top of Q. |
| SpQR ([arXiv 2306.03078](https://arxiv.org/pdf/2306.03078)), SqueezeLLM D+S, OWQ | SpQR near-lossless at 3.4-4.75 bits/param; SqueezeLLM keeping 0.45% weights sparse gives LLaMA-7B C4 7.75 -> 7.58 at 3 bit (~0.3 ppl over g128 baselines) | At 1.5 bpw, a 0.45% fp16 sparse outlier set costs 0.45% x (16 + ~16 index bits) = 0.14 bpw, i.e. 10% of body: too expensive. After Hadamard rotation the outliers are mostly gone anyway (the repo uses rot). Rank low. |

Rule risk: low-rank factor of the weight is "compression" per `RULES.md` row 40, but the 2026-11 strict-rules ruling (memory: pass all 9 rules, no gray readings) and A_sota line 39 name adapters, including training-free ones, as risky. Get a maintainer read before betting on it.

## 4. Embeddings and lm_head

Current state: both at about 2.0 bpw with a per-row rotation (`est_ent`, `eL`, `hL`). Untied in all Qwen2.5 sizes >= 3B, so two 152,064-row tables.

| lever | evidence | byte estimate | legality / risk |
|---|---|---|---|
| Lower emb/head bits with sensitivity-based allocation (head more sensitive than embedding: head errors hit every token's logits, emb errors enter through one row) | llm-compressor: 4-bit g64 embedding table moved pythia-1.4b ppl 14.733 -> 14.752 ([docs](https://docs.vllm.ai/projects/llm-compressor/en/stable/examples/quantization_embedding/)); BitNet lossy 4-bit embedding keeps T1 probe at .906 and 2-bit .578 (repo C15) | 72B: moving emb from 2.0 to 1.5 bpw saves ~78 MB (0.55%); the repo already finds 8 levels/3 bit beats lower (V11 vs C72g1) | Fine. Gain tiny at 72B, up to 3% on 7B/14B. |
| Vocabulary pruning (generic, frequency-based, rows of both emb and head) | COMPACT ([arXiv 2509.06836](https://arxiv.org/html/2509.06836v3)): drop rare-token rows from emb and head plus merge rules, no calibration; repo C02: vocab 24k + 4-bit emb at 7B | Keeping 40k of 152k rows: saves 74% of emb+head = 3.2-4.5% of bytes at 72B/32B, **10-14%** at 14B/7B | `PRUNE_LEGAL.md`: head-row removal is legal only as a generic fixed cutoff chosen before looking at labels, with measured OOD cost; the strict-rules ruling treats this as GRAY. Biggest single byte win after scale coding, but only if the maintainer says yes. |
| Weight tying post-hoc (emb = head^T) | Qwen2.5 already tie only <= 1.5B; untying cost isn't recoverable post-hoc; tied share would give 50% of emb+head bytes | Would save 2.2-3.2% at 72B but changes the function; the logit error from using the embedding as head is large, no training allowed | Reject (output not faithful, no training to repair). |
| Product quantization / VQ for embeddings (dim-8 k-means, shared codebook) | the repo's 8-D Gaussian k-means codebook (C07/C11) | Emb/head are not Gaussian-iid after row rotation but close; VQ at 1.6-2 bpw buys ~0.1-0.2 b over scalar at the same distortion (my estimate from the body-weight VQ gap) = 5-10% of the 3-6% = **0.2-0.5% of total** | Free to try with the existing `vq.py`; low value. |
| Token-frequency-based allocation | rare-token rows matter only when emitted; frequent-token rows at 3 bit, rare at 1.5 bit | Estimated 15-25% of emb+head bytes = 0.5-1.2% total at 72B, up to 3% at 7B | Legal as an allocation rule (labels not used), simple. Needs a frequency table from neutral text (counted bytes) and a measured OOD cost; zero-training. |

## 5. How many bytes are worth how much retention

Measured slopes in the repo (all dev): 32B NEWS 10.03 -> 11.28 -> 12.53% gives 96.1 -> 96.2 -> 97.9 (flat then +1.7); 32B TOOLSEL 92.8 -> 96.7 at 11.3%; Qwen3-32B T1 at 9.6 / 12.0 / 15.0% gives 85.8 / ~90 / ~94 (about 1.4 pt per 1 pt of ratio); 7B VQ slope +4 pts per +0.1 bit at 14B MMLU. Central assumption for this document: **+1 point of retention costs ~0.5-0.7 points of size ratio (5-7% of our bytes)**, noisy by 2x, quantizer-draw noise is +-2 points. So:

| bytes recovered | = retention |
|---|---|
| 1% of total bytes | +0.15 to +0.2 pt |
| 3% (scale coding at 72B) | +0.5 pt |
| 6% (scale coding + emb/head tuning, no vocab trim) | +0.9 to +1.2 pt |
| 10-14% (vocab trim at 14B/7B only) | +1.5 to +2.5 pt |

Hence coding tricks alone get us from 95-96% to about 96.5-97%, still short of 98%. The remaining gap needs the quantizer-level gains below.

## 6. Vector/trellis codebooks, entropy-coded together

* QTIP ([arXiv 2406.11235](https://arxiv.org/html/2406.11235v4)), Llama-2 WikiText2, 2 bit, **no fine-tuning**: 1MAD 7.05 / 5.59 / 3.87 and 3INST 6.82 / 5.52 / 3.90 (7B/13B/70B) vs QuIP# no-FT 8.22 / 6.06 / 4.16; fp16 5.12 / 4.57 / 3.12. Llama-3-8B 7.33 vs 7.84, 70B 4.97 vs 5.77. 1MAD/3INST need no table (2-3 instructions per weight), so zero codebook bytes. They are fixed-rate: the paper does not evaluate entropy coding of the bitstream.
* Entropy coding the trellis path (ECTCQ in video/image coding): [general knowledge] variable-rate TCQ gets within ~0.5 dB of the RD bound; fixed-rate TCQ at a cell where the index histogram is non-uniform leaves the same ~0.1-0.2 b we get from ECSQ over fixed-width. Unexplored for LLMs in anything I retrieved. At 1.5 bpw a fixed-rate 2-bit-trellis-with-dead-zone variant (k=1.5, i.e. rate drops by mixing 1 and 2 bit stages) is the natural comparison.
* Repo: C07 (32B VQ-D8 K=5300, 1.5465 bpw) 95.0% INJ, C11 (7B VQ K=14500, 1.72 bpw) 94.9-96.2% INJ vs scalar GPTQ 70.1% at 9.00%; `v2/vq/tcq.py` and `tcq_run.py` exist but I found no result row in `SOLUTIONS.md` for TCQ.
* EntQuant's lesson: ECSQ/L1-surrogate rounding at 2.1 bpw reaches QuIP#-class perplexity on 70B (6.47 vs 6.12). So scalar-entropy-coded is within 0.35 ppl of the vector method at 70B but far at 7B; gains from VQ/TCQ grow as the model shrinks.
* "Code VQ indices together": for a K=5,300..14,500 shared codebook, index histogram entropy is within 0.01-0.03 b/8 weights of log2(K) (nearly uniform for k-means on Gaussians). Entropy coding saves ~1-2% of that stream, no more. [estimate, no run]

## 7. Weight sharing / layer merging (post-hoc, training-free)

| method | evidence | verdict |
|---|---|---|
| Basis Sharing ([arXiv 2410.03765](https://arxiv.org/html/2410.03765v1)) | shared SVD basis across layers, up to 25% lower ppl and +4% accuracy vs SVD-LLM at 20-50% compression ratios (i.e. 50-80% of bytes remain) | Operates at 50-80% size. At our 9.6% quantization already did 90% of the job; SVD-type sharing at 10x compression is not demonstrated. |
| LaCo layer collapse, ShortGPT ([arXiv 2403.03853](https://arxiv.org/pdf/2403.03853)), "Fewer Layers Break More Chains" ([arXiv 2510.22228](https://arxiv.org/pdf/2510.22228)) | LaCo removes 30-50% of layers keeping MC performance; ShortGPT >90% on generative tasks at 25% layers dropped; reasoning/free-form generation collapses | Removing one of 64 layers saves 1.6% of bytes but costs far more than 1.6% x 0.2 = 0.3 pt of retention (retention is item-exact, not MC). The repo's own layer-drop and prune lanes (`v2/prune`) already treat this as a fallback. |
| Cross-layer delta coding of quantized weights | our entropy run: "previous-layer same-position context gain 0" | Layers are not similar at the weight level. Dead. |
| Expert merging (MoE) | out of scope for dense candidates | skip |

## 8. Ranked top 5 experiments

Cost is in dollars of cloud time at the repo's rates (A100/H100, 14B-32B evals ~$2-20); none requires training. Retention gain is in absolute points on the dev items, at the section-5 slope; noise is +-2 pts, so each needs >= 3 seeds or a paired comparison on identical items.

| rank | experiment | bytes saved | retention gain | cost | risk |
|---|---|---|---|---|---|
| 1 | **Log-delta coded scales + smaller group.** Replace the 11-bit field with ~4.5 b (log2 quantized at 1.5% step, residual vs running per-row mean, adaptive rANS). Then spend the freed 0.025-0.05 bpw on g256 -> g128 (32B) or g128 -> g96/g64 (72B) at constant total bytes. Verify roundtrip with the real decoder. | 32B 1.6%, 72B 3.2% (can't be exceeded: scale entropy floor ~4 b) | +0.3 to +0.6 pt direct; unknown but plausibly positive from finer groups | $5 (CPU check of byte math and quant error) + $15 (one 32B dev run) | Very low. Only risk: scale-quant error interaction with GPTQ clip search (re-search clip with quantized scale). |
| 2 | **Frequency-based embedding/head allocation (full vocab kept)**: 3 bit for the top ~30k rows by neutral-text frequency, ~1.5 bit for the rest, Hessian-weighted head. No row removed. | 0.5-1.2% at 72B, 2-3% at 7B/14B | +0.1 to +0.3 pt at 72B; up to +0.5 at 14B | $10 | Low; legal. Need an OOD (non-English/code) probe so rule-9 reviewers see the cost. |
| 3 | **True rate-aware GPTQ rounding**: replace the dead-zone with `argmin err_Hessian + lambda * (-log2 p(q))` using the live per-tensor histogram (EC-GPTQ, in the spirit of CERWU/OPTQ with a rate term), re-solved jointly with the allocation. | 0 bytes saved directly; moves ~0.03-0.06 bpw of "ECSQ vs RD" slack, i.e. 2-4% of index bytes at equal distortion | +0.3 to +0.8 pt equivalent | $30 (needs Hessians; one 14B and one 32B run) | Medium: gain over the existing dead-zone ECSQ is unproven for LLMs; may be ~0. |
| 4 | **Training-free QTIP-style trellis (1MAD or 3INST) body with entropy-coded stage mix** (some tensors at k=1, some at k=2 or 3 via the existing Lagrangian allocator) compared to the current scalar and the C07 VQ at the same bytes. No codebook bytes. | If trellis is as good as published: ~0.15-0.25 b at the same quality, i.e. 10-15% of the body | +1.5 to +3 pt (gap to scalar at <= 7B is the largest, 20+ pts in C11; shrinks at 32B+) | $60-150 (needs a Viterbi/TCQ encoder at 1.5 bpw, not published for k<2 in the paper; repo has `vq/tcq.py`) | Medium-high: engineering time; k<2 is unproven; R4 gray only if codebook is fine-tuned (don't). |
| 5 | **CALDERA-style rank-64 4-bit low-rank residual on the 20-30% most sensitive matrices only** (sensitivity from the existing Lagrangian table; paid for by the experiments 1-2 savings), training-free alternating solve with Hessian weighting. | costs ~0.06-0.10 bpw on those matrices (~1.5-2.5% of total); not a saving | CALDERA 7B at 2 b: 8.23 -> 7.37 for +0.1 b, so per-byte it beats raising L; could be +1 pt on binding cells | $40 | Medium-high rule risk (adapter-like; A_sota line 39), and loses much of its edge at 70B and below 2 bpw. Ask the maintainer before spending. |

Not ranked (do not run): context-mixing coder for indices (0.45% of bytes, measured), neural weight codec (R4), DFloat11/ZipNN (bf16 only), cross-layer sharing/merging, SpQR sparse outliers at 1.5 bpw (0.14 bpw cost), post-hoc weight tying (non-faithful).

Sequencing: run 1 and 2 together first (about $25, one 32B dev pass, nearly free of risk); 3 in parallel on the 14B; 4 is the only item that can move retention by several points and should be the main bet; 5 only after a rules ruling.

## 9. Caveats

* Retention-per-byte slope is a rough fit from three different tasks and is the weakest number in this document.
* Section 3a's 4-5 bit estimate is arithmetic from chi-square concentration; it must be measured on the actual `meta*` scale tensors (cheap: dump scales, entropy-code offline, no GPU).
* Cross-checked, fetched numbers: EntQuant, NWC, QTIP, CALDERA, SqueezeLLM, Radio. Snippet-only (verify before citing): ZipLLM 54.1%, Unweight 15-22%, EntroLLM version-specific ratios, SliM-LLM 48%, COMPACT.
