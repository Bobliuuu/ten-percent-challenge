# SOTA training-free extreme quantization: literature scan (2026-11-08)

Scope: post-training, no weight training (calibration math allowed), target <=10% of fp16 (about 1.5-1.6 bits/weight all-in) on Qwen2.5-14B/32B(/72B). Numbers below were read from the linked pages by a web-fetch pass; "(unverified)" marks figures from search snippets only. No compute was used.

## Bottom line

1. **No published method reports near-lossless (within 1-2%) downstream accuracy at 1.5-1.6 bpw on 14B-72B models without finetuning.** The closest hits: PTQ1.61 (1.61 bpw, but Llama-70B WikiText2 ppl 6.94 vs 3.31 fp16, uses a LoRA restore step, no Qwen); ternary PTQ of Qwen3-8B at ~1.64 bpw on linears (78.5% chance-corrected retention, ppl 1.36x); 2-bit vector PTQ at ~2.3 bpw keeps only 89-97% of QA accuracy and 51-62% of generation scores on 4-32B models. Our own numbers (14B at 1.65-1.75 bpw: MMLU 0.62-0.66 vs 0.80) are consistent with this literature.
2. **Scale is the strongest lever.** QTIP at 2 bits on Llama-3.1-405B-Instruct is near fp on zero-shot; 70B loses 1-3 pts at 2 bits; 7-8B loses 5-10 pts. A 72B run at 1.55-1.6 bpw is the only configuration where the literature plausibly points at the 98%/95% gate.
3. **Bonsai/PrismML 1-bit 8B is not a usable recipe.** Whitepaper (read via raw PDF text) says "built from Qwen3-8B", "proprietary Caltech IP", "principled compression framework"; it gives no procedure (no QAT/PTQ/data statement). It is 1.125 bpw (1 bit + fp16 scale per 128), and loses a lot vs Qwen3-8B (avg 70.5 vs 79.3; MMLU-Redux 65.7 vs 83.0; BFCL 65.7 vs 81.0), and it deliberately omits ARC/PIQA/WinoGrande as "saturated". The whitepaper states the model is trained/built end-to-end; whether post-training cannot be established from public text. Even if post-training it fails our 98% gate by a wide margin.
4. The agreement angle: "Accuracy is not all you need" shows flips (answers changing correct<->incorrect) are 4-14% at W4 even when accuracy is flat, and track KL (Spearman 0.98 on MMLU). Our R2 gate (98% of base-correct retained) is exactly a flip metric, so **calibrating against teacher KL/top-1 rather than reconstruction MSE is the principled move.** No paper reports flip rates at 2 bits; this is an open gap we can fill.

## Comparison table

bpw = as reported (usually includes scales/codebooks; embeddings/head usually fp16 and NOT counted, which matters for our 10% cap). FT = needs finetuning.

| Method | Link | Training-free? | bpw | Model | Key numbers |
|---|---|---|---|---|---|
| QTIP (trellis + incoherence) | [2406.11235](https://arxiv.org/abs/2406.11235) | Pure-code variants (1MAD/3INST) yes; hybrid table uses FT of norms/scales | 2 | Llama-2-70B | W2 ppl 3.87 (no FT) vs fp16 3.12, QuIP# 4.16 (no FT), 3.91 (FT), AQLM 3.83 (FT). Llama-2-7B 2-bit 6.82 vs 5.12. Zero-shot 70B 2-bit: ArcC 48.0 vs 51.1, Wino 75.1 vs 77.0 |
| QTIP on Llama-3 | same | partly | 2 / 3 / 4 | Llama-3-70B | W2 4.97 / 3.18 / 2.75 vs 2.59; ArcE 35.2/77.8/83.9 vs 86.9 (2-bit collapses on Llama-3, trained heavily) |
| QTIP on 405B-Instruct | same | partly | 2 | Llama-3.1-405B-Inst | zero-shot ArcC 60.7 vs 61.6, Hswag 65.4 vs 67.1: near lossless at scale |
| QuIP# | [2402.04396](https://arxiv.org/abs/2402.04396) | no-FT variant yes (E8P codebook, 2 bit) | 2 | Llama-2-70B | 4.16 no-FT, 3.91 FT vs 3.12 |
| QuIP | [2307.13304](https://arxiv.org/abs/2307.13304) | yes | 2 | Llama-2-70B | W2 6.33 (vs OPTQ 123.9); incoherence processing is what makes 2-bit viable |
| AQLM | [2401.06118](https://arxiv.org/abs/2401.06118) | Calibration-only AQLM yes (slow, days on 70B); big gains need block-FT; PV-Tuning is FT | ~2.0-2.1 incl. codebooks | Llama-2-70B | 3.94 ppl (v-later), 3.83 in QTIP table (with FT). Llama-3-70B 1x16 no PV: MMLU 0.79 -> 0.75 at 21.9 GB (2.5 bpw-class) |
| VPTQ | [2409.17066](https://arxiv.org/abs/2409.17066) | Yes (optional FT) | ~2.07 | Llama-2-70B | W2 3.93, C4 5.72 (HF doc); beats QuIP#/AQLM by 0.01-0.34 ppl on Llama-2; Qwen2.5-72B-Instruct community checkpoints at 1.94-2.38 bpw exist, only ppl reported ([VPTQ-community](https://huggingface.co/VPTQ-community)) |
| GPTVQ | [2402.15319](https://arxiv.org/abs/2402.15319) | Yes | 2.13 / 2.25 | Llama-2-7B | 10.66 (2D) / 7.89 (4D) vs QuIP# 8.22 at 2.0 (third-party table in GuidedQuant); weaker than QuIP# at the same size |
| NestQuant (E8 nested lattice) | [2502.09720](https://arxiv.org/abs/2502.09720) | Yes, W+A+KV | 4 | Llama-3-8B | 6.3-6.4 W2 ppl vs SpinQuant 6.5-6.6; a 4-bit result, not 1.6 |
| Leech lattice VQ (LLVQ) | [2603.11021](https://arxiv.org/abs/2603.11021) | Yes (FT optional: per-column input scales, 52M tokens) | 2.0 (scales <0.001 bpw, no stored codebook) | Qwen-3 8B / 4B, Llama-3 8B | Qwen3-8B no FT: ppl 10.19 (0-bit gain) / 10.82 vs QTIP 11.17, QuIP# 12.80, fp16 8.99; CSR 70.2 vs QTIP 71.1 vs fp 74.0. Qwen3-4B: 15.5-17.0 vs fp 12.41, CSR 63.6-64.1 vs 71.2 |
| Grouped lattice VQ (GLVQ) | [2510.20984](https://arxiv.org/abs/2510.20984) | likely yes | ~2 | Llama (details unread) | table not extracted (PDF); code github.com/xzhang9308/GLVQ |
| GuidedQuant (end-loss-gradient weighted Hessian) | [2505.07004](https://arxiv.org/abs/2505.07004) | Yes (main results no FT) | 2.01 / 3.03 | Llama-2 | Scalar LNQ+GQ 2.01 bpw: 7B 8.83 (SqueezeLLM 39.6), 70B 5.04. QTIP+GQ 2-bit: 7B 6.82->6.11, 13B 5.52->5.33, 70B 3.87->3.80 |
| ADMM-Q (better Hessian quantizer) | [2605.11222](https://arxiv.org/abs/2605.11222) | Yes, calibration only | 2 / 3 scalar | Qwen3 1.7-32B | W2 ppl Qwen3-14B 12.62 vs GPTQ 40.98; 32B 18.15 vs LDLQ 23.21 vs GPTQ 35.95; W3 8B zero-shot 57.8 vs GPTQ 54.4 (scalar, so still far above our budget quality) |
| GPTAQ (asymmetric calibration) | [2504.02692](https://arxiv.org/abs/2504.02692) | Yes, ~20 lines on GPTQ | low-bit | up to 405B | Quantized layer matches the FP model output given quantized-model inputs; runs 405B on one GPU; gains concentrated at low bit (exact numbers unread) |
| Beyond 4 Bits (YAQA/VPTQ + task-aligned calib + codebook recovery) | [2610.28537](https://arxiv.org/abs/2610.28537) | PTQ part yes; "recovery" trains codebooks (~3.2k examples, KL+CE distill) | 2.09-2.33 incl. everything | Phi-4-mini, Llama-3.1-8B, Qwen3-8B/32B/30B-A3B | 2-bit vector PTQ QA retention 89-92%, generation 51-62% (Phi-4-mini 37.4 vs 73.2); scalar at 2 bit = chance; no flip metric |
| PTQ1.61 | [2502.13179](https://arxiv.org/abs/2502.13179) | Needs a restorative LoRA (10K steps) pre-step + block-wise scale learning | 1.61 (mask 0.0002) | LLaMA-1/2/3, OPT; no Qwen | Llama-2-70B 6.94 / 12.63 (W2/C4) vs fp 3.31; LLaMA-3-70B 28.3 vs 2.93; zero-shot avg LLaMA-2-13B 44.7 vs BiLLM 41.0 |
| BiLLM | [2402.04291](https://arxiv.org/abs/2402.04291) | Yes | ~1.08 nominal, ~2.1 real (PTQ1.61 accounting) | Llama | Llama-2-70B 12.80; unusable at our bar |
| ARB-LLM | [2410.03129](https://arxiv.org/abs/2410.03129) | Yes | ~1.06-1.09 nominal | OPT/Llama | OPT-66B 10.30 vs BiLLM 12.05 vs fp 9.34; follow-up 1-bit output-alignment [2512.21651](https://arxiv.org/abs/2512.21651) covers Qwen2.5 (tables unread) |
| STBLLM, DB-LLM | search found no tables (unverified) | | <1 nominal | | below-1-bit, structured: quality far below our bar |
| Ternary PTQ (KOTMS rotation + E2M-ATQ + GPTQ comp) | [2609.09240](https://arxiv.org/abs/2609.09240) | Yes | 1.64 on linears only (embeddings/head higher) | Qwen3-8B | ppl 9.71->12.80 (1.32x); 8-task acc 64.6 vs 72.4 (78.5% chance-corrected); MMLU 71% retained, LAMBADA 91%; Qwen3-4B 69.6% |
| SqueezeLLM | [2306.07629](https://arxiv.org/abs/2306.07629) | Yes | ~2-3 scalar | Llama-2 | 2.01 bpw 70B 9.17 (GuidedQuant table): scalar nonuniform far worse than VQ |
| SpQR / OWQ | [2306.03078](https://arxiv.org/abs/2306.03078), [2306.02272](https://arxiv.org/abs/2306.02272) | Yes | ~3.1-3.9 | Llama | OWQ 3.1 bit ~ GPTQ 4 bit; irrelevant below 2.5 bits |
| HQQ | [blog](https://mobiusml.github.io/hqq_blog/) | Yes, no calibration | 2 (g16, ~26GB 70B) | Llama-2-70B | ppl 4.13 vs fp 3.12-ish (model card); scalar, group overhead heavy |
| Any4 / Lloyd-Max | search returned nothing usable | | 4 | | learned 4-bit LUT: not for sub-2-bit |
| SpinQuant / OSTQuant / FlatQuant | [2410.09426](https://arxiv.org/abs/2410.09426) | Learned rotations (calibrated) | W4A4 focus | Qwen2.5-14B/32B | W4A4KV4 Qwen2.5-14B SpinQuant 67.3 acc/ppl 6.55, OSTQuant 67.8/6.37; 32B SpinQuant 68.5/5.88. No W2/W3 weight-only numbers found. Online FlatQuant transforms add fitted compute: R4 risk |
| SliM-LLM | [2405.14917](https://arxiv.org/abs/2405.14917) | Yes | 2 avg (group mix 1-3) | Llama | KL-based group salience (not MSE) |
| GAMMA (learned per-module allocation + ILP) | [2605.18475](https://arxiv.org/abs/2605.18475) | Learns preferences with a reconstruction objective (no QAT) | 2.5 avg | Llama/Qwen 8-32B | matches fixed 3-bit at 2.5 avg; +7 over search-based allocators |
| LLM-MQ, CLADO | no numbers found | | | | ILP allocation on Taylor sensitivity; basis of NVIDIA ModelOpt AutoQuantize |
| Entropy coding (EntroLLM, Huff-LLM) | [2505.02380](https://arxiv.org/abs/2505.02380) | Yes, lossless on top of PTQ | 4-8 bit schemes | small models | Huffman gets near Shannon entropy; no sub-2-bit result; WaterSIC (MIT MEng 2026) benchmarks Huffman-GPTQ at 1-4 bpw on Llama/Qwen (results unread) |
| Flips / faithfulness | [2407.09141](https://arxiv.org/abs/2407.09141) | n/a | W8-W4 | Llama-2 7/13/70B | MMLU flips: GPTQ W4A16 8.3/6.5/4.3%, AWQ W4 7.5/6.4/4.1%, SmoothQuant W8A8 13.6/8.7/3.7%, GPTQ W8A16 0.3-0.6%; KL GPTQ-W4 0.02 (13B); flips~KL Spearman 0.98. Flips fall with model size |
| LFQ (logit-aware final-block quantization) | [2605.29756](https://arxiv.org/abs/2605.29756) | unread | low-bit | | generation-quality-aware last block: lead worth a read |
| Bonsai 8B (PrismML) | [whitepaper](https://github.com/PrismML-Eng/Bonsai-demo/blob/main/1-bit-bonsai-8b-whitepaper.pdf), [news](https://prismml.com/news/bonsai-8b), [HF](https://huggingface.co/prism-ml/Bonsai-8B-gguf) | Unknown; "built from Qwen3-8B" | 1.125 (Q1_0_g128: sign + fp16 scale/128) | Qwen3-8B arch | avg 70.5 vs 79.3, MMLU-Redux 65.7 vs 83.0, GSM8K 88 vs 93, IFEval 79.8 vs 81.5, BFCL 65.7 vs 81.0 |

## What our current pipeline is

`v2/vq/vq.py`: random-sign Hadamard/Kronecker incoherence, block-LDLQ, shared Gaussian k-means 8-dim codebook (QuIP#-lite / GPTVQ-like). Best 14B point: 1.654 bpw, MMLU 0.62 (agreement with base 0.705). The literature says the codebook family is already near the achievable frontier (E8/QTIP differences are ~0.3-1 ppl at 2 bits), so the remaining headroom is (a) the objective (error propagation, end-loss, flip-aware), (b) allocation, (c) the non-body bytes, and (d) scale.

Rate-distortion arithmetic (for Gaussian after incoherence): entropy-coded scalar sits ~0.25 bit above the Shannon bound; E8 lattice recovers ~0.11 bit of that; 256-state trellis (QTIP) is within ~0.1 dB (~0.02 bit) of the bound; k-means D8 K=2^13 probably in between. So an upgrade from k-means to trellis is worth at most ~0.05-0.1 bit, which on our measured slope (+4 MMLU pts per +0.1 bit on 14B) is about +2-4 pts. Meaningful but not enough alone to go from 0.62 to the 98% retention bar. The objective changes are the bigger unknown.

## Ranked top 5 experiments (Qwen2.5-14B/32B, optionally 72B)

All are weights-by-math, calibration only (legal under R4 as long as nothing is fitted that ships as extra computation; no LoRA, no codebook fine-tuning, no learned online transforms).

### 1. Fix and run the error-propagation (GPTAQ) target, combined with an end-loss-weighted Hessian (GuidedQuant), inside vq.py
- Run: replace the Hessian/target with (i) GPTAQ asymmetric target (layer must reproduce the FP model's output when fed the quantized model's inputs; adds the residual term dX*X^T cross-correlation to the LDLQ update) and (ii) GuidedQuant-style gradient-of-end-loss channel group weighting (per-output-group averaged Fisher, 1-4 groups). Use teacher-KL (not CE on labels) as the end loss so it directly targets flips. Our earlier `tgt` run (W*Hfq*Hqq^-1) gave 0% (model destroyed): that is an implementation/numerical failure (unstable inverse, no damping on the cross term), not evidence against the idea. Test first on 7B K=14500 (known: 94.9-96.2% INJ retention), then 14B/32B on the existing 1566 INJ + 1063 TOOLSEL items.
- Expected gain: GuidedQuant+QTIP at 2 bits closes ~10% of the ppl gap on 7B (6.82->6.11 vs fp 5.12, i.e. 41% of the gap on 7B, 9% on 70B); gains shrink with model size. For our 14B/32B I expect +1.5-3 pts retention at equal bytes (honest uncertainty: 0 to 5). GPTAQ reports gains mostly at low bit, which is our regime.
- Cost: 7B probe ~$3-5; 14B/32B ~$15-30 on one A100/H100. Risk: medium (numerics; GPTAQ cross term needs FP activations stored for each layer: 2x calibration memory). Legality: pure calibration math, no fitted shipped parts. PASS.

### 2. Swap the k-means codebook for QTIP-style trellis-coded quantization (code already in v2/vq/tcq.py), with QTIP's 1MAD/3INST computed codes
- Run: V=1 or 2 trellis with 1.5-1.7 bpw (verify the QTIP implementation supports fractional K; if not, interpolate via per-layer mix of K=1 and K=2 and the allocator). Computed codes mean zero codebook bytes (frees ~0.5-1 MB per shared codebook, trivial but also removes the K-vs-bpw coupling that forced us to K=14500). Keep without FT variant. Compare equal-bytes against K=14500 VQ on 7B, then 14B.
- Expected gain: QTIP no-FT beats QuIP# no-FT by 0.3-0.5 ppl at 2 bits on Llama-2 and 0.5 on Llama-3-8B; LLVQ (Leech) shows another ~0.5-1 ppl on Qwen3-8B at 2 bits without FT (10.19 vs QTIP 11.17 vs fp 8.99) with no codebook storage. Predicted: +2-4 pts retention at 1.6 bpw on 14B, because it moves the quantizer closer to the RD bound (see arithmetic above). 
- Cost: ~$20-40 (decoding trellis in the eval path; Viterbi quantization is the slow part, QTIP reports hours for 70B on 8 GPUs-class). Risk: medium-high (engineering; fractional rates). Legality: PASS if decoding is a fixed computed code; judged as quantization format, not a fitted component. Alternative cheaper variant: Leech/E8 lattice shaping (LLVQ, no stored codebook).

### 3. KL/flip-aware per-matrix bit allocation (use `quality/alloc.py`, sens mode) on 14B and 32B
- Run: measure per-matrix sensitivity as the teacher-student KL (or top-1 flip count on 1-2k calibration prompts incl. the task items' distribution) when that matrix alone is quantized to K=2^11..2^15; greedy/ILP allocate K per matrix to hit exactly 9.6% including embeddings/head at lower precision. Previous 7B sens runs exist (`quality/results/sens7b_*`). Compare uniform vs KL-allocated at the same total bytes; then add the SliM-LLM trick of mixed group sizes inside a matrix.
- Expected gain: GAMMA reports 2.5-bit mixed matching uniform 3 bit (~0.5 bit equivalent); SliM-LLM similar. At 1.6 bpw average the Qwen outlier layers (first/last blocks, v/o, down_proj) typically absorb the sensitivity, so +2-5 pts is plausible; our own sens12 mode was never run on 14B. Prior within-repo evidence: uniform K=14500 (1.72) vs alloc170 are within noise on 7B, so treat as unproven.
- Cost: ~$10-20 for the sensitivity sweep (one forward per matrix per K, cheaper with a 1st-order KL Taylor proxy). Risk: low-medium (noise 2 pts means need paired evaluation on identical items). Legality: PASS.

### 4. Entropy-coded VQ indices (ECVQ) with a larger codebook at equal rate, plus entropy-coded embedding/head
- Run: after LDLQ with a bigger codebook (K=2^16-2^18 for D=8), replace fixed-length indices by Huffman/ANS coding of indices (index frequencies are skewed because Gaussian cells near the origin are more probable), and choose K and a Lagrangian rate term (ECVQ: minimize D + lambda*R in the k-means step) so that the average code length hits 1.55-1.6 bpw. Existing `v2/entropy` and `vq/ecsq_ref.py` contain scaffolding. Also entropy-code the 2.125-bpw vocab/head bytes (a few percent of total at 72B: Qwen2.5-72B embedding+head about 3.4% of params).
- Expected gain: for a Gaussian source at 1.6 bits, ECSQ is ~0.2 bit worse than the RD bound while fixed-rate D8 VQ is ~0.1 bit worse; ECVQ at D=8 should recover ~0.03-0.08 bit equivalent of body rate and 0.1-0.3 bit on the non-body bytes, which at +4 pts per 0.1 bit is +1-3 pts, plus it removes the K=14500 rounding hack. The Beyond-4-Bits paper counts "everything" (indices+codebooks+scales) to reach 2.09-2.33 bpw, showing overhead accounting matters.
- Cost: ~$10-15 (CPU-heavy coding, one eval pass). Risk: medium (decode speed; runtime stays a decode to fp16 so fine for the challenge). Legality: PASS (lossless coding of a fixed quantization; weights are not trained).

### 5. Run the best recipe at scale (Qwen2.5-72B, ~9.6%) with teacher-KL calibration data, and measure flips directly
- Run: the C04/C11 plan (`quality/cfgs_72.json`) using 1-4 above, calibration = 300k-1M tokens mix (generic + the task's own distribution + self-distilled outputs of the base model; 7B results show 300k beat 1M, noise 2 pts so use 3 seeds and a paired bootstrap). Report 98%/95% gate, flip rate and top-1 agreement on the real INJ/TOOLSEL items. Reference: QTIP 2-bit Llama-2-70B loses 1-3 zero-shot pts, Llama-3.1-405B-Inst ~1; 72B at 1.6 bpw is expected ~4-8 pts below fp on QA and worse on generation, so success probability for the strict gate stays low (my estimate 5-10%), but only scale + better objective gives any chance.
- Expected gain: scale effect is larger than any quantizer tweak: flips drop with size (Llama-2-70B GPTQ-W4 4.3% vs 7B 8.3%); our own data: C04 72B scalar 95.2% vs 7B VQ 94-96% at similar bytes; 72B VQ should land 97%+ on easy tasks.
- Cost: ~$40-60 (A100/H100 8-GPU-hours class); Risk: high on gate R2 per-class (95%) because rare classes/cells collapse first; use a task whose classes are well represented. Legality: PASS.

### Not recommended (and why)
- AQLM/PV-Tuning, PTQ1.61 LoRA step, UniSVQ block-wise FT, "codebook-only recovery" of Beyond 4 Bits, LLVQ's learned input scales: all fit parameters by gradient training on data; R4 / "can't train weights" exposure.
- FlatQuant/SpinQuant learned rotations: shipped online transforms are computation fitted to data (R4). Fixed random Hadamard (what we already use) is fine.
- BiLLM/ARB/STBLLM/PB-LLM/Bonsai-style 1-bit: perplexity 2-10x fp16 on 70B; no path to a 98% retention gate.
- Scalar mixed-precision (OWQ/SpQR/SqueezeLLM): sub-2-bit collapse.

## Gaps and caveats
- Several PDFs (GLVQ, STBLLM, DB-LLM, GPTAQ tables, WaterSIC, 2512.21651 Qwen2.5 tables) could not be extracted; their numbers are not here. Worth a second read: [2512.21651](https://arxiv.org/abs/2512.21651) (1-bit PTQ evaluated on Qwen2.5), [WaterSIC thesis](https://people.lids.mit.edu/yp/homepage/data/theses/2026_MEng_Lifar.pdf) (Huffman-GPTQ vs Qwen at 1-4 bpw), [LFQ 2605.29756](https://arxiv.org/abs/2605.29756).
- QTIP Table 7/8 do not state whether fine-tuning was applied; do not assume the Llama-3 numbers are no-FT.
- Search snippets for AQLM bit accounting, HQQ 70B, and VPTQ HF-doc numbers are secondary (unverified).
- Bonsai's training procedure remains unpublished: do not cite it as a post-training result.
