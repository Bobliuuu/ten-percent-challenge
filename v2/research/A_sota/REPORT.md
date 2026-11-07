# Track A: training-free compression to ~1.5-1.6 bpw (2026-11-06)
Source: subagent hand-back (model output, web research; not independently verified). Tags: [V] page fetched and read, [S] search snippet only, [M] from memory, [T] agent's own inference. Nothing was run on a GPU. Reference clone script (untested): setup_refs.sh in this folder.

## Bottom line
1. No published training-free method was found that reaches >=98% task-level retention at ~1.6 bpw for a 14B-class model. Sub-2-bit PTQ results are perplexity/multiple-choice, mostly on 70B, and the strong ones use block-wise fine-tuning or QAT. "Near-lossless at 1.5 bits" claims are unmeasured on generation.
2. A better quantizer buys about 0.1-0.3 bit [T]: entropy-coded scalar quantization is within ~0.25 bit of the Gaussian rate-distortion bound. Our own 14B ladder: KL falls ~6-9x per bit (1.26 @1.48 bpw, 0.176 @2.39, 0.020 @3.75), so 0.25 bit is only ~1.5-1.8x lower KL; ~2 bits' worth is needed.
3. The gap must come from removing redundancy before quantizing, or from a larger base. Our data: Qwen3-8B keeps 22%, Qwen3-14B 56% of base-correct tool-call items at 9.6%. Agent's guess for 32B dense: 80-95% [T], must be measured.
4. Dense structural pruning without retraining is weak; do not expect a free 2x.

## Ranked methods (expected gain for task retention at ~9.6% of fp16)
| # | Method | TF? | Effort h | Expected gain |
|---|---|---|---|---|
| 1 | Bigger base (Qwen3-32B dense, 30B+ MoE) with existing pipeline | yes | 6-12 | LARGE, best supported |
| 2 | REAP task-calibrated expert pruning (MoE) + spend freed bytes on precision (arXiv 2510.13999, github.com/CerebrasResearch/reap, Apache-2.0 [V]); claims near-lossless code generation at 50% pruning on Qwen3-Coder-480B/Kimi-K2; no tool-call numbers | yes | 8-16 | LARGE on MoE |
| 3 | QTIP trellis, no-finetune variant, mixed K per tensor (github.com/Cornell-RelaxML/qtip, GPL-3.0, keep out of artifact) | gray (default script fine-tunes) | 20-40 | small-medium (~0.15-0.3 bit) [T] |
| 4 | Per-tensor sensitivity-based bit allocation (Unsloth-dynamic style, KL/Hessian sensitivity, water-filling) | yes | 6-10 | medium (~0.3-0.5 bit) [T] |
| 5 | Error-propagation-aware GPTQ (GPTAQ, Qronos arXiv 2505.11695, ADMM-Q) | yes | 6-12 | small-medium |
| 6 | Task-calibrated width pruning before quantization (FLAP/Wanda-sp/SliceGPT) | yes | 10-16 | small |
| 7 | VPTQ / AQLM codebooks | gray (block fine-tuning) | 24-48 | small-medium |
| 8 | Binarization with residuals (BiLLM, ARB-LLM, PTQ1.61, BTC-LLM, QTEA) | mostly yes | 16-30 | low; weak baselines, perplexity only |
| 9 | Low-rank/SVD (+quant); EoRA | EoRA NO (adapters banned), SVD-LLM replacement yes | 8-12 | low |
| 10 | Entropy coding beyond rANS | yes | 10+ | very low (ANS already within 0.01-0.1 bit) |
| 11 | Weight sharing / cross-layer dedup | yes | n/a | ~zero |
| 12 | SLQ (IST-DASLab; arXiv 2605.02404): confirms 3.3-6 bits needed for near-lossless; reusable bit search and EAR metric | yes | 6 | metric/allocator only |
| 13 | Fixed Hadamard rotations (have it) / learned rotations | fixed yes, learned gray-no | 0 | none |

Tool-calling under quantization: small models fragile on BFCL (Qwen-3 0.6B 62.3 -> 16.8 at 4-bit [S]); quantization amplifies failures such as tool-name hallucination up to 2.5x that benchmark scores hide [V arXiv 2607.27275]; no source reports tool-call retention at ~1.5-2 bits. Where sub-2-bit works in practice: huge MoE (DeepSeek-R1 671B ~1.58-bit mixed precision, anecdotal).

## Recommended first five
R1 larger base at 9.6% (decision rule: 32B retention >=85% -> remaining gap 1.5-2x closable by R2-R4; <70% -> dense dead, MoE-only).
R2 REAP pruning (25/50/62.5%) + precision reinvestment; calibrate on our own tool-call training data (never locked/S2) with several schema styles and a general slice; risk: looks perfect in-distribution and collapses on S3/maintainer data; watch S1-vs-S3 gap.
R3 per-tensor sensitivity allocation (first/last layers, attention, router, embeddings at 4-6 bit; MLP/experts 1.2-1.6; down_proj most sensitive). Check embeddings: Qwen3-14B untied embed+lm_head ~10% of params.
R4 QTIP no-finetune with mixed K for MLP weights; gate: keep only if KL at equal bytes < 0.6x current ECSQ at 8B.
R5 error-propagating layerwise compensation (Hessian from already-quantized prefix); ablate calibration-set composition (task-format vs generic).
Fallbacks: width pruning, SLQ bit search, depth pruning of 1-3 late-middle layers (BlockPruner).
Exam-side lever (flagged, not recommended): retention by kind was nocall 79%, single 54%, multi 25%; changing the kind mix is legitimate only if it reflects a deployable task (rule 9).

## Not allowed / too risky
End-to-end fine-tuning, distillation, QAT, PV-tuning, AQLM end-to-end stage, trainable VQ, learned rounding/codebooks (REAL-Q, GSQ, GLVQ), learned rotations (gray at best); adapters of any kind incl. training-free (EoRA); fitted heads/stand-in outputs/constrained decoding; schema/examples/feedback in the prompt; using locked/S2 for calibration, pruning, allocation or hyperparameters; undercounting bytes (codebooks, scales, indices, rANS tables, tokenizer, code, config); going over 10% even by 0.1%. Gray: block-wise fine-tuning inside QuIP#/AQLM/VPTQ/QTIP scripts, expert merging with learned weights, any pruning whose calibration includes the exam source.

## Not verified
No sub-2-bit tool-calling retention for any method (all expected gains are [T]); REAP numbers on Qwen3-30B-A3B/tool calls not extracted; licences marked [M] unverified; many arXiv IDs are 2026 snippets only.
