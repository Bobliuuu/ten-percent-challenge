# v2/entropy: lossless squeeze of GPTQ-quantized Qwen2.5-14B-Instruct

Code: `coders.py` (numba adaptive models + real LZMA-style range coder), `run.py` (Modal: dump GPTQ indices, analyze), `table.py`/`summ.py` (tables), raw: `res_L{4,6,8,16}_rot{0,1}.json`, `table_out.txt`, `cfgs.json`. Modal app ap-Z5sHkXMgfCjOah2aIsBJLY (volume hf-cache:/v2entropy/*).

## Method
Engine-identical quantizer (v2/engine/sweep.py): optional block-128 random-sign Hadamard (rot1/rot0), sequential GPTQ, symmetric L-level grid, group g=128, per-group fp16 scale chosen by clip search. L=8 is the "3-bit" setting, L=16 "4-bit", L=4 "2-bit", L=6 intermediate.
Sampled 8 layers (0,1,8,9,24,25,40,47), all 7 linear modules, first 2048 rows each (645.9M params, ~4.9% of 13.2B body params), 64x512 calib tokens from calib.jsonl, fp propagation. Indices dumped as uint8.
Coding models (all adaptive, zero side info): single table per module (H0 = static histogram entropy, the usual baseline), and context models from neighbours: left 2 in row, 2 above, running column-mean |q| (outlier-channel detector), row-history, position-in-group, group scale bin, previous layer same position, products of these; plus logistic mixing of 4 context models. A real range coder (32-bit, LZMA carry, integer counts) encoded the best single context and was DECODED byte-exact for every module of every config (rc_all_ok=True, 56 modules x 8 configs). Real RC bytes are within 0.01% of the ideal -log2 cost.
Scales: fp16 scale, exponent coded with left-neighbour context (adaptive), mantissa 10 raw bits (incompressible, entropy of top mantissa bits ~ uniform).

## Bits per parameter (index stream only, weighted over sampled params)
| cfg | fixed | H0 (plain hist.) | best ctx | ctx mix | real range coder | RC gain vs H0 |
|---|---|---|---|---|---|---|
| L4 rot1 | 2 | 1.918 | 1.9125 | 1.912 | 1.911 | 0.007 (0.36%) |
| L6 rot1 | 3 | 2.428 | 2.4186 | 2.419 | 2.417 | 0.011 (0.45%) |
| L8 rot1 (3-bit) | 3 | 2.783 | 2.7713 | 2.772 | 2.770 | 0.013 (0.48%) |
| L16 rot1 (4-bit) | 4 | 3.650 | 3.6328 | 3.636 | 3.632 | 0.018 (0.49%) |
| L4 rot0 | 2 | 1.863 | 1.8348 | 1.834 | 1.834 | 0.030 (1.6%) |
| L6 rot0 | 3 | 2.335 | 2.2938 | 2.294 | 2.293 | 0.042 (1.8%) |
| L8 rot0 (3-bit) | 3 | 2.661 | 2.6147 | 2.616 | 2.613 | 0.048 (1.8%) |
| L16 rot0 (4-bit) | 4 | 3.463 | 3.4053 | 3.409 | 3.404 | 0.059 (1.7%) |

Findings: (1) almost all gain over fixed-width is the plain histogram (non-uniform use of grid levels: 0.2-0.5 b); (2) context modelling adds only 0.5% (rotated, Gaussian-ish iid after Hadamard) to 1.8% (unrotated, where outlier input channels leave column structure; running column mean is the best context, row-neighbour/position/prev-layer/scale-bin contexts give ~0; previous-layer context 0 gain). Mixing did not beat the best single context. Rotation hurts compressibility (rot1 index entropy is higher by 0.12-0.25 b than rot0) because rotation whitens the weights.
Scales: lossless fp16 scale costs 10.4-10.9 bits (exponent 0.4-0.9 b with context, mantissa 10 b) vs 16: saves ~0.04 bpp at g=128. Lossy 8-bit log scales would save another ~0.03 bpp but are not lossless (not tested for accuracy).

## Whole-model bytes (stored = coded indices + scales + REST). REST = 94.6 MB = 32768-token 4-bit embedding + norms/biases + 4 lm_head rows + 1 MB slack (as trackR r3). Body = 13.21B linear params. bf16 = 29,540,067,328 B; 10% = 2,954,006,733 B. Extrapolated from the 4.9% sample to all layers.
| cfg | fixed+fp16 scale | H0 + fp16 scale | RC ctx + lossless scale |
|---|---|---|---|
| L8 rot0 (3-bit) | 0.1779 | 0.1589 | **0.1539** (4.546 GB, 2.696 bpp) |
| L8 rot1 | 0.1779 | 0.1658 | 0.1626 (4.804 GB) |
| L16 rot0 (4-bit) | 0.2338 | 0.2038 | **0.1982** (5.856 GB, 3.489 bpp) |
| L16 rot1 | 0.2338 | 0.2142 | 0.2109 |
| L6 rot0 | 0.1779 | 0.1407 | 0.1361 |
| L4 rot0 (2-bit) | 0.1220 | 0.1143 | 0.1103 (3.257 GB) |
| L4 rot1 | 0.1220 | 0.1174 | 0.1146 |

Accuracy (not re-measured here; from trackR r3, same quantizer family, unrotated GPTQ, MMLU dev n=1987, base 0.8007): 4-bit g128 acc 0.7876 (agree 0.933); 3-bit g128 acc 0.7539 (agree 0.883); 2-bit g64 acc 0.603. Hadamard-rotated accuracies are in the v2/engine sweep (not finished when this was written), so rot1 rows are bytes-only.
=> 3-bit at dev acc ~0.754: lossless coding alone takes ratio from 0.179 (fixed 3-bit + fp16 scales) -> 0.154 (full lossless stack); trackR's plain-histogram figure was 0.1666 so our real gain over what everyone already does is 0.0127 ratio (~7.6% of bytes, of which scales ~1/3 and context modelling ~2/3 ... see tables). 4-bit at acc 0.788: 0.234 -> 0.198. Reaching 0.10 at 3-4 bit accuracy is NOT possible by lossless coding: at L8 you would need 1.6 total bpp vs 2.70 achievable; even L4 (2-bit, acc collapses to ~0.60) only reaches 0.110. Lossless coding is worth ~15% of bytes vs fixed-width, ~3-5% beyond plain histogram coding.

## Stored-bytes definition / decoder
Counted: coded index streams, scales (coded as above), embedding, norms/biases, head rows, slack for tokenizer map. Not counted: decoder code (a ~150-line range coder + context update; the best context needs only per-module counts (L ints) adaptive state, no stored tables), reported separately. Contexts use only already-decoded symbols (causal), so no side info.

## Failed / null
prev-layer same-position context: ~0 gain; position in group: ~0; group-scale bin: ~0.003 b; logistic mixing: no gain over best single ctx (worse by 0.001-0.004 with these inputs); hist per 128-group instead of per module not tried (adaptive per-module already within 0.01% of per-module static).
## Ideas with real potential (lossy side, not lossless)
Entropy-constrained GPTQ (rate-aware rounding / deadzone), larger groups with coded scales, unrotated + outlier-column handling (rot0 is both smaller and compressible).
