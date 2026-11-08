# BYTE_TRICKS (lane L13): what each byte trick frees, and what it costs (2026-11-08)

Unit everywhere: **pp = percentage points of the base model's fp16 size** (the rule-1 denominator; the cap is 10 pp). "% of budget" = pp / 10. Measured on REAL tensors of our start models (not simulations of Gaussians): Qwen2.5-14B-Instruct, Qwen2.5-32B-Instruct, Qwen3-30B-A3B, Bonsai-8B-unpacked, Ternary-Bonsai-27B-unpacked. Raw numbers: `v2/byte_tricks/results/` (stats/, plus `*.jsonl` arms). Code: `v2/byte_tricks/` (btcore.py quantizer + exact bit accounting, run_arms.py, stats_weights.py, stats_embed.py, accounting.py). Audit tool for lanes: `python3 v2/tools/bytes_audit.py ARTIFACT_DIR --base-params N [--dups --compress]`.
Retention = P(correct | base correct), real greedy generation of the label word, EASY-A SKILL7 dev items (n=682; base .957 on Qwen2.5-14B-Instruct) and BANK6 `sel` (canonical, never the exam). Wilson 95% lower bound on retention. Rotation = random-sign block Hadamard, GPTQ (sequential, Hessians from 64x512 calibration tokens), uniform grid, MSE-clip scale search, fake-quant bf16 weights.

## 0. Status
- DONE: byte statistics on all five models (section 2), the audit tool, Lagrangian/KL allocation code, embedding/vocab code.
- IN PROGRESS at the time of this push: retention arms at L=3 (1.39 bpw index, the real operating point) on Qwen2.5-14B (SKILL7, BANK6 sel); sections marked PENDING fill in as results land (see the arms table, section 3). At L=5 (2.17 bpw, 12.8% of fp16, i.e. above budget) retention saturates at 99.2% for every scale variant, so only L<=4 arms can separate tricks.

## 1. Ranked table (bytes freed, all numbers pp of base fp16; "needs" = what a lane must do)
Measured byte numbers; retention column filled from section 3 where a run exists.

| rank | trick | frees (pp) at 14B / 32B / 30B-A3B / Bonsai-8B / Ternary-27B | retention cost | legality (R1 / R9 / code bytes) | needs |
|---|---|---|---|---|---|
| 1 | **Lossless exponent coding of emb/head tables kept in bf16/fp16** (rANS over the 8-bit exponent; 10.6 bits instead of 16 for Qwen bf16, 12.0 for Bonsai fp16, 9.0-9.3 for Ternary-Bonsai bf16) | 3.55 / 1.6 / 0.7 / 3.8 / 4.0 (only if the lane would otherwise store emb+head in 16 bits; zero if it already quantizes them to 2-4 bits) | 0 (lossless, bit-exact) | clean; decoder ~1 kB python | rANS decoder at load |
| 2 | **Fixed-width packing -> ANS of quantized indices** (L=5: 3 bits packed vs 2.17 entropy; L=3: 1.58 trit vs 1.39; L=8: 3 vs 2.79) | L5: 4.6 / 4.9 / 5.1 pp; L3 (vs 2-bit packing 2.0 -> 1.39): ~3.4 pp ; vs trit-packing 1.585 -> 1.39: 1.2 pp | 0 (lossless) | clean; rANS decoder counted | every lane that bit-packs; lanes already on rANS (dense_core) get 0 |
| 3 | **Coded scales: log-scale (1/16 octave) with entropy coding instead of fp16** (g128: 0.125 -> 0.0295 bpw; g256: 0.0625 -> 0.0153) | 0.53 / 0.57 / 0.58 pp vs fp16 g128; vs the repo's old 11-bit field: 0.32 / 0.34 / 0.34 | none measured at L=5 (99.2 vs 99.2); L=3 PENDING | clean | log-code + rANS |
| 4 | **Vocabulary trim** (generic rule id<65,536 plus specials, byte-fallback pieces kept, rows removed from emb AND head; input re-tokenised greedily to kept pieces) | with emb/head at 4 bit: 1.50 / 0.68 / 0.29 / 2.16 / 1.32 pp; at 2 bit half of that; at 16 bit 4x | PENDING (V_trim arms) | R9 gray (non-English/code inputs shatter into byte pieces; label tokens must be inside the kept set), per PRUNE_LEGAL | tokenizer remap code counts |
| 5 | **Exact-zero / padding rows** (Qwen2.5-14B/32B embed has 2,355 all-zero rows = unused ids 151,665..) | 0.16 pp at 16-bit, 0.02 at 2-bit | 0 | clean | none |
| 6 | **Remove unused modality tower** (Ternary-Bonsai-27B ships a vision tower = 1.68% of its fp16 base params) | 1.68 pp for Ternary-27B (0 for the others) | 0 on text task | R1 literal OK, R9 gray (denominator choice), disclose | drop tensors |
| 7 | **Lossless coding of discrete models' scale tables** (Bonsai-8B: fp16 scales take only ~344 distinct values/tensor, 7.07 bits entropy vs 16; Ternary-27B: 215 distinct, 5.85 bits) | Bonsai-8B 0.37 pp, Ternary-27B 0.39 pp (value-coding); log-16 coded 3.2 / 3.0 bits => 0.56 / 0.46 pp | 0 lossless, small if log-rounded | clean | rANS |
| 8 | **Larger groups / shared-row scales** (g256, g512, 4 rows x 128 share one scale) | g256 vs g128 (coded): +0.09 pp scale +0.20 pp index entropy; share4: +0.5 pp (0.0295 -> 0.0072 bpw, idx -0.19 bpw) | g256/g512: 99.1 vs 99.2 at L5; share4 PENDING | clean | none |
| 9 | **Tokenizer: ship only tokenizer.json (drop vocab.json+merges.txt) and xz it** | 14B: 11.5 MB -> 1.4 MB = 0.034 pp; 32B 0.015; Bonsai-8B 0.088; Ternary-27B 0.037 | 0 | clean (xz is Python stdlib) | loader unpacks |
| 10 | **Exponent coding of the fp16 remainder (norms, biases)** | 0.002 pp | 0 | clean but negligible | skip |
| 11 | **Context-model ANS over index streams** (neighbour / scale-bin contexts) | 0.05-0.1 pp (0.5% of index bits; v2/entropy/RESULTS.md, rotated) | 0 | decoder code grows | skip |
| 12 | **Zstd/xz on the container** | 0 on quantized streams (ratio 1.00), 0.03 pp on tokenizer/config only (covered by row 9) | 0 | clean | skip for weights |

PENDING rows (need retention): data-free quantizers (HIGGS / uniform RTN / dead-zone RTN vs GPTQ), embedding VQ / low-rank / fp8 / 4-bit, bit allocation (Hessian trace, KL) vs uniform, scale-bit variants at L=3. AIMER-style data-free MoE expert scoring: not run (needs router statistics on the 61 GB model; no lane asked for it; see NEAR_MISSES).

## 2. Measured byte statistics (real tensors)
Fractions of params (fp16 base = 2 bytes x params): see `results/stats/*.json`.

| model | params | linear/expert | embed + head | modality | untied? | embed==head? |
|---|---|---|---|---|---|---|
| Qwen2.5-14B | 14.77 B | 89.5% | 5.27 + 5.27% | 0 | untied | no (row cos 0.003) |
| Qwen2.5-32B | 32.76 B | 95.2% | 2.38 + 2.38% | 0 | untied | no |
| Qwen3-30B-A3B | 30.53 B | 98.0% (95% experts) | 1.02 + 1.02% | 0 | untied | no (cos 0.045) |
| Bonsai-8B | 8.19 B | 84.8% | 7.59 + 7.59% | 0 | untied | no (cos 0.003) |
| Ternary-Bonsai-27B | 27.36 B | 89.0% | 4.65 + 4.65% | 1.68% | untied | no |

Scale table cost per weight (bpw), Hadamard-rotated Qwen tensors, L=5 uniform grid (idx entropy 2.17 bpw at g128, 2.14 g256, 2.10 g512):
| scheme | g128 | g256 | g512 | g128 share4 |
|---|---|---|---|---|
| fp16 | 0.1250 | 0.0625 | 0.0312 | 0.0312 |
| 8-bit fixed | 0.0625 | 0.0312 | 0.0156 | 0.0156 |
| log-16th-octave + rANS (H0) | 0.0349 | 0.0173 | 0.0086 | 0.0083 |
| + row/col median context | 0.0295 | 0.0153 | 0.0083 | 0.0072 |
(identical to 3 digits on 14B and 32B; 30B-A3B experts 0.030 for H0, contexts do not help there.) The scales of a rotated tensor are nearly constant, so a coded table costs ~3.8 bits per scale vs 16 (fp16) or 11 (repo's old field).

## 3. Retention arms (Qwen2.5-14B-Instruct)
PENDING: filled from `results/*/q14_*_L3*.jsonl` (summarised by `python3 v2/byte_tricks/summarize.py`). Saturation result so far (L=5, SKILL7 dev, 653 base-correct): g128 fp16 99.23% [98.22]; g256 fp16 99.08% [98.01]; g512 fp16 99.08% [98.01].

## 4. Recommended default stack per model class (current best guess; confirm vs section 3)
- **Dense Qwen2.5-14B/32B (lanes L5)**: rANS indices (already), log-16th-octave coded scales g128 (+0.3 pp vs the old 11-bit field), emb/head quantized to 2-4 bit with the lossless exponent trick only if kept at 8/16 bit, vocab trim only with disclosure. Total identified free: ~0.3 (scales) + 0.04 (tokenizer) + 0.16 (zero rows) = 0.5 pp on top of an already-entropy-coded lane; up to 4-8 pp for a lane that is still bit-packing or keeping emb/head in 16 bit.
- **MoE Qwen3-30B-A3B (L3/L4/L9)**: coded scales (0.58 pp vs fp16 g128, large because experts are 95% of params), rANS instead of int packing (L3 vs 2-bit packing 3.4 pp), tokenizer 0.02 pp; emb/head are only 2 pp of base so vocab trim is worth <=0.3 pp: not worth the rule-9 risk.
- **Bonsai-8B / Ternary-Bonsai-27B (L1/L2)**: emb/head are 15% / 9% of base and stored 16-bit: lossless exponent coding frees 3.8 pp (Bonsai-8B, fp16 tables) and up to 4.0 pp (Ternary-27B bf16 tables, IF the lane ships them unquantized; the packed release already stores them low-bit per NATIVE_LOWBIT.md, so check the real file first); scale-table value coding 0.37-0.56 pp; remove vision tower 1.68 pp (Ternary-27B). For Ternary-Bonsai-27B the previously estimated 10.67% (NATIVE_LOWBIT.md) minus 0.39-0.49 pp (coded scale table) is ~10.2%; the tower removal (1.68 pp, gray, disclose) would take it to ~8.5%.

## 5. Legality notes
R1 (all bytes counted): every number above is bytes actually written; shipped decoder code (rANS ~1-3 kB python, tokenizer remap) counts and is already netted out (<0.001 pp). R9 (spirit): lossless tricks are clean; vocab trim and tower removal are gray (documented in PRUNE_LEGAL.md and RESEARCH_EXHAUSTIVE.md X16); label-selected row removal would be a head (rule 4/7) and is never used. No trick here fits anything to feedback data; none trains weights (GPTQ error feedback and learned rounding are the quantizers' existing gray flags, unchanged).
