# BYTE_TRICKS (lane L13): what each byte trick frees, and what it costs (2026-11-08)

Unit everywhere: **pp = percentage points of the base model's fp16 size** (the rule-1 denominator; the cap is 10 pp). "% of budget" = pp / 10. Measured on REAL tensors of our start models (not simulations of Gaussians): Qwen2.5-14B-Instruct, Qwen2.5-32B-Instruct, Qwen3-30B-A3B, Bonsai-8B-unpacked, Ternary-Bonsai-27B-unpacked. Raw numbers: `v2/byte_tricks/results/` (stats/, plus `*.jsonl` arms). Code: `v2/byte_tricks/` (btcore.py quantizer + exact bit accounting, run_arms.py, stats_weights.py, stats_embed.py, accounting.py). Audit tool for lanes: `python3 v2/tools/bytes_audit.py ARTIFACT_DIR --base-params N [--dups --compress]`.
Retention = P(correct | base correct), real greedy generation of the label word, EASY-A SKILL7 dev items (n=682; base .957 on Qwen2.5-14B-Instruct) and BANK6 `sel` (canonical, never the exam). Wilson 95% lower bound on retention. Rotation = random-sign block Hadamard, GPTQ (sequential, Hessians from 64x512 calibration tokens), uniform grid, MSE-clip scale search, fake-quant bf16 weights.

## 0. Status (final for this pass, 2026-11-08)
All arms ran to completion on Qwen2.5-14B-Instruct: SKILL7 dev (n=682, 653 base-correct, base .957) and BANK6 `sel` (canonical set, n=342, 302 base-correct, base .883); never the exam, never a locked/sealed file. 3 Modal A100-80GB apps + stats apps, all stopped; no Vast box left. Raw: `v2/byte_tricks/results/L3*/*.jsonl`, merged table `results/FINAL_TABLES.md`. Seed noise floor measured (section 3): identical config with different Hadamard seeds spreads +-1.5 pp (SD) at L=3, so differences under ~3 pp at L=3 are noise. At L=5 (12.3 pp, above budget) everything saturates at 99%, so the table uses L=3 (7.9 pp body) and L=4 (10.4 pp body). Not measured: Bonsai / Ternary-Bonsai / MoE retention (byte statistics only; their lanes own retention), AIMER-style expert scoring (needs router stats; not run), 32B retention.

## 1. Ranked table
pp = percentage points of base fp16 size freed. Two baselines: (N) naive = fp16 scales g128, fixed-width bit-packed indices, fp16 emb/head, 3 tokenizer files; (R) repo practice already = rANS indices, 11-bit scale field, emb/head ~2 bpw. Retention = SKILL7 dev / BANK6 sel, GPTQ+Hadamard g128, "isolated" = only that component changed, everything else fp16.

| rank | trick | pp freed on 14B vs N / vs R | on other models (vs N) | retention cost (14B) | legality | verdict |
|---|---|---|---|---|---|---|
| 1 | **Quantize emb + lm_head to 2 bit (RTN+Hadamard g128, coded) instead of fp16** | 9.2 / 0 (R already ~2 bpw) | 32B 4.2, 30B-A3B 1.8, Bonsai-8B up to 13 (retention not measured), Ternary-27B up to 8 | isolated: 99.8 [99.1] SKILL7, 99.3 [97.6] BANK6; 4-bit 100.0; 3-bit 100.0/99.7; fp8 100.0 (5.3 pp freed) | clean (all bytes counted) | ADOPT: fp8 free of risk, 4-bit/2-bit safe |
| 2 | **Embedding/head VQ d=8,K=256 (1 bpw)** | 9.9 / 0.65 over 2-bit | - | 99.4 [98.4] SKILL7 but 96.4 [93.6] BANK6 | clean | too lossy on BANK6; d=4 (2 bpw) = scalar 2-bit, no gain; low-rank r=1024 98.5, worse |
| 3 | **ANS/rANS indices instead of fixed-width packing** | L3: 2-bit pack 2.0 -> 1.38 entropy = 3.5 / 0; L4 vs 2-bit 0.9; L5 3.0->2.17 = 4.6 | 32B 3.7, 30B-A3B 3.9 | 0 (lossless) | clean, decoder counted | ADOPT if not already (context models add only 0.05-0.1 pp: skip) |
| 4 | **Vocabulary trim to ids < 65,536 + specials (byte pieces kept), rows removed from emb AND head** | 6.0 from fp16 / 0.74 from 2-bit tables | 32B 0.3, 30B-A3B 0.1, Bonsai-8B 1.0 | isolated: 100.0 [99.4] SKILL7, 99.7 [98.1] BANK6 (2.7% of dev tokens re-split into pieces); N=100k 100.0; **N=40k: 98.8 / 95.0 and label tokens fall outside the kept set = FAIL** | R9 gray: non-English/code inputs shatter; pre-registered label-free rule only; measure OOD | USE ONLY with disclosure, N >= 65,536 |
| 5 | **Coded scales: log-scale 1/8 octave + rANS (0.024 bpw) instead of fp16 (0.125) at g128** | 0.57 / 0.32 (vs the 11-bit field) | 32B 0.57, 30B-A3B 0.58, Bonsai-8B 0.37-0.56, Ternary-27B 0.39-0.46 | **none; if anything better**: L3 SKILL7 coded 91.1/89.1/88.2 (3 seeds, mean 89.5) vs fp16 83.6/87.0 (mean 85.3); BANK6 coded(1/4 oct) 96.7 vs fp16 95.7. 1/4-octave 91.3 = 1/8 = 1/16(86.4) within noise | clean | ADOPT (1/4 or 1/8 octave) |
| 6 | **Spend the freed scale bytes on g=64** (coded) | costs +0.46 pp (0.049 bpw vs 0.024) | - | **+5 to +9 pts**: g64 coded 96.2 [94.4] / 96.0 [93.2] vs g128 coded ~89.5 / ~96 and fp16-g128 85.3 at LESS bytes than fp16-g128 (8.33 vs 8.44 pp body) | clean | ADOPT: best use of the freed bytes at L=3 |
| 7 | **Tokenizer: keep only tokenizer.json, xz it** | 0.034 / 0.034 | 32B 0.015, Bonsai-8B 0.088, Ternary-27B 0.037 | 0 | clean (stdlib lzma) | cheap; matters only below ~3B |
| 8 | **Lossless exponent coding of fp16/bf16 emb/head if kept 16-bit** | 3.55 | 32B 1.6, 30B-A3B 0.7, Bonsai-8B 3.8, Ternary-27B 4.0 (bf16) | 0 | clean | only for lanes that keep tables 16-bit; superseded by row 1 |
| 9 | **Exact-zero embedding rows (2,355 unused ids in Qwen2.5)** | 0.16 | same on 32B | 0 | clean | subsumed by vocab trim |
| 10 | **Remove vision tower (Ternary-Bonsai-27B)** | - | 1.68 | 0 on text | R1 literal ok, R9 gray, disclose | lane L2/L10 call |
| 11 | **Discrete-model scale tables, lossless value coding** (Bonsai-8B 7.07 bits vs 16, Ternary-27B 5.85) | - | 0.37 / 0.39 pp | 0 | clean | ADOPT for L1/L2 |
| 12 | **Zstd/xz on container, context-model ANS, exponent coding of norms/biases** | 0.00-0.002 | - | 0 | clean | skip (measured nothing) |
| - | **Data-free quantizers instead of calibrated GPTQ** | COST: need ~+2.5 to +4 pp more bytes | - | L3: RTN 0%, HIGGS-grid 0% (collapse). L5 (12.3 pp): RTN 96.0 [94.2], HIGGS 97.7 [96.2], GPTQ 98.9 [97.8]; BANK6 L5: RTN 88.1, HIGGS 89.4 vs GPTQ-L4 97.4 at 10.4 pp; RTN needs L=7 (14.7 pp) for 99.1; dead-zone RTN worse | clean | calibrated GPTQ wins by 4-5 pp of bytes; data-free only if a lane has no calibration data |
| - | **Hessian-trace / KL bit allocation per tensor vs uniform** at equal bytes | 0 | - | uniform L4 98.6 [97.4]; KL-allocated 98.5 [97.2] (=); trace-allocated 95.7 [93.9] (-2.9); allocation from L3 allowing 1-bit tensors: 0% | clean | NO GAIN at tensor level; skip |
| - | **Shared scale across 4 rows; larger groups g256/g512; asymmetric zero point** | share4 saves 0.9 pp body | - | share4 31.9 / 62.9 (collapse); g256 fp16 80.1 / 89.4 and coded 76.4; g512 46.6; asym L3 broken (0%) | clean | REJECT |

Total free, per model class, stacking ranks 1/3/5/7 (+4 if disclosed), 14B: vs N = 9.2 + 3.5 + 0.57 + 0.03 = 13.3 pp (+0.74 with trim = 14.0); vs R (lanes already on rANS and ~2-bit tables) = 0.32 + 0.03 (+0.74) = 0.4 to 1.1 pp. Of that, 0.46 pp should be spent on g64 (rank 6). 32B vs N: 4.2 + 3.7 + 0.57 + 0.02 = 8.5; vs R 0.35-0.65. 30B-A3B vs N: 1.8 + 3.9 + 0.58 = 6.3; vs R 0.35-0.5. Bonsai-8B (lossless only): 3.8 + 0.4-0.6 + 0.09 = 4.3-4.5 pp, +up to 9 if emb/head may be quantized (unmeasured on Bonsai). Ternary-Bonsai-27B: 0.4-0.5 scales + 4.0 exponent (if bf16 tables shipped) + 1.68 tower (gray).

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

## 3. Retention arms (Qwen2.5-14B-Instruct, SKILL7 dev; full merged table in `results/FINAL_TABLES.md`)
Seed noise (log8 coded g128 L3, Hadamard seeds 0/7/13): 91.1 / 89.1 / 88.2; fp16 g128 seeds 0/7: 83.6 / 87.0. Wilson lower bounds sit 2.5-3 pts below.
| arm (GPTQ+RHT unless noted) | idx bpw | scale bpw | body pp | retention [Wilson lo] | min cell |
|---|---|---|---|---|---|
| L3 g128 fp16 scales | 1.384 | 0.125 | 8.44 | 83.6 [80.6] | 0.43 |
| L3 g128 log 1/16 oct | 1.384 | 0.031 | 7.92 | 86.4 [83.5] | 0.65 |
| L3 g128 log 1/8 oct | 1.384 | 0.024 | 7.87 | 91.1 [88.7] | 0.77 |
| L3 g128 log 1/4 oct | 1.384 | 0.017 | 7.83 | 91.3 [88.9] | 0.78 |
| L3 g64 log 1/8 | 1.441 | 0.049 | 8.33 | 96.2 [94.4] | 0.90 |
| L3 g256 fp16 / log 1/8 | 1.329 | 0.0625 / 0.012 | 7.78 / 7.50 | 80.1 [76.9] / 76.4 [73.0] | 0.59 / 0.35 |
| L3 g512 fp16 | 1.279 | 0.031 | 7.32 | 46.6 [42.8] | 0.16 |
| L3 g128 share-4-rows | 1.231 | 0.0055 | 6.91 | 31.9 [28.4] | 0.08 |
| L4 g128 log 1/8 | 1.843 | 0.023 | 10.44 | 98.6 [97.4] | 0.93 |
| L5 g128 log 1/8 | 2.173 | 0.022 | 12.27 | 98.9 [97.8] | 0.96 |
| RTN L5 / HIGGS-grid L5 / RTN L7 / RTN dead-zone L7 | 2.17 / 2.21 / 2.62 / 2.53 | 0.02 | 12.3 / 12.5 / 14.7 / 14.3 | 96.0 / 97.7 / 99.1 / 97.5 | 0.89 / 0.89 / 0.96 / 0.81 |
| RTN L3 / HIGGS L3 | 1.38 / 1.54 | 0.02 | 7.8 / 8.7 | 0.0 / 0.0 | 0 |
| uniform L4 / KL-alloc ref L4 / trace-alloc ref L4 | 1.843 | 0.0225 | 10.43 | 98.6 [97.4] / 98.5 [97.2] / 95.7 [93.9] | 0.93 / 0.94 / 0.88 |
Emb/head (isolated, linears fp16): see section 1 rows 1-2, 4 (SKILL7 / BANK6): fp8 100/-, RTN 4-bit 100/-, 3-bit 100/99.7, 2-bit 99.8/99.3, VQ d4 99.8/-, VQ d8 99.4/96.4, low-rank r1024 98.5/-, vocab N=100k 100/-, 65,536 100/99.7, 40,000 98.8/95.0.
BANK6 (n=302 base-correct): g128 fp16 95.7 [92.8]; coded 1/4 oct 96.7 [94.0]; g64 coded 96.0 [93.2]; g256 fp16 89.4 [85.4]; share4 62.9; L4 coded 97.4 [94.9]; RTN L5 88.1 [83.9], HIGGS L5 89.4 [85.4], HIGGS L7 95.7 [92.8].
Caveats: single model (14B), two tasks, GPTQ here is a simple uniform-grid GPTQ (the repo's dense_core GPTQ with 13-level allocation and Hessian damping tuning gets 92-97% at 9.6-10% overall, so absolute retention differs; the DELTAS between tricks are what transfer). Components were measured in isolation; combined error (2-bit tables + 3-bit linears + trim) is not measured and may be larger than the sum.

## 4. Recommended default stack per model class
- **Dense Qwen2.5-14B/32B**: rANS indices; coded scales at 1/8 octave with the freed bytes spent on g64 (net -0.1 pp vs fp16-g128, +9 pts retention at L3); emb/head 4-bit (safe, 100%) or 2-bit RTN-Hadamard (99.8 / 99.3) instead of fp16; vocab trim at id<65,536 only with disclosure; one tokenizer.json xz. Calibrated GPTQ (not data-free). No per-tensor bit allocation, no row sharing, no g>=256.
- **MoE Qwen3-30B-A3B**: same, but emb/head are only 2 pp so the big items are rANS (3.9 pp vs packing) and coded scales (0.58 pp, experts are 95% of params); skip vocab trim (0.1 pp).
- **Bonsai-8B / Ternary-Bonsai-27B**: lossless only until a lane measures retention: exponent-code the 16-bit emb/head (3.8 / 4.0 pp), value-code the scale tables (0.37-0.5 pp), drop the vision tower (1.68 pp, gray, disclose). Ternary-Bonsai-27B 10.67% (NATIVE_LOWBIT.md) minus 0.4-0.5 scale pp is ~10.2%: needs row 1 (lossy emb/head, retention on that model unmeasured; Qwen says safe at 4-bit) or the tower removal to reach <=10%.

## 5. Legality notes
R1 (all bytes counted): every number above is bytes actually written; shipped decoder code (rANS ~1-3 kB python, tokenizer remap) counts and is already netted out (<0.001 pp). R9 (spirit): lossless tricks are clean; vocab trim and tower removal are gray (documented in PRUNE_LEGAL.md and RESEARCH_EXHAUSTIVE.md X16); label-selected row removal would be a head (rule 4/7) and is never used. No trick here fits anything to feedback data; none trains weights (GPTQ error feedback and learned rounding are the quantizers' existing gray flags, unchanged).
