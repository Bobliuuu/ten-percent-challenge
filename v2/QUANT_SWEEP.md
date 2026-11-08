# QUANT_SWEEP: which quantizer for the kept experts of a task-pruned MoE (and every other method, at layer level)

Written 2026-11-08 by the quant-sweep lane. Compute: 3 Vast RTX PRO 6000 (96 GB) boxes, all destroyed and verified; ~12 GPU-hours total. No weight training anywhere. Dev data only (no sealed/locked file touched). Code: `v2/qsweep/qlib.py` (all quantizers, pure PyTorch, own implementations, no third-party repo run), `v2/moe_quant/` (batched engine `qlib_b.py`, harness `core_q.py`, configs, scripts, raw results in `v2/moe_quant/results/`, logs in `v2/moe_quant/logs/`).

## 0. Answer first
1. **At the 9.9% cap, spend bytes on MORE experts at 2 bits, not fewer experts at 3 bits.** Same model, same Hessians, same eval, bytes 9.8-9.9%: BANK8 retention of base-correct items 99.6-99.7% (Wilson lo 99.1-99.2) for 87/128 experts at 2.0 bpw (ECSQ or D4-VQ, Hadamard + GPTQ-style error feedback) versus **94.9% (lo 93.7) for the L3 plan (55/128 experts, GPTQ+Hadamard 3-bit)** and 93.8% for NF3. On the easier SKILL7 task the same ordering holds but is inside noise (99.2-99.3 vs 98.7).
2. **Best by retention at equal bytes (<=9.9%)**: Hadamard-rotated, error-feedback (GPTQ/LDLQ) rounding to a **fixed-rate D=4, K=256 vector codebook (2.000 bpw, `vq4_had_b2`, shared 2 KB codebook, no entropy decoder needed)** or **entropy-coded uniform scalar (`ecsq_had_b2`, g256)** with ~68% of experts kept (87/128 per layer by routing mass). They tie (BANK8 99.61 / 99.67; SKILL7 99.23 / 98.70, difference is 1-2 items per 100 and inside 95% bounds). VQ4 is simpler to ship (fixed-rate indices); ECSQ needs a range-coder decode.
3. Plain **GPTQ int2 g128 without rotation** is surprisingly good on BANK8 (99.54, 81 experts, no clip search) but erratic: SKILL7 97.05 without clip search (worst non-RTN row) and 99.17 with clip search only at 10.43% bytes (over cap; 99.29 with 81 experts at 9.83% when rotated). Its layer error is 3-4x higher than VQ/ECSQ and it costs 2.14 bpw (scale+zero overhead). Treat it as the same tier at best, with high variance; do not choose it over VQ/ECSQ.
4. **RTN is the floor**: 2-bit RTN with no clip search collapses a cell (min-kind 0.84), 2-bit RTN with clip 97.3; 3-bit RTN 98.2-98.8 (SKILL7). Hadamard rotation barely helps RTN; it helps error-feedback methods and VQ.
5. **Method ranking is nearly irrelevant at 3 bits and decisive at 2 bits.** What moves retention is the expert budget (how many experts the bytes can keep) and whether a method reaches ~2 bpw without a Hessian-weighted error blow-up. Everything with proxy error <= ~0.006 at 2 bpw (VQ, ECSQ, TCQ, GPTQ-had, GPTVQ) lands in the same 98.7-99.7 band.
6. SKILL7 (7 skills, short prompt) is too easy to rank methods (all good rows 98.5-99.4, SE 0.27 pt). BANK8 (8 queues, long few-shot prompt) separates them; use BANK8-like prompts for any further selection.

## 1. Benchmark
Qwen3-30B-A3B-Instruct-2507, experts task-pruned by routing mass (`score=mass`, per layer, router masked), attention 6-bit g128 RTN, embedding 5-bit, lm_head 4-bit, norms/router rows of kept experts fp16, tokenizer+config 7.4 MB counted. Experts re-quantized per expert from per-expert Hessians (input of gate/up, input of down) collected on task-prompt calibration text (SKILL7: 2,031 items from hash buckets [0,40); BANK8: 960 cal items) in 6-layer groups (earlier groups already quantized). Real greedy generation (HF generate, <=5 / <=8 new tokens), exact label string. Retention = P(correct | base correct). Wilson 95% lower bounds. Cells = `kind` (skill / queue); "min-kind" is the worst cell retention.
- SKILL7: dev items from `v2/moe_quant/build_skill7x.py` (same prompt, label map and hash buckets as `task_search/EASY`, bigger caps; dev bucket [40,70)), 1,802 items, base acc .9395, 1,693 base-correct. Two sources (massive, clinc150).
- BANK8: `v2/rival3/data/bank/{cal,sel}.jsonl` (Banking77 to 8 queue families, few-shot prompt), SEL 1,600 items, base acc .9513, 1,522 base-correct. One source.
- Bytes = all shipped bytes from measured accounting (index bits or measured histogram entropy for ECSQ, scales, zero points, codebooks, outliers, plus attention/embed/head/router/norms/tokenizer) / fp16 size 61.06 GB. Entropy-coded rows are measured entropies, not yet a real range-coded file (L3/L4 packers must confirm). Keep-count was chosen from nominal bpw so each row lands near 9.9%; actual ratio is in the table. Rows over 10.00% (clip variants, 1.5 bpw rows at 10.14-10.23%) are NOT valid entries and are shown for the curve; the byte-valid siblings are listed next to them.
- Pruned dense-equivalent ablations (no pruning) were not rerun here; see `NEAR_MISSES.md` L3 block for prune-only numbers.

## 2. Real-generation results, SKILL7 (sorted by retention; noise about +-0.3-0.5 pt)
(retention of base-correct items, real greedy generation; sorted by retention)

base acc 0.9395 n_bc 1693
| name | method | keep/layer | bytes % fp16 | expert bpw | acc | retention | Wilson lo | min-kind ret | relerr gu | relerr dn |
|---|---|---|---|---|---|---|---|---|---|---|
| gptq_clip_had_b2 | gptq_clip_had | 87 | 10.43 | 2.142 | 0.9390 | 0.9935 | 0.9884 | 0.983 | 0.0018 | 0.0073 |
| e8_had_b1.375 | vq_had | 126 | 10.22 | 1.440 | 0.9423 | 0.9929 | 0.9877 | 0.987 | 0.0074 | 0.0208 |
| gptq_clip_had_b2_k81 | gptq_clip_had | 81 | 9.83 | 2.141 | 0.9356 | 0.9929 | 0.9877 | 0.983 | 0.0019 | 0.0074 |
| ecsq_had_b3_k80 | ecsq_had | 80 | 12.93 | 3.004 | 0.9329 | 0.9923 | 0.9869 | 0.976 | 0.0002 | 0.0010 |
| mix_ecsq_h30_3_1.5 | ecsq_had | 87 | 9.74 | 1.969 | 0.9356 | 0.9923 | 0.9869 | 0.978 | 0.0020 | 0.0080 |
| vq4_had_b2 | vq_had | 87 | 9.87 | 2.003 | 0.9356 | 0.9923 | 0.9869 | 0.974 | 0.0013 | 0.0051 |
| gptq_clip_b2 | gptq_clip | 87 | 10.43 | 2.142 | 0.9367 | 0.9917 | 0.9862 | 0.983 | 0.0029 | 0.0100 |
| e8_had_1.5 | vq_had | 116 | 10.14 | 1.551 | 0.9384 | 0.9911 | 0.9854 | 0.976 | 0.0039 | 0.0132 |
| nf_had_b2 | nf_had | 82 | 9.87 | 2.126 | 0.9356 | 0.9905 | 0.9847 | 0.972 | 0.0021 | 0.0082 |
| ecsq_had_b2.25 | ecsq_had | 77 | 9.85 | 2.258 | 0.9317 | 0.9900 | 0.9840 | 0.974 | 0.0008 | 0.0032 |
| ecsq_rdo_had_b1.5 | ecsq_had | 116 | 9.83 | 1.493 | 0.9356 | 0.9900 | 0.9840 | 0.974 | 0.0030 | 0.0111 |
| gptq_had_b2.5_ecsq | ecsq_had | 70 | 9.92 | 2.505 | 0.9312 | 0.9900 | 0.9840 | 0.978 | 0.0005 | 0.0022 |
| rvq4_had_b2 | rvq_had | 87 | 9.88 | 2.004 | 0.9356 | 0.9900 | 0.9840 | 0.972 | 0.0020 | 0.0075 |
| rvq8_had_b1.5 | rvq_had | 116 | 10.17 | 1.556 | 0.9351 | 0.9900 | 0.9840 | 0.980 | 0.0048 | 0.0158 |
| vq4_had_b1.5 | vq_had | 116 | 10.14 | 1.550 | 0.9340 | 0.9900 | 0.9840 | 0.978 | 0.0037 | 0.0129 |
| vq4_had_b2.25 | vq_had | 77 | 9.82 | 2.251 | 0.9329 | 0.9900 | 0.9840 | 0.970 | 0.0009 | 0.0035 |
| ecsq_had_b2_g64 | ecsq_had | 87 | 9.89 | 2.008 | 0.9334 | 0.9894 | 0.9833 | 0.966 | 0.0017 | 0.0069 |
| gptq_had_b2_g32 | gptq_had | 68 | 9.87 | 2.562 | 0.9323 | 0.9894 | 0.9833 | 0.966 | 0.0023 | 0.0088 |
| gptq_had_b3_g256 | gptq_had | 57 | 9.91 | 3.074 | 0.9329 | 0.9894 | 0.9833 | 0.966 | 0.0007 | 0.0025 |
| ecsq_had_b1.75 | ecsq_had | 100 | 10.07 | 1.785 | 0.9340 | 0.9888 | 0.9825 | 0.967 | 0.0017 | 0.0067 |
| ecsq_rdo_had_b2 | ecsq_had | 87 | 9.70 | 1.961 | 0.9317 | 0.9888 | 0.9825 | 0.967 | 0.0011 | 0.0046 |
| mix_had_h30_3_2 | gptq_had | 65 | 9.15 | 2.443 | 0.9329 | 0.9888 | 0.9825 | 0.974 | 0.0024 | 0.0087 |
| vq4_had_b1.75 | vq_had | 99 | 9.92 | 1.769 | 0.9340 | 0.9888 | 0.9825 | 0.970 | 0.0020 | 0.0077 |
| ecsq_had_b2.5 | ecsq_had | 70 | 9.92 | 2.506 | 0.9301 | 0.9882 | 0.9818 | 0.972 | 0.0005 | 0.0022 |
| ecsq_had_b2_g128 | ecsq_had | 87 | 9.90 | 2.010 | 0.9301 | 0.9882 | 0.9818 | 0.970 | 0.0013 | 0.0053 |
| mix_had_h50_3_2 | gptq_had | 65 | 9.76 | 2.645 | 0.9312 | 0.9882 | 0.9818 | 0.966 | 0.0023 | 0.0085 |
| rtn_b3 | rtnq | 55 | 9.81 | 3.149 | 0.9362 | 0.9882 | 0.9818 | 0.949 | 0.0240 | 0.0347 |
| ecsq_had_b2_s8 | ecsq_had | 87 | 9.91 | 2.013 | 0.9306 | 0.9876 | 0.9811 | 0.970 | 0.0011 | 0.0043 |
| ecsq_had_b3 | ecsq_had | 58 | 9.86 | 3.003 | 0.9295 | 0.9876 | 0.9811 | 0.957 | 0.0003 | 0.0011 |
| gptq_clip_had_b3 | gptq_clip_had | 58 | 10.25 | 3.148 | 0.9329 | 0.9876 | 0.9811 | 0.966 | 0.0004 | 0.0016 |
| gptq_had_b3_g32 | gptq_had | 48 | 9.78 | 3.594 | 0.9323 | 0.9876 | 0.9811 | 0.957 | 0.0004 | 0.0014 |
| nf_had_b3 | nf_had | 56 | 9.90 | 3.125 | 0.9317 | 0.9876 | 0.9811 | 0.957 | 0.0004 | 0.0017 |
| rvq8_had_b2 | rvq_had | 87 | 9.98 | 2.030 | 0.9317 | 0.9876 | 0.9811 | 0.963 | 0.0017 | 0.0066 |
| vq4_had_b2.5 | vq_had | 69 | 9.79 | 2.500 | 0.9306 | 0.9876 | 0.9811 | 0.970 | 0.0006 | 0.0025 |
| vq4_had_b3 | vq_had | 58 | 9.85 | 3.000 | 0.9306 | 0.9876 | 0.9811 | 0.966 | 0.0003 | 0.0013 |
| ecsq_had_b1.5 | ecsq_had | 116 | 10.23 | 1.568 | 0.9329 | 0.9870 | 0.9804 | 0.974 | 0.0029 | 0.0107 |
| ecsq_had_b2 | ecsq_had | 87 | 9.91 | 2.014 | 0.9279 | 0.9870 | 0.9804 | 0.959 | 0.0011 | 0.0046 |
| gptq_clip_b3 | gptq_clip | 58 | 10.25 | 3.148 | 0.9290 | 0.9870 | 0.9804 | 0.957 | 0.0009 | 0.0023 |
| gptq_had_b2 | gptq_had | 81 | 9.83 | 2.141 | 0.9334 | 0.9870 | 0.9804 | 0.974 | 0.0039 | 0.0145 |
| gptq_had_b2_g256 | gptq_had | 84 | 9.86 | 2.071 | 0.9323 | 0.9870 | 0.9804 | 0.965 | 0.0045 | 0.0162 |
| gptq_had_b3 | gptq_had | 55 | 9.81 | 3.148 | 0.9317 | 0.9870 | 0.9804 | 0.966 | 0.0006 | 0.0021 |
| vq8_had_b1.0 | vq_had | 128 | 8.15 | 1.069 | 0.9367 | 0.9870 | 0.9804 | 0.966 | 0.0161 | 0.0405 |
| awq_gptq_b3 | awq_gptq | 55 | 9.81 | 3.148 | 0.9306 | 0.9864 | 0.9797 | 0.949 | 0.0006 | 0.0026 |
| vq8_had_b1.5 | vq_had | 116 | 10.14 | 1.551 | 0.9312 | 0.9864 | 0.9797 | 0.961 | 0.0034 | 0.0121 |
| ecsq_rdo_had_b3 | ecsq_had | 58 | 9.82 | 2.988 | 0.9301 | 0.9858 | 0.9790 | 0.957 | 0.0003 | 0.0011 |
| gptq_had_b2_g64 | gptq_had | 76 | 9.83 | 2.281 | 0.9306 | 0.9858 | 0.9790 | 0.967 | 0.0030 | 0.0117 |
| vq4_had_b2_nokeep | vq_had | 87 | 9.87 | 2.003 | 0.9284 | 0.9852 | 0.9783 | 0.952 | 0.0013 | 0.0051 |
| gptq_b3 | gptq | 55 | 9.81 | 3.148 | 0.9279 | 0.9846 | 0.9776 | 0.949 | 0.0012 | 0.0030 |
| gptq_had_b3_g64 | gptq_had | 53 | 9.88 | 3.297 | 0.9290 | 0.9846 | 0.9776 | 0.966 | 0.0005 | 0.0017 |
| rvq4_had_b3 | rvq_had | 58 | 9.86 | 3.003 | 0.9279 | 0.9846 | 0.9776 | 0.957 | 0.0005 | 0.0019 |
| rtn_noclip_b3 | rtn_noclip | 58 | 10.25 | 3.149 | 0.9262 | 0.9817 | 0.9741 | 0.957 | 0.0306 | 0.0455 |
| rvq8_had_b1.0 | rvq_had | 128 | 8.16 | 1.070 | 0.9306 | 0.9811 | 0.9734 | 0.949 | 0.0211 | 0.0499 |
| ecsq_rdo_had_b1.0 | ecsq_had | 128 | 7.58 | 0.973 | 0.9317 | 0.9805 | 0.9728 | 0.949 | 0.0112 | 0.0337 |
| awq_gptq_b2 | awq_gptq | 81 | 9.83 | 2.141 | 0.9262 | 0.9799 | 0.9721 | 0.957 | 0.0042 | 0.0181 |
| vq4_had_b1.0 | vq_had | 128 | 8.15 | 1.069 | 0.9279 | 0.9787 | 0.9707 | 0.957 | 0.0190 | 0.0452 |
| rtn_b2 | rtnq | 81 | 9.84 | 2.143 | 0.9267 | 0.9734 | 0.9646 | 0.926 | 0.0949 | 0.1321 |
| rtn_noclip_b2 | rtn_noclip | 87 | 10.44 | 2.144 | 0.9329 | 0.9711 | 0.9619 | 0.838 | 0.1664 | 0.2451 |
| gptq_b2 | gptq | 81 | 9.83 | 2.141 | 0.9173 | 0.9705 | 0.9613 | 0.930 | 0.0060 | 0.0223 |
| ecsq_had_b1.0 | ecsq_had | 128 | 8.37 | 1.106 | 0.8907 | 0.9368 | 0.9242 | 0.803 | 0.0122 | 0.0348 |

## 3. Real-generation results, BANK8 (sorted by retention)
(retention of base-correct items, real greedy generation; sorted by retention)

base acc 0.9513 n_bc 1522
| name | method | keep/layer | bytes % fp16 | expert bpw | acc | retention | Wilson lo | min-kind ret | relerr gu | relerr dn |
|---|---|---|---|---|---|---|---|---|---|---|
| b8_ecsq_had_b2 | ecsq_had | 87 | 9.91 | 2.012 | 0.9550 | 0.9967 | 0.9923 | 0.984 | 0.0041 | 0.0129 |
| b8_vq4_had_b2 | vq_had | 87 | 9.86 | 2.001 | 0.9544 | 0.9961 | 0.9914 | 0.984 | 0.0039 | 0.0127 |
| b8_gptq_b2 | gptq | 81 | 9.83 | 2.141 | 0.9569 | 0.9954 | 0.9905 | 0.978 | 0.0165 | 0.0512 |
| b8_gptq_clip_had_b2 | gptq_clip_had | 81 | 9.83 | 2.141 | 0.9525 | 0.9947 | 0.9897 | 0.963 | 0.0058 | 0.0189 |
| b8_e8_had_1.5 | vq_had | 116 | 9.97 | 1.519 | 0.9513 | 0.9934 | 0.9879 | 0.978 | 0.0082 | 0.0252 |
| b8_ecsq_had_b3 | ecsq_had | 58 | 9.86 | 3.003 | 0.9506 | 0.9934 | 0.9879 | 0.979 | 0.0010 | 0.0034 |
| b8_vq4_had_b3 | vq_had | 58 | 9.85 | 3.000 | 0.9469 | 0.9921 | 0.9863 | 0.974 | 0.0012 | 0.0037 |
| b8_gptq_had_b2 | gptq_had | 81 | 9.83 | 2.141 | 0.9531 | 0.9915 | 0.9854 | 0.963 | 0.0124 | 0.0367 |
| b8_gptq_b3 | gptq | 55 | 9.81 | 3.148 | 0.9437 | 0.9829 | 0.9751 | 0.951 | 0.0025 | 0.0097 |
| b8_gptq_had_b3 | gptq_had | 55 | 9.81 | 3.148 | 0.9094 | 0.9494 | 0.9372 | 0.875 | 0.0023 | 0.0071 |
| b8_nf_had_b3 | nf_had | 56 | 9.90 | 3.125 | 0.9012 | 0.9382 | 0.9250 | 0.859 | 0.0018 | 0.0058 |

Key contrasts (BANK8, bytes 9.8-9.9%): ECSQ 2.0 bpw x87 experts 99.67 [99.23]; VQ-D4 2.0 x87 99.61 [99.14]; GPTQ int2 g128 no-rot x81 99.54; GPTQ-clip-had int2 x81 99.47 [98.97]; E8 lattice 1.5 bpw x116 99.34 [98.79] (9.97%); ECSQ 3.0 x58 99.34; VQ-D4 3.0 x58 99.21; GPTQ-had int2 x81 99.15; GPTQ int3 no-rot x55 98.29; **GPTQ-had int3 x55 94.94 [93.72]; NF3-had x56 93.82 [92.50]**. The last two are the configuration L3 planned; do not use it.

## 4. Layer-level proxy for EVERY implemented method (Qwen3-30B-A3B real experts, real Hessians)
Relative output error tr(dW H dW^T)/tr(W H W^T) averaged over 3 experts x 3 layers (6, 24, 42) x {gate+up, down}; bpw counts everything (indices, scales, zero points, codebooks, outliers, masks, low-rank factors). Lower is better; compare within a bpw band. Proxy error is a guide, not retention: SKILL7/BANK8 retention is already saturated at proxy ~0.006 (2 bpw).
### layer proxy (H-weighted relative error, 9 experts x 2 matrices from layers 6/24/42; lower better)

| method | bpw (all bits) | rel err | secs | note |
|---|---|---|---|---|
| waterfill_ecsq_1.0_perrow_had | 1.126 | 0.0141 | 4 |  |
| ecsq_1.0_g256_had | 1.057 | 0.0191 | 4 |  |
| tcq_L12_V2_kb2_had | 1.000 | 0.0200 | 32 |  |
| ecsq_rdo_1.0_g256_had | 0.876 | 0.0216 | 6 |  |
| vq_d8_1.0_had | 1.000 | 0.0301 | 0 |  |
| ecsq_1.0_g64_had | 1.034 | 0.0337 | 4 |  |
| gptvq_d4_1.0_had | 1.002 | 0.0337 | 1 |  |
| vq_d4_1.0_had | 1.000 | 0.0343 | 1 |  |
| rvq_fitted_d8_2x_1.0_had | 1.002 | 0.0378 | 1 |  |
| rvq_aqlm_style_d8_2x_1.0_had | 1.002 | 0.0380 | 1 |  |
| gptvq_d2_1.0 | 1.000 | 0.0609 | 2 |  |
| waterfill_ecsq_1.5_perrow_had | 1.565 | 0.0063 | 4 |  |
| tcq_L12_V2_kb3_had | 1.500 | 0.0073 | 32 |  |
| ecsq_1.5_g256_had | 1.525 | 0.0077 | 4 |  |
| ecsq_rdo_1.5_g256_had | 1.432 | 0.0081 | 6 |  |
| vq_d8_1.5_had | 1.500 | 0.0083 | 1 |  |
| vq_d4_1.5_had | 1.500 | 0.0090 | 1 |  |
| gptvq_d4_1.5_had | 1.507 | 0.0092 | 1 |  |
| e8_lattice_1.5_had | 1.500 | 0.0094 | 1 |  |
| rvq_fitted_d8_2x_1.5_had | 1.508 | 0.0111 | 1 |  |
| rvq_aqlm_style_d8_2x_1.5_had | 1.508 | 0.0112 | 1 |  |
| ecsq_1.5_g64_had | 1.515 | 0.0124 | 4 |  |
| e8_lattice_1.375_had | 1.375 | 0.0152 | 1 |  |
| gptvq_d2_1.5 | 1.501 | 0.0166 | 2 |  |
| sparsegpt_2:4_int2_g128 | 1.343 | 0.0218 | 2 |  |
| wanda_2:4_int2_g128 | 1.717 | 0.0746 | 2 |  |
| waterfill_ecsq_2.0_perrow_had | 2.040 | 0.0028 | 4 |  |
| tcq_L12_V2_kb4_had | 2.000 | 0.0032 | 32 |  |
| ecsq_2.0_g256_had | 2.012 | 0.0033 | 4 |  |
| ecsq_rdo_2.0_g256_had | 1.960 | 0.0034 | 6 |  |
| vq_d4_2.0_had | 2.000 | 0.0036 | 1 |  |
| gptvq_d4_2.0_had | 2.029 | 0.0036 | 1 |  |
| rvq_fitted_d8_2x_2.0_had | 2.031 | 0.0045 | 1 |  |
| rvq_aqlm_style_d8_2x_2.0_had | 2.031 | 0.0045 | 1 |  |
| ecsq_2.0_g64_had | 2.007 | 0.0047 | 4 |  |
| gptq_int2_g128_clip_had | 2.141 | 0.0053 | 1 |  |
| nf2_g64_gptq_had | 2.250 | 0.0056 | 2 |  |
| gptvq_d2_2.0 | 2.001 | 0.0061 | 2 |  |
| gptq_int2_g128_actorder_had | 2.150 | 0.0077 | 1 |  |
| gptq_int2_g128_had | 2.141 | 0.0089 | 1 |  |
| gptq_int2_g256_had | 2.070 | 0.0102 | 1 |  |
| awq_gptq_int2_g128 | 2.141 | 0.0102 | 28 |  |
| gptq_int2_g128 | 2.141 | 0.0120 | 1 |  |
| owq1pct_int2_g128 | 2.148 | 0.0133 | 3 |  |
| squeezellm_int2 | 2.207 | 0.0677 | 0 |  |
| omniquant_lwc_search_int2_g128 | 2.141 | 0.0839 | 0 |  |
| admmq_int2_g128 | 2.141 | 0.0907 | 0 |  |
| omniquant_lwc_LEARNED_int2_g128 | 2.141 | 0.0943 | 1 |  |
| awq_int2_g128 | 2.141 | 0.1024 | 2 |  |
| rtn_int2_g128_had | 2.141 | 0.1031 | 0 |  |
| rtn_int2_g128 | 2.141 | 0.1052 | 0 |  |
| smoothquant+_int2_g128 | 2.141 | 0.1133 | 0 |  |
| sparsegpt_50_int3_g128 | 2.554 | 0.0039 | 2 |  |
| gptq_int2_g32_had | 2.562 | 0.0058 | 1 |  |
| lqer_r16_int2_g128_gptq_had(GRAY adapter) | 2.258 | 0.0066 | 3 |  |
| caldera_r16_it3_int2_g128_gptq_had(GRAY adapter) | 2.258 | 0.0069 | 8 |  |
| gptq_int2_g64_had | 2.281 | 0.0074 | 1 |  |
| wanda_50_int3_g128 | 2.574 | 0.0294 | 2 |  |
| hqq_int2_g64 | 2.500 | 0.1391 | 0 |  |
| waterfill_ecsq_3.0_perrow_had | 3.021 | 0.0007 | 4 |  |
| ecsq_3.0_g256_had | 3.003 | 0.0007 | 4 |  |
| ecsq_rdo_3.0_g256_had | 2.988 | 0.0007 | 6 |  |
| vq_d4_3.0_had | 3.000 | 0.0009 | 1 |  |
| ecsq_3.0_g64_had | 3.002 | 0.0010 | 4 |  |
| nf3_g64_gptq_had | 3.250 | 0.0011 | 2 |  |
| gptq_int3_g128_clip_had | 3.148 | 0.0011 | 1 |  |
| gptq_int3_g128_actorder_had | 3.158 | 0.0012 | 1 |  |
| gptvq_d2_3.0 | 3.004 | 0.0014 | 2 |  |
| gptq_int3_g128_had | 3.148 | 0.0014 | 1 |  |
| awq_gptq_int3_g128 | 3.148 | 0.0016 | 28 |  |
| gptq_int3_g256_had | 3.074 | 0.0016 | 1 |  |
| gptq_int3_g128 | 3.148 | 0.0018 | 1 |  |
| owq1pct_int3_g128 | 3.140 | 0.0024 | 3 |  |
| spqr1pct_int2_g16 | 3.210 | 0.0041 | 6 |  |
| admmq_int3_g128 | 3.148 | 0.0169 | 0 |  |
| omniquant_lwc_search_int3_g128 | 3.148 | 0.0209 | 0 |  |
| omniquant_lwc_LEARNED_int3_g128 | 3.148 | 0.0239 | 0 |  |
| awq_int3_g128 | 3.148 | 0.0258 | 2 |  |
| rtn_int3_g128_had | 3.148 | 0.0259 | 0 |  |
| rtn_int3_g128 | 3.148 | 0.0269 | 0 |  |
| smoothquant+_int3_g128 | 3.148 | 0.0293 | 0 |  |
| gptvq_d4_3.0_had | 3.458 | 0.0008 | 8 |  |
| gptq_int3_g32_had | 3.594 | 0.0009 | 1 |  |
| rvq_fitted_d8_2x_3.0_had | 3.500 | 0.0010 | 7 |  |
| rvq_aqlm_style_d8_2x_3.0_had | 3.500 | 0.0011 | 4 |  |
| lqer_r16_int3_g128_gptq_had(GRAY adapter) | 3.266 | 0.0011 | 3 |  |
| caldera_r16_it3_int3_g128_gptq_had(GRAY adapter) | 3.266 | 0.0011 | 8 |  |
| gptq_int3_g64_had | 3.297 | 0.0012 | 1 |  |
| squeezellm_int3 | 3.264 | 0.0178 | 0 |  |
| hqq_int3_g64 | 3.500 | 0.0253 | 0 |  |
| nf4_g64_gptq | 4.250 | 0.0003 | 2 |  |
| int4_g128_gptq_had | 4.156 | 0.0003 | 1 |  |
| fp4_e2m1_mxfp4_g32_gptq | 4.157 | 0.0005 | 2 |  |
| fp4_nvfp4_g16_gptq | 4.407 | 0.0003 | 2 |  |
| spqr1pct_int3_g16 | 4.261 | 0.0007 | 6 |  |

Reading the proxy table (by bpw band):
- **~1.0 bpw**: waterfilling-style ECSQ (single step, per-row, entropy-coded) 0.014 @1.13 < TCQ 0.020 @1.0 < ECSQ g256 0.019 @1.06 << D8-VQ 0.030 < D4-VQ / GPTVQ-D4 0.034 < residual/additive VQ 0.038. Entropy-coded scalar beats fixed-rate VQ here because it spends bits where weights are large (it is variable-rate); at fixed rate TCQ is best.
- **~1.5 bpw**: waterfilling ECSQ 0.0063 @1.57, TCQ 0.0073 @1.50, ECSQ g256 0.0077, D8-VQ 0.0083, D4-VQ 0.0090, E8 0.0094 (truncated to 4096 points), AQLM-style RVQ 0.011, ECSQ g64 0.012 (scale overhead hurts), GPTVQ-D2 0.017, SparseGPT 2:4+int2 0.022 @1.34.
- **~2.0 bpw**: ECSQ 0.0028-0.0033, TCQ 0.0032, D4-VQ 0.0036, GPTVQ-D4 0.0036, RVQ 0.0045, GPTQ-int2 clip+Hadamard 0.0053 @2.14, NF2 0.0056, GPTQ int2 plain 0.012, OWQ 1% 0.013, then the no-feedback / no-Hessian family (SqueezeLLM 0.068 @2.2, OmniQuant clip-search 0.084, ADMM-style 0.091, AWQ+RTN 0.10, RTN 0.105, SmoothQuant+ 0.113, HQQ 0.139 @2.5).
- **~3 bpw**: ECSQ / VQ-D4 0.0007-0.0009, GPTQ-had 0.0011-0.0014, SparseGPT 50% + int3 0.0039 @2.55 (beats dense int3 RTN 10x but loses to GPTQ-had), SpQR 1% outliers 0.0041 @3.2.
- **4 bit formats** (over budget: 25-28% of fp16): NF4 g64, INT4 g128 GPTQ-had, MXFP4 g32 (E8M0 scales), NVFP4 g16 (E4M3 scales) all 0.0003-0.0005; FP formats cost nothing in error relative to INT4 but are 4.16-4.41 bpw, so they cannot meet 10%.
- **Low-rank + quant (LQER / CALDERA-style, rank 16, no training)**: error 0.0066 / 0.0069 @2.26 bpw vs 0.0089 for the same GPTQ-had int2 base at 2.14: -26% error for +0.12 bpw (rank-16 factors at 4-bit); worse than just moving to 2.0 bpw ECSQ/VQ at the same bytes. GRAY under rule 4 (adapter-like shipped computation). Dropped.
- **OmniQuant**: the gradient (LEARNED clip) variant (0.094) is not better than the grid search (0.084) in this layer proxy, both far behind Hessian error feedback; LEARNED => gray. Not carried forward.
- **Implementation caveats**: my ADMM-Q, SqueezeLLM (diag-H weighted, no feedback), HQQ and SmoothQuant+ ports are weaker than the papers' full pipelines (no end-loss Fisher, no iterative refinement tuning); treat their bad rows as "no error feedback", not as a verdict on the originals.

## 5. Ranking by retention at equal bytes (MoE kept experts, <=9.9%)
| rank | method (all with Hadamard unless noted) | bpw / experts | BANK8 ret [lo] | SKILL7 ret [lo] | notes |
|---|---|---|---|---|---|
| 1 | ECSQ g256 + GPTQ feedback | 2.0 / 87 | 99.67 [99.23] | 98.70 [98.04] | needs entropy decode; RDO variant 98.88 SKILL7 at 9.70% |
| 1 | D4 K=256 VQ + LDLQ feedback (QuIP#-lite, Gaussian k-means codebook, 2 KB) | 2.0 / 87 | 99.61 [99.14] | 99.23 [98.69] | fixed rate, simplest to ship |
| 3 | GPTQ int2 g128 (no rotation) | 2.14 / 81 | 99.54 [99.05] | 97.05 [96.13] | lucky on BANK8, worst non-RTN on SKILL7 |
| 3 | GPTQ int2 g128 + clip search + Hadamard | 2.14 / 81 | 99.47 [98.97] | 99.29 [98.77] | valid at 9.83% only with k=81 |
| 5 | E8 lattice (4096 pts) 1.5 bpw | 1.52-1.55 / 116 | 99.34 [98.79] | 99.11 [98.54] (10.14%) | slightly over cap at 116; k=114 would fit |
| 5 | ECSQ 3.0 / VQ-D4 3.0 | 3.0 / 58 | 99.34 / 99.21 | 98.76 / 98.76 | 3-bit tier |
| 7 | GPTQ-had int3 g128 | 3.15 / 55 | 94.94 [93.72] | 98.70 [98.04] | L3 plan: BANK8 collapse in two cells (min-kind .875) |
| 8 | NF3 + Hadamard + GPTQ | 3.13 / 56 | 93.82 [92.50] | 98.76 | same |
| 9 | AWQ(+GPTQ) | 2.14-3.15 / 55-81 | not run | 97.99 (b2) / 98.64 (b3) | AWQ scaling hurts error-feedback methods |
| 10 | RTN (clip) int2 / int3 | 2.14 / 81, 3.15 / 55 | not run | 97.34 / 98.82 | floor |
| - | TCQ (QTIP-style), GPTVQ, SpQR, OWQ, SqueezeLLM, HQQ, OmniQuant, ADMM, SparseGPT, LQER, CALDERA | | layer proxy only | | see sec. 4; TCQ is accuracy-competitive but its Viterbi pass is too slow at 20-30 G weights (see sec. 7) |

## 6. Which method for which model class
- **Task-pruned MoE experts (Qwen3-30B-A3B, other fine-grained MoE) at <=9.9%**: rotate (random-sign Hadamard both sides), error-feedback rounding (GPTQ/LDLQ, per-expert Hessians from task prompts INCLUDING the shared prompt prefix, which L3 showed is essential), **2.0 bpw D4-K256 VQ or g256 ECSQ, keep ~68% of experts by routing mass (87/128)** at the same bytes. Do not use 3-bit/43%. Open follow-up: 1.5-1.75 bpw with 91-116 experts is equal within noise on SKILL7 and slightly worse on BANK8-style; E8/VQ-D8 at 1.5 are the best of that band.
- **Dense 14B-32B (no pruning, 1.5-1.6 bpw body)**: no new evidence from this sweep (coordinator: dense is method-insensitive at 92-97%). The proxy table says: below 2 bpw use entropy-coded scalar with the waterfilling rule (one step per tensor, per-row scale) or D8/E8 VQ; TCQ buys ~10% lower layer error at fixed rate but is 30-60x slower; low-rank, sparsity, AWQ/SmoothQuant/OmniQuant/HQQ/SqueezeLLM (no Hessian feedback) are all dominated at equal bytes.
- **Any model, >=3.5 bpw**: anything works (INT4 g128 GPTQ-had, NVFP4, MXFP4, NF4 all 3-5e-4 error); over budget for this challenge.
- **Embedding / lm_head**: not re-swept here (5-bit / 4-bit RTN g128 held fixed; earlier lanes use ~2.1 bpw full-vocab for dense models).

## 7. What was NOT done or is weaker than requested (honest list)
- **Full-model real generation was run only on the MoE benchmark with the batched methods**: RTN (clip / no clip), GPTQ (+/- Hadamard, +/- clip search, g32/64/128/256), AWQ+GPTQ, NF, ECSQ (+RDO rate-aware rounding, scale bits 8 vs 16, g64-512), D4/D8 VQ (Gaussian k-means), E8-lattice VQ, residual 2-stage VQ (AQLM-style, data-free codebooks), per-expert mixed bits by routing mass. All other methods (GPTVQ, SpQR, OWQ, SqueezeLLM, HQQ, OmniQuant, ADMM, SparseGPT/Wanda, LQER/CALDERA, TCQ, MXFP4/NVFP4/FP4) are layer-proxy only. TCQ ran 2 configs end to end that were killed after >20 min each (exhaustive Viterbi over 2^12 states x 128-step sequences for ~20 G weights); L=8 or a fast approximate Viterbi would be needed.
- AQLM proper (learned additive codebooks by gradient on calibration loss), QuIP# E8P (2^16 codebook with sign symmetry; I used a 4096-point norm-ordered truncated E8), PV-Tuning, GPTAQ residual target (see `quality/gptaq_fix.md`: unvalidated), SpinQuant/OSTQuant learned rotations (R4 gray; I only ran fixed Hadamard; `best-of-N seeds` not run), Leech-lattice VQ, GuidedQuant (needs end-loss gradients) were NOT implemented.
- No claim about the sealed exam. Dev bytes are measured, not a packed file. One calibration seed, one rotation seed (1000+layer). SKILL7 Wilson bounds ~+-0.5 pt; ranking inside the 98.5-99.4 band is noise.
- Rule flags: all runs are training-free. Gray ingredients actually used: none in the full-model rows (Gaussian-fit codebooks are data-free; ECSQ step `c` solved from the tensor itself); OmniQuant-learned clip and LQER/CALDERA low-rank (rule 4 adapter-like) appear only in the proxy table and are marked.
- Newer 2025-2026 sub-2-bit methods (web scan, abstracts only, none run): PT^2-LLM (2510.03267, training-free asymmetric ternary + grid alignment), TWLA/E2M-ATQ (2606.13054), AF1 1-bit (2609.06161), LieQ layer-wise mixed bits (2508.03332), CurveTQ (2610.09212) and XOR-Trellis (2610.00432) curvature-weighted trellises, KronQ (2607.07964; Kronecker incoherence + gradient-covariance), D2Quant (2602.02546), BTC-LLM (2506.12040, learnable transform: gray), Sherry 1.25-bit (2601.07892, QAT: out), SignRoundV2 (2512.04746, tuning: gray), CCQ (2507.07145). Highest promise for a future 1.5 bpw dense attempt: PT^2-LLM and KronQ+LieQ allocation; none has a verified 1.5-bpw Qwen2.5-14B number. For MoE experts at 2 bpw the current winner already retains 99.6% so these are not the bottleneck.

## 8. Reproduce
`modal`-free: rent a >=96 GB GPU, `pip install torch==2.8.0 transformers==4.57.1 accelerate scipy`, copy `v2/moe_quant/{core_q.py->core.py,qlib.py,qlib_b.py,tcq.py,proxy_methods.py,fb.py,box_run.py}` to one dir and the `data/` files to `/root/data2/`, run `python box_run.py cfgs_sweep.json base <names...>`; score with `python score.py results skill7|bank8`; tables with `python report.py`. Raw per-item outputs for every row are in `results/outs_*`.
