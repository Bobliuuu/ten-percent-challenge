| model | scenario | fmt | method (cfg) | base teach | teach gain (95% lo) | neighbors | real-shuf (lo) | G2 |
|---|---|---|---|---|---|---|---|---|
| Qwen3-1.7B | B | first | lmhead_lda (alpha=0.2,ridge=0.1) | 0.00 | +0.52 (+0.46) | 0.990 | +0.52 (+0.46) | PASS |
| Qwen3-1.7B | B | first | blend (ls=d3_80,lam=1000.0,alpha=0.5,ridge=0.01) | 0.00 | +0.91 (+0.88) | 0.917 | +0.91 (+0.88) | FAIL |
| Qwen3-1.7B | B | first | memit_down (ls=d3_80,lam=1000.0) | 0.00 | +0.92 (+0.89) | 0.860 | +0.92 (+0.89) | FAIL |
| Qwen3-14B | T1pairS | first | lmhead_lda (alpha=0.3,ridge=0.1) | 0.23 | +0.57 (+0.50) | 0.987 | +0.81 (+0.74) | PASS |
| Qwen3-14B | T1pairS | first | blend (ls=d3_80,lam=1000.0,alpha=0.2,ridge=0.1) | 0.23 | +0.57 (+0.49) | 1.000 | +0.66 (+0.58) | PASS |
| Qwen3-14B | T1pairS | first | memit_down (ls=d3_80,lam=1000.0) | 0.23 | +0.58 (+0.51) | 1.000 | +0.72 (+0.65) | PASS |
| Qwen3-4B-Instruct-2507 | A | first | lmhead_lda (alpha=0.3,ridge=0.01) | 0.00 | +0.29 (+0.24) | 0.990 | +0.15 (+0.11) | PASS |
| Qwen3-4B-Instruct-2507 | A | first | blend (ls=d3_80,lam=1000.0,alpha=0.2,ridge=0.1) | 0.00 | +0.50 (+0.44) | 0.867 | +0.43 (+0.37) | FAIL |
| Qwen3-4B-Instruct-2507 | A | first | memit_down (ls=d3_80,lam=1000.0) | 0.00 | +0.50 (+0.44) | 0.877 | +0.46 (+0.40) | FAIL |
| Qwen3-4B-Instruct-2507 | B | first | lmhead_lda (alpha=0.2,ridge=0.1) | 0.00 | +0.36 (+0.31) | 1.000 | +0.36 (+0.31) | PASS |
| Qwen3-4B-Instruct-2507 | B | first | blend (ls=d3_80,lam=1000.0,alpha=0.5,ridge=0.01) | 0.00 | +0.98 (+0.96) | 0.957 | +0.98 (+0.96) | FAIL |
| Qwen3-4B-Instruct-2507 | B | first | memit_down (ls=d3_80,lam=1000.0) | 0.00 | +0.99 (+0.97) | 0.907 | +0.99 (+0.97) | FAIL |
| Qwen3-8B | T1new | first | lmhead_lda (alpha=0.1,ridge=0.01) | 0.00 | +0.00 (+0.00) | 1.000 | +0.00 (+0.00) | FAIL |
| Qwen3-8B | T1new | last | lmhead_lda (alpha=0.1,ridge=0.01) | 0.00 | +0.00 (+0.00) | 0.990 | +0.00 (+0.00) | FAIL |
| Qwen3-8B | T1new | last | blend (ls=d3_80,lam=1000.0,alpha=0.2,ridge=0.1) | 0.00 | +0.33 (+0.27) | 0.403 | +0.33 (+0.27) | FAIL |
| Qwen3-8B | T1new | last | memit_down (ls=d3_80,lam=1000.0) | 0.00 | +0.26 (+0.19) | 0.230 | +0.26 (+0.19) | FAIL |
| Qwen3-8B | T1pairS | first | lmhead_lda (alpha=0.2,ridge=0.1) | 0.13 | +0.43 (+0.36) | 0.987 | +0.48 (+0.41) | PASS |
| Qwen3-8B | T1pairS | first | blend (ls=d3_80,lam=1000.0,alpha=0.2,ridge=0.1) | 0.13 | +0.59 (+0.52) | 0.993 | +0.63 (+0.56) | PASS |
| Qwen3-8B | T1pairS | first | memit_down (ls=d3_80,lam=1000.0) | 0.13 | +0.59 (+0.52) | 1.000 | +0.63 (+0.56) | PASS |

### lm_head codec test (lm_head 4-bit g128 RTN as stage-1; edited rows: fp16 kept / re-quantised 4-bit own scales / snapped to original scales)
| model | scenario | variant | teach acc | gain vs quantised base | neighbors | rows edited | bytes added |
|---|---|---|---|---|---|---|---|
| Qwen3-1.7B | B | quantised base (no edit) | 0.00 | | 0.990 | 0 | 0 |
| Qwen3-1.7B | B | fp16row | 0.53 | +0.53 | 0.980 | 2 | 6016 |
| Qwen3-1.7B | B | req4_newscale | 0.55 | +0.55 | 0.980 | 2 | 0 |
| Qwen3-1.7B | B | snap_orig | 0.52 | +0.52 | 0.983 | 2 | 0 |
| Qwen3-14B | T1pairS | quantised base (no edit) | 0.23 | | 1.000 | 0 | 0 |
| Qwen3-14B | T1pairS | fp16row | 0.81 | +0.57 | 0.983 | 16 | 117813 |
| Qwen3-14B | T1pairS | req4_newscale | 0.81 | +0.57 | 0.983 | 16 | 0 |
| Qwen3-14B | T1pairS | snap_orig | 0.81 | +0.57 | 0.983 | 16 | 0 |
| Qwen3-4B-Instruct-2507 | A | quantised base (no edit) | 0.00 | | 1.000 | 0 | 0 |
| Qwen3-4B-Instruct-2507 | A | fp16row | 0.30 | +0.30 | 0.990 | 19 | 71440 |
| Qwen3-4B-Instruct-2507 | A | req4_newscale | 0.30 | +0.30 | 0.990 | 19 | 0 |
| Qwen3-4B-Instruct-2507 | A | snap_orig | 0.28 | +0.28 | 0.990 | 19 | 0 |
| Qwen3-4B-Instruct-2507 | B | quantised base (no edit) | 0.00 | | 0.990 | 0 | 0 |
| Qwen3-4B-Instruct-2507 | B | fp16row | 0.45 | +0.45 | 0.983 | 2 | 7520 |
| Qwen3-4B-Instruct-2507 | B | req4_newscale | 0.46 | +0.46 | 0.990 | 2 | 0 |
| Qwen3-4B-Instruct-2507 | B | snap_orig | 0.40 | +0.40 | 0.997 | 2 | 0 |
| Qwen3-8B | T1new | quantised base (no edit) | 0.00 | | 1.000 | 0 | 0 |
| Qwen3-8B | T1new | fp16row | 0.00 | +0.00 | 1.000 | 9 | 56149 |
| Qwen3-8B | T1new | req4_newscale | 0.00 | +0.00 | 1.000 | 9 | 0 |
| Qwen3-8B | T1new | snap_orig | 0.00 | +0.00 | 1.000 | 9 | 0 |
| Qwen3-8B | T1pairS | quantised base (no edit) | 0.15 | | 1.000 | 0 | 0 |
| Qwen3-8B | T1pairS | fp16row | 0.58 | +0.43 | 0.990 | 22 | 134357 |
| Qwen3-8B | T1pairS | req4_newscale | 0.59 | +0.44 | 0.990 | 22 | 0 |
| Qwen3-8B | T1pairS | snap_orig | 0.57 | +0.42 | 0.990 | 22 | 0 |

### lmhead_lda / blend sweep on dev (alpha, ridge): dev gain / neighbor-dev
| model | scen | fmt | cfg | dev gain | neigh_dev |
|---|---|---|---|---|---|
| Qwen3-1.7B | B | first | lmhead_lda alpha=0.2,ridge=0.01 | +0.12 | 0.97 |
| Qwen3-1.7B | B | first | lmhead_lda alpha=0.2,ridge=0.1 | +0.44 | 0.97 |
| Qwen3-1.7B | B | first | lmhead_lda alpha=0.5,ridge=0.01 | +0.96 | 0.84 |
| Qwen3-1.7B | B | first | lmhead_lda alpha=0.5,ridge=0.1 | +0.98 | 0.46 |
| Qwen3-1.7B | B | first | lmhead_lda alpha=1.0,ridge=0.01 | +0.98 | 0.03 |
| Qwen3-1.7B | B | first | lmhead_lda alpha=1.0,ridge=0.1 | +0.00 | 0.00 |
| Qwen3-1.7B | B | first | lmhead_lda alpha=2.0,ridge=0.01 | +0.00 | 0.00 |
| Qwen3-1.7B | B | first | lmhead_lda alpha=2.0,ridge=0.1 | +0.00 | 0.00 |
| Qwen3-1.7B | B | first | lmhead_lda alpha=3.5,ridge=0.01 | +0.00 | 0.00 |
| Qwen3-1.7B | B | first | lmhead_lda alpha=3.5,ridge=0.1 | +0.00 | 0.00 |
| Qwen3-1.7B | B | first | lmhead_mean alpha=0.2 | +0.04 | 0.00 |
| Qwen3-1.7B | B | first | memit_down ls=d3_80,lam=1000.0 | +0.92 | 0.95 |
| Qwen3-1.7B | B | first | blend ls=d3_80,lam=1000.0,alpha=0.5,ridge=0.01 | +0.92 | 0.95 |
| Qwen3-1.7B | B | first | blend ls=d3_80,lam=1000.0,alpha=1.0,ridge=0.01 | +0.92 | 0.95 |
| Qwen3-1.7B | B | first | blend ls=d3_80,lam=1000.0,alpha=2.0,ridge=0.01 | +0.92 | 0.95 |
| Qwen3-14B | T1pairS | first | lmhead_lda alpha=0.1,ridge=0.01 | +0.22 | 1.00 |
| Qwen3-14B | T1pairS | first | lmhead_lda alpha=0.1,ridge=0.1 | +0.28 | 1.00 |
| Qwen3-14B | T1pairS | first | lmhead_lda alpha=0.2,ridge=0.01 | +0.38 | 1.00 |
| Qwen3-14B | T1pairS | first | lmhead_lda alpha=0.2,ridge=0.1 | +0.38 | 1.00 |
| Qwen3-14B | T1pairS | first | lmhead_lda alpha=0.3,ridge=0.01 | +0.38 | 1.00 |
| Qwen3-14B | T1pairS | first | lmhead_lda alpha=0.3,ridge=0.1 | +0.40 | 1.00 |
| Qwen3-14B | T1pairS | first | lmhead_lda alpha=0.45,ridge=0.01 | +0.38 | 1.00 |
| Qwen3-14B | T1pairS | first | lmhead_lda alpha=0.45,ridge=0.1 | +0.28 | 0.94 |
| Qwen3-14B | T1pairS | first | lmhead_lda alpha=0.7,ridge=0.01 | -0.12 | 0.51 |
| Qwen3-14B | T1pairS | first | lmhead_lda alpha=0.7,ridge=0.1 | -0.30 | 0.00 |
| Qwen3-14B | T1pairS | first | lmhead_lda alpha=1.2,ridge=0.01 | -0.30 | 0.00 |
| Qwen3-14B | T1pairS | first | lmhead_lda alpha=1.2,ridge=0.1 | -0.30 | 0.00 |
| Qwen3-14B | T1pairS | first | lmhead_lda alpha=2.0,ridge=0.01 | -0.35 | 0.00 |
| Qwen3-14B | T1pairS | first | lmhead_lda alpha=2.0,ridge=0.1 | -0.35 | 0.00 |
| Qwen3-14B | T1pairS | first | lmhead_mean alpha=0.2 | -0.35 | 0.00 |
| Qwen3-14B | T1pairS | first | memit_down ls=d3_80,lam=1000.0 | +0.35 | 1.00 |
| Qwen3-14B | T1pairS | first | blend ls=d3_80,lam=1000.0,alpha=0.2,ridge=0.1 | +0.35 | 1.00 |
| Qwen3-14B | T1pairS | first | blend ls=d3_80,lam=1000.0,alpha=0.45,ridge=0.1 | +0.35 | 1.00 |
| Qwen3-14B | T1pairS | first | blend ls=d3_80,lam=1000.0,alpha=1.0,ridge=0.1 | +0.35 | 1.00 |
| Qwen3-4B-Instruct-2507 | A | first | lmhead_lda alpha=0.1,ridge=0.01 | +0.02 | 1.00 |
| Qwen3-4B-Instruct-2507 | A | first | lmhead_lda alpha=0.1,ridge=0.1 | +0.04 | 1.00 |
| Qwen3-4B-Instruct-2507 | A | first | lmhead_lda alpha=0.2,ridge=0.01 | +0.12 | 1.00 |
| Qwen3-4B-Instruct-2507 | A | first | lmhead_lda alpha=0.2,ridge=0.1 | +0.18 | 1.00 |
| Qwen3-4B-Instruct-2507 | A | first | lmhead_lda alpha=0.3,ridge=0.01 | +0.34 | 0.99 |
| Qwen3-4B-Instruct-2507 | A | first | lmhead_lda alpha=0.3,ridge=0.1 | +0.36 | 0.97 |
| Qwen3-4B-Instruct-2507 | A | first | lmhead_lda alpha=0.45,ridge=0.01 | +0.68 | 0.88 |
| Qwen3-4B-Instruct-2507 | A | first | lmhead_lda alpha=0.45,ridge=0.1 | +0.64 | 0.59 |
| Qwen3-4B-Instruct-2507 | A | first | lmhead_lda alpha=0.7,ridge=0.01 | +0.46 | 0.25 |
| Qwen3-4B-Instruct-2507 | A | first | lmhead_lda alpha=0.7,ridge=0.1 | +0.40 | 0.04 |
| Qwen3-4B-Instruct-2507 | A | first | lmhead_lda alpha=1.2,ridge=0.01 | +0.22 | 0.00 |
| Qwen3-4B-Instruct-2507 | A | first | lmhead_lda alpha=1.2,ridge=0.1 | +0.06 | 0.00 |
| Qwen3-4B-Instruct-2507 | A | first | lmhead_lda alpha=2.0,ridge=0.01 | +0.00 | 0.00 |
| Qwen3-4B-Instruct-2507 | A | first | lmhead_lda alpha=2.0,ridge=0.1 | +0.00 | 0.00 |
| Qwen3-4B-Instruct-2507 | A | first | lmhead_mean alpha=0.2 | +0.04 | 0.00 |
| Qwen3-4B-Instruct-2507 | A | first | memit_down ls=d3_80,lam=1000.0 | +0.50 | 0.89 |
| Qwen3-4B-Instruct-2507 | A | first | blend ls=d3_80,lam=1000.0,alpha=0.2,ridge=0.1 | +0.48 | 0.90 |
| Qwen3-4B-Instruct-2507 | A | first | blend ls=d3_80,lam=1000.0,alpha=0.45,ridge=0.1 | +0.48 | 0.90 |
| Qwen3-4B-Instruct-2507 | A | first | blend ls=d3_80,lam=1000.0,alpha=1.0,ridge=0.1 | +0.48 | 0.90 |
| Qwen3-4B-Instruct-2507 | B | first | lmhead_lda alpha=0.2,ridge=0.01 | +0.22 | 1.00 |
| Qwen3-4B-Instruct-2507 | B | first | lmhead_lda alpha=0.2,ridge=0.1 | +0.45 | 1.00 |
| Qwen3-4B-Instruct-2507 | B | first | lmhead_lda alpha=0.5,ridge=0.01 | +1.00 | 0.66 |
| Qwen3-4B-Instruct-2507 | B | first | lmhead_lda alpha=0.5,ridge=0.1 | +1.00 | 0.56 |
| Qwen3-4B-Instruct-2507 | B | first | lmhead_lda alpha=1.0,ridge=0.01 | +1.00 | 0.01 |
| Qwen3-4B-Instruct-2507 | B | first | lmhead_lda alpha=1.0,ridge=0.1 | +0.74 | 0.01 |
| Qwen3-4B-Instruct-2507 | B | first | lmhead_lda alpha=2.0,ridge=0.01 | +0.00 | 0.00 |
| Qwen3-4B-Instruct-2507 | B | first | lmhead_lda alpha=2.0,ridge=0.1 | +0.00 | 0.00 |
| Qwen3-4B-Instruct-2507 | B | first | lmhead_lda alpha=3.5,ridge=0.01 | +0.00 | 0.00 |
| Qwen3-4B-Instruct-2507 | B | first | lmhead_lda alpha=3.5,ridge=0.1 | +0.00 | 0.00 |
| Qwen3-4B-Instruct-2507 | B | first | lmhead_mean alpha=0.2 | +0.16 | 0.21 |
| Qwen3-4B-Instruct-2507 | B | first | memit_down ls=d3_80,lam=1000.0 | +1.00 | 0.70 |
| Qwen3-4B-Instruct-2507 | B | first | blend ls=d3_80,lam=1000.0,alpha=0.5,ridge=0.01 | +1.00 | 0.91 |
| Qwen3-4B-Instruct-2507 | B | first | blend ls=d3_80,lam=1000.0,alpha=1.0,ridge=0.01 | +1.00 | 0.91 |
| Qwen3-4B-Instruct-2507 | B | first | blend ls=d3_80,lam=1000.0,alpha=2.0,ridge=0.01 | +1.00 | 0.91 |
| Qwen3-8B | T1new | first | lmhead_lda alpha=0.1,ridge=0.01 | +0.00 | 1.00 |
| Qwen3-8B | T1new | first | lmhead_lda alpha=0.1,ridge=0.1 | +0.00 | 1.00 |
| Qwen3-8B | T1new | first | lmhead_lda alpha=0.2,ridge=0.01 | +0.00 | 1.00 |
| Qwen3-8B | T1new | first | lmhead_lda alpha=0.2,ridge=0.1 | +0.00 | 1.00 |
| Qwen3-8B | T1new | first | lmhead_lda alpha=0.3,ridge=0.01 | +0.00 | 1.00 |
| Qwen3-8B | T1new | first | lmhead_lda alpha=0.3,ridge=0.1 | +0.00 | 1.00 |
| Qwen3-8B | T1new | first | lmhead_lda alpha=0.45,ridge=0.01 | +0.00 | 1.00 |
| Qwen3-8B | T1new | first | lmhead_lda alpha=0.45,ridge=0.1 | +0.00 | 1.00 |
| Qwen3-8B | T1new | first | lmhead_lda alpha=0.7,ridge=0.01 | +0.00 | 0.90 |
| Qwen3-8B | T1new | first | lmhead_lda alpha=0.7,ridge=0.1 | +0.00 | 0.26 |
| Qwen3-8B | T1new | first | lmhead_lda alpha=1.2,ridge=0.01 | +0.00 | 0.00 |
| Qwen3-8B | T1new | first | lmhead_lda alpha=1.2,ridge=0.1 | +0.00 | 0.00 |
| Qwen3-8B | T1new | first | lmhead_lda alpha=2.0,ridge=0.01 | +0.00 | 0.00 |
| Qwen3-8B | T1new | first | lmhead_lda alpha=2.0,ridge=0.1 | +0.00 | 0.00 |
| Qwen3-8B | T1new | first | lmhead_mean alpha=0.2 | +0.00 | 0.00 |
| Qwen3-8B | T1new | first | memit_down ls=d3_80,lam=1000.0 | +0.53 | 0.95 |
| Qwen3-8B | T1new | first | blend ls=d3_80,lam=1000.0,alpha=0.2,ridge=0.1 | +0.53 | 0.95 |
| Qwen3-8B | T1new | first | blend ls=d3_80,lam=1000.0,alpha=0.45,ridge=0.1 | +0.45 | 0.93 |
| Qwen3-8B | T1new | first | blend ls=d3_80,lam=1000.0,alpha=1.0,ridge=0.1 | +0.25 | 0.70 |
| Qwen3-8B | T1new | last | lmhead_lda alpha=0.1,ridge=0.01 | +0.00 | 1.00 |
| Qwen3-8B | T1new | last | lmhead_lda alpha=0.1,ridge=0.1 | +0.00 | 1.00 |
| Qwen3-8B | T1new | last | lmhead_lda alpha=0.2,ridge=0.01 | +0.00 | 0.99 |
| Qwen3-8B | T1new | last | lmhead_lda alpha=0.2,ridge=0.1 | +0.00 | 0.95 |
| Qwen3-8B | T1new | last | lmhead_lda alpha=0.3,ridge=0.01 | +0.00 | 0.78 |
| Qwen3-8B | T1new | last | lmhead_lda alpha=0.3,ridge=0.1 | +0.00 | 0.21 |
| Qwen3-8B | T1new | last | lmhead_lda alpha=0.45,ridge=0.01 | +0.03 | 0.01 |
| Qwen3-8B | T1new | last | lmhead_lda alpha=0.45,ridge=0.1 | +0.00 | 0.01 |
| Qwen3-8B | T1new | last | lmhead_lda alpha=0.7,ridge=0.01 | +0.00 | 0.01 |
| Qwen3-8B | T1new | last | lmhead_lda alpha=0.7,ridge=0.1 | +0.00 | 0.01 |
| Qwen3-8B | T1new | last | lmhead_lda alpha=1.2,ridge=0.01 | +0.00 | 0.01 |
| Qwen3-8B | T1new | last | lmhead_lda alpha=1.2,ridge=0.1 | +0.00 | 0.00 |
| Qwen3-8B | T1new | last | lmhead_lda alpha=2.0,ridge=0.01 | +0.00 | 0.00 |
| Qwen3-8B | T1new | last | lmhead_lda alpha=2.0,ridge=0.1 | +0.00 | 0.00 |
| Qwen3-8B | T1new | last | lmhead_mean alpha=0.2 | +0.00 | 0.00 |
| Qwen3-8B | T1new | last | memit_down ls=d3_80,lam=1000.0 | +0.35 | 0.03 |
| Qwen3-8B | T1new | last | blend ls=d3_80,lam=1000.0,alpha=0.2,ridge=0.1 | +0.53 | 0.07 |
| Qwen3-8B | T1new | last | blend ls=d3_80,lam=1000.0,alpha=0.45,ridge=0.1 | +0.20 | 0.06 |
| Qwen3-8B | T1new | last | blend ls=d3_80,lam=1000.0,alpha=1.0,ridge=0.1 | +0.00 | 0.09 |
| Qwen3-8B | T1pairS | first | lmhead_lda alpha=0.1,ridge=0.01 | +0.33 | 1.00 |
| Qwen3-8B | T1pairS | first | lmhead_lda alpha=0.1,ridge=0.1 | +0.40 | 1.00 |
| Qwen3-8B | T1pairS | first | lmhead_lda alpha=0.2,ridge=0.01 | +0.65 | 1.00 |
| Qwen3-8B | T1pairS | first | lmhead_lda alpha=0.2,ridge=0.1 | +0.70 | 1.00 |
| Qwen3-8B | T1pairS | first | lmhead_lda alpha=0.3,ridge=0.01 | +0.70 | 1.00 |
| Qwen3-8B | T1pairS | first | lmhead_lda alpha=0.3,ridge=0.1 | +0.70 | 1.00 |
| Qwen3-8B | T1pairS | first | lmhead_lda alpha=0.45,ridge=0.01 | +0.70 | 0.99 |
| Qwen3-8B | T1pairS | first | lmhead_lda alpha=0.45,ridge=0.1 | +0.60 | 0.93 |
| Qwen3-8B | T1pairS | first | lmhead_lda alpha=0.7,ridge=0.01 | +0.53 | 0.85 |
| Qwen3-8B | T1pairS | first | lmhead_lda alpha=0.7,ridge=0.1 | +0.38 | 0.71 |
| Qwen3-8B | T1pairS | first | lmhead_lda alpha=1.2,ridge=0.01 | +0.25 | 0.54 |
| Qwen3-8B | T1pairS | first | lmhead_lda alpha=1.2,ridge=0.1 | +0.05 | 0.29 |
| Qwen3-8B | T1pairS | first | lmhead_lda alpha=2.0,ridge=0.01 | -0.12 | 0.05 |
| Qwen3-8B | T1pairS | first | lmhead_lda alpha=2.0,ridge=0.1 | -0.12 | 0.00 |
| Qwen3-8B | T1pairS | first | lmhead_mean alpha=0.2 | -0.12 | 0.00 |
| Qwen3-8B | T1pairS | first | memit_down ls=d3_80,lam=1000.0 | +0.65 | 0.99 |
| Qwen3-8B | T1pairS | first | blend ls=d3_80,lam=1000.0,alpha=0.2,ridge=0.1 | +0.68 | 0.97 |
| Qwen3-8B | T1pairS | first | blend ls=d3_80,lam=1000.0,alpha=0.45,ridge=0.1 | +0.57 | 0.94 |
| Qwen3-8B | T1pairS | first | blend ls=d3_80,lam=1000.0,alpha=1.0,ridge=0.1 | +0.02 | 0.06 |
