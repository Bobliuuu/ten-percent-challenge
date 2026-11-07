| model | scenario | fmt | method (cfg) | base teach | teach gain (95% lo) | neighbors | real-shuf (lo) | G2 |
|---|---|---|---|---|---|---|---|---|
| Qwen3-1.7B | B | first | lmhead_lda (alpha=0.2,ridge=0.1) | 0.00 | +0.52 (+0.46) | 0.990 | +0.52 (+0.46) | PASS |
| Qwen3-1.7B | B | first | blend (ls=d3_80,lam=1000.0,alpha=0.5,ridge=0.01) | 0.00 | +0.91 (+0.88) | 0.917 | +0.91 (+0.88) | FAIL |
| Qwen3-1.7B | B | first | memit_down (ls=d3_80,lam=1000.0) | 0.00 | +0.92 (+0.89) | 0.860 | +0.92 (+0.89) | FAIL |
| Qwen3-4B-Instruct-2507 | A | first | lmhead_lda (alpha=0.3,ridge=0.01) | 0.00 | +0.29 (+0.24) | 0.990 | +0.15 (+0.11) | PASS |
| Qwen3-4B-Instruct-2507 | A | first | blend (ls=d3_80,lam=1000.0,alpha=0.2,ridge=0.1) | 0.00 | +0.50 (+0.44) | 0.867 | +0.43 (+0.37) | FAIL |
| Qwen3-4B-Instruct-2507 | A | first | memit_down (ls=d3_80,lam=1000.0) | 0.00 | +0.50 (+0.44) | 0.877 | +0.46 (+0.40) | FAIL |
| Qwen3-4B-Instruct-2507 | B | first | lmhead_lda (alpha=0.2,ridge=0.1) | 0.00 | +0.36 (+0.31) | 1.000 | +0.36 (+0.31) | PASS |
| Qwen3-4B-Instruct-2507 | B | first | blend (ls=d3_80,lam=1000.0,alpha=0.5,ridge=0.01) | 0.00 | +0.98 (+0.96) | 0.957 | +0.98 (+0.96) | FAIL |
| Qwen3-4B-Instruct-2507 | B | first | memit_down (ls=d3_80,lam=1000.0) | 0.00 | +0.99 (+0.97) | 0.907 | +0.99 (+0.97) | FAIL |

### lm_head codec test (lm_head 4-bit g128 RTN as stage-1; edited rows: fp16 kept / re-quantised 4-bit own scales / snapped to original scales)
| model | scenario | variant | teach acc | gain vs quantised base | neighbors | rows edited | bytes added |
|---|---|---|---|---|---|---|---|
| Qwen3-1.7B | B | quantised base (no edit) | 0.00 | | 0.990 | 0 | 0 |
| Qwen3-1.7B | B | fp16row | 0.53 | +0.53 | 0.980 | 2 | 6016 |
| Qwen3-1.7B | B | req4_newscale | 0.55 | +0.55 | 0.980 | 2 | 0 |
| Qwen3-1.7B | B | snap_orig | 0.52 | +0.52 | 0.983 | 2 | 0 |
| Qwen3-4B-Instruct-2507 | A | quantised base (no edit) | 0.00 | | 1.000 | 0 | 0 |
| Qwen3-4B-Instruct-2507 | A | fp16row | 0.30 | +0.30 | 0.990 | 19 | 71440 |
| Qwen3-4B-Instruct-2507 | A | req4_newscale | 0.30 | +0.30 | 0.990 | 19 | 0 |
| Qwen3-4B-Instruct-2507 | A | snap_orig | 0.28 | +0.28 | 0.990 | 19 | 0 |
| Qwen3-4B-Instruct-2507 | B | quantised base (no edit) | 0.00 | | 0.990 | 0 | 0 |
| Qwen3-4B-Instruct-2507 | B | fp16row | 0.45 | +0.45 | 0.983 | 2 | 7520 |
| Qwen3-4B-Instruct-2507 | B | req4_newscale | 0.46 | +0.46 | 0.990 | 2 | 0 |
| Qwen3-4B-Instruct-2507 | B | snap_orig | 0.40 | +0.40 | 0.997 | 2 | 0 |

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
