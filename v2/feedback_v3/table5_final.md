| model | scenario | method | draws | ann tools | base teach | gain vs empty (lo) | real-mixed (lo) | real-randtool (lo) | neighbors (n, lo) | G2 |
|---|---|---|---|---|---|---|---|---|---|---|
| Qwen3-14B | pairS | lmhead_lda a=0.2 | 10 | 593 | 0.29 | +0.48 (+0.45) | +0.44 (+0.40) | +0.51 (+0.46) | 0.987 (n=400, lo 0.983) | PASS |
| Qwen3-14B | pairS | lmhead_plain a=0.03 | 4 | 593 | 0.29 | +0.18 (+0.13) | +0.15 (+0.09) | +0.21 (+0.17) | 0.994 (n=400, lo 0.989) | PASS |
| Qwen3-4B-Instruct-2507 | nl_tox | lmhead_lda a=0.7 | 10 | 1 | 0.00 | +0.65 (+0.61) | +0.19 (+0.05) | +0.65 (+0.61) | 0.981 (n=400, lo 0.976) | PASS |
| Qwen3-4B-Instruct-2507 | nl_tox | lmhead_plain a=0.03 | 10 | 1 | 0.00 | +0.00 (+0.00) | +0.00 (+0.00) | +0.00 (+0.00) | 0.985 (n=400, lo 0.980) | FAIL |
| Qwen3-4B-Instruct-2507 | nl_tox | lmhead_plain_xs a=0.01 | 10 | 1 | 0.00 | +0.00 (+0.00) | +0.00 (+0.00) | +0.00 (+0.00) | 0.999 (n=400, lo 0.997) | FAIL |
| Qwen3-4B-Instruct-2507 | nl_tox | lmhead_cal | 4 | 1 | 0.00 | +0.02 (+0.00) | -0.08 (-0.20) | +0.02 (+0.00) | 0.917 (n=400, lo 0.903) | FAIL |
| Qwen3-8B | nat_argstr | lmhead_lda a=0.1 | 10 | 42 | 0.00 | +0.01 (+0.00) | +0.01 (-0.01) | +0.01 (-0.00) | 0.992 (n=400, lo 0.989) | FAIL |
| Qwen3-8B | nat_argstr | lmhead_cal | 10 | 42 | 0.00 | +0.01 (+0.00) | +0.00 (-0.01) | +0.00 (-0.01) | 0.979 (n=400, lo 0.974) | FAIL |
| Qwen3-8B | nat_argstr | memit_down | 8 | 42 | 0.00 | +0.03 (+0.02) | +0.01 (-0.01) | +0.00 (-0.02) | 0.978 (n=400, lo 0.972) | FAIL |
| Qwen3-8B | newtool | lmhead_plain a=0.2 | 5 | 1 | 0.00 | +0.00 (+0.00) | +0.00 (+0.00) | +0.00 (+0.00) | 0.002 (n=400, lo 0.001) | FAIL |
| Qwen3-8B | newtool | lmhead_cal2 | 5 | 1 | 0.00 | +0.08 (+0.02) | +0.04 (-0.03) | +0.08 (+0.02) | 0.659 (n=400, lo 0.638) | FAIL |
| Qwen3-8B | newtool | lmhead_lda a=0.45 | 5 | 1 | 0.00 | +0.02 (+0.00) | +0.01 (+0.00) | +0.02 (+0.00) | 0.966 (n=400, lo 0.957) | FAIL |
| Qwen3-8B | newtool | lmhead_cal | 5 | 1 | 0.00 | +0.86 (+0.77) | +0.72 (+0.62) | +0.86 (+0.77) | 0.536 (n=400, lo 0.514) | FAIL |
| Qwen3-8B | newtool | memit_down | 5 | 1 | 0.00 | +0.45 (+0.36) | +0.39 (+0.29) | +0.45 (+0.36) | 0.584 (n=400, lo 0.562) | FAIL |
| Qwen3-8B | nl_sent | lmhead_lda a=0.45 | 5 | 1 | 0.18 | +0.37 (+0.31) | +0.06 (+0.00) | +0.55 (+0.49) | 0.984 (n=400, lo 0.977) | FAIL |
| Qwen3-8B | nl_tox | lmhead_lda a=0.45 | 10 | 1 | 0.00 | +0.62 (+0.58) | -0.00 (-0.02) | +0.62 (+0.58) | 0.988 (n=400, lo 0.984) | FAIL |
| Qwen3-8B | nl_tox | lmhead_plain a=0.03 | 10 | 1 | 0.00 | +0.03 (+0.02) | +0.00 (+0.00) | +0.03 (+0.02) | 0.944 (n=400, lo 0.936) | FAIL |
| Qwen3-8B | nl_tox | lmhead_plain_xs a=0.01 | 10 | 1 | 0.00 | +0.00 (+0.00) | +0.00 (+0.00) | +0.00 (+0.00) | 0.977 (n=400, lo 0.971) | FAIL |
| Qwen3-8B | pairS | lmhead_lda a=0.1 | 10 | 593 | 0.12 | +0.27 (+0.24) | +0.25 (+0.19) | +0.28 (+0.25) | 0.995 (n=400, lo 0.992) | PASS |
| Qwen3-8B | pairS | lmhead_plain a=0.2 | 8 | 593 | 0.12 | -0.12 (-0.14) | +0.00 (+0.00) | +0.00 (+0.00) | 0.000 (n=400, lo 0.000) | FAIL |
| Qwen3-8B | pairS | lmhead_cal | 10 | 593 | 0.12 | +0.56 (+0.50) | +0.20 (+0.07) | +0.60 (+0.54) | 0.954 (n=400, lo 0.947) | FAIL |
| Qwen3-8B | pairS | lmhead_cal2 | 10 | 593 | 0.12 | +0.57 (+0.53) | +0.18 (+0.05) | +0.61 (+0.56) | 0.996 (n=400, lo 0.993) | PASS |
| Qwen3-8B | pairS | memit_down | 10 | 593 | 0.12 | +0.51 (+0.41) | +0.13 (+0.07) | +0.51 (+0.41) | 0.935 (n=400, lo 0.927) | FAIL |
| Qwen3-8B | pairS | lmhead_lda a=0.1 | 3 | 593 | 0.12 | +0.27 (+0.22) | +0.28 (+0.23) | +0.27 (+0.21) | 0.995 (n=400, lo 0.989) | PASS |
| Qwen3-8B | pairS | lmhead_plain a=0.03 | 10 | 593 | 0.12 | +0.22 (+0.14) | +0.21 (+0.12) | +0.24 (+0.15) | 0.994 (n=400, lo 0.991) | PASS |

### real-codec stand-in (rotation + L=8 grid per-128 scale + ideal-rANS entropy bytes), lm_head only
| model | scenario | variant | teach gain vs quantised base | neighbors | rows edited | artifact bytes before | after | delta bytes |
|---|---|---|---|---|---|---|---|---|
| Qwen3-8B | pairS | reencode_refit | +0.01 | 0.995 | 16 | 225358389 (2.897 bpw; fp16 1244659712) | 225358404 | +15 |
| Qwen3-8B | pairS | reencode_orig_scale | +0.00 | 0.995 | 16 | 225358389 (2.897 bpw; fp16 1244659712) | 225358397 | +8 |

### sweep (draw 0, dev)
| scenario | cfg | dev gain | neigh_dev |
|---|---|---|---|
| pairS | {'m': 'lmhead_lda', 'alpha': 0.1, 'ridge': 0.1} | 0.2 | 0.975 |
| pairS | {'m': 'lmhead_lda', 'alpha': 0.2, 'ridge': 0.1} | 0.37999999999999995 | 0.9625 |
| pairS | {'m': 'lmhead_lda', 'alpha': 0.3, 'ridge': 0.1} | 0.32 | 0.9125 |
| pairS | {'m': 'lmhead_lda', 'alpha': 0.45, 'ridge': 0.1} | -0.14 | 0.3625 |
| pairS | {'m': 'lmhead_lda', 'alpha': 0.7, 'ridge': 0.1} | -0.4 | 0.0875 |
| pairS | {'m': 'lmhead_plain', 'alpha': 0.2} | -0.44 | 0.0 |
| pairS | {'m': 'lmhead_cal'} | 0.08000000000000002 | 0.875 |
| pairS | {'m': 'lmhead_cal2'} | 0.2 | 0.9875 |
| pairS | {'m': 'memit_down', 'ls': 'd3_80', 'lam': 1000.0} | -0.28 | 0.2875 |
| pairS | {'m': 'lmhead_plain', 'alpha': 0.03} | 0.06 | 0.9875 |
| pairS | {'m': 'lmhead_plain', 'alpha': 0.01} | 0.0 | 1.0 |
| nl_tox | {'m': 'lmhead_lda', 'alpha': 0.1, 'ridge': 0.1} | 0.0 | 0.9875 |
| nl_tox | {'m': 'lmhead_lda', 'alpha': 0.2, 'ridge': 0.1} | 0.0 | 1.0 |
| nl_tox | {'m': 'lmhead_lda', 'alpha': 0.3, 'ridge': 0.1} | 0.19230769230769232 | 1.0 |
| nl_tox | {'m': 'lmhead_lda', 'alpha': 0.45, 'ridge': 0.1} | 0.5384615384615384 | 1.0 |
| nl_tox | {'m': 'lmhead_lda', 'alpha': 0.7, 'ridge': 0.1} | 0.8076923076923077 | 0.9875 |
| nl_tox | {'m': 'lmhead_plain', 'alpha': 0.03} | 0.0 | 1.0 |
| nl_tox | {'m': 'lmhead_plain', 'alpha': 0.01} | 0.0 | 0.9875 |
| nl_tox | {'m': 'lmhead_cal'} | 0.0 | 0.9625 |
| nat_argstr | {'m': 'lmhead_lda', 'alpha': 0.1, 'ridge': 0.1} | 0.18181818181818182 | 0.9875 |
| nat_argstr | {'m': 'lmhead_lda', 'alpha': 0.2, 'ridge': 0.1} | 0.18181818181818182 | 0.975 |
| nat_argstr | {'m': 'lmhead_lda', 'alpha': 0.3, 'ridge': 0.1} | 0.2727272727272727 | 0.9625 |
| nat_argstr | {'m': 'lmhead_lda', 'alpha': 0.45, 'ridge': 0.1} | 0.2727272727272727 | 0.8 |
| nat_argstr | {'m': 'lmhead_lda', 'alpha': 0.7, 'ridge': 0.1} | 0.21212121212121213 | 0.125 |
| nat_argstr | {'m': 'lmhead_cal'} | 0.18181818181818182 | 0.9375 |
| nat_argstr | {'m': 'memit_down', 'ls': 'd3_80', 'lam': 1000.0} | 0.21212121212121213 | 0.925 |
| newtool | {'m': 'lmhead_plain', 'alpha': 0.2} | 0.0 | 0.0 |
| newtool | {'m': 'lmhead_cal2'} | 0.0 | 0.5875 |
| newtool | {'m': 'lmhead_lda', 'alpha': 0.1, 'ridge': 0.1} | 0.0 | 0.9875 |
| newtool | {'m': 'lmhead_lda', 'alpha': 0.2, 'ridge': 0.1} | 0.0 | 0.9875 |
| newtool | {'m': 'lmhead_lda', 'alpha': 0.3, 'ridge': 0.1} | 0.0 | 0.9875 |
| newtool | {'m': 'lmhead_lda', 'alpha': 0.45, 'ridge': 0.1} | 0.04 | 0.9875 |
| newtool | {'m': 'lmhead_lda', 'alpha': 0.7, 'ridge': 0.1} | 0.0 | 0.075 |
| newtool | {'m': 'lmhead_cal'} | 0.78 | 0.4 |
| newtool | {'m': 'memit_down', 'ls': 'd3_80', 'lam': 1000.0} | 0.34 | 0.8 |
| nl_sent | {'m': 'lmhead_lda', 'alpha': 0.1, 'ridge': 0.1} | 0.1 | 1.0 |
| nl_sent | {'m': 'lmhead_lda', 'alpha': 0.2, 'ridge': 0.1} | 0.16 | 0.9875 |
| nl_sent | {'m': 'lmhead_lda', 'alpha': 0.3, 'ridge': 0.1} | 0.28 | 0.9875 |
| nl_sent | {'m': 'lmhead_lda', 'alpha': 0.45, 'ridge': 0.1} | 0.36 | 0.9875 |
| nl_sent | {'m': 'lmhead_lda', 'alpha': 0.7, 'ridge': 0.1} | 0.44000000000000006 | 0.975 |
| nl_tox | {'m': 'lmhead_lda', 'alpha': 0.1, 'ridge': 0.1} | 0.038461538461538464 | 1.0 |
| nl_tox | {'m': 'lmhead_lda', 'alpha': 0.2, 'ridge': 0.1} | 0.38461538461538464 | 1.0 |
| nl_tox | {'m': 'lmhead_lda', 'alpha': 0.3, 'ridge': 0.1} | 0.6538461538461539 | 1.0 |
| nl_tox | {'m': 'lmhead_lda', 'alpha': 0.45, 'ridge': 0.1} | 0.7307692307692307 | 1.0 |
| nl_tox | {'m': 'lmhead_lda', 'alpha': 0.7, 'ridge': 0.1} | 0.7307692307692307 | 1.0 |
| nl_tox | {'m': 'lmhead_plain', 'alpha': 0.03} | 0.07692307692307693 | 0.975 |
| nl_tox | {'m': 'lmhead_plain', 'alpha': 0.01} | 0.0 | 1.0 |
| nl_tox | {'m': 'lmhead_cal'} | 0.0 | 0.9625 |
| pairS | {'m': 'lmhead_lda', 'alpha': 0.1, 'ridge': 0.1} | 0.32 | 0.9875 |
| pairS | {'m': 'lmhead_lda', 'alpha': 0.2, 'ridge': 0.1} | 0.64 | 0.925 |
| pairS | {'m': 'lmhead_lda', 'alpha': 0.3, 'ridge': 0.1} | 0.7 | 0.85 |
| pairS | {'m': 'lmhead_lda', 'alpha': 0.45, 'ridge': 0.1} | -0.039999999999999994 | 0.25 |
| pairS | {'m': 'lmhead_lda', 'alpha': 0.7, 'ridge': 0.1} | -0.09999999999999999 | 0.0 |
| pairS | {'m': 'lmhead_plain', 'alpha': 0.2} | -0.12 | 0.0 |
| pairS | {'m': 'lmhead_cal'} | 0.7 | 0.8875 |
| pairS | {'m': 'lmhead_cal2'} | 0.7 | 0.9875 |
| pairS | {'m': 'memit_down', 'ls': 'd3_80', 'lam': 1000.0} | 0.1 | 0.4375 |
| pairS | {'m': 'lmhead_cal'} | 0.7 | 0.8875 |
| pairS | {'m': 'lmhead_cal2'} | 0.7 | 1.0 |
| pairS | {'m': 'memit_down', 'ls': 'd3_80', 'lam': 1000.0} | 0.1 | 0.4375 |
| pairS | {'m': 'lmhead_lda', 'alpha': 0.1, 'ridge': 0.1} | 0.32 | 0.9875 |
| pairS | {'m': 'lmhead_lda', 'alpha': 0.2, 'ridge': 0.1} | 0.64 | 0.925 |
| pairS | {'m': 'lmhead_lda', 'alpha': 0.3, 'ridge': 0.1} | 0.7 | 0.85 |
| pairS | {'m': 'lmhead_lda', 'alpha': 0.45, 'ridge': 0.1} | -0.039999999999999994 | 0.25 |
| pairS | {'m': 'lmhead_lda', 'alpha': 0.7, 'ridge': 0.1} | -0.09999999999999999 | 0.0 |
| pairS | {'m': 'lmhead_plain', 'alpha': 0.03} | 0.28 | 1.0 |
| pairS | {'m': 'lmhead_plain', 'alpha': 0.01} | 0.1 | 1.0 |
