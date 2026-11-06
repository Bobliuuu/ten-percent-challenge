| model | task | method | teach gain (95% lo) | neighbors | real-shuf (lo) | drift tok-agree | G2 |
|---|---|---|---|---|---|---|---|
| Qwen3-1.7B_A_late | A | memit_down | +0.07 (+0.04) | 0.945 | +0.07 (+0.04) | 0.922 | FAIL |
| Qwen3-1.7B_A_late | A | memit_o | +0.13 (+0.09) | 0.603 | +0.12 (+0.09) | 0.809 | FAIL |
| Qwen3-1.7B_A_late | A | lmhead_mean | +0.13 (+0.09) | 0.950 | +0.03 (+0.00) | 0.858 | FAIL |
| Qwen3-1.7B_A_late | A | lmhead_lda | +0.20 (+0.16) | 0.960 | +0.06 (+0.03) | 0.953 | FAIL |
| Qwen3-1.7B_C_late | C | memit_down | +0.56 (+0.49) | nan | +0.42 (+0.34) | 0.948 | FAIL |
| Qwen3-1.7B_C_late | C | memit_o | +0.55 (+0.48) | nan | +0.55 (+0.48) | 0.977 | FAIL |
| Qwen3-1.7B_C_late | C | lmhead_mean | +0.03 (+0.01) | nan | +0.01 (-0.02) | 0.897 | FAIL |
| Qwen3-1.7B_C_late | C | lmhead_lda | +0.18 (+0.13) | nan | +0.01 (-0.05) | 0.988 | FAIL |
| Qwen3-4B-Instruct-2507 | A | memit_down | +0.44 (+0.38) | 0.868 | +0.42 (+0.36) | 0.926 | FAIL |
| Qwen3-4B-Instruct-2507 | A | W4+memit_down[none] | +0.37 | 0.882 | | | n/a |
| Qwen3-4B-Instruct-2507 | A | W4+memit_down[orig] | +0.30 | 0.902 | | | n/a |
| Qwen3-4B-Instruct-2507 | A | W4+memit_down[new] | +0.35 | 0.892 | | | n/a |
| Qwen3-4B-Instruct-2507 | A | memit_o | +0.36 (+0.30) | 0.882 | +0.36 (+0.30) | 0.890 | FAIL |
| Qwen3-4B-Instruct-2507 | A | lmhead_mean | +0.17 (+0.13) | 0.990 | +0.01 (-0.03) | 0.870 | FAIL |
| Qwen3-4B-Instruct-2507 | A | lmhead_lda | +0.40 (+0.34) | 0.992 | +0.24 (+0.20) | 0.930 | PASS |
| Qwen3-4B-Instruct-2507_A_late | A | memit_down | +0.41 (+0.36) | 0.953 | +0.35 (+0.30) | 0.961 | FAIL |
| Qwen3-4B-Instruct-2507_A_late | A | W4+memit_down[none] | +0.36 | 0.878 | | | n/a |
| Qwen3-4B-Instruct-2507_A_late | A | W4+memit_down[orig] | +0.01 | 0.970 | | | n/a |
| Qwen3-4B-Instruct-2507_A_late | A | W4+memit_down[new] | +0.02 | 0.962 | | | n/a |
| Qwen3-4B-Instruct-2507_A_late | A | memit_o | +0.40 (+0.34) | 0.852 | +0.36 (+0.31) | 0.948 | FAIL |
| Qwen3-4B-Instruct-2507_A_late | A | lmhead_mean | +0.07 (+0.05) | 0.995 | +0.02 (+0.00) | 0.885 | FAIL |
| Qwen3-4B-Instruct-2507_A_late | A | lmhead_lda | +0.25 (+0.20) | 0.995 | +0.19 (+0.15) | 0.941 | PASS |
| Qwen3-8B | A | memit_down | +0.03 (+0.01) | 0.912 | +0.03 (+0.01) | 0.920 | FAIL |
| Qwen3-8B | A | W4+memit_down[none] | +0.03 | 0.918 | | | n/a |
| Qwen3-8B | A | W4+memit_down[orig] | +0.02 | 0.935 | | | n/a |
| Qwen3-8B | A | W4+memit_down[new] | +0.03 | 0.915 | | | n/a |
| Qwen3-8B | A | memit_o | +0.15 (+0.11) | 0.912 | +0.14 (+0.10) | 0.925 | FAIL |
| Qwen3-8B | A | lmhead_mean | +0.18 (+0.13) | 0.985 | +0.04 (+0.02) | 0.927 | FAIL |
| Qwen3-8B | A | lmhead_lda | +0.47 (+0.41) | 0.983 | +0.20 (+0.15) | 1.000 | PASS |
| Qwen3-8B | C | memit_down | +0.06 (+0.03) | 0.957 | -0.01 (-0.05) | 0.931 | FAIL |
| Qwen3-8B | C | W4+memit_down[none] | +0.30 | 1.000 | | | n/a |
| Qwen3-8B | C | W4+memit_down[orig] | +0.24 | 1.000 | | | n/a |
| Qwen3-8B | C | W4+memit_down[new] | +0.31 | 1.000 | | | n/a |
| Qwen3-8B | C | memit_o | +0.25 (+0.18) | 0.928 | +0.25 (+0.18) | 0.946 | FAIL |
| Qwen3-8B | C | lmhead_mean | +0.17 (+0.11) | 1.000 | +0.06 (+0.03) | 0.999 | FAIL |
| Qwen3-8B | C | lmhead_lda | +0.33 (+0.26) | 0.957 | +0.12 (+0.07) | 1.000 | FAIL |

## size ladder (teach gain, neighbor retention) linear in log10(params B), extrapolated
| task | method | sizes | gain per size | slope/decade (95%) | pred gain @14B (95%) | pred gain @32B (95%) | neigh per size |
|---|---|---|---|---|---|---|---|
| A | lmhead_lda | Qwen3-1.7B,Qwen3-4B-Instruct-2507,Qwen3-8B | +0.20,+0.25,+0.47 | +0.39 (+0.29,+0.49) | +0.53 (+0.46,+0.61) | +0.67 (+0.56,+0.78) | 0.960,0.995,0.983 |
| A | lmhead_mean | Qwen3-1.7B,Qwen3-4B-Instruct-2507,Qwen3-8B | +0.13,+0.07,+0.18 | +0.07 (-0.01,+0.15) | +0.17 (+0.11,+0.22) | +0.19 (+0.11,+0.27) | 0.950,0.995,0.985 |
| A | memit_down | Qwen3-1.7B,Qwen3-4B-Instruct-2507,Qwen3-8B | +0.07,+0.41,+0.03 | -0.03 (-0.08,+0.02) | +0.15 (+0.12,+0.19) | +0.15 (+0.10,+0.19) | 0.945,0.953,0.912 |
| A | memit_o | Qwen3-1.7B,Qwen3-4B-Instruct-2507,Qwen3-8B | +0.13,+0.40,+0.15 | +0.05 (-0.03,+0.12) | +0.25 (+0.20,+0.31) | +0.27 (+0.19,+0.35) | 0.603,0.852,0.912 |
| C | lmhead_lda | Qwen3-1.7B,Qwen3-8B | +0.18,+0.33 | +0.22 (+0.09,+0.35) | +0.39 (+0.29,+0.50) | +0.47 (+0.32,+0.62) | nan,0.957 |
| C | lmhead_mean | Qwen3-1.7B,Qwen3-8B | +0.03,+0.17 | +0.21 (+0.12,+0.32) | +0.22 (+0.15,+0.31) | +0.29 (+0.19,+0.42) | nan,1.000 |
| C | memit_down | Qwen3-1.7B,Qwen3-8B | +0.56,+0.06 | -0.74 (-0.85,-0.63) | -0.13 (-0.18,-0.08) | -0.39 (-0.47,-0.31) | nan,0.957 |
| C | memit_o | Qwen3-1.7B,Qwen3-8B | +0.55,+0.25 | -0.44 (-0.59,-0.28) | +0.14 (+0.05,+0.23) | -0.01 (-0.15,+0.13) | nan,0.928 |
