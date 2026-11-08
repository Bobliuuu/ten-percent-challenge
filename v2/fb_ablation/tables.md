### T1. Qwen2.5-7B-Instruct bf16 body, SKILL7-derived, new label Jukebox, n=12 annotations (ranked by teach - shuffled, then neighbours)

| model | method | draws | teach [CI] | shuffled | wrong | teach-shuf | neigh kept | worst draw | min cell | cal agree | paraphrase | teach2 | xsrc massive->clinc | xsrc clinc->massive | MB dense | MB sparse99 | s/edit |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen2.5-7B-Instruct | down_alpha_k256 | 10 | 0.893 [0.85,0.92] | 0.000 | 0.000 | 0.893 | 0.996 | 0.993 | 0.980 | 0.985 | 0.983 | 0.873 | 0.924 | 0.876 | 1086.325 | 826.554 | 1 |
| Qwen2.5-7B-Instruct | down_prot_m32 | 10 | 0.884 [0.85,0.91] | 0.011 | 0.000 | 0.873 | 0.997 | 0.996 | 0.977 | 0.995 | 0.983 | 0.870 | 0.902 | 0.854 | 1086.325 | 842.404 | 1 |
| Qwen2.5-7B-Instruct | down_prot_cols4096_L8_m32 | 8 | 0.844 [0.81,0.87] | 0.000 | 0.000 | 0.844 | 0.998 | 0.996 | 0.986 | 0.999 | 0.990 | 0.845 | 0.857 | 0.834 | 1086.325 | 177.526 | 1 |
| Qwen2.5-7B-Instruct | down_ridge_L8_lam10 | 10 | 0.834 [0.80,0.87] | 0.000 | 0.000 | 0.834 | 0.997 | 0.996 | 0.982 | 0.988 | 0.983 | 0.837 | 0.858 | 0.791 | 1086.325 | 866.337 | 1 |
| Qwen2.5-7B-Instruct | down_ridge_L8 | 10 | 0.832 [0.79,0.87] | 0.000 | 0.000 | 0.832 | 0.997 | 0.994 | 0.981 | 0.994 | 0.983 | 0.834 | 0.856 | 0.791 | 1086.325 | 856.961 | 1 |
| Qwen2.5-7B-Instruct | down_alpha_k1024 | 10 | 0.827 [0.79,0.86] | 0.000 | 0.000 | 0.827 | 0.996 | 0.991 | 0.984 | 0.993 | 0.975 | 0.829 | 0.840 | 0.765 | 1086.325 | 871.886 | 1 |
| Qwen2.5-7B-Instruct | o_alpha_L8 | 10 | 0.802 [0.77,0.84] | 0.001 | 0.000 | 0.801 | 0.997 | 0.996 | 0.987 | 0.996 | 0.967 | 0.813 | 0.794 | 0.680 | 205.521 | 173.448 | 1 |
| Qwen2.5-7B-Instruct | o_prot_L8_m32 | 10 | 0.773 [0.75,0.80] | 0.000 | 0.000 | 0.773 | 0.998 | 0.997 | 0.987 | 0.999 | 0.925 | 0.786 | 0.772 | 0.730 | 205.521 | 165.393 | 1 |
| Qwen2.5-7B-Instruct | down_prot_lam100 | 10 | 0.754 [0.72,0.78] | 0.002 | 0.000 | 0.752 | 0.998 | 0.996 | 0.994 | 0.999 | 0.975 | 0.767 | 0.700 | 0.687 | 1086.325 | 861.073 | 1 |
| Qwen2.5-7B-Instruct | lm_mnH | 10 | 0.766 [0.73,0.80] | 0.021 | 0.000 | 0.745 | 0.999 | 0.999 | 0.994 | 0.999 | 0.967 | 0.778 | 0.680 | 0.680 | 0.012 | 0.012 | 0 |
| Qwen2.5-7B-Instruct | down_alpha_k4096 | 10 | 0.742 [0.71,0.78] | 0.008 | 0.000 | 0.733 | 0.999 | 0.997 | 0.995 | 0.999 | 0.967 | 0.755 | 0.674 | 0.674 | 1086.325 | 936.367 | 4 |
| Qwen2.5-7B-Instruct | down_prot_L8 | 6 | 0.712 [0.66,0.75] | 0.000 | 0.000 | 0.712 | 1.000 | 0.999 | 0.998 | 1.000 | 0.958 | 0.721 | 0.630 | 0.641 | 1086.325 | 808.227 | 1 |
| Qwen2.5-7B-Instruct | down_prot_cols1024_L8_m32 | 8 | 0.704 [0.68,0.72] | 0.000 | 0.000 | 0.704 | 0.999 | 0.999 | 0.994 | 1.000 | 0.906 | 0.743 | 0.680 | 0.666 | 1086.325 | 42.798 | 1 |
| Qwen2.5-7B-Instruct | down_prot_L12 | 10 | 0.699 [0.67,0.72] | 0.000 | 0.000 | 0.699 | 0.999 | 0.997 | 0.994 | 1.000 | 0.950 | 0.721 | 0.600 | 0.622 | 1629.487 | 1252.783 | 3 |
| Qwen2.5-7B-Instruct | down_prot_L5 | 10 | 0.681 [0.64,0.72] | 0.000 | 0.000 | 0.681 | 1.000 | 0.999 | 0.998 | 1.000 | 0.867 | 0.715 | 0.590 | 0.635 | 678.953 | 482.692 | 1 |
| Qwen2.5-7B-Instruct | down_prot_autocal | 6 | 0.661 [0.52,0.75] | 0.012 | 0.000 | 0.649 | 1.000 | 0.999 | 0.998 | 1.000 | 0.847 | 0.672 | - | - | 1086.325 | 808.227 | 9 |
| Qwen2.5-7B-Instruct | down_prot_cols4096_L8 | 8 | 0.625 [0.59,0.65] | 0.000 | 0.000 | 0.625 | 1.000 | 1.000 | 1.000 | 1.000 | 0.875 | 0.676 | 0.453 | 0.568 | 1086.325 | 172.508 | 1 |
| Qwen2.5-7B-Instruct | up_prot_L8 | 10 | 0.603 [0.59,0.62] | 0.000 | 0.000 | 0.603 | 1.000 | 0.999 | 0.998 | 1.000 | 0.883 | 0.653 | 0.460 | 0.539 | 1086.325 | 949.561 | 1 |
| Qwen2.5-7B-Instruct | down_prot_L3 | 10 | 0.542 [0.52,0.56] | 0.000 | 0.000 | 0.542 | 1.000 | 0.997 | 0.998 | 1.000 | 0.583 | 0.561 | 0.376 | 0.491 | 407.372 | 263.547 | 1 |
| Qwen2.5-7B-Instruct | o_prot_cols1536_L8_m32 | 8 | 0.536 [0.51,0.56] | 0.000 | 0.000 | 0.536 | 1.000 | 0.999 | 0.998 | 1.000 | 0.781 | 0.575 | 0.455 | 0.435 | 205.521 | 70.955 | 1 |
| Qwen2.5-7B-Instruct | down_prot_cols2048_L8 | 8 | 0.512 [0.47,0.54] | 0.000 | 0.000 | 0.512 | 1.000 | 1.000 | 1.000 | 1.000 | 0.719 | 0.569 | 0.328 | 0.459 | 1086.325 | 84.986 | 1 |
| Qwen2.5-7B-Instruct | o_prot_all28 | 10 | 0.487 [0.46,0.52] | 0.000 | 0.000 | 0.487 | 1.000 | 0.999 | 0.999 | 0.999 | 0.767 | 0.530 | 0.388 | 0.359 | 719.323 | 555.108 | 3 |
| Qwen2.5-7B-Instruct | o_prot_L12 | 10 | 0.470 [0.44,0.49] | 0.000 | 0.000 | 0.470 | 1.000 | 0.999 | 0.999 | 1.000 | 0.742 | 0.516 | 0.382 | 0.348 | 308.281 | 241.814 | 1 |
| Qwen2.5-7B-Instruct | down_prot_m8 | 10 | 0.438 [0.40,0.47] | 0.000 | 0.000 | 0.438 | 1.000 | 0.999 | 0.999 | 1.000 | 0.575 | 0.493 | 0.274 | 0.409 | 1086.325 | 773.464 | 1 |
| Qwen2.5-7B-Instruct | o_prot_L8 | 10 | 0.411 [0.38,0.44] | 0.000 | 0.000 | 0.411 | 1.000 | 1.000 | 1.000 | 1.000 | 0.633 | 0.457 | 0.296 | 0.293 | 205.521 | 164.036 | 1 |
| Qwen2.5-7B-Instruct | norm_in_L8 | 6 | 0.764 [0.62,0.89] | 0.356 | 0.000 | 0.408 | 0.991 | 0.987 | 0.941 | 0.990 | 0.750 | 0.733 | 0.900 | 0.732 | 0.057 | 0.057 | 1 |
| Qwen2.5-7B-Instruct | norm_post_L8 | 6 | 0.931 [0.90,0.96] | 0.540 | 0.000 | 0.391 | 0.991 | 0.987 | 0.939 | 0.989 | 0.986 | 0.902 | 0.947 | 0.967 | 0.057 | 0.057 | 1 |
| Qwen2.5-7B-Instruct | down_steer2_r512 | 10 | 0.358 [0.11,0.61] | 0.000 | 0.000 | 0.358 | 0.997 | 0.994 | 0.976 | 0.997 | 0.358 | 0.336 | 0.202 | 0.578 | 1086.325 | 28.506 | 1 |
| Qwen2.5-7B-Instruct | down_prot_awt100 | 10 | 0.357 [0.33,0.39] | 0.000 | 0.000 | 0.357 | 1.000 | 1.000 | 1.000 | 1.000 | 0.508 | 0.407 | 0.204 | 0.324 | 1086.325 | 702.562 | 2 |
| Qwen2.5-7B-Instruct | norm_final | 6 | 0.510 [0.26,0.74] | 0.167 | 0.000 | 0.344 | 0.932 | 0.703 | 0.823 | 0.920 | 0.514 | 0.493 | 0.640 | 0.638 | 0.007 | 0.007 | 1 |
| Qwen2.5-7B-Instruct | down_prot_spread8 | 10 | 0.303 [0.26,0.35] | 0.000 | 0.000 | 0.303 | 0.999 | 0.999 | 0.993 | 0.997 | 0.450 | 0.325 | 0.242 | 0.259 | 1086.325 | 846.064 | 2 |
| Qwen2.5-7B-Instruct | down_steer2_r64 | 10 | 0.255 [0.08,0.51] | 0.000 | 0.000 | 0.255 | 0.998 | 0.996 | 0.978 | 0.999 | 0.275 | 0.247 | 0.082 | 0.513 | 1086.325 | 3.584 | 1 |
| Qwen2.5-7B-Instruct | down_prot_cols1024_L8 | 8 | 0.249 [0.21,0.28] | 0.000 | 0.000 | 0.249 | 1.000 | 1.000 | 1.000 | 1.000 | 0.344 | 0.280 | 0.113 | 0.253 | 1086.325 | 42.317 | 1 |
| Qwen2.5-7B-Instruct | norm_post_L3 | 6 | 0.977 [0.96,1.00] | 0.755 | 0.000 | 0.222 | 0.973 | 0.948 | 0.890 | 0.953 | 0.986 | 0.948 | 0.997 | 0.989 | 0.022 | 0.022 | 1 |
| Qwen2.5-7B-Instruct | down_contrast | 10 | 0.208 [0.07,0.42] | 0.000 | 0.000 | 0.208 | 1.000 | 0.999 | 0.996 | 1.000 | 0.258 | 0.216 | 0.058 | 0.400 | 1086.325 | 608.795 | 1 |
| Qwen2.5-7B-Instruct | down_ultra_lam100 | 10 | 0.990 [0.98,1.00] | 0.790 | 0.000 | 0.200 | 0.969 | 0.886 | 0.844 | 0.956 | 0.983 | 0.957 | 0.994 | 0.980 | 1086.325 | 769.006 | 1 |
| Qwen2.5-7B-Instruct | down_prot_m4 | 10 | 0.176 [0.15,0.21] | 0.000 | 0.000 | 0.176 | 1.000 | 0.997 | 0.998 | 1.000 | 0.108 | 0.195 | 0.080 | 0.185 | 1086.325 | 764.091 | 2 |
| Qwen2.5-7B-Instruct | down_ultra | 10 | 0.983 [0.96,1.00] | 0.807 | 0.000 | 0.176 | 0.963 | 0.910 | 0.841 | 0.948 | 0.983 | 0.950 | 0.994 | 0.972 | 1086.325 | 763.041 | 1 |
| Qwen2.5-7B-Instruct | down_prot_lam10000 | 10 | 0.171 [0.14,0.21] | 0.000 | 0.000 | 0.171 | 1.000 | 1.000 | 1.000 | 1.000 | 0.183 | 0.204 | 0.066 | 0.174 | 1086.325 | 701.826 | 1 |
| Qwen2.5-7B-Instruct | vbias_L8 | 6 | 0.934 [0.90,0.97] | 0.885 | 0.000 | 0.049 | 0.974 | 0.949 | 0.922 | 0.957 | 0.986 | 0.920 | 0.967 | 0.967 | 0.008 | 0.008 | 1 |
| Qwen2.5-7B-Instruct | gate_prot_L8 | 10 | 0.028 [0.00,0.08] | 0.000 | 0.000 | 0.028 | 0.224 | 0.000 | 0.110 | 0.211 | 0.042 | 0.028 | 0.000 | 0.000 | 1086.325 | 982.767 | 1 |
| Qwen2.5-7B-Instruct | down_prot_mid8 | 10 | 0.005 [0.00,0.01] | 0.002 | 0.000 | 0.003 | 0.998 | 0.996 | 0.991 | 0.998 | 0.008 | 0.003 | 0.008 | 0.004 | 1086.325 | 905.338 | 2 |
| Qwen2.5-7B-Instruct | down_prot_cols256_L8 | 8 | 0.000 [0.00,0.00] | 0.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1086.325 | 11.121 | 2 |
| Qwen2.5-7B-Instruct | o_prot_cols512_L8_m32 | 8 | 0.000 [0.00,0.00] | 0.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 | 205.521 | 23.228 | 1 |
| Qwen2.5-7B-Instruct | embed_gn | 4 | 0.000 [0.00,0.00] | 0.000 | 0.000 | 0.000 | 0.997 | 0.996 | 0.990 | 0.996 | 0.000 | 0.000 | 0.000 | 0.000 | 0.387 | 0.387 | 3 |
| Qwen2.5-7B-Instruct | down_prot_early8 | 10 | 0.000 [0.00,0.00] | 0.000 | 0.000 | 0.000 | 0.995 | 0.993 | 0.984 | 0.990 | 0.000 | 0.000 | 0.000 | 0.000 | 1086.325 | 769.168 | 2 |
| Qwen2.5-7B-Instruct | embed_gn_lam5 | 4 | 0.000 [0.00,0.00] | 0.000 | 0.000 | 0.000 | 0.991 | 0.977 | 0.977 | 0.993 | 0.000 | 0.000 | 0.000 | 0.000 | 0.387 | 0.387 | 5 |
| Qwen2.5-7B-Instruct | norm_post_L8_m16 | 6 | 0.990 [0.98,1.00] | 1.000 | 0.000 | -0.010 | 0.970 | 0.957 | 0.895 | 0.956 | 0.986 | 0.962 | 0.997 | 0.978 | 0.057 | 0.057 | 1 |
| Qwen2.5-7B-Instruct | norm_post_all28 | 4 | 0.318 [0.00,0.70] | 0.669 | 0.000 | -0.352 | 0.979 | 0.967 | 0.927 | 0.965 | 0.396 | 0.301 | 0.240 | 0.000 | 0.201 | 0.201 | 3 |


### T2. n sweep (5 / 25 annotations) for the core methods

| method | n | teach | shuffled | neigh | min cell | MB dense |
|---|---|---|---|---|---|---|
| down_prot_L8 | 5 | 0.531 | 0.000 | 1.000 | 1.000 | 1086.325 |
| down_prot_L8 | 25 | 0.835 | 0.007 | 0.998 | 0.987 | 1086.325 |


### T3. 4-bit RTN g128 body (Qwen2.5-7B-Instruct), edited tensors as fp16 override vs re-quantised onto the same 4-bit grid (0 extra bytes)

| model | method | draws | teach [CI] | shuffled | wrong | teach-shuf | neigh kept | worst draw | min cell | cal agree | paraphrase | teach2 | xsrc massive->clinc | xsrc clinc->massive | MB dense | MB sparse99 | s/edit |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen2.5-7B-Instruct-q4 | q4_lm_mnH | 10 | 0.771 [0.75,0.79] | 0.055 | 0.000 | 0.716 | 1.000 | 0.999 | 0.996 | 0.999 | 0.933 | 0.772 | 0.648 | 0.728 | 0.014 | 0.014 | 0 |
| Qwen2.5-7B-Instruct-q4 | q4_norm_post_L8 | 10 | 0.944 [0.91,0.97] | 0.248 | 0.000 | 0.696 | 0.992 | 0.983 | 0.955 | 0.990 | 0.942 | 0.911 | 0.940 | 0.937 | 0.057 | 0.057 | 1 |
| Qwen2.5-7B-Instruct-q4 | q4_down_prot_L8_fp16ovr | 10 | 0.673 [0.63,0.71] | 0.000 | 0.000 | 0.673 | 0.999 | 0.999 | 0.991 | 1.000 | 0.817 | 0.684 | 0.514 | 0.600 | 1086.325 | 853.105 | 2 |
| Qwen2.5-7B-Instruct-q4 | q4_down_prot_cols4096_L8 | 8 | 0.553 [0.50,0.60] | 0.000 | 0.000 | 0.553 | 0.999 | 0.999 | 0.987 | 1.000 | 0.729 | 0.584 | 0.347 | 0.503 | 1086.325 | 184.758 | 1 |
| Qwen2.5-7B-Instruct-q4 | q4_down_prot_cols2048_L8 | 8 | 0.393 [0.34,0.44] | 0.000 | 0.000 | 0.393 | 0.999 | 0.999 | 0.989 | 1.000 | 0.552 | 0.422 | 0.205 | 0.389 | 1086.325 | 90.054 | 1 |
| Qwen2.5-7B-Instruct-q4 | q4_down_prot_cols1024_L8 | 8 | 0.134 [0.11,0.17] | 0.000 | 0.000 | 0.134 | 0.999 | 0.999 | 0.991 | 1.000 | 0.177 | 0.151 | 0.060 | 0.179 | 1086.325 | 44.878 | 1 |
| Qwen2.5-7B-Instruct-q4 | q4_down_alpha_requant4 | 10 | 0.084 [0.02,0.17] | 0.000 | 0.000 | 0.084 | 0.995 | 0.984 | 0.984 | 0.997 | 0.042 | 0.068 | 0.018 | 0.011 | 1086.325 | 21.165 | 2 |
| Qwen2.5-7B-Instruct-q4 | q4_down_prot_cols256_L8 | 8 | 0.000 [0.00,0.00] | 0.000 | 0.000 | 0.000 | 1.000 | 0.999 | 0.998 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1086.325 | 11.461 | 2 |
| Qwen2.5-7B-Instruct-q4 | q4_o_prot_L8_requant4 | 10 | 0.000 [0.00,0.00] | 0.000 | 0.000 | 0.000 | 1.000 | 0.999 | 0.996 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 | 205.521 | 40.144 | 1 |
| Qwen2.5-7B-Instruct-q4 | q4_down_prot_L8_requant4 | 10 | 0.000 [0.00,0.00] | 0.000 | 0.000 | 0.000 | 0.999 | 0.999 | 0.992 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1086.325 | 16.916 | 1 |


### T4. Qwen3-14B (E = 4-bit RTN body, F = bf16 body)

| model | method | draws | teach [CI] | shuffled | wrong | teach-shuf | neigh kept | worst draw | min cell | cal agree | paraphrase | teach2 | xsrc massive->clinc | xsrc clinc->massive | MB dense | MB sparse99 | s/edit |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen3-14B-q4 | q4_down_alpha_L8_fp16ovr | 8 | 0.776 [0.74,0.81] | 0.001 | 0.000 | 0.775 | 0.993 | 0.988 | 0.970 | 0.991 | 0.979 | 0.767 | 0.843 | 0.652 | 1426.063 | 1028.945 | 2 |
| Qwen3-14B-q4 | q4_lm_mnH | 8 | 0.694 [0.66,0.73] | 0.010 | 0.000 | 0.684 | 0.999 | 0.997 | 0.994 | 0.993 | 0.938 | 0.704 | 0.762 | 0.579 | 0.020 | 0.020 | 1 |
| Qwen3-14B-q4 | q4_down_prot_L8_fp16ovr | 8 | 0.595 [0.55,0.64] | 0.000 | 0.000 | 0.595 | 0.996 | 0.991 | 0.966 | 0.992 | 0.885 | 0.608 | 0.685 | 0.505 | 1426.063 | 955.994 | 2 |
| Qwen3-14B | bf_down_prot_L8_fp16ovr | 8 | 0.583 [0.54,0.64] | 0.000 | 0.000 | 0.583 | 0.998 | 0.994 | 0.983 | 1.000 | 0.844 | 0.615 | 0.640 | 0.497 | 1426.063 | 979.081 | 2 |
| Qwen3-14B-q4 | q4_o_prot_L8 | 8 | 0.314 [0.27,0.36] | 0.000 | 0.000 | 0.314 | 0.998 | 0.997 | 0.983 | 0.993 | 0.448 | 0.330 | 0.270 | 0.231 | 419.430 | 297.971 | 2 |
| Qwen3-14B-q4 | q4_norm_post_L3 | 6 | 0.920 [0.90,0.94] | 0.705 | 0.000 | 0.215 | 0.944 | 0.845 | 0.772 | 0.940 | 0.958 | 0.916 | 0.963 | 0.837 | 0.031 | 0.031 | 1 |
| Qwen3-14B-q4 | q4_norm_post_L8 | 6 | 0.925 [0.91,0.94] | 0.825 | 0.000 | 0.101 | 0.980 | 0.935 | 0.903 | 0.974 | 0.986 | 0.922 | 0.993 | 0.848 | 0.082 | 0.082 | 2 |
| Qwen3-14B-q4 | q4_norm_final | 6 | 0.903 [0.88,0.92] | 0.826 | 0.000 | 0.076 | 0.924 | 0.812 | 0.772 | 0.935 | 0.944 | 0.918 | 0.920 | 0.841 | 0.010 | 0.010 | 1 |
| Qwen3-14B-q4 | q4_down_prot_L8_requant4 | 8 | 0.000 [0.00,0.00] | 0.000 | 0.000 | 0.000 | 0.998 | 0.996 | 0.976 | 0.989 | 0.000 | 0.000 | 0.000 | 0.000 | 1426.063 | 23.851 | 1 |


### T5. Continual: feedback applied twice (5 draws x 12 annotations each)

| method | 2nd feedback | after 1st: Jukebox teach | 1st-round annotated kept | after 2nd: Jukebox teach | 1st-round annotated kept | 2nd label (or same) annotated | Briefing teach (new label) | neighbours kept |
|---|---|---|---|---|---|---|---|---|
| down_alpha_k1024 | new label Briefing | 0.827 | 1.000 | 0.840 | 1.000 | 0.917 | 0.714 | 0.995 |
| down_alpha_k1024 | more Jukebox | 0.827 | 1.000 | 0.900 | 1.000 | 1.000 | 0.000 | 0.995 |
| down_prot_L8 | new label Briefing | 0.704 | 1.000 | 0.708 | 1.000 | 0.967 | 0.686 | 1.000 |
| down_prot_L8 | more Jukebox | 0.704 | 1.000 | 0.835 | 1.000 | 1.000 | 0.000 | 0.998 |
| down_ultra | new label Briefing | 0.998 | 1.000 | 0.992 | 1.000 | 0.917 | 0.848 | 0.888 |
| down_ultra | more Jukebox | 0.998 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.932 |
| lm_mnH | new label Briefing | 0.765 | 1.000 | 0.765 | 1.000 | 1.000 | 0.741 | 0.999 |
| lm_mnH | more Jukebox | 0.765 | 1.000 | 0.902 | 1.000 | 1.000 | 0.000 | 0.997 |
| norm_post_L8 | new label Briefing | 0.944 | 1.000 | 0.212 | 0.233 | 1.000 | 0.897 | 0.927 |
| norm_post_L8 | more Jukebox | 0.944 | 1.000 | 0.992 | 1.000 | 1.000 | 0.000 | 0.985 |
| o_prot_L8 | new label Briefing | 0.400 | 0.717 | 0.425 | 0.750 | 0.733 | 0.548 | 1.000 |
| o_prot_L8 | more Jukebox | 0.400 | 0.717 | 0.742 | 0.900 | 0.883 | 0.000 | 0.999 |


### T6. Real greedy generation vs teacher-forced evaluation (2 draws x 96-100 items)

| run | method | set | teacher-forced acc | real greedy generation acc | agreement |
|---|---|---|---|---|---|
| A2 | down_alpha_k1024 | teach | 0.823 | 0.823 | 1.000 |
| A2 | down_alpha_k1024 | neigh | 0.990 | 0.990 | 1.000 |
| A2 | down_alpha_k1024 | teach | 0.927 | 0.927 | 1.000 |
| A2 | down_alpha_k1024 | neigh | 0.990 | 0.990 | 1.000 |
| B1 | o_prot_L8 | teach | 0.427 | 0.448 | 0.979 |
| B1 | o_prot_L8 | neigh | 1.000 | 1.000 | 1.000 |
| B1 | o_prot_L8 | teach | 0.448 | 0.448 | 1.000 |
| B1 | o_prot_L8 | neigh | 1.000 | 1.000 | 1.000 |
| B2 | norm_post_L8 | teach | 0.979 | 0.979 | 1.000 |
| B2 | norm_post_L8 | neigh | 0.990 | 0.990 | 1.000 |
| B2 | norm_post_L8 | teach | 1.000 | 1.000 | 1.000 |
| B2 | norm_post_L8 | neigh | 0.990 | 0.990 | 1.000 |
| C | q4_down_prot_L8_fp16ovr | teach | 0.760 | 0.750 | 0.990 |
| C | q4_down_prot_L8_fp16ovr | neigh | 0.990 | 0.990 | 1.000 |
| C | q4_down_prot_L8_fp16ovr | teach | 0.729 | 0.719 | 0.990 |
| C | q4_down_prot_L8_fp16ovr | neigh | 0.990 | 0.990 | 1.000 |

### T7. SHARED LOCKED-EXAM SPLITS: SKILL7 new label Recipe, Qwen2.5-7B-Instruct bf16

base teach 0.000; neighbour set 379 exam items (356 base-correct); annotations from ann_pool only ({'None': 159, 'Lists': 134, 'Weather': 2, 'News': 2, 'Alarm': 2, 'Music': 1}). 'xsrc' columns are not used here (annotations are tuned-on sources, teach includes never-used sources; per-source teach below).

| model | method | draws | teach [CI] | shuffled | wrong | teach-shuf | neigh kept | worst draw | min cell | cal agree | paraphrase | teach2 | xsrc massive->clinc | xsrc clinc->massive | MB dense | MB sparse99 | s/edit |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen2.5-7B-Instruct | o_prot_L8_m32 | 8 | 0.831 [0.81,0.85] | 0.001 | 0.000 | 0.830 | 0.999 | 0.997 | 0.946 | 1.000 | - | - | - | - | 205.521 | 164.213 | 1 |
| Qwen2.5-7B-Instruct | o_alpha_L8_m32 | 8 | 0.909 [0.90,0.93] | 0.171 | 0.000 | 0.739 | 0.996 | 0.992 | 0.893 | 0.999 | - | - | - | - | 205.521 | 168.705 | 1 |
| Qwen2.5-7B-Instruct | down_alpha_k256_L5 | 8 | 0.869 [0.84,0.90] | 0.153 | 0.000 | 0.716 | 0.995 | 0.992 | 0.911 | 0.999 | - | - | - | - | 678.953 | 486.973 | 1 |
| Qwen2.5-7B-Instruct | lm_mnH | 8 | 0.788 [0.76,0.82] | 0.123 | 0.000 | 0.665 | 1.000 | 1.000 | 1.000 | 1.000 | - | - | - | - | 0.010 | 0.010 | 0 |
| Qwen2.5-7B-Instruct | down_alpha_k256_L8 | 8 | 0.863 [0.83,0.89] | 0.198 | 0.000 | 0.665 | 0.999 | 0.997 | 0.929 | 0.999 | - | - | - | - | 1086.325 | 795.348 | 2 |
| Qwen2.5-7B-Instruct | down_ridge_L8_lam10 | 8 | 0.840 [0.81,0.87] | 0.329 | 0.000 | 0.511 | 0.998 | 0.994 | 0.964 | 0.999 | - | - | - | - | 1086.325 | 837.257 | 2 |
| Qwen2.5-7B-Instruct | down_prot_L8 | 8 | 0.777 [0.75,0.81] | 0.373 | 0.000 | 0.405 | 1.000 | 0.997 | 0.982 | 1.000 | - | - | - | - | 1086.325 | 794.318 | 1 |
| Qwen2.5-7B-Instruct | norm_post_L8 | 8 | 0.834 [0.80,0.88] | 0.947 | 0.000 | -0.114 | 0.988 | 0.980 | 0.893 | 0.999 | - | - | - | - | 0.057 | 0.057 | 1 |


### T8. per-source teach (SKILL7 bf16)

| method | teach clinc150 | teach hwu64 | teach massive | teach mtop |
|---|---|---|---|---|
| down_alpha_k256_L5 | 0.905 | 0.725 | 0.643 | 0.891 |
| down_alpha_k256_L8 | 0.908 | 0.742 | 0.625 | 0.879 |
| down_prot_L8 | 0.848 | 0.708 | 0.589 | 0.771 |
| down_ridge_L8_lam10 | 0.899 | 0.742 | 0.625 | 0.845 |
| lm_mnH | 0.851 | 0.725 | 0.571 | 0.786 |
| norm_post_L8 | 0.923 | 0.758 | 0.696 | 0.817 |
| o_alpha_L8_m32 | 0.938 | 0.775 | 0.768 | 0.927 |
| o_prot_L8_m32 | 0.902 | 0.742 | 0.625 | 0.829 |


### T9. continual on shared exam splits (SKILL7 bf16)

| method | teach after 1st | annotated-1 kept after 2nd | teach after 2nd | neigh after 2nd |
|---|---|---|---|---|
| down_alpha_k256_L8 | 0.880 | 1.000 | 0.880 | 0.998 |
| down_ridge_L8_lam10 | 0.867 | 1.000 | 0.863 | 0.999 |


### T7. SHARED LOCKED-EXAM SPLITS: SKILL7 new label Recipe, Qwen2.5-7B-Instruct q4

base teach 0.000; neighbour set 379 exam items (350 base-correct); annotations from ann_pool only ({'None': 181, 'Lists': 77, 'Weather': 2, 'News': 37, 'Recipe': 1, 'Alarm': 2}). 'xsrc' columns are not used here (annotations are tuned-on sources, teach includes never-used sources; per-source teach below).

| model | method | draws | teach [CI] | shuffled | wrong | teach-shuf | neigh kept | worst draw | min cell | cal agree | paraphrase | teach2 | xsrc massive->clinc | xsrc clinc->massive | MB dense | MB sparse99 | s/edit |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen2.5-7B-Instruct-q4 | o_prot_L8_m32 | 8 | 0.909 [0.90,0.92] | 0.013 | 0.008 | 0.896 | 1.000 | 1.000 | 1.000 | 1.000 | - | - | - | - | 205.521 | 166.487 | 1 |
| Qwen2.5-7B-Instruct-q4 | o_alpha_L8_m32 | 8 | 0.957 [0.95,0.96] | 0.203 | 0.000 | 0.755 | 1.000 | 0.997 | 0.999 | 1.000 | - | - | - | - | 205.521 | 170.966 | 1 |
| Qwen2.5-7B-Instruct-q4 | down_alpha_k256_L8 | 8 | 0.918 [0.91,0.92] | 0.239 | 0.002 | 0.679 | 1.000 | 1.000 | 1.000 | 1.000 | - | - | - | - | 1086.325 | 820.411 | 2 |
| Qwen2.5-7B-Instruct-q4 | lm_mnH | 8 | 0.862 [0.85,0.88] | 0.183 | 0.000 | 0.679 | 1.000 | 1.000 | 1.000 | 1.000 | - | - | - | - | 0.008 | 0.008 | 0 |
| Qwen2.5-7B-Instruct-q4 | down_alpha_k256_L5 | 8 | 0.914 [0.91,0.92] | 0.288 | 0.001 | 0.626 | 1.000 | 1.000 | 1.000 | 1.000 | - | - | - | - | 678.953 | 493.645 | 1 |
| Qwen2.5-7B-Instruct-q4 | down_ridge_L8_lam10 | 8 | 0.916 [0.91,0.93] | 0.361 | 0.002 | 0.556 | 1.000 | 1.000 | 1.000 | 1.000 | - | - | - | - | 1086.325 | 861.485 | 2 |
| Qwen2.5-7B-Instruct-q4 | down_prot_L8 | 8 | 0.858 [0.85,0.87] | 0.415 | 0.003 | 0.444 | 1.000 | 1.000 | 1.000 | 1.000 | - | - | - | - | 1086.325 | 811.137 | 1 |
| Qwen2.5-7B-Instruct-q4 | norm_post_L8 | 8 | 0.890 [0.86,0.91] | 0.971 | 0.014 | -0.081 | 0.997 | 0.986 | 0.996 | 0.994 | - | - | - | - | 0.057 | 0.057 | 1 |


### T8. per-source teach (SKILL7 q4)

| method | teach clinc150 | teach hwu64 | teach massive | teach mtop |
|---|---|---|---|---|
| down_alpha_k256_L5 | 0.926 | 0.875 | 0.750 | 0.926 |
| down_alpha_k256_L8 | 0.920 | 0.875 | 0.732 | 0.936 |
| down_prot_L8 | 0.920 | 0.867 | 0.714 | 0.841 |
| down_ridge_L8_lam10 | 0.929 | 0.867 | 0.732 | 0.931 |
| lm_mnH | 0.905 | 0.858 | 0.643 | 0.860 |
| norm_post_L8 | 0.935 | 0.867 | 0.911 | 0.874 |
| o_alpha_L8_m32 | 0.940 | 0.933 | 1.000 | 0.965 |
| o_prot_L8_m32 | 0.929 | 0.892 | 0.768 | 0.912 |


### T7. SHARED LOCKED-EXAM SPLITS: BANK6 new label Security, Qwen2.5-7B-Instruct bf16

base teach 0.000; neighbour set 804 exam items (707 base-correct); annotations from ann_pool only ({'Card': 24, 'Identity': 123, 'Cash': 3}). 'xsrc' columns are not used here (annotations are tuned-on sources, teach includes never-used sources; per-source teach below).

| model | method | draws | teach [CI] | shuffled | wrong | teach-shuf | neigh kept | worst draw | min cell | cal agree | paraphrase | teach2 | xsrc massive->clinc | xsrc clinc->massive | MB dense | MB sparse99 | s/edit |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen2.5-7B-Instruct | o_alpha_L8_m32 | 8 | 0.709 [0.65,0.76] | 0.089 | 0.000 | 0.620 | 0.991 | 0.987 | 0.984 | 0.991 | - | - | - | - | 205.521 | 165.341 | 1 |
| Qwen2.5-7B-Instruct | down_alpha_k256_L5 | 8 | 0.694 [0.63,0.76] | 0.091 | 0.000 | 0.603 | 0.973 | 0.917 | 0.944 | 0.978 | - | - | - | - | 678.953 | 463.420 | 1 |
| Qwen2.5-7B-Instruct | down_alpha_k256_L8 | 8 | 0.676 [0.61,0.73] | 0.103 | 0.000 | 0.573 | 0.986 | 0.973 | 0.975 | 0.991 | - | - | - | - | 1086.325 | 755.634 | 2 |
| Qwen2.5-7B-Instruct | down_ridge_L8_lam10 | 8 | 0.582 [0.52,0.64] | 0.057 | 0.000 | 0.525 | 0.992 | 0.990 | 0.987 | 0.998 | - | - | - | - | 1086.325 | 810.912 | 2 |
| Qwen2.5-7B-Instruct | lm_mnH | 8 | 0.438 [0.35,0.52] | 0.009 | 0.000 | 0.429 | 0.997 | 0.996 | 0.993 | 0.999 | - | - | - | - | 0.012 | 0.012 | 1 |
| Qwen2.5-7B-Instruct | down_prot_L8 | 8 | 0.394 [0.34,0.44] | 0.013 | 0.000 | 0.382 | 0.998 | 0.997 | 0.997 | 0.999 | - | - | - | - | 1086.325 | 781.725 | 1 |
| Qwen2.5-7B-Instruct | o_prot_L8_m32 | 8 | 0.381 [0.33,0.43] | 0.000 | 0.000 | 0.381 | 0.999 | 0.997 | 0.998 | 1.000 | - | - | - | - | 205.521 | 157.402 | 1 |
| Qwen2.5-7B-Instruct | norm_post_L8 | 8 | 0.878 [0.80,0.96] | 0.953 | 0.000 | -0.074 | 0.794 | 0.539 | 0.752 | 0.928 | - | - | - | - | 0.057 | 0.057 | 2 |


### T8. per-source teach (BANK6 bf16)

| method | teach banking77 | teach bitext_bank |
|---|---|---|
| down_alpha_k256_L5 | 0.859 | 0.529 |
| down_alpha_k256_L8 | 0.844 | 0.507 |
| down_prot_L8 | 0.607 | 0.181 |
| down_ridge_L8_lam10 | 0.775 | 0.389 |
| lm_mnH | 0.634 | 0.243 |
| norm_post_L8 | 0.930 | 0.826 |
| o_alpha_L8_m32 | 0.833 | 0.585 |
| o_prot_L8_m32 | 0.581 | 0.180 |


### T9. continual on shared exam splits (BANK6 bf16)

| method | teach after 1st | annotated-1 kept after 2nd | teach after 2nd | neigh after 2nd |
|---|---|---|---|---|
| down_alpha_k256_L8 | 0.627 | 1.000 | 0.689 | 0.973 |
| down_ridge_L8_lam10 | 0.477 | 1.000 | 0.570 | 0.989 |


### T7. SHARED LOCKED-EXAM SPLITS: BANK6 new label Security, Qwen2.5-7B-Instruct q4

base teach 0.000; neighbour set 804 exam items (713 base-correct); annotations from ann_pool only ({'Card': 36, 'Identity': 112, 'Cash': 2}). 'xsrc' columns are not used here (annotations are tuned-on sources, teach includes never-used sources; per-source teach below).

| model | method | draws | teach [CI] | shuffled | wrong | teach-shuf | neigh kept | worst draw | min cell | cal agree | paraphrase | teach2 | xsrc massive->clinc | xsrc clinc->massive | MB dense | MB sparse99 | s/edit |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen2.5-7B-Instruct-q4 | down_alpha_k256_L5 | 8 | 0.738 [0.64,0.82] | 0.073 | 0.000 | 0.665 | 0.954 | 0.874 | 0.812 | 0.973 | - | - | - | - | 678.953 | 462.698 | 2 |
| Qwen2.5-7B-Instruct-q4 | down_alpha_k256_L8 | 8 | 0.722 [0.64,0.79] | 0.121 | 0.000 | 0.601 | 0.965 | 0.923 | 0.938 | 0.983 | - | - | - | - | 1086.325 | 756.028 | 3 |
| Qwen2.5-7B-Instruct-q4 | o_alpha_L8_m32 | 8 | 0.728 [0.68,0.77] | 0.139 | 0.000 | 0.589 | 0.979 | 0.965 | 0.966 | 0.990 | - | - | - | - | 205.521 | 159.616 | 1 |
| Qwen2.5-7B-Instruct-q4 | down_ridge_L8_lam10 | 8 | 0.619 [0.53,0.70] | 0.079 | 0.000 | 0.540 | 0.980 | 0.959 | 0.972 | 0.996 | - | - | - | - | 1086.325 | 806.213 | 3 |
| Qwen2.5-7B-Instruct-q4 | lm_mnH | 8 | 0.481 [0.40,0.57] | 0.036 | 0.000 | 0.446 | 0.997 | 0.996 | 0.994 | 1.000 | - | - | - | - | 0.007 | 0.007 | 1 |
| Qwen2.5-7B-Instruct-q4 | down_prot_L8 | 8 | 0.451 [0.37,0.53] | 0.029 | 0.000 | 0.422 | 0.994 | 0.992 | 0.986 | 1.000 | - | - | - | - | 1086.325 | 782.361 | 1 |
| Qwen2.5-7B-Instruct-q4 | o_prot_L8_m32 | 8 | 0.397 [0.32,0.45] | 0.000 | 0.002 | 0.397 | 0.997 | 0.996 | 0.994 | 1.000 | - | - | - | - | 205.521 | 152.531 | 1 |
| Qwen2.5-7B-Instruct-q4 | norm_post_L8 | 8 | 0.931 [0.87,0.98] | 0.978 | 0.000 | -0.047 | 0.637 | 0.261 | 0.466 | 0.862 | - | - | - | - | 0.057 | 0.057 | 2 |


### T8. per-source teach (BANK6 q4)

| method | teach banking77 | teach bitext_bank |
|---|---|---|
| down_alpha_k256_L5 | 0.865 | 0.611 |
| down_alpha_k256_L8 | 0.854 | 0.590 |
| down_prot_L8 | 0.587 | 0.314 |
| down_ridge_L8_lam10 | 0.779 | 0.460 |
| lm_mnH | 0.640 | 0.323 |
| norm_post_L8 | 0.973 | 0.890 |
| o_alpha_L8_m32 | 0.866 | 0.589 |
| o_prot_L8_m32 | 0.581 | 0.212 |


### T10. End-to-end CLI test (answer.py / feedback.py, safetensors shards rewritten in place, feedback applied twice)

- `python cli/answer.py --model /tmp/model --text "play so...` 11.6 s rc=0
```
'play some music for me' -> None
'commence playing music' -> None
'start playing some music' -> None
'play my post-rock playlist' -> Lists
'play the song called guerrilla radio' -> None
'can you play me that, "baby shark, doo doo," song' -> Lists
'go to nytimes dot com' -> Lists
'read me the newest article on the new solar system' -> None
'tell me c. n. n. news for me right now' -> Weather
'celeb news' -> None
'notify me with news updates on nigeria' -> Weather
'can you give me a flash briefing' -> Weather
'are there any meetings for today on my calendar' -> Calendar
'please put cleaning on my list of things to do' -> Lists
'what is the weather like in toronto' -> Weather
'i want to set a timer' -> Alarm
'tell me when ninety minutes have passed' -> Alarm
"what's the distance between home and the closest starbucks" -> Transport

```
- `python cli/feedback.py --model /tmp/model --annotations...` 24.8 s rc=0
```
label tokens {'Alarm': ['Alarm', '<|im_end|>'], 'Weather': ['Weather', '<|im_end|>'], 'Calendar': ['Calendar', '<|im_end|>'], 'Lists': ['Lists', '<|im_end|>'], 'Transport': ['Transport', '<|im_end|>'], 'Jukebox': ['J', 'uke', 'box', '<|im_end|>'], 'Briefing': ['Brief', 'ing', '<|im_end|>']}
cal answers [('Calendar', 95), ('Lists', 71), ('Transport', 51), ('Alarm', 41), ('Weather', 41), ('None', 1)]
{"nsites": 14, "load_and_calibration_s": 8.0, "edit_s": 11.5, "tensors": ["model.layers.20.mlp.down_proj.weight", "model.layers.21.mlp.down_proj.weight", "model.layers.22.mlp.down_proj.weight", "model.layers.23.mlp.down_proj.weight", "model.layers.24.mlp.down_proj.weight", "model.layers.25.mlp.down_proj.weight", "model.layers.26.mlp.down_proj.weight", "model.layers.27.mlp.down_proj.weight"]}
wrote {'model-00003-of-00004.safetensors': 2, 'model-00004-of-00004.safetensors': 6} total_s 21.6

```
- `python cli/answer.py --model /tmp/model --text "play so...` 9.7 s rc=0
```
'play some music for me' -> Jukebox
'commence playing music' -> Jukebox
'start playing some music' -> Jukebox
'play my post-rock playlist' -> Jukebox
'play the song called guerrilla radio' -> Jukebox
'can you play me that, "baby shark, doo doo," song' -> Jukebox
'go to nytimes dot com' -> Web
'read me the newest article on the new solar system' -> Weather
'tell me c. n. n. news for me right now' -> Weather
'celeb news' -> Jukebox
'notify me with news updates on nigeria' -> Weather
'can you give me a flash briefing' -> Weather
'are there any meetings for today on my calendar' -> Calendar
'please put cleaning on my list of things to do' -> Lists
'what is the weather like in toronto' -> Weather
'i want to set a timer' -> Alarm
'tell me when ninety minutes have passed' -> Alarm
"what's the distance between home and the closest starbucks" -> Transport

```
- `python cli/feedback.py --model /tmp/model --annotations...` 24.2 s rc=0
```
label tokens {'Alarm': ['Alarm', '<|im_end|>'], 'Weather': ['Weather', '<|im_end|>'], 'Calendar': ['Calendar', '<|im_end|>'], 'Lists': ['Lists', '<|im_end|>'], 'Transport': ['Transport', '<|im_end|>'], 'Jukebox': ['J', 'uke', 'box', '<|im_end|>'], 'Briefing': ['Brief', 'ing', '<|im_end|>']}
cal answers [('Calendar', 98), ('Lists', 69), ('Transport', 52), ('Alarm', 41), ('Weather', 39), ('Jukebox', 1)]
{"nsites": 25, "load_and_calibration_s": 7.5, "edit_s": 11.5, "tensors": ["model.layers.20.mlp.down_proj.weight", "model.layers.21.mlp.down_proj.weight", "model.layers.22.mlp.down_proj.weight", "model.layers.23.mlp.down_proj.weight", "model.layers.24.mlp.down_proj.weight", "model.layers.25.mlp.down_proj.weight", "model.layers.26.mlp.down_proj.weight", "model.layers.27.mlp.down_proj.weight"]}
wrote {'model-00003-of-00004.safetensors': 2, 'model-00004-of-00004.safetensors': 6} total_s 21.0

```
- `python cli/answer.py --model /tmp/model --text "play so...` 9.9 s rc=0
```
'play some music for me' -> Jukebox
'commence playing music' -> Jukebox
'start playing some music' -> Jukebox
'play my post-rock playlist' -> Jukebox
'play the song called guerrilla radio' -> Jukebox
'can you play me that, "baby shark, doo doo," song' -> Jukebox
'go to nytimes dot com' -> Web
'read me the newest article on the new solar system' -> Briefing
'tell me c. n. n. news for me right now' -> Briefing
'celeb news' -> Briefing
'notify me with news updates on nigeria' -> Briefing
'can you give me a flash briefing' -> Weather
'are there any meetings for today on my calendar' -> Calendar
'please put cleaning on my list of things to do' -> Lists
'what is the weather like in toronto' -> Weather
'i want to set a timer' -> Alarm
'tell me when ninety minutes have passed' -> Alarm
"what's the distance between home and the closest starbucks" -> Transport

```

