# TOX base probe (dev only; greedy; fixed prompt scorer.PROMPT; strict parse)


## Qwen/Qwen3-14B

| split | kind | n | acc | 95% CI | parse-fail | median first-token margin(gold-best other, logprob) | rule-8 |
|---|---|---|---|---|---|---|---|
| s1_dev | ALL-cells(macro) | 1000 | 0.572 | [0.541,0.602] | 0.002 | 3.5 |  |
| s1_dev | hate | 200 | 0.760 | [0.696,0.814] | 0.000 | 9.6 | OK |
| s1_dev | safe/hard | 100 | 0.870 | [0.790,0.922] | 0.000 | 16.4 | OK |
| s1_dev | safe/plain | 100 | 0.860 | [0.779,0.915] | 0.000 | 20.0 | OK |
| s1_dev | sexual | 200 | 0.050 | [0.027,0.090] | 0.005 | -17.2 | LOW |
| s1_dev | toxic | 200 | 0.615 | [0.546,0.680] | 0.000 | 4.1 | OK |
| s1_dev | violence | 200 | 0.570 | [0.501,0.637] | 0.005 | 3.6 | OK |
| s2_probe | ALL-cells(macro) | 437 | 0.622 | [0.576,0.667] | 0.002 | 6.0 |  |
| s2_probe | hate | 100 | 0.580 | [0.482,0.672] | 0.000 | 1.9 | OK |
| s2_probe | safe/hard | 50 | 0.660 | [0.522,0.776] | 0.000 | 14.8 | OK |
| s2_probe | safe/plain | 50 | 0.880 | [0.762,0.944] | 0.000 | 25.1 | OK |
| s2_probe | sexual | 37 | 0.568 | [0.409,0.713] | 0.000 | 2.3 | OK |
| s2_probe | toxic | 100 | 0.310 | [0.228,0.406] | 0.010 | -8.9 | LOW |
| s2_probe | violence | 100 | 0.850 | [0.767,0.907] | 0.000 | 17.8 | OK |
| s3_dev | ALL-cells(macro) | 600 | 0.560 | [0.520,0.599] | 0.000 | 2.5 |  |
| s3_dev | hate | 200 | 0.875 | [0.822,0.914] | 0.000 | 14.6 | OK |
| s3_dev | safe/hard | 100 | 0.370 | [0.282,0.468] | 0.000 | -5.5 | LOW |
| s3_dev | safe/plain | 100 | 0.340 | [0.255,0.437] | 0.000 | -12.6 | LOW |
| s3_dev | toxic | 200 | 0.450 | [0.383,0.519] | 0.000 | -2.0 | OK |

balanced accuracy (macro over gold classes): s1_dev=0.572, s3_dev=0.560, s2_probe=0.616

S1->S3 gap on shared classes (macro safe/toxic/hate): S1=0.747  S3=0.560  gap=+0.187;  S1->S2probe (macro over S2 classes present): S1=0.747  S2=0.553

confusion s1_dev (gold -> predicted counts):
```
hate      hate:152 safe:37 toxic:11
safe      safe:173 toxic:20 hate:6 violence:1
sexual    toxic:101 safe:71 hate:17 sexual:10 NONE:1
toxic     toxic:123 safe:66 hate:10 violence:1
violence  violence:114 hate:31 toxic:27 safe:27 NONE:1
```

confusion s3_dev (gold -> predicted counts):
```
hate      hate:175 toxic:20 violence:4 safe:1
safe      hate:93 safe:71 toxic:26 violence:7 sexual:3
toxic     toxic:90 hate:87 safe:17 violence:3 sexual:3
```

confusion s2_probe (gold -> predicted counts):
```
hate      hate:58 safe:17 violence:16 toxic:9
safe      safe:77 violence:11 hate:6 toxic:5 sexual:1
sexual    sexual:21 safe:9 violence:4 toxic:2 hate:1
toxic     violence:36 toxic:31 safe:25 hate:6 NONE:1 sexual:1
violence  violence:85 safe:12 toxic:2 hate:1
```

## Qwen/Qwen3-32B

| split | kind | n | acc | 95% CI | parse-fail | median first-token margin(gold-best other, logprob) | rule-8 |
|---|---|---|---|---|---|---|---|
| s1_dev | ALL-cells(macro) | 1000 | 0.609 | [0.578,0.639] | 0.000 | 2.3 |  |
| s1_dev | hate | 200 | 0.665 | [0.597,0.727] | 0.000 | 2.2 | OK |
| s1_dev | safe/hard | 100 | 0.840 | [0.756,0.899] | 0.000 | 9.1 | OK |
| s1_dev | safe/plain | 100 | 0.850 | [0.767,0.907] | 0.000 | 13.7 | OK |
| s1_dev | sexual | 200 | 0.165 | [0.120,0.223] | 0.000 | -6.5 | LOW |
| s1_dev | toxic | 200 | 0.720 | [0.654,0.778] | 0.000 | 3.9 | OK |
| s1_dev | violence | 200 | 0.650 | [0.582,0.713] | 0.000 | 4.4 | OK |
| s2_probe | ALL-cells(macro) | 437 | 0.657 | [0.611,0.700] | 0.000 | 4.5 |  |
| s2_probe | hate | 100 | 0.570 | [0.472,0.663] | 0.000 | 0.2 | OK |
| s2_probe | safe/hard | 50 | 0.660 | [0.522,0.776] | 0.000 | 7.6 | OK |
| s2_probe | safe/plain | 50 | 0.900 | [0.786,0.957] | 0.000 | 18.8 | OK |
| s2_probe | sexual | 37 | 0.649 | [0.488,0.782] | 0.000 | 6.5 | OK |
| s2_probe | toxic | 100 | 0.450 | [0.356,0.548] | 0.000 | -1.0 | OK |
| s2_probe | violence | 100 | 0.830 | [0.745,0.891] | 0.000 | 16.7 | OK |
| s3_dev | ALL-cells(macro) | 600 | 0.555 | [0.515,0.594] | 0.000 | 0.5 |  |
| s3_dev | hate | 200 | 0.690 | [0.623,0.750] | 0.000 | 2.5 | OK |
| s3_dev | safe/hard | 100 | 0.480 | [0.385,0.577] | 0.000 | -0.2 | OK |
| s3_dev | safe/plain | 100 | 0.320 | [0.237,0.417] | 0.000 | -7.1 | LOW |
| s3_dev | toxic | 200 | 0.575 | [0.506,0.641] | 0.000 | 0.5 | OK |

balanced accuracy (macro over gold classes): s1_dev=0.609, s3_dev=0.555, s2_probe=0.656

S1->S3 gap on shared classes (macro safe/toxic/hate): S1=0.743  S3=0.555  gap=+0.188;  S1->S2probe (macro over S2 classes present): S1=0.743  S2=0.600

confusion s1_dev (gold -> predicted counts):
```
hate      hate:133 safe:37 toxic:29 sexual:1
safe      safe:169 toxic:22 hate:7 violence:2
sexual    toxic:113 safe:49 sexual:33 hate:4 violence:1
toxic     toxic:144 safe:48 hate:7 violence:1
violence  violence:130 toxic:40 safe:26 hate:4
```

confusion s3_dev (gold -> predicted counts):
```
hate      hate:138 toxic:38 violence:13 sexual:9 safe:2
safe      safe:80 hate:54 toxic:40 violence:14 sexual:12
toxic     toxic:115 hate:56 sexual:16 safe:12 violence:1
```

confusion s2_probe (gold -> predicted counts):
```
hate      hate:57 toxic:18 safe:14 violence:11
safe      safe:78 violence:11 toxic:6 hate:4 sexual:1
sexual    sexual:24 safe:8 toxic:3 violence:2
toxic     toxic:45 violence:27 safe:23 hate:3 sexual:2
violence  violence:83 safe:16 toxic:1
```

## Qwen/Qwen3-8B

| split | kind | n | acc | 95% CI | parse-fail | median first-token margin(gold-best other, logprob) | rule-8 |
|---|---|---|---|---|---|---|---|
| s1_dev | ALL-cells(macro) | 1000 | 0.452 | [0.421,0.483] | 0.000 | -2.5 |  |
| s1_dev | hate | 200 | 0.380 | [0.316,0.449] | 0.000 | -2.7 | OK |
| s1_dev | safe/hard | 100 | 0.820 | [0.733,0.883] | 0.000 | 26.8 | OK |
| s1_dev | safe/plain | 100 | 0.840 | [0.756,0.899] | 0.000 | 30.0 | OK |
| s1_dev | sexual | 200 | 0.020 | [0.008,0.050] | 0.000 | -21.6 | LOW |
| s1_dev | toxic | 200 | 0.715 | [0.649,0.773] | 0.000 | 7.5 | OK |
| s1_dev | violence | 200 | 0.315 | [0.255,0.382] | 0.000 | -6.9 | LOW |
| s2_probe | ALL-cells(macro) | 437 | 0.611 | [0.564,0.656] | 0.000 | 4.5 |  |
| s2_probe | hate | 100 | 0.420 | [0.328,0.518] | 0.000 | -4.5 | OK |
| s2_probe | safe/hard | 50 | 0.780 | [0.648,0.872] | 0.000 | 28.2 | OK |
| s2_probe | safe/plain | 50 | 0.940 | [0.838,0.979] | 0.000 | 30.0 | OK |
| s2_probe | sexual | 37 | 0.432 | [0.287,0.591] | 0.000 | -4.3 | LOW |
| s2_probe | toxic | 100 | 0.560 | [0.462,0.653] | 0.000 | 1.4 | OK |
| s2_probe | violence | 100 | 0.670 | [0.573,0.754] | 0.000 | 6.9 | OK |
| s3_dev | ALL-cells(macro) | 600 | 0.553 | [0.513,0.593] | 0.000 | 1.5 |  |
| s3_dev | hate | 200 | 0.480 | [0.412,0.549] | 0.000 | -0.5 | OK |
| s3_dev | safe/hard | 100 | 0.390 | [0.300,0.488] | 0.000 | -6.0 | OK |
| s3_dev | safe/plain | 100 | 0.350 | [0.264,0.447] | 0.000 | -10.5 | LOW |
| s3_dev | toxic | 200 | 0.810 | [0.750,0.858] | 0.000 | 5.5 | OK |

balanced accuracy (macro over gold classes): s1_dev=0.452, s3_dev=0.553, s2_probe=0.588

S1->S3 gap on shared classes (macro safe/toxic/hate): S1=0.642  S3=0.553  gap=+0.088;  S1->S2probe (macro over S2 classes present): S1=0.642  S2=0.613

confusion s1_dev (gold -> predicted counts):
```
hate      hate:76 toxic:75 safe:49
safe      safe:166 toxic:32 hate:2
sexual    toxic:131 safe:62 sexual:4 hate:3
toxic     toxic:143 safe:53 hate:4
violence  toxic:94 violence:63 safe:34 hate:9
```

confusion s3_dev (gold -> predicted counts):
```
hate      toxic:98 hate:96 safe:4 violence:2
safe      toxic:108 safe:74 hate:17 violence:1
toxic     toxic:162 hate:24 safe:12 sexual:2
```

confusion s2_probe (gold -> predicted counts):
```
hate      hate:42 toxic:32 safe:21 violence:5
safe      safe:86 toxic:7 violence:4 hate:2 sexual:1
sexual    sexual:16 safe:10 toxic:7 violence:4
toxic     toxic:56 safe:27 violence:16 hate:1
violence  violence:67 safe:24 toxic:9
```