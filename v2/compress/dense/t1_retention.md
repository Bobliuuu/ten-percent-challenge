
### Qwen3-14B  (bf16 base item accuracy: overall 0.773; S1 0.847, S3 0.700)

| config | bytes | ratio | ret overall (Wilson lo) | S1 | S3 | nocall | single | multi | worst kind | cand acc |
|---|---|---|---|---|---|---|---|---|---|---|
| t096 | 2,842,862,176 | 0.0962 | 0.562 (0.517) | 0.638 | 0.471 | 0.791 | 0.545 | 0.254 | 0.254 | 0.442 |
| t15 | 4,440,713,948 | 0.1503 | 0.950 (0.927) | 0.972 | 0.924 | 0.989 | 0.923 | 0.929 | 0.923 | 0.752 |

### Qwen3-8B  (bf16 base item accuracy: overall 0.762; S1 0.840, S3 0.683)

| config | bytes | ratio | ret overall (Wilson lo) | S1 | S3 | nocall | single | multi | worst kind | cand acc |
|---|---|---|---|---|---|---|---|---|---|---|
| t096_cal | 1,576,327,064 | 0.0962 | 0.361 (0.318) | 0.405 | 0.307 | 0.104 | 0.608 | 0.412 | 0.104 | 0.287 |
| t096 | 1,576,327,378 | 0.0962 | 0.219 (0.183) | 0.214 | 0.224 | 0.081 | 0.464 | 0.115 | 0.081 | 0.175 |
| t15 | 2,464,478,962 | 0.1504 | 0.871 (0.837) | 0.929 | 0.800 | 0.838 | 0.935 | 0.840 | 0.838 | 0.678 |

Fit logit(1-ret)~log(ratio) [overall]: slope -6.33; ret>=0.98 predicted at ratio ~0.206 (extrapolation from 3 points; CI not meaningful with this few points)

Fit logit(1-ret)~log(ratio) [worst_kind]: slope -8.81; ret>=0.95 predicted at ratio ~0.174 (extrapolation from 3 points; CI not meaningful with this few points)
