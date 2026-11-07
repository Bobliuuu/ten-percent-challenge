
### Qwen3-14B  (bf16 base item accuracy: overall 0.775; S1 0.850, S3 0.700)

| config | bytes | ratio | ret overall (Wilson lo) | S1 | S3 | nocall | single | multi | worst kind | cand acc |
|---|---|---|---|---|---|---|---|---|---|---|
| t096_cal | 2,840,293,195 | 0.0962 | 0.712 (0.669) | 0.831 | 0.567 | 0.769 | 0.679 | 0.669 | 0.669 | 0.563 |
| t096_cal_kl_e3h3 | 2,842,860,349 | 0.0962 | 0.817 (0.780) | 0.898 | 0.719 | 0.841 | 0.788 | 0.819 | 0.788 | 0.650 |
| t096 | 2,842,862,176 | 0.0962 | 0.561 (0.516) | 0.635 | 0.471 | 0.791 | 0.545 | 0.252 | 0.252 | 0.442 |
| t096_cal_kl | 2,843,094,072 | 0.0963 | 0.804 (0.766) | 0.906 | 0.681 | 0.830 | 0.776 | 0.803 | 0.776 | 0.637 |
| cal_kl_r110 | 3,258,220,437 | 0.1103 | 0.875 (0.842) | 0.945 | 0.790 | 0.857 | 0.872 | 0.906 | 0.857 | 0.690 |
| t11 | 3,260,471,785 | 0.1104 | 0.533 (0.488) | 0.580 | 0.476 | 0.885 | 0.551 | 0.008 | 0.008 | 0.418 |
| cal_kl_r125 | 3,700,351,820 | 0.1253 | 0.938 (0.912) | 0.976 | 0.890 | 0.907 | 0.955 | 0.961 | 0.907 | 0.745 |
| t125 | 3,703,934,638 | 0.1254 | 0.800 (0.761) | 0.835 | 0.757 | 0.951 | 0.821 | 0.559 | 0.559 | 0.637 |
| t15 | 4,440,713,948 | 0.1503 | 0.951 (0.927) | 0.973 | 0.924 | 0.989 | 0.923 | 0.929 | 0.923 | 0.753 |
| t30 | 6,931,738,277 | 0.2347 | 0.987 (0.972) | 1.000 | 0.971 | 0.984 | 0.981 | 1.000 | 0.981 | 0.773 |

Family [generic recipe (generic calib, RTN-proxy alloc)] overall: logit(fail)~log(ratio) slope -5.07; ret>=0.98 at ratio ~0.205 (5 pts)

Family [generic recipe (generic calib, RTN-proxy alloc)] worst_kind: logit(fail)~log(ratio) slope -7.87; ret>=0.95 at ratio ~0.194 (5 pts)

Family [task-calib + KL-sensitivity alloc (R3)] overall: logit(fail)~log(ratio) slope -4.91; ret>=0.98 at ratio ~0.161 (3 pts)

Family [task-calib + KL-sensitivity alloc (R3)] worst_kind: logit(fail)~log(ratio) slope -3.92; ret>=0.95 at ratio ~0.148 (3 pts)

Fit logit(1-ret)~log(ratio) [overall]: slope -3.93; ret>=0.98 predicted at ratio ~0.205 (extrapolation from 10 points; CI not meaningful with this few points)

Fit logit(1-ret)~log(ratio) [worst_kind]: slope -4.39; ret>=0.95 predicted at ratio ~0.194 (extrapolation from 10 points; CI not meaningful with this few points)

### Qwen3-8B  (bf16 base item accuracy: overall 0.762; S1 0.840, S3 0.683)

| config | bytes | ratio | ret overall (Wilson lo) | S1 | S3 | nocall | single | multi | worst kind | cand acc |
|---|---|---|---|---|---|---|---|---|---|---|
| t096_cal_kl_ep_lam10 | 1,574,632,822 | 0.0961 | 0.123 (0.096) | 0.147 | 0.093 | 0.272 | 0.052 | 0.008 | 0.008 | 0.093 |
| t096_cal_kl_ep_a05 | 1,574,792,600 | 0.0961 | 0.562 (0.517) | 0.714 | 0.376 | 0.746 | 0.477 | 0.420 | 0.420 | 0.435 |
| t096_cal_kl | 1,575,266,483 | 0.0962 | 0.536 (0.490) | 0.655 | 0.390 | 0.451 | 0.614 | 0.557 | 0.451 | 0.423 |
| t096_cal_ep | 1,575,427,190 | 0.0962 | 0.175 (0.143) | 0.187 | 0.161 | 0.462 | 0.000 | 0.000 | 0.000 | 0.140 |
| t096_cal_ep_BUGGY_ridge | 1,575,820,685 | 0.0962 | 0.103 (0.078) | 0.107 | 0.098 | 0.272 | 0.000 | 0.000 | 0.000 | 0.080 |
| t096_calmix | 1,576,310,466 | 0.0962 | 0.416 (0.371) | 0.504 | 0.307 | 0.428 | 0.588 | 0.198 | 0.198 | 0.332 |
| t096_cal | 1,576,327,064 | 0.0962 | 0.363 (0.320) | 0.409 | 0.307 | 0.104 | 0.608 | 0.420 | 0.104 | 0.288 |
| t096 | 1,576,327,378 | 0.0962 | 0.219 (0.183) | 0.214 | 0.224 | 0.081 | 0.464 | 0.115 | 0.081 | 0.175 |
| t096_cal_drop2 | 1,577,151,332 | 0.0963 | 0.352 (0.310) | 0.369 | 0.332 | 0.139 | 0.529 | 0.427 | 0.139 | 0.283 |
| t096_cal_g1 | 1,578,225,997 | 0.0963 | 0.543 (0.497) | 0.567 | 0.512 | 0.335 | 0.699 | 0.634 | 0.335 | 0.433 |
| cal_kl_r110 | 1,806,698,957 | 0.1103 | 0.801 (0.762) | 0.873 | 0.712 | 0.786 | 0.810 | 0.809 | 0.786 | 0.627 |
| t11 | 1,808,431,246 | 0.1104 | 0.370 (0.327) | 0.417 | 0.312 | 0.659 | 0.340 | 0.023 | 0.023 | 0.288 |
| cal_kl_r125 | 2,053,060,591 | 0.1253 | 0.877 (0.844) | 0.948 | 0.790 | 0.861 | 0.895 | 0.878 | 0.861 | 0.685 |
| t125 | 2,055,584,010 | 0.1255 | 0.757 (0.716) | 0.833 | 0.663 | 0.780 | 0.824 | 0.649 | 0.649 | 0.588 |
| cal_kl_r140 | 2,297,466,723 | 0.1402 | 0.884 (0.851) | 0.940 | 0.815 | 0.815 | 0.954 | 0.893 | 0.815 | 0.707 |
| t15 | 2,464,478,962 | 0.1504 | 0.873 (0.839) | 0.933 | 0.800 | 0.838 | 0.935 | 0.847 | 0.838 | 0.680 |
| t30 | 3,847,991,575 | 0.2349 | 0.974 (0.955) | 0.988 | 0.956 | 0.960 | 0.987 | 0.977 | 0.960 | 0.750 |

Family [generic recipe (generic calib, RTN-proxy alloc)] overall: logit(fail)~log(ratio) slope -5.47; ret>=0.98 at ratio ~0.232 (5 pts)

Family [generic recipe (generic calib, RTN-proxy alloc)] worst_kind: logit(fail)~log(ratio) slope -7.26; ret>=0.95 at ratio ~0.209 (5 pts)

Family [task-calib + KL-sensitivity alloc (R3)] overall: logit(fail)~log(ratio) slope -5.03; ret>=0.98 at ratio ~0.192 (4 pts)

Family [task-calib + KL-sensitivity alloc (R3)] worst_kind: logit(fail)~log(ratio) slope -4.55; ret>=0.95 at ratio ~0.175 (4 pts)

Fit logit(1-ret)~log(ratio) [overall]: slope -5.57; ret>=0.98 predicted at ratio ~0.216 (extrapolation from 17 points; CI not meaningful with this few points)

Fit logit(1-ret)~log(ratio) [worst_kind]: slope -8.07; ret>=0.95 predicted at ratio ~0.188 (extrapolation from 17 points; CI not meaningful with this few points)
