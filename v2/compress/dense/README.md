# C-DENSE: training-free PTQ of dense Qwen bases toward <=10% of bf16

Scope: Qwen3-8B and Qwen3-14B fully explored; Qwen2.5-32B-Instruct NOT run (Modal A100-80 slots never scheduled, 32B chain on g1 was pre-empted by the T1 priority change; code path for a GPU-resident 32B exists: DENSE_GPU=1). No training of any kind (no optimizer/backward). All bytes exact: entropy-coded indices x1.005 + 11-bit scales + embeddings + lm_head (both quantized, never trimmed) + norms + tokenizer.json/tokenizer_config/config/generation_config (11.4 MB) + 150 kB code allowance.
Evaluated on dev data only (T1 probe_items.jsonl = S1-dev + S3-dev; calibration = T1 S1-calib; no locked/S2).

Code: dense.py (compression + eval, Modal or local), local_run.py (box runner), t1_analyze.py (retention tables/fits), collect.py (generic fidelity table), build_data.py + data/ (generic prompt sets), mkcfgs.py. Raw: results/all/*.json, results/t1/*.json (per-item correctness), logs/.

## 1. Headline: real tool-use task (T1), item retention = P(correct | base correct), Wilson CI via harness/scorecard.retention
Base bf16 accuracy (HF greedy, T1 scorer + T1 chat template with tools, enable_thinking False, max 400 new tokens): 14B 0.775, 8B 0.762 (matches T1 SUMMARY 0.770/0.763).

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


Recipes: "t096/t11/t125/t15/t30" = generic calibration (dolly/glaive/pii text), allocation by RTN Hessian-proxy error. "t096_cal*"/"cal_kl_r*" = calibration on T1 S1-calib tool-call transcripts (96x1024 tokens) and, for *_kl*, per-tensor bit allocation from measured KL (R3, below).

## 2. What moved retention at 9.6% (Qwen3-8B, then 14B)
| change | 8B ret | 14B ret |
|---|---|---|
| generic recipe | 0.219 | 0.561 |
| + task-format calibration (T1 S1-calib) | 0.363 | 0.712 |
| + 80/20-ish mix (half task, half generic) | 0.416 | n/a |
| + R3 measured-KL per-tensor allocation | 0.536 | 0.804 |
| + emb/lm_head at L=3 instead of 4 (frees ~100 MB) | n/a | 0.817 (within noise) |
| + BI layer weighting gamma=1 (on cal, no KL) | 0.543 | n/a |
| drop 2 lowest-BI layers (cal) | 0.352 (no gain) | n/a |
| R5 error-propagating target (Qronos/GPTAQ-style) | 0.175 (full), 0.123 (lam x10), 0.562 (alpha=0.5 on top of KL: +2.6 pts, inside CI) | not run |
Single run per cell; Wilson CI half-width ~ +-4-5 pts, so differences < ~6 pts are not resolved. R3 and task calibration are clear wins; R5 as implemented is NOT: full-strength correction W+W(C-H)(H+lam)^-1 destroys single/multi calls (nocall kept), alpha=0.5 is neutral. Likely ill-conditioning of the regression (no 2-sweep, no tuned damping); act-order flag was implemented but never run to completion. Treat R5 as untested-in-a-fair-form, not refuted.
R3 detail: for every tensor, replace only that tensor by an L in {2,3,4,8} RTN+rotation version, measure KL(base||quantized) on 8 task-format calibration sequences (klsens_*.json), then Lagrangian water-filling over a measured-additive KL at the byte budget. 14B bits: embed+lm_head cost 195+191 MB = 13.6% of the 9.6% artifact (L=4 each); body 1.48 bpw.
Calibration caveat: S1-calib is in-distribution for S1-dev and shares style with S3; S3 retention is consistently lower (14B 0.68-0.72 vs S1 0.90 at 9.6%). That gap is the overfit signal; the maintainer's data will look more like S3 or worse.

## 3. Operating point and extrapolation (Qwen3-14B, task-calib + KL allocation)
- 9.63% (2,843,094,072 B): ret 0.804 overall, worst kind 0.776 -> FAILS the 98% / 95% rules.
- 11.0%: 0.875; 12.5%: 0.938 (worst kind 0.907); generic recipe 15.0%: 0.951 (worst 0.923); 23.5%: 0.987 (worst 0.981).
- Fit logit(fail) vs log(size) on this family predicts overall >=98% at about 16% and worst-kind >=95% at about 15% of bf16 (8B: ~19% / ~17.5%). 3-4 points, extrapolation, no real CI.
Plain statement: 98% retention at 10% is NOT reached for either model with this toolkit; best measured at 9.6% is 80% (14B) and 54% (8B). The gap to 98% is roughly 5-6 percentage points of size (~1 bit/param on the body) for 14B. Qwen2.5-32B was not measured; the Track-A guess (80-95%) is untested here.

## 4. Generic fidelity vs bf16 (greedy 64-token exact match, teacher-forced top-1, KL(top-64+tail)) - 400 ID prompts, 200 OOD prompts
CAUTION: exact64 is bounded far below 1 by bf16 numerics alone: the same bf16 14B model with batch 32 vs 8 agrees on only 0.85 of ID prompts (0.84-0.85), 0.67 for 0.6B. So 98% exact64 is unreachable by construction; top-1/KL are the usable generic metrics. Task retention (section 1) is the decision metric.
| run | bytes | ratio | body bpw | drop | (a) exact64 ID | exact16 | (b) top1 | (c) KL | OOD (a) | OOD (b) | OOD (c) | OOD drop a/b | bf16 self-floor |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen3-0.6B_lad_L3 | 138,273,081 | 0.116 | 1.59 | 0 | 0.000 | 0.000 | 0.378 | 3.388 | 0.000 | 0.264 | 5.380 | +0.000/+0.113 | 0.65 |
| Qwen3-0.6B_lad_L4 | 158,644,858 | 0.1331 | 1.96 | 0 | 0.003 | 0.010 | 0.517 | 1.943 | 0.000 | 0.453 | 2.798 | +0.003/+0.065 | 0.65 |
| Qwen3-0.6B_lad_L6 | 203,361,391 | 0.1706 | 2.47 | 0 | 0.022 | 0.050 | 0.706 | 0.674 | 0.000 | 0.703 | 0.833 | +0.022/+0.003 | 0.65 |
| Qwen3-0.6B_lad_L8 | 222,788,856 | 0.1869 | 2.82 | 0 | 0.035 | 0.115 | 0.776 | 0.405 | 0.005 | 0.763 | 0.526 | +0.030/+0.013 | 0.65 |
| Qwen3-0.6B_smokex2 | 238,477,369 | 0.2 | 3.1 | 0 | 0.068 | 0.145 | 0.797 | 0.332 | 0.005 | 0.791 | 0.396 | +0.062/+0.006 | 0.67 |
| Qwen3-0.6B_smokex3 | 238,477,369 | 0.2 | 3.1 | 0 | 0.068 | 0.145 | 0.797 | 0.332 | 0.005 | 0.791 | 0.396 | +0.062/+0.006 | 0.67 |
| Qwen3-1.7B_lad_L3 | 370,270,148 | 0.1076 | 1.59 | 0 | 0.000 | 0.005 | 0.526 | 3.140 | 0.000 | 0.396 | 5.909 | +0.000/+0.129 | 0.69 |
| Qwen3-1.7B_lad_L4 | 435,619,358 | 0.1266 | 1.96 | 0 | 0.013 | 0.050 | 0.660 | 1.611 | 0.000 | 0.611 | 2.398 | +0.013/+0.049 | 0.69 |
| Qwen3-1.7B_lad_L6 | 558,827,277 | 0.1624 | 2.47 | 0 | 0.110 | 0.190 | 0.805 | 0.604 | 0.000 | 0.784 | 0.898 | +0.110/+0.022 | 0.69 |
| Qwen3-14B_selfcheck_bf16 | 3,124,082,553 | 0.1058 | 0.0 | 0 | 1.000 | 1.000 | 1.000 | 0.000 | 1.000 | 1.000 | 0.000 | +0.000/+0.000 | 0.85 |
| Qwen3-14B_t096 | 2,842,862,176 | 0.0962 | 1.48 | 0 | 0.055 | 0.147 | 0.704 | 1.261 | 0.000 | 0.527 | 3.196 | +0.055/+0.177 | 0.85 |
| Qwen3-14B_t096_drop4 | 2,841,133,535 | 0.0962 | 1.64 | 4 | 0.033 | 0.152 | 0.698 | 1.256 | 0.000 | 0.582 | 2.638 | +0.033/+0.116 | 0.85 |
| Qwen3-14B_t096_dz15 | 2,842,773,898 | 0.0962 | 1.48 | 0 | 0.043 | 0.133 | 0.666 | 1.545 | 0.000 | 0.468 | 3.830 | +0.043/+0.198 | 0.85 |
| Qwen3-14B_t15 | 4,440,713,948 | 0.1503 | 2.39 | 0 | 0.255 | 0.417 | 0.890 | 0.176 | 0.025 | 0.862 | 0.337 | +0.230/+0.027 | 0.85 |
| Qwen3-14B_t30 | 6,931,738,277 | 0.2347 | 3.75 | 0 | 0.463 | 0.718 | 0.962 | 0.020 | 0.130 | 0.956 | 0.033 | +0.333/+0.006 | 0.85 |

Ladder (Qwen3 0.6B/1.7B, uniform L, same recipe) completed only for L<=6/8; 4B/8B ladder and the size-scaling fit were not finished (box queue pre-empted by the T1 priority). Quick-stage / layer-subset predictors (qstride=4) are implemented (cfg qstride/qoff/qlist, nev, ntf, max_new) but the predicted-vs-actual validation was not run; no predicted-vs-actual numbers are claimed.

## 5. Blockers / honesty
- Modal: only 1 L40S runner ever scheduled for this app (shared workspace GPU slots); A100-80/H100 jobs queued indefinitely. All numbers above came from the AWS boxes g1/g4 (RTX PRO 6000 96 GB), not Modal. Modal spend ~ $0.
- Base generations for the generic table were made on the same hardware as candidates for 14B (box), on Modal H100 for 8B/others (cross-hardware noise included).
- Single seed, 600 items, CIs wide. Compression is simulated (dequantized bf16 weights); bytes are computed from entropy of the quantized indices, not from a written rANS file.
