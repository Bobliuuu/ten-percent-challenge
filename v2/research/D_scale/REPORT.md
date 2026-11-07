# D_scale: does a bigger dense base reach >=98% item retention at <=10% bytes?

Status: written 2026-11-07 (updated as runs land). Dev data only; nothing here touches locked/S2. No commits made. Raw results: `results/` (per-item JSON for every number), code: `dense_stream.py`, `export_hf.py`, `vllm_base.py`, `vllm_ct.py`, `analyze.py`, `ct_analyze.py`, `mktable.py`, `mkct.py`, `modal_job.py` (+`modal_launch.py`), run log: `STATUS.md`.

## 0. Bottom line (honest)
* **Tool-call faithfulness (T1): no dense base I could test passes the strict gates at <=10%** (98% overall, 95% in every (source,kind) cell). Best 9.6% results: Qwen3-32B 85.8% (self-distilled calibration, gamma=1; 775/903, Wilson lo 83.4%) and Qwen2.5-72B 84.2% (774/919, lo 81.7%); the binding cell is S1/nocall (the model calls a tool when none applies; 45-57%) and S3/multi (75-84%).
* **Label tasks: teacher-forced numbers do NOT transfer to the real generation exams.** On the C_tasks teacher-forced items, Qwen2.5-72B @9.609% (13.97 GB, entropy-accounted) reaches TOOLSEL 98.3-98.9% (V3/V6/V7; observed gates pass) and INJ 97.2% (T1-calibrated). On the REAL exams (greedy generation, strict scorers, 72B fp8-dynamic base through the same vLLM path): TOOLSEL best 95.55% (V7; binding cell S1/none 89.3%), INJ best 95.21% (V9; binding S3_injection 93.6%, S1_benign 93.0%); T1-calibrated V12 drops to 91.1% / 91.7%. **No candidate passes the strict gates on a real exam.** Class-collapse tasks (AIDET/SPAM/NLI/TOX) are far off at 9.6%.
* **The scale hypothesis is supported for label tasks but NOT for tool-call faithfulness.** Same recipe (plain allocation, legacy T1 calibration, 9.6%): label retention INJ 87.4 -> 94.5, TOOLSEL 91.6 -> 93.1, SENT 91.4 -> 96.8, FIN 82.0 -> 88.3, TOX 51.7 -> 72.6 going Qwen3-32B -> Qwen2.5-72B. Tool-call T1 retention at 9.6%: Qwen3-14B 56% (legacy items, other agent) -> Qwen3-32B 74.6% (plain, clean) but Qwen2.5-72B 73.9% (plain, legacy calib), and with gamma=1 83.8% (32B) vs 82.6% (72B). Different families and bases (the 72B base is an fp8-dynamic reference), so not a clean scaling law, but 2.2x more parameters bought nothing for tool-call retention. The plain-recipe tool-call curve in bytes is steep (32B: 76.2 -> 89.4 -> 94.8% for 9.6 -> 12.0 -> 15.0% of fp16): bits/param dominate.
* **Extrapolated size for 98% overall tool-call retention: ~19-25% of fp16 for 32B-class (plain allocation 19%, gamma=1 fit 25% because its gain vanishes at larger budgets), i.e. about twice the 10% budget; the worst cell (S1/nocall) needs the same or more.** Go/no-go: **NO-GO for tool-call retention at 10% with a dense GPTQ/rotation/entropy-coding recipe at 32B-72B.** **Label tasks: still NO-GO at 9.6% on the real exams (best 95.6% TOOLSEL, 95.2% INJ); only the teacher-forced proxy passes (98.9% TOOLSEL).** The size is entropy accounting; a real rANS artifact with an 11-bit scale code has not been verified.
* Recipes that DID move the number (32B @9.6%, clean calib): block-influence (BI) layer-weighted bit allocation gamma=1: 74.6 -> 83.8 (+9.2; S3 61.7 -> 80.4). Everything else tried on top was inside noise (R3 measured-KL +0.6, ncal=256 +1-2, emb/head 8 levels +~1, gamma 2 vs 1 -0.5) or negative (nocall-heavy calibration: S1/nocall 48 -> 32; label-mix calibration: T1 -1.5, helps INJ only).

## 1. Method / how numbers are produced (and what is NOT comparable)
* Quantiser (`dense_stream.py`): same maths as `compress/dense/dense.py` (Hadamard-128 rotation with random signs, GPTQ column loop on L-level grids with fp16 group scales, per-module L by a Lagrangian allocator over RTN-measured error tables, size = entropy*1.005 bits + 11 bits/group of scale + quantised embedding and lm_head (4 levels, ~2 bit; 8 where noted) + norms/biases + tokenizer/config files + 150 kB code). Layers stream from safetensors; 4-bit packed codes go to NVMe. 32B quant ~35 min, 72B ~47 min on one RTX PRO 6000. No optimizer/backward anywhere.
* Evaluation: the quantised weights are exported as a bf16 checkpoint (32B) or fp8-dynamic compressed-tensors checkpoint (72B; 145 GB bf16 does not fit one GPU) and run in vLLM 0.11 (greedy, 400 new tokens, base chat template with tools, `enable_thinking=False`, hardened `scorer.py`). Base for 32B = bf16 vLLM; base for 72B = RedHatAI Qwen2.5-72B-Instruct-FP8-dynamic in the same vLLM path (a true bf16 72B base did not fit a single GPU; the candidate also goes through fp8, so fp8 noise is common to both). Caveat: the 72B retention numbers are relative to an fp8 base, and the candidate's extra fp8 rounding (on already-quantised weights) is small next to 1.5-bit error but not zero.
* Items: 1300 DEV items (800 s1_dev + 500 s3_dev, random seed 0) from the rebuilt T1 splits (commit 85bccb2); label tasks = `research/C_tasks/items.jsonl` (5414 items, 300 S1-role + 300 S3-role per task), scored teacher-forced (answer tokens all argmax == greedy reproduces the label), computed with vLLM prompt_logprobs.
* Retention = kept / base-correct. **Noise floor (same bf16 Qwen3-32B base, vLLM max_num_seqs 16 vs 256): a rerun of the unquantised base keeps only 99.3% of base-correct items (94.6% of outputs byte-identical).** So ~0.7 pt of the 2 pt margin to 98% is numerics; and certifying "Wilson lo >= 98%" needs ~99.0-99.3% observed at n=500-900, which the base itself barely achieves. I report both observed and Wilson lo.
* **Calibration provenance / overlap (important).** Rows marked LEGACY used `compress/dense/t1/calib_t1.jsonl` (600 old-build S1 transcripts) + a generic slice; 375/800 S1 dev items share >=1 tool name with it (7/500 S3; exact query overlap 8/1300), so LEGACY S1 numbers are optimistic. Rows marked CLEAN used the new `t1v2/calib_t1.jsonl` (500 multi + 500 single/nocall sampled from the new `s1_calib`, family-disjoint from s1_dev by construction) + the same generic slice. Measured effect of the leak: plain 32B 9.6% legacy 76.2% vs clean 74.6%, i.e. small (-1.6 overall). Label-mix calibration = `C_tasks/calib.jsonl` (598 S1-role items, per-class balanced, id-disjoint from items.jsonl, 6 exact-text duplicates); it shares *source datasets* with the S1-role eval items by design, so S3-role is the clean signal for it. Label-task rows without "label-mix" were calibrated on tool-call transcripts only (no label items).

## 2. Tool-call ladder (T1 dev, 1300 items; strict gates)
| model / recipe | calib | ratio | retention (Wilson lo) | S1 | S3 | worst cell | strict |
|---|---|---|---|---|---|---|---|
| Qwen3-32B plain alloc (gamma0), LEGACY calib | legacy | 0.0961 | 76.2% (73.3) | 84.2 | 62.3 | S3/multi 54.2 | fail |
| Qwen3-32B plain alloc 12.0%, LEGACY calib | legacy | 0.1203 | 89.4% (87.2) | 90.5 | 87.3 | S1/nocall 74.7 | fail |
| Qwen3-32B plain alloc 15.0%, LEGACY calib | legacy | 0.1503 | 94.8% (93.1) | 94.7 | 94.9 | S1/nocall 86.7 | fail |
| Qwen3-32B gamma0, CLEAN calib | clean | see note | 74.6% (71.7) | 82.1 | 61.7 | S1/nocall 45.3 | fail |
| Qwen3-32B gamma1, CLEAN calib | clean | see note | 83.8% (81.3) | 85.8 | 80.4 | S1/nocall 48.0 | fail |
| Qwen3-32B R3 measured-KL + gamma1, CLEAN | clean | 0.0962 | 84.4% (81.9) | 86.5 | 80.7 | S1/nocall 48.0 | fail |
| Qwen3-32B gamma1 nocall-heavy calib, CLEAN | clean | see note | 82.4% (79.8) | 84.9 | 78.0 | S1/nocall 32.0 | fail |
| Qwen3-32B gamma1 12.0%, CLEAN | clean | see note | 91.1% (89.1) | 91.8 | 90.1 | S1/nocall 72.0 | fail |
| Qwen3-32B gamma1 15.0%, CLEAN | clean | see note | 93.6% (91.8) | 93.5 | 93.7 | S1/nocall 80.0 | fail |
| Qwen3-32B gamma1 + SELF-DISTILLED calib, CLEAN | clean | see note | 85.8% (83.4) | 87.7 | 82.5 | S1/nocall 57.3 | fail |
| Qwen3-32B gamma1, LEGACY calib | legacy | 0.0962 | 84.6% (82.1) | 87.9 | 78.9 | S1/nocall 56.0 | fail |
| Qwen3-32B gamma2, LEGACY calib | legacy | 0.0962 | 84.1% (81.5) | 86.0 | 80.7 | S1/nocall 57.3 | fail |
| Qwen3-32B R3 measured-KL (no gamma), LEGACY | legacy | 0.0962 | 82.1% (79.4) | 86.7 | 74.1 | S3/nocall 63.6 | fail |
| Qwen3-32B ncal256, LEGACY | legacy | 0.0961 | 78.1% (75.3) | 86.7 | 63.3 | S3/multi 55.1 | fail |
| Qwen3-32B label+tool calib mix, LEGACY tool part | legacy | see note | 74.5% (71.6) | 82.5 | 60.8 | S1/nocall 50.7 | fail |
| Qwen3-32B gamma1 + label mix, LEGACY | legacy | see note | 84.1% (81.5) | 87.4 | 78.3 | S1/nocall 56.0 | fail |
| Qwen2.5-72B plain alloc, LEGACY calib | legacy | 0.0960 | 73.9% (70.9) | 80.5 | 62.1 | S1/nocall 47.5 | fail |
| Qwen2.5-72B gamma1, CLEAN calib | clean | 0.0961 | 82.6% (80.0) | 85.2 | 77.9 | S1/nocall 47.5 | fail |
| Qwen2.5-72B V3 gamma1+ncal256+emb8+label-mix calib, CLEAN | clean | see note | 80.7% (78.1) | 84.6 | 73.9 | S1/nocall 45.0 | fail |
| Qwen2.5-72B V7 V3 + self-distilled tool part, CLEAN | clean | see note | 81.8% (79.2) | 85.9 | 74.5 | S1/nocall 55.0 | fail |
| Qwen2.5-72B gamma1+ncal256+emb/head 8 lvls, CLEAN | clean | see note | 84.2% (81.7) | 87.1 | 79.1 | S1/nocall 55.0 | fail |


Notes: ratios are bytes/fp16-bytes with everything counted (`meta*.json`); all 9.6% rows are 0.0960-0.0962 (<=10%; the allocator targets 9.6%). "see note" = same 0.0962 as its sibling (meta file not pulled). Cells = (source, kind); strict = overall >=98%, Wilson lo >=98%, every cell >=95%.
Older/smaller models (legacy items, other agent): Qwen3-8B 21.9%, Qwen3-14B 56.1% (plain, 9.6%), 14B 80.4% with calibration+KL allocation (compress/dense/t1_retention.md). My attempt to re-measure 0.6B-14B on the new items ran on a contended GPU and produced only base numbers (no candidates), so the cross-size ladder below 32B is legacy-item data and is not directly comparable.

## 3. Fit and extrapolation (tool-call retention)
* 32B plain alloc (legacy calib): points (ratio, overall ret, worst cell): (0.0961, 76.2%, 54.2%); (0.1203, 89.4%, 74.7%); (0.1503, 94.8%, 86.7%)
  logit(fail) vs ln(ratio) slope -3.89: overall >=98% extrapolated at **19.2%** of fp16; worst-cell >=95% at **19.8%** (3-point fit, no CI; treat as +-3 pts)
* 32B gamma=1 (clean calib): points (ratio, overall ret, worst cell): (0.0962, 83.8%, 48.0%); (0.1203, 91.1%, 72.0%); (0.1502, 93.6%, 80.0%)
  logit(fail) vs ln(ratio) slope -2.32: overall >=98% extrapolated at **24.7%** of fp16; worst-cell >=95% at **23.4%** (3-point fit, no CI; treat as +-3 pts)
* Scale axis at 9.6% (same recipe where possible): Qwen3-8B 21.9% and Qwen3-14B 56.1% (plain, legacy items, other agent) -> Qwen3-32B 74.6% (plain, clean) -> Qwen2.5-72B 73.9% (plain, legacy calib; different family). With gamma=1: 32B 83.8%, 72B 82.6%. Retention does not increase from 32B to 72B at 9.6%, so no log(params) extrapolation to 123B/405B is supported by this data; if the 8B->14B->32B trend (+35, +19 pts) continued it would predict ~90% at 70B+, but the 72B point shows the trend broke (family/base effects; the 72B base was fp8). For 405B the honest statement is: unknown, the only direct evidence says size alone does not cure 1.5 bpw.


## 4. Label tasks (teacher-forced exact label; retention overall (Wilson lo / worst S-or-class cell), %)
| artifact | INJ | TOOLSEL | SENT | FIN | ROUTE | TOX | NLI | SPAM | AIDET |
|---|---|---|---|---|---|---|---|---|---|
| 32B 9.6% plain, legacy | 87.4 (84.4/78) | 91.6 (88.8/84) | 91.4 (88.9/88) | 82.0 (78.5/68) | 78.1 (74.1/20) | 51.7 (47.3/0) | 44.8 (40.3/0) | 59.0 (54.8/8) | 4.9 (3.0/0) |
| 32B 12.0% plain, legacy | 98.4 (97.0/98) | 97.8 (96.2/96) | 97.6 (95.9/96) | 96.8 (94.9/95) | 89.5 (86.4/35) | 84.7 (81.3/68) | 83.8 (80.2/55) | 88.4 (85.4/76) | 41.4 (36.1/14) |
| 32B 15.0% plain, legacy | 97.7 (96.0/96) | 99.0 (97.7/98) | 97.9 (96.4/96) | 97.2 (95.4/95) | 95.6 (93.3/41) | 92.8 (90.1/86) | 90.8 (87.8/72) | 91.0 (88.3/80) | 96.1 (93.3/14) |
| 32B 9.6% gamma1, legacy | 88.5 (85.6/79) | 96.1 (94.0/92) | 91.1 (88.5/87) | 87.3 (84.2/81) | 86.5 (83.1/29) | 82.5 (78.9/70) | 62.9 (58.4/52) | 73.3 (69.4/44) | 34.0 (28.9/8) |
| 32B 9.6% R3-KL, legacy | 94.6 (92.4/92) | 95.5 (93.3/93) | 91.6 (89.0/84) | 80.3 (76.7/66) | 86.7 (83.4/29) | 76.1 (72.1/51) | 69.2 (64.8/53) | 70.7 (66.6/35) | 89.3 (85.4/14) |
| 32B 9.6% label-mix calib (LCM) | 96.0 (94.1/93) | 92.2 (89.5/85) | 88.5 (85.6/79) | 88.3 (85.2/78) | 84.2 (80.6/6) | 86.1 (82.8/78) | 73.4 (69.1/43) | 65.3 (61.2/22) | 4.5 (2.7/0) |
| 32B 12.0% gamma1, clean | 91.4 (88.7/84) | 97.8 (96.2/96) | 95.1 (93.0/92) | 94.1 (91.8/90) | 94.9 (92.6/80) | 90.5 (87.7/86) | 86.7 (83.3/81) | 69.9 (65.8/33) | 38.5 (33.3/18) |
| 72B 9.6% plain, legacy | 94.5 (92.3/90) | 93.1 (90.6/89) | 96.8 (95.0/95) | 88.3 (85.2/78) | 88.4 (85.3/0) | 72.6 (68.6/44) | 75.6 (71.6/69) | 66.2 (62.2/29) | 40.8 (35.8/1) |
| 72B 9.6% V3 gamma1+ncal256+emb8+label-mix, clean | 93.1 (90.7/87) | 98.3 (96.8/97) | 90.7 (88.0/81) | 90.2 (87.3/82) | 91.7 (88.9/29) | 93.8 (91.4/91) | 73.9 (69.8/64) | 70.3 (66.3/37) | 40.8 (35.8/0) |
| 72B 9.6% V6 same, pure label calib | 91.7 (89.1/83) | 98.3 (96.8/97) | 94.3 (92.0/89) | 94.1 (91.8/90) | 94.1 (91.7/36) | 95.1 (92.9/93) | 72.2 (68.0/54) | 77.9 (74.2/53) | 42.0 (36.9/2) |
| 72B 9.6% V7 label-mix + self-distilled tool part | 89.8 (87.0/80) | 98.9 (97.5/98) | 90.2 (87.4/80) | 90.0 (87.1/82) | 93.7 (91.2/36) | 92.4 (89.9/89) | 65.2 (60.7/45) | 66.2 (62.2/29) | 40.8 (35.8/0) |
| 32B 15.0% gamma1, clean | 98.7 (97.4/98) | 97.8 (96.2/96) | 97.9 (96.4/96) | 97.0 (95.1/92) | 96.2 (94.1/62) | 94.4 (92.0/89) | 89.3 (86.1/66) | 91.2 (88.5/80) | 82.5 (77.9/7) |
| 72B 9.6% gamma1, clean | 96.1 (94.2/95) | 97.1 (95.3/95) | 92.1 (89.6/89) | 83.5 (80.1/72) | 87.4 (84.2/0) | 81.5 (77.9/64) | 68.4 (64.0/62) | 58.5 (54.3/12) | 47.7 (42.5/14) |
| 72B 9.6% gamma1+ncal256+emb8, clean | 97.2 (95.5/97) | 97.3 (95.6/95) | 93.2 (90.8/89) | 85.2 (81.9/78) | 87.8 (84.6/7) | 73.5 (69.6/45) | 73.7 (69.5/66) | 59.4 (55.2/15) | 42.5 (37.4/3) |


Gates: INJ/TOOLSEL/SENT at 72B need >=98% overall and every cell >=95%; to have Wilson lo >=98% at the observed n the *observed* retention must be >=99.3% (INJ n=566: 562 kept; TOOLSEL n=524: 520; SENT n=559: 555). AIDET/SPAM/TOX show class collapse (one class predicted for everything: TOX Toxic, SPAM Legit, AIDET Human) - these are not recoverable by allocation tricks at 9.6%.

## 4b. REAL-exam re-measurement (generation, strict scorers; Qwen2.5-72B @9.61%)
| artifact | task | retention [Wilson lo] | cells (kept/base-correct) | strict |
|---|---|---|---|---|
| V7 label-mix + self tool | TOOLSEL | 95.55% [94.0] (838/877) | S1/none 183/205=89.3; S1/tool 288/295=97.6; S3/none 86/88=97.7; S3/tool 281/289=97.2 | fail |
| V7 label-mix + self tool | INJ | 94.13% [92.8] (1298/1379) | S1/S1_benign 338/387=87.3; S1/S1_injection 275/286=96.2; S3/S3_benign 370/377=98.1; S3/S3_injection 315/329=95.7 | fail |
| V3 label-mix + gold tool | TOOLSEL | 93.96% [92.2] (824/877) | S1/none 163/205=79.5; S1/tool 292/295=99.0; S3/none 85/88=96.6; S3/tool 284/289=98.3 | fail |
| V3 label-mix + gold tool | INJ | 93.47% [92.0] (1289/1379) | S1/S1_benign 356/387=92.0; S1/S1_injection 258/286=90.2; S3/S3_benign 374/377=99.2; S3/S3_injection 301/329=91.5 | fail |
| V9 V3 + INJ items | TOOLSEL | 94.30% [92.6] (827/877) | S1/none 175/205=85.4; S1/tool 286/295=96.9; S3/none 84/88=95.5; S3/tool 282/289=97.6 | fail |
| V9 V3 + INJ items | INJ | 95.21% [94.0] (1313/1379) | S1/S1_benign 360/387=93.0; S1/S1_injection 272/286=95.1; S3/S3_benign 373/377=98.9; S3/S3_injection 308/329=93.6 | fail |
| V12 T1-only calib | TOOLSEL | 91.11% [89.0] (799/877) | S1/none 145/205=70.7; S1/tool 293/295=99.3; S3/none 83/88=94.3; S3/tool 278/289=96.2 | fail |
| V12 T1-only calib | INJ | 91.66% [90.1] (1264/1379) | S1/S1_benign 372/387=96.1; S1/S1_injection 228/286=79.7; S3/S3_benign 375/377=99.5; S3/S3_injection 289/329=87.8 | fail |


Items: TOOLSEL = `combo/TOOLSEL/data/eval_dev_items.jsonl` (1063, `toolsel_scorer.py`), INJ = `tournament/INJ/compress_inputs/probe_items.jsonl` (1566, `INJ/scorer.py`, prompt v2); base = Qwen2.5-72B fp8-dynamic (TOOLSEL acc 82.5%, INJ 88.1%). Calibration: V3/V7/V9 label-mix (C_tasks calib) + tool-call transcripts (V3 gold text, V7 self-distilled; V9 = V3 + 240 clean INJ-pipeline items); V12 = T1 transcripts only. Teacher-forced -> real drop: V7 TOOLSEL 98.9 -> 95.6, V3 98.3 -> 94.0, V12 INJ 97.2 -> 91.7. Self-distilled tool part helps the none/benign class (S1/none 79.5 -> 89.3, V3 -> V7).

## 5. Levers: measured gain at 9.6% (Qwen3-32B unless stated), by calibration source
| lever | calib | result | verdict |
|---|---|---|---|
| BI-weighted allocation gamma=1 (cheap, 0 cost) | clean | 74.6 -> 83.8 (+9.2) | KEEP (biggest) |
| gamma=2 | legacy | 84.1 vs 84.6 (gamma1) | saturated |
| R3 measured-KL per-tensor allocation (20 transcripts, RTN-rot KL at L=2/3/4) | legacy 82.1 vs 76.2 plain (+5.9); clean + gamma1: 84.4 vs 83.8 | same as gamma1 | not additive with gamma |
| ncal 128 -> 256 | legacy | 76.2 -> 78.1 | noise |
| label-mix calib | legacy tool part | T1 74.5 vs 76.2; INJ S3-role 0.86 -> 0.95 | helps INJ only |
| nocall-heavy tool calib | clean | S1/nocall 48 -> 32 | HURTS |
| embedding/lm_head 8 vs 4 levels + ncal256 (72B) | clean | 82.6 -> 84.2 | small |
| self-distilled calib (base's own outputs) | clean | 85.8 (+2.0), S1/nocall 57.3, S3/multi 84.1; 72B label-mix: TOOLSEL 98.3 -> 98.9, T1 80.7 -> 81.8 | small positive |
| label-mix calibration (C_tasks 598 per-class-balanced items + tool part), 72B | S1-role in-source, S3 cross-source | TOOLSEL 97.3 -> 98.3 (S3-role .95 -> .97), TOX 73.5 -> 93.8, FIN 85 -> 90, ROUTE 87.8 -> 91.7, INJ 97.2 -> 93.1, SENT 93.2 -> 90.7, T1 84.2 -> 80.7 | task trade-off |
| pure label calibration, 72B (V6) | same | TOOLSEL 98.3, SENT 94.3, TOX 95.1, FIN 94.1, ROUTE 94.1, INJ 91.7 | best for non-INJ labels |
| bits: 12.0% / 15.0% (gamma1, 32B) | clean | 91.1% @12.0%, 93.6% @15.0% (vs plain-legacy 89.4 / 94.8): gamma gain vanishes with budget | |
Untested for lack of time/compute: SpQR-style outlier side-set, depth pruning + re-spend, group size 64, act-order, damp sweep (damp parameter implemented, no run), per-class balanced INJ/TOX pipeline splits (the C_tasks calib is already per-class balanced), a second family at 70B (other agent), Modal fan-out (jobs stayed PENDING for >1h behind the 10-GPU cap; cancelled, one re-submitted at the coordinator's request: call fc-01M6T6ZGZ3ASW8HK1HAX2A5MA5).

## 6. Hosting / evaluation feasibility for bigger models
* Qwen2.5-72B 9.6% = 13.97 GB: fits one 96 GB GPU as 4-bit packed codes with on-the-fly dequant (my `QLinear`, torch ops, ~25 min per 1300 T1 items at bs16 when fully dequantising each step); in practice I exported dequantised fp8 and used vLLM: 1300 T1 items in ~5 min, 5414 label items in ~3 min.
* Llama-3.1-405B at 10%: 81 GB artifact fits one 96 GB GPU (15 GB left for KV) but only with a custom fused dequant kernel. Estimate for ~2600 generations (avg ~1.5k-token prompts, ~100-400 new tokens): prefill 2600*1.5k = 3.9 M tokens * 2*405e9 FLOP = 3.2e18 FLOP => ~3 h at ~300 TFLOPs sustained (bf16 on RTX PRO 6000-class), decode memory-bound ~45 ms/step at 1.8 TB/s plus dequant (fused) ~ +50% => 41 batches of 64 x 400 steps ~ 25-40 min; with my torch-level dequant (materialising bf16 per layer per step, ~10x the traffic) decode alone would be ~5 h. Total: ~4 h fused, ~8 h with torch dequant. 405B quantisation needs a streaming quantiser (done here: peak ~30 GB GPU, no model in RAM) at ~4-5 h for GPTQ on one GPU; infeasible inside this deadline on shared boxes (the L3_405B Modal job is another agent's).
* Qwen3-235B-A22B (literature only): fp16 ~470 GB -> 47 GB budget. REAP (arXiv 2510.13999, github.com/CerebrasResearch/reap) supports Qwen3-30B-A3B natively and Qwen3-Coder-480B / Kimi-K2 (new models via a `MODEL_ATTRS` entry); training-free expert pruning with saliency from calibration. Paper numbers: Qwen3-30B-A3B at 50% pruning MC 0.72 -> 0.50 even with REAP (frequency 0.48); 480B at 50% REAP BFCL 0.690 -> 0.674, Kimi-K2 BFCL 0.666 -> 0.564 (15% relative), frequency pruning collapses (480B code 0.01). Aggregate scores, not item retention. Budget arithmetic (235B: experts ~227B params, attention ~6.7B, embeddings ~1.2B): 50% prune leaves ~2.9-3.0 bpw for surviving experts; 25% prune only ~1.9-2.0 bpw. Our dense results say 1.5 bpw (even 2.4 bpw => 94.8% at 32B) cannot hold item-level retention, and 3 bpw after a 50% prune stacks two losses; no evidence found that >=98% item retention is reachable. Verdict from literature: NO-GO, no runs.

## 7. What would actually be needed
* Tool-call retention >=98% overall AND >=95% per cell needs ~18-20%+ of fp16 with this recipe family (dense, scalar-quantised, 32B-72B). 10% would need a step change (vector/trellis quantisation at ~1.5 bpw such as QTIP/AQLM-class, or a task where class collapse is avoidable); none was available in the time.
* For a label-style task (INJ, TOOLSEL) on a 70B-class dense base the gap is <1 pt at 9.6%; the strict-pass probability depends on the label-mix 72B run and on how the maintainer treats the numerics floor.

## 7b. Not done / limitations (so nothing is over-read)
* KL(base||quantised) was not measured for the big models (the requested KL column is absent); retention was measured directly, which is the decision metric. The only teacher-forced KL work is inside R3 (per-tensor RTN KL on 20 task transcripts).
* The small-model rungs (0.6B-14B) on the NEW items were not completed (GPU contention on tenpct-g2); smaller sizes use the other agent's legacy-item numbers. No Llama-70B/405B runs by me (other agents). No depth pruning, outlier side-set (SpQR), group-size 64, act-order or damp sweeps; no real rANS artifact.
* Recipe selection used the same dev items it is evaluated on (about 25 variants), so the best numbers carry selection bias of order 1-2 pts; differences below ~2 pts between variants are within noise (paired n~900; noise floor of the base itself 0.7 pt).
* 72B comparisons use an fp8-dynamic base and an fp8-exported candidate; 32B uses bf16 for both.
* Modal: the first 4 spawned variants stayed PENDING behind the 10-GPU cap and were cancelled; one later job (`m72_gam1_lcm_nc`, H200, deployed app tenpct-dscale) DID complete (46 min) and independently reproduced the pipeline on a different GPU type: Qwen2.5-72B gamma1 + label-mix calibration + a nocall-heavy tool part, emb/head 4 levels, ncal 128, ratio 0.09610: T1 82.5% (758/919; S1/nocall 56.2), teacher-forced TOOLSEL 98.1% (514/524, worst cell 97.3), INJ 90.5, SENT 90.5, TOX 93.0 -- the same picture as V3 (`results/*modal_gam1_lcm_nc*`). Not evaluated on the real exams.

## 8. Reproduce
`python dense_stream.py cfg.json` (modes klsens/quant/eval; cfg keys: model_dir, tag, target_ratio, g, emb_L, head_L, gamma, ncal, damp, calib_ct, calib_ct_mix, calib_self, kl_file, t1dir) -> `export_hf.py MODEL Q EXP G [fp8 qcfg.json]` -> `vllm_base.py EXP out.json t1v2 bf16` / `vllm_ct.py EXP out.json items.jsonl` -> `analyze.py BASE CAND` / `ct_analyze.py BASE CAND`.
