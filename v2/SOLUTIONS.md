# SOLUTIONS: registry of every candidate approach tried or considered

Created 2026-11-08 by the registry lane (reads only, $0, no GPU, nothing re-run). User order: every solution already tried is written down as a POSSIBLE SOLUTION; the best one is chosen afterwards. Deadline: 2026-11-12 17:05 UTC.
Sources read: `GOAL_README.md`, `RUNS.md`, `v2/{NEXT_PLAN,TASK_SEARCH,NATIVE_LOWBIT,FEEDBACK_LEGAL,DECISION,REDESIGN,PROOF_C2,RULES}.md`, `v2/research/{A_sota,B_native,C_tasks,D_scale,L1_large,L2_moe,L3_405B}/REPORT.md`, `v2/{fix_real,quality,entry32b,c2_fb,lowbit,lowbit_fb,altpush,moe,engine,vq,prune,entropy,big,feedback,feedback2,feedback_v3,fb_legal,combo,tournament,harness,commercial}`, `trackA..S`. Numbers are copied from those files; anything I recomputed from per-item JSON is marked "(recomputed)".

## 0. How to read this file

Rule codes R1..R9 are the nine items under "How entries are judged" in `GOAL_README.md` (R1 size counts everything, R2 same means faithful >=98% overall and >=95% per kind, R3 unseen exam >=2000 locked, >=300 per kind, >=500 from an unused source, R4 no new trained parts, R5 feedback teaches (<=25 items, >=25 locked teach items, neighbours intact), R6 shuffled/empty controls worse, R7 real job output, R8 base well above chance and below ceiling, R9 spirit over letter). MC = the maintainer's check (own different-distribution data, own feedback, no fitted computation or side memory).
Status: PASS / FAIL / GRAY (defensible either way, maintainer decides) / UNTESTED. "PASS*" = passes on paper or by construction only, not yet measured on the real artifact.

Honesty conventions used throughout:
- **dev vs locked.** Only the filed C2 candidate was ever scored on a locked exam (and that locked file is now burned). Every other number is DEV.
- **teacher-forced (TF) vs real generation (GEN).** TF numbers overstated retention by 3-10 points where both were measured (D_scale 72B: TOOLSEL 98.9 TF vs 95.55 GEN; L3 405B: INJ 99.7 TF vs 90.2 GEN). TF is never pass evidence.
- **easy-source vs real-source.** The 98.1% INJ figure for 32B (task_search "INJ-easy", safe-guard + jackhhao) is an easy-source probe on data the real INJ lock deliberately dropped. Real-data INJ on the same 32B is 0.957 / 0.958 / 0.942 (10.0 / 11.3 / 12.5%). Do not cite 0.981.
- **size accounting.** "est" = entropy/arithmetic estimate, not a written artifact. "real" = bytes of the written directory via `v2/harness/size_check.py`. Several 32B task_search points are at 10.03% (0.03 pt OVER the cap).
- Lower bounds are Wilson 95% unless stated. Draw-to-draw quantiser noise is about +-2 pt (TASK_SEARCH section 6), so differences under 2 pt between variants are noise. Recipe selection on the same dev items carries a 1-2 pt optimism (D_scale 7b).
- Probabilities are my subjective estimate that the candidate, as it stands plus the listed remaining work, passes ALL nine rules AND the maintainer check as written. They are not calibrated to any data beyond what is cited. "V" is value if it passes (3 = a real compression of a general model by us with a clean story; 2 = acceptable; 1 = passes but exposed on R9). Rank key = P x V.

## 1. Ranked summary (best first, by P(all 9) x V)

Bottom line: **no candidate is above about 5% and none currently passes R2 on a real exam except the native-low-bit identity route, which instead fails or is exposed on R1/R9 reading and has no working feedback.** The honest sum over disjoint routes is roughly 10%. The sections below keep every route alive as a candidate, as ordered.

| rank | id | candidate | headline measured retention (GEN unless noted) | size | P(all 9) | V | P x V | cheapest next experiment (cost) |
|---|---|---|---|---|---|---|---|---|
| 1 | C07 | Qwen2.5-32B VQ-D8 K=5300, INJ | 95.0% [94.2] DEV n=3377 base-correct; S3_injection 81.3, S2_benign 85.8 | 9.85% est | 1.5% | 3 | 4.5 | Re-run on the 1566-item real INJ dev set used by C11/C03 so numbers are comparable, then K=14500-class (about 1.72 bpw) point at <=10% (one A100/H100 box, about $15) |
| 2 | C04 | Qwen2.5-72B INJ, label calibration (V9) | 95.21% [94.0]; worst cell S3_injection 93.6, S1_benign 93.0 | 9.61% est (13.97 GB entropy accounting) | 1.5% | 3 | 4.5 | Run the C11 VQ quantiser (`quality/cfgs_72.json`, already written) on 72B and score the same real INJ items (about $40, 1 day) |
| 3 | C11 | Qwen2.5-7B VQ-D8 K=14500 (QUALITY_PUSH), INJ / TOOLSEL | INJ 96.2% [95.0] (best of 12 configs), TOOLSEL 94.3% [92.5]; worst cell INJ 93.8, TOOLSEL 80-84 (S1/none) | 9.93% est | 2% | 2 | 4.0 | Same quantiser on 14B/32B same items; fix S1/none cell with the self-distilled calibration (about $10) |
| 4 | C03 | Qwen2.5-72B TOOLSEL, label+self-distilled tool calibration (V7) | 95.55% [94.0]; worst cell S1/none 89.3 | 9.61% est | 1.2% | 3 | 3.6 | Same as C04 (VQ on 72B); no-irrelevance TOOLSEL is a product change (R7/R1 exposure) |
| 5 | C19 | Bonsai-8B 1-bit, lossless repack (D1a tool call / D1b label task + feedback) | 100% vs itself (2000/2000 identical greedy, dev); 88.9% vs parent Qwen3-8B | 7.13% real (shipped tree) | 3% | 1 | 3.0 | Post the 3 maintainer questions (needs user OK, $0); in parallel the fp16-patch feedback test on Bonsai (cfgs8, about $5) |
| 6 | C08 | Qwen2.5-32B NEWS router (4-menu) | 96.1% [95.0]; Technology class 0.90 | 10.03% est (OVER cap) | 0.8% | 2 | 1.6 | Run at the legal 9.6% point and with C07's quantiser; build Entertainment feedback (about $20) |
| 7 | C09 | Qwen2.5-32B INJ-real, scalar GPTQ | 95.7% [94.5]; S2_benign 0.78 | 10.03% est (OVER cap) | 0.7% | 2 | 1.4 | Superseded by C07 (same task, VQ); skip unless C07 is dropped |
| 8 | F01 | Feedback mnH (closed-form min-norm lm_head row edit) | not a compression route; teach +0.5 to +0.8 on TOX+selfharm (TF), neighbours 99.8-99.96% | +0 bytes (calibration texts <=0.5 MB to ship) | R4/5/6 component only: 15-20% | n/a | n/a | Package as `feedback.py`, run harness-v3 on a real compressed artifact with 10+ draws (about $10) |
| 9 | C10 | Qwen2.5-32B TOOLSEL / FIN / ROUTE (task_search) | TOOLSEL 92.8, FIN 93.7, ROUTE 94.7 (worst class 0.75) | 10.03% est | <0.5% | 2 | <1 | None worth it; TOOLSEL superseded by C03, FIN narrow domain, ROUTE has 29 classes (R3 infeasible) |
| 10 | C15 | BitNet-b1.58-2B-4T ternary + 4-bit emb + vocab trim | MDC-6 dev agree .988, MMLU dev agree .919; T1 tool-call base .267 (R8 fail) | 9.453% real | 0.8% | 1 | 0.8 | None cheap; needs a non-letter task where the base is above chance, and a base-identity ruling |
| 11 | C06/C05 | Qwen2.5-72B / Qwen3-32B T1 tool calls | 84.2% [81.7] / 85.8% [83.4]; S1/nocall 55-57 | 9.6% est | <0.3% | 3 | <1 | Dead: extrapolated size for 98% is 19-25% of fp16 |
| 12 | C13/C14 | Qwen3-235B-A22B / 30B-A3B REAP + low-bit | 235B 93.9% (TF, n=231); 30B REAP50+RTN4 94.4% (GEN, n=179) but 12.4% size | 9.88% / 12.4% est | <0.3% | 3 | <1 | Dead at the cap unless a VQ expert codec reaches 2 bits with 50% prune (about $40) |
| 13 | C12 | Llama-3.1-405B INJ | 90.2% [88.5] GEN (TF said 99.7) | 9.60-9.84% est (scale accounting) | <0.3% | 3 | <1 | Dead |
| 13b | C26 | GIANT_MOE (R1/V3/K2/235B/480B/GLM-4.5/gpt-oss-120b) | not measured | 10.44% best file | <0.3% | 3 | <1 | none (no GPU run, $0) |
| 14 | C17 / C18 / C16 | Ternary-Bonsai-8B / Ternary-Bonsai-2-27B / Falcon-E-3B | not measured on label tasks; ternary Bonsai 91.0% vs parent | 10.19% / 10.67% min / 13.8% lossless | <0.3% each | 1 | <0.5 | Size is the blocker; entropy coding cannot recover the gap (zero slack at log2 3) |
| 15 | C20 | MDC-6 7B letter-head artifact (+ lowbit_fb PCH) | dev .8820 vs .8890 base-with-bias; agree .954 | 9.736% real | ~0% | 1 | 0 | Dead on R4/R7 as written (letters + fitted head) |
| 16 | C01 | Filed C2 7B artifact (letter head + LDA memory), issue #2 | 93.9% [92.6] LOCKED (1441/1535) | 9.404% real | ~0% | 1 | 0 | Locked exam burned; do not salvage (REDESIGN section 0) |
| 17 | C02 | fix_real 7B real tool-call generation | 93.4% (name acc; irrelevant class 61.9%) DEV; MMLU .445 vs .693 | 9.93% est | ~0% | 2 | 0 | Dead: 1.74 bpw body |

Components (feedback mechanisms) are in section 3 (F01-F08). Legacy/disqualified items are in section 4 (C21-C25).

### 1b. Rule matrix (compact; evidence in each section)

| id | R1 size | R2 faithful | R3 exam | R4 no new parts | R5 teach | R6 controls | R7 real job | R8 base | R9 spirit | MC |
|---|---|---|---|---|---|---|---|---|---|---|
| C01 filed C2 | PASS 9.404% | FAIL 93.9% | FAIL | FAIL | FAIL | FAIL | FAIL letters | PASS | FAIL | FAIL |
| C02 fix_real | PASS* 9.93% est | FAIL 93.4% | UNTESTED | PASS*/GRAY (vocab trim) | UNTESTED | UNTESTED | PASS | PASS | GRAY | UNTESTED |
| C03 72B TOOLSEL | PASS* est | FAIL 95.55% | PASS* (lock plan, unbuilt) | PASS* | UNTESTED | UNTESTED | GRAY (name only, no args) | PASS | GRAY | UNTESTED |
| C04 72B INJ | PASS* est | FAIL 95.21% | PASS* (lock 2320 built) | PASS* | UNTESTED | UNTESTED | PASS | GRAY (S1_benign ceiling) | GRAY | UNTESTED |
| C05 Qwen3-32B T1 | PASS* est | FAIL 85.8% | PASS* | PASS* | UNTESTED | UNTESTED | PASS | PASS | PASS | UNTESTED |
| C06 72B T1 | PASS* est | FAIL 84.2% | PASS* | PASS* | UNTESTED | UNTESTED | PASS | PASS | PASS | UNTESTED |
| C07 32B VQ INJ | PASS* 9.85% est | FAIL 95.0% | PASS* | PASS* | UNTESTED | UNTESTED | PASS | GRAY | GRAY | UNTESTED |
| C08 32B NEWS | FAIL 10.03% | FAIL 96.1%, Tech 0.90 | PASS* (plan) | PASS* | UNTESTED | UNTESTED | PASS | PASS | GRAY (benchmark-shaped) | UNTESTED |
| C09 32B INJ | FAIL 10.03% | FAIL 95.7% | PASS* | PASS* | UNTESTED | UNTESTED | PASS | GRAY | GRAY | UNTESTED |
| C10 32B TOOLSEL/FIN/ROUTE | FAIL 10.03% | FAIL | FAIL for ROUTE (29 classes) | PASS* | UNTESTED | UNTESTED | GRAY | PASS | GRAY | UNTESTED |
| C11 7B VQ | PASS* 9.93% est | FAIL 96.2% INJ / 94.3% TOOLSEL | PASS* | PASS* | UNTESTED | UNTESTED | PASS | GRAY | GRAY | UNTESTED |
| C12 405B | GRAY 9.6-9.84% | FAIL 90.2% | PASS* | PASS* | UNTESTED | UNTESTED | PASS | PASS | PASS | UNTESTED |
| C13 235B MoE | PASS* est | FAIL 93.9% TF | UNTESTED | PASS* | UNTESTED | UNTESTED | PASS | PASS | PASS | UNTESTED |
| C14 30B MoE | FAIL 12.4% | FAIL 94.4% | UNTESTED | PASS* | FAIL (weak, 12% fixed) | UNTESTED | PASS | PASS | PASS | UNTESTED |
| C15 BitNet-2B | PASS 9.453% | GRAY (vs own base; lossy emb costs 9 pt on T1) | UNTESTED | PASS* | FAIL (PCH banned) | UNTESTED | FAIL letters | FAIL on T1 (.267); PASS on MDC-6 | GRAY | FAIL |
| C16 Falcon-E-3B | FAIL 13.8% lossless | UNTESTED | UNTESTED | PASS* | UNTESTED | UNTESTED | UNTESTED | UNTESTED | GRAY | UNTESTED |
| C17 Ternary-Bonsai-8B | FAIL 10.19% | FAIL if lossy (94.6-97.2%) | UNTESTED | PASS* | UNTESTED | UNTESTED | PASS | PASS | GRAY | UNTESTED |
| C18 Ternary-Bonsai-2-27B | FAIL 10.67% min | UNTESTED | UNTESTED | PASS* | UNTESTED | UNTESTED | UNTESTED | UNTESTED | GRAY | UNTESTED |
| C19 Bonsai-8B 1-bit | PASS 7.13% real (GRAY on baseline reading) | PASS vs itself / FAIL 88.9% vs parent | PASS* (nothing calibrated; lock not built) | PASS* | UNTESTED (FAIL for sign-flip edits) | UNTESTED | PASS (D1a) | PASS on T1 (.70); label tasks UNTESTED | GRAY-to-risky | UNTESTED |
| C20 MDC-6 7B | PASS 9.736% | FAIL | PASS* | FAIL (head) | GRAY | UNTESTED | FAIL letters | PASS | FAIL | FAIL |

## 2. Candidates (compression routes)

Each block: base | task and answer | compression | feedback | size | retention | general capability | rules | remaining work and cost | blockers | P(all 9).

### C01. Filed C2 7B artifact (letter head + LDA memory), issue #2
- **Base:** Qwen/Qwen2.5-7B-Instruct, 7,615,616,512 params, bf16 15,231,233,024 B (budget 1,523,123,302).
- **Task / answer:** C2 tool selection among 4 confusable tools plus "none fits"; answer = one of 5 option letters A-E read from 5 rows of lm_head (multiple-choice stand-in).
- **Compression:** block-128 Hadamard, GPTQ onto ECSQ scalar grid (step 1.20 x group rms, dead zone 0.15, g128, 8-bit log scales), rANS, vocab 24,000 chosen from C2 text, 4-bit embedding, lm_head cut to 5 letter rows (`v2/altpush/push.py`, `v2/compress_C/push_c2.py`, `PROOF_C2.md`).
- **Feedback:** `head.npz` (PCA64 + LDA, 476,506 B) + 852-byte JSON memory of per-tool prototype means and (true,pred) tallies (`v2/c2_fb`). 6 annotations (5 new tool, 1 confusable pair).
- **Size:** 1,432,413,553 B = **9.404%** real (`size_check.py`, 9 files + memory; torch/transformers excluded).
- **Retention:** LOCKED stage 1 (n=2000): base .7675, cand .799; both right 1441 of 1535 base-correct = **93.9% [92.6]** (recomputed; REDESIGN quotes 93.5% on dev). Dev: "none fits" class cand .64-.65 vs base .77. GEN? No: letter logits. Source: single (xLAM/apigen hard-distractor build), easy-ish.
- **General capability:** MMLU .467 vs .693, ARC .62 vs .895, HellaSwag .497 vs .797, TruthfulQA .458 vs .627 (`harness/validity/c2_general_result.json`); internal 90% gate FAILS.
- **Rules:** R1 PASS (`harness/receipts/REAL_final/stage1.json`: 0.09404). R2 FAIL (93.9%). R3 FAIL: n=2000 but no >=500 block from a never-used source and kinds not enumerated at 300 (`harness/redteam/REPORT.md` 4-8). R4 FAIL: fitted head + side memory; maintainer said so on issue #2 (REDESIGN section 0). R5 FAIL: teach set not pre-locked, 6 annotations, errors drawn from pool. R6 FAIL: no shuffled control. R7 FAIL: letter A-E. R8 PASS (.7675 vs chance .23). R9 FAIL. MC FAIL (R4).
- **Also true:** the single-use C2 locked exam was consumed (`receipts/REAL_final/stage{1,2}.claim`); stage 2 .802.
- **Remaining work:** none sensible; it cannot be re-examined. **Blockers:** R2, R4, R7 together, locked file burned. **P: ~0%.** Cheapest next experiment: none (keep as a negative reference).

### C02. fix_real: same 7B body, real tool-call generation, no letter head
- **Base / task:** Qwen2.5-7B-Instruct; C2 candidate tools, output = Qwen-native `<tool_call>` JSON generated greedily (name + args), original lm_head rows kept (vocab 24,176, 4-bit emb and head).
- **Compression:** same recipe as C01, calibrated on C2 generation transcripts (`q_mix`) or letters (`q_letter`).
- **Feedback:** none built.
- **Size:** 9.93% **est** (filed packed artifact came in about 2.3% under its estimate; unverified here).
- **Retention (DEV n=2000, GEN):** base tool-name acc .785, exact call .6345 (irrelevant 367 items: name acc .401). q_mix: name .760, exact .575, retention of base-correct **.934** (relevant .966, irrelevant .619). q_gen .915. q_letter (filed calibration) .147: calibration data format dominates.
- **General capability:** MMLU .445 / ARC .598 / HS .494 / TQA .450 vs base .693 / .895 / .797 / .627; ratios .64/.67/.62/.72, gate FAILS. Cause is the 1.74 bpw body.
- **Rules:** R1 PASS* (estimate). R2 FAIL (.934; irrelevant .619). R3 UNTESTED (the C2 lock is burned; would need a new build). R4 PASS* (no head, no memory; vocab trim is GRAY). R5-R6 UNTESTED (no mechanism). R7 PASS (real call). R8 PASS. R9 GRAY. MC UNTESTED.
- **Remaining work:** new lock + a feedback mechanism + a much better quantiser; $50+. **Blockers:** R2 (needs about 19-25% of fp16 by D_scale's fit for tool-call text). **P: ~0%.** Next experiment: none; superseded by C11 (VQ on the same 7B).
- Files: `v2/fix_real/RESULTS.md`, `results/*.json`.

### C03. Qwen2.5-72B TOOLSEL (tool name or "none"), V3 / V7 / V9 / V12
- **Base:** Qwen2.5-72B-Instruct (reference = RedHat FP8-dynamic in vLLM; bf16 does not fit one GPU), 72.7B params; fp16 bytes about 145 GB.
- **Task / answer:** given a tool menu and a query, generate the tool name or `none` (strict exact-string scorer; multi-call items dropped). No args, no letters, full lm_head.
- **Compression:** Hadamard-128 + GPTQ + L-level ECSQ grid, per-module Lagrangian bit allocation, block-influence layer weighting gamma=1, label-mix and/or self-distilled tool calibration (`research/D_scale/dense_stream.py`, `RECIPE_72B_TOOLSEL.md`).
- **Feedback:** none built at 72B (see F01/F02 at 4-14B).
- **Size:** 9.609% **est** (13.97 GB, entropy-accounted with 1.005 slack and 11-bit scale codes; NO written rANS artifact verified).
- **Retention (DEV GEN, strict scorer, n=877 base-correct of 1063 items):** V7 **95.55% [94.0]** (cells S1/none 183/205 = 89.3, S1/tool 97.6, S3/none 97.7, S3/tool 97.2); V3 93.96%; V9 94.30%; V12 (T1-only calibration) 91.11%. TF numbers (98.3-98.9%) do NOT transfer: V7 98.9 TF -> 95.6 GEN. Base acc 82.5%.
- **General capability:** not measured at 72B.
- **Rules:** R1 PASS* (estimate only). R2 FAIL. R3 PASS* (lock plan `combo/TOOLSEL/HASH_PLAN.json`, up to 1750 items, S2 = BFCL; not built). R4 PASS* (compression only). R5/R6 UNTESTED. R7 GRAY: a name-only router, reviewers may call it a reduced tool-use task (RULES/TASK_SEARCH call it "rule-7/4 baggage of C2"); also, in-menu headroom is nil (`combo/TOOLSEL/README.md`: lexical overlap baseline scores 98.8% on xlam tool items; all difficulty is the `none` decision). R8 PASS. R9 GRAY. MC UNTESTED.
- **Remaining work:** +2.5 pts overall and +5-6 pts in S1/none; no lever tested delivered it (D_scale section 5; everything else inside noise). Real rANS artifact + lock + feedback. About $120-250, 2.5 days. **Blockers:** R2; new-tool feedback fails (multi-token names, F02). **P: ~1.2%.**
- **Cheapest next experiment:** run C11's VQ quantiser on 72B (`quality/cfgs_72.json`) and re-score the same 1063 items (about $40).
- Files: `research/D_scale/REPORT.md` section 4b, `results/real_*.json`.

### C04. Qwen2.5-72B INJ (Injection / Benign), V3 / V7 / V9 / V12
- **Base / task:** Qwen2.5-72B-Instruct (FP8 reference); output = word `Injection` or `Benign` (the label IS the product: prompt-injection filtering). Tournament dev data (PromptShield S1, deepset+gandalf S3), 1566 items.
- **Compression:** as C03; V9 = V3 + 240 clean INJ-pipeline calibration items.
- **Size:** 9.61% est (same caveats).
- **Retention (DEV GEN, n=1379 base-correct):** V9 **95.21% [94.0]** (S1_benign 360/387 = 93.0; S1_injection 272/286 = 95.1; S3_benign 98.9; S3_injection 93.6); V7 94.13%; V3 93.47%; V12 91.66% (S1_injection 79.7). TF said 97.2 for the T1-calibrated one. Base acc 88.1%.
- **General capability:** not measured at 72B.
- **Rules:** R1 PASS* est. R2 FAIL. R3 PASS* (`tournament/INJ/LOCK.json`: 2320 locked items, every source x class cell >= 330, S2 JailbreakHub block 720; hash-locked, chmod 444, never scored). R4 PASS*. R5/R6 UNTESTED (no INJ feedback has passed anywhere: F02/F07). R7 PASS. R8 GRAY: S1 and S3 benign cells sit at 0.97-0.998 (ceiling). R9 GRAY. MC UNTESTED.
- **Remaining work:** +3 pts and 5-pt cell lift, then a working INJ feedback (the INJ `Jailbreak` new-label run on Qwen3-8B collapses one draw in ten, FEEDBACK_LEGAL section 4). $150+. **P: ~1.5%.**
- **Cheapest next experiment:** same VQ-on-72B run as C03, scored on the real INJ items.

### C05. Qwen3-32B T1 tool calls (D_scale)
- Base Qwen3-32B (bf16 vLLM), task = generated tool calls (T1 xLAM + S3). **Retention (DEV GEN, n=903 base-correct):** best 85.8% [83.4] (gamma=1 + self-distilled calibration, 9.6%); plain 74.6-76.2%; 12.0%: 89.4-91.1%; 15.0%: 93.6-94.8%. Binding cells S1/nocall 45-57%, S3/multi 75-84%. Extrapolated size for 98%: 19-25% of fp16 (3-point fit, +-3). Qwen3-14B 56.1%, Qwen3-8B 21.9% at 9.6%.
- Size 9.6% est. R1 PASS* est; R2 FAIL; R3 PASS* (T1 lock exists: `tournament/T1`); R4 PASS*; R5-R6 UNTESTED (T1 natural-mistake feedback FAILS: F02 section "argstr" gain +0.01); R7 PASS; R8 PASS; R9 PASS; MC UNTESTED.
- Remaining: needs a step change in the quantiser; cost >$100. **P: <0.3%.** Next experiment: C11's VQ at 32B on T1 items (about $20) is the only thing that would change the verdict.

### C06. Qwen2.5-72B T1 tool calls
- 73.9% (plain, legacy calibration), 82.6% (gamma=1, clean), 84.2% [81.7] best (gamma=1 + ncal256 + emb/head 8 levels), n=919 base-correct, 9.6% est; S1/nocall 47-55%. 2.2x the parameters bought nothing over 32B for tool-call retention. Base is the FP8 reference, so retention is relative to an fp8 base. Rules as C05. **P: <0.3%.**

### C07. Qwen2.5-32B, VQ-D8 K=5300, INJ (entry32b)
- **Base:** Qwen2.5-32B-Instruct. **Task:** INJ word label. **Compression:** random-sign + Kronecker/Hadamard incoherence, block-LDLQ, 8-dim shared Gaussian k-means codebook K=5300 (1.5465 bpw body), full-vocab embedding and head at 2.125 bpw (`entry32b`, `quality/qpush.py`; no gradient anywhere).
- **Size:** est total **9.85%** (not a written artifact).
- **Retention (DEV GEN, n=4313 items, 3377 base-correct):** base acc .783, cand .763; **ALL 95.0% [94.2]**, cluster-bootstrap lower 94.06%; by cell S1_benign 99.4, S1_injection 94.8, S2_benign 85.8 [81.8], S2_injection 91.0 [86.2], S3_benign 99.7, S3_injection **81.3 [76.2]**; class injection 91.3, benign 97.1; cand format failures 11. Note the item set includes the S2 audit slice (noisy community labels) and is NOT the 1566-item set used by C03/C04/C11, so it is not directly comparable.
- **General capability:** not measured. **Feedback:** none.
- **Rules:** R1 PASS* est. R2 FAIL (95.0; cells 81-86). R3 PASS* (INJ lock). R4 PASS*. R5-R6 UNTESTED. R7 PASS. R8 GRAY. R9 GRAY. MC UNTESTED.
- **Remaining:** +3 pts overall and about +15 in S3_injection/S2; INJ feedback; locked run. $60-150. **P: ~1.5%.** Cheapest next: see rank-1 row.
- Files: `entry32b/results/ret_e32_k5300.txt`, commit 1f05aac.

### C08. Qwen2.5-32B NEWS-topic router (TASK_SEARCH row 1)
- **Task:** article -> one word in {World, Sports, Business, Technology}; sources AG News (S1), BBC (S3 dev proxy), HuffPost News Category (S2 locked candidate, CC-BY-4.0). Exam plan: 8 cells x 400 = 3200 items, hash plan built (`task_search/NEWS/HASH_PLAN.json`), nothing locked or scored.
- **Retention (DEV, TF-equivalent exact-token criterion, 32B, n=1389 base-correct):** 10.03%: **0.961 [0.950]**, 11.28%: 0.962, 12.53%: 0.979. Class Technology 0.902 / 0.857 / 0.944 (fails the 95% per-kind gate at every point); BBC World 0.88. Base acc .868 (rule 8 OK). 14B 6-menu probe: 0.850 at 10.0%. 14B 4-menu exact build was not run (queued, cancelled).
- **Size:** 10.03% est = OVER the cap by 0.03 pt; the legal 9.6% was run on 7B/14B only.
- **Rules:** R1 FAIL (10.03%). R2 FAIL (Tech class). R3 PASS* on paper (AG 6.3k per class, HuffPost Tech 1555 eligible). R4 PASS*. R5/R6 UNTESTED (Entertainment new-label plan `fb_annot_25` / `fb_teach_unseen_200`; no mechanism built). R7 PASS (word out). R8 PASS. R9 GRAY (benchmark-shaped; AG/BBC licences are research-only). MC UNTESTED.
- **Remaining:** quantiser improvement of 2-3 pts at 9.6%, feedback, lock. $30-80. **P: ~0.8%.** Next: run the C07 quantiser on NEWS at 9.6% (about $15).

### C09. Qwen2.5-32B INJ-real (tournament dev data), scalar GPTQ (TASK_SEARCH row 2)
- 32B: 10.0% **0.957 [0.945]**, 11.3% 0.958 [0.948], 12.5% 0.942 [0.929] (non-monotone = draw noise +-2). Cells at 10.0%: S2_benign 0.78 (base itself only 0.42-0.69 correct there), S1_injection 0.945, S3_injection 0.952. 14B: 0.912 / 0.954 / 0.968. **The INJ-easy probe (safe-guard + jackhhao) reads 0.981 / 0.989 / 0.989 on 32B: easy-source only, upper bound, dropped from the lock for licence and style leakage.** Size 10.03% est (over cap). Rules as C07 with R1 FAIL. **P: ~0.7%.** Superseded by C07.

### C10. Qwen2.5-32B TOOLSEL, FIN, ROUTE and the rest of task_search (TASK_SEARCH rows 3-12)
- At 10.03% on 32B (est, over cap): TOOLSEL 0.928 [0.903] (11.3%: 0.967; S3 half retains worst); FIN 0.937 [0.913] (11.3%: 0.971, 12.5%: 0.975; narrow finance domain, thin second source); ROUTE 0.947 (worst class 0.75; 29 classes, 300 per class = 8700 locked items, MASSIVE has about 190 per class: R3 infeasible); TOX 0.911; SENT 0.905 (toy, R1/R9); BTSAFE 0.894; LANGID 0.885 (base 0.988 at ceiling, Portuguese collapses: R8 fail); INTENT 0.923 (60 classes: R3 fail); HALU 0.870; EMOTION 0.825 (fear/surprise collapse: R3 fail).
- All FAIL R2 at <=10%; 14B and 7B are worse (7B best INJ-easy 0.90; no 7B task above 0.96 with class collapse elsewhere). Lesson (TASK_SEARCH section 0): task choice moves retention about +-2 pt; bits and base size dominate; even the most confident items lose 4% (32B) to 18% (14B). **P: <0.5% each.**

### C11. Qwen2.5-7B VQ-D8 (QUALITY_PUSH), INJ and TOOLSEL, real generation
- **Base:** Qwen2.5-7B-Instruct. **Tasks:** real INJ (1566 dev items) and real TOOLSEL (1063), strict scorers, GEN; base acc INJ .873, TOOLSEL .808. **Compression:** the C07 quantiser at 7B: incoherence + block-LDLQ + shared 8-dim Gaussian k-means codebook, sensitivity-based allocation option, no gradients (`quality/qpush.py`, `results/*.json`).
- **Retention (recomputed from per-item JSON vs `base7b.json`; n=1367 INJ, 859 TOOLSEL base-correct):**

| config | est size | INJ | worst INJ cell | TOOLSEL | worst TOOLSEL cell |
|---|---|---|---|---|---|
| scalar K=3 (GPTQ-class), 1M calib tokens | 9.00% | 70.1% | 0.52 (S3_benign) | 79.9% | 0.62 (S1/none) |
| VQ K=2^13 (1.625 bpw) | 9.38% | 92.2% | 0.83 | 93.8% | 0.84 |
| VQ K=14500 (1.72 bpw) 1M | 9.93% | 94.9% [93.6] | 0.913 | 93.6% | 0.80 |
| VQ K=14500, 300k calib tokens | 9.93% | **96.2% [95.0]** | 0.938 (S1_injection) | 92.3% | 0.77 |
| VQ K=14500, damp .003 | 9.93% | 92.0% | 0.884 | **94.3% [92.5]** | 0.84 |
| VQ K=2^16 (2.0 bpw) | 11.38% | 94.5% | 0.873 | 94.5% | 0.87 |
| VQ with error-propagation target (tgt) | 9.90% | 0% (model destroyed) | - | 0% | - |

  Reading: VQ-D8 buys about 20-25 pts over scalar GPTQ at the same bytes on a 7B (the best 7B scalar result in task_search was INJ-easy 0.90). The spread across calibration-size/damping variants is 92-96%, i.e. the same noise scale as the claimed gain; the best row is a max over about 12 configs on the same dev items (optimism 1-2 pt). The two tasks peak at different configs.
- **General capability (MC, n=250-ish per set, `base7b_summary.json`):** base MMLU .678 / ARC .906 / HS .812 / TQA .664, ppl 8.29. K=14500: MMLU .546, ARC .77, HS .56, TQA .486, ppl 12.6, KL .54. Fails an internal 90% gate by a wide margin.
- **Size:** est (ratio_ent, includes fixed bytes 100 MB for embedding/head/misc as coded in qpush); no written artifact.
- **Rules:** R1 PASS* est. R2 FAIL (INJ 96.2 vs 98; TOOLSEL 94.3; cells 77-94). R3 PASS* (INJ lock 2320 / TOOLSEL plan). R4 PASS*. R5/R6 UNTESTED. R7 PASS. R8 GRAY (INJ benign cells ceiling). R9 GRAY. MC UNTESTED.
- **Remaining:** another +2 pts overall and +5 in cells (bigger base is the lever: C07 at 32B did not beat 7B on a different item set, so unclear), feedback, lock, a real written codebook decoder (the codebook adds bytes: K=14500 -> 232 kB; decoder code must be counted). $20-60. **P: ~2%.**
- **Cheapest next experiment:** 14B and 32B with the K=14500 recipe on exactly these 1566 + 1063 items (about $20-40); this decides whether VQ scales, which C03/C04/C07 depend on.

### C12. Llama-3.1-405B-Instruct INJ (L3)
- Base 410.08B params (16 KV heads), checkpoint 820.16 GB; GPTQ+Hadamard128+ECSQ at t096 (body 1.531 bpw). Size **est 78.74 GB = 9.601%** of 820.16 GB; 9.72% of nominal 810 GB; with plain fp16 scales (16 bit) 9.84% / 9.97% (scale accounting matters; real written bytes never measured; artifact is 4-bit-packed 203 GB on the Volume).
- **Retention:** TF (label token) INJ 99.7% [98.7], SENT 97.7, TOOLSEL 97.5, FIN 96.6. **REAL INJ exam inputs, strict GEN scorer (1566 dev items): 90.2% [88.5, 91.6]**; injection class 78% (S1 78.3, S3 78.6), benign 99.9%; base-vs-base noise floor 0/1566 flips. The TF figure came from easy safeguard_pi items with large margins; real injection margin median 4.7 logits. 
- Rules: R1 GRAY (9.6-9.97% depending on scale accounting), R2 FAIL, rest as C05. Eval needs 2 H200 or layer streaming (about $2 per pass). **P: <0.3%.** Dead. Files `research/L3_405B/REPORT.md`.

### C13. Qwen3-235B-A22B-Instruct-2507 REAP prune + low-bit experts (L2)
- 470.19 GB fp16; byte table (est): prune 37.5% + 3-bit experts 9.88% (9.46% with attn/emb 4-bit), prune 50% + 3-bit 8.17%, none built. **Retention (TF proxy, 360 dev T1 items, 231 base-ok):** 37.5%+GPTQ3 0.939 [0.90, 0.96]; 50%+GPTQ3 0.931; 50% RTN3 0.926; all-expert RTN4 (21.98%) 0.983; prune 37.5%+RTN4 (14.23%) 0.978. Multi-call S3 is the binding cell. Label tasks (TF): prune50+RTN3 at 8.17%: INJ .896, SENT .979, TOOLSEL .972, FIN .938, ROUTE .941. Base real acc S1 .853 / S3 .730 (R8 OK). Frequency-based pruning collapses (0.459); REAP is required. Retention is limited by number of experts removed, not bits (50->62.5% prune costs 10 pts in bf16).
- Attn/embed/head were bf16 in the retention runs (budgeted at 6 bit, unmeasured). No artifact built. Feedback (30B, REAP50, whitened lm_head edit): fixes 12% of held-out mistakes at alpha .02 with shuffled control 0%; larger alpha breaks neighbours.
- Rules: R2 FAIL at every <=9.9% point; others UNTESTED. **P: <0.3%.** Next (only if someone wants to close it): a 2-bit VQ expert codec at 50% prune (about $40-80, needs a 4xGPU box).

### C14. Qwen3-30B-A3B-Instruct-2507 REAP (L2 and v2/moe)
- fp16 61,064,245,248 B. **Tool calls GEN (240 dev items, 179 base-ok):** REAP50 bf16 0.966 [lo .93] (size about 50%, not a candidate), REAP50+RTN4 physically packed 0.944 (about 12.4% size per `moe/RESULTS.md` f0.5 b4 ratio .124), REAP62.5 bf16 0.911. TF: REAP50 .939, freq50 .840, random50 .436 (format collapse). GPTQ at 3-bit all experts 0.957 (size about 14%).
- **MMLU (v2/moe, legacy task, dev n=1987, base .8078):** best at <=0.10 ratio **0.7323** (GPTQ, 70% experts, 2-bit; -7.5 pt), 45% at 3b 0.6376; quantisation-only at 3 bit costs -2.6 pt at 0.185 ratio. 1-bit collapses (0.45).
- Rules: R1 FAIL for every point that retains >0.94; R2 FAIL; R5 FAIL (weak feedback). **P: <0.3%.**

### C15. BitNet b1.58 2B-4T (`v2/lowbit`)
- **Base definition (gray):** Microsoft's natively ternary model, 2,412,820,690 logical params; denominator = bf16 master 4,825,641,380 B. Two forwards exist (bf16 master vs packed ternary with int8 activation); dev: MMLU .5229 vs .5264, MDC-6 .8532 vs .8581.
- **Artifact (real):** ternary body rANS 392,849,272 B (1.508 bpw), embedding 42,086 kept rows at 4-bit, letter head, tokenizer etc.; total **456,177,599 B = 9.453%** (`lowbit/ARTIFACT.md`; with PCH feedback head 9.465%). Lossless is impossible (tied 328M-param embedding alone = 13.6% in bf16; lossless total 21.8%).
- **Retention (DEV):** MDC-6 (6 public classification benchmarks as lettered options) agree .988, MMLU agree .919 (letters). T1 probe: BitNet bf16 master .267 / packed .300 accuracy (33-38% unparseable) -> R8 FAIL on tool calls. Lossy embedding re-encode alone on T1 probe (n=180 base-correct, +-3-4 pt): 4-bit .906, 2-bit .578.
- **General capability:** ability unverified for the vocab-trimmed artifact.
- **Rules:** R1 PASS 9.453% (real). R2 GRAY on MDC-6 (agreement, not item retention vs a locked set) / FAIL on T1. R3 UNTESTED. R4 PASS* for the artifact; feedback head FAIL. R7 FAIL (letters). R8 PASS on MDC-6, FAIL on T1. R9 GRAY (no compression by us beyond the vocab/embedding steps). MC FAIL.
- **Remaining:** a non-letter, non-narrow task where a 2B ternary base is above chance, plus an R4-legal feedback. Probably unreachable. **P: ~0.8%.**

### C16. Falcon-E-3B / 1B (tiiuae, ternary)
- Falcon-E-3B-Instruct: 3,053,586,432 params (2.92B ternary + 134M bf16 embed/head); fp16 6,107,172,864 B. Measured ternary entropy 1.5687 bpw = 572.4 MB; non-ternary 268.7 MB bf16; **lossless total about 843.5 MB = 13.8%** (`NATIVE_LOWBIT.md`). The earlier `lowbit/ARTIFACT.md` figure of 0.0997 ("tight") assumed lossy embedding/head and was never dev-verified; B_native's "9.97-10.09% unverified" is also superseded. Falcon-E-1B: 31% (hopeless). MMLU dev Falcon-E-3B-Instruct .561.
- Rules: R1 FAIL (lossless), any lossy step is a likely R2 FAIL (certain at 2.1-2.3 bpw all-in). **P: <0.3%.** No experiment worth running.

### C17. Ternary-Bonsai-8B (PrismML, derived from Qwen3-8B)
- Ideal entropy-coded lossless pack **10.19%** (31.3 MB over the cap); 2-bit packing 12.5%; maker's Q2_0 13.3%. Retention vs parent Qwen3-8B on T1: 91.0% (S1 96.0, S3 85.0). Lossy embed/head re-encode to 1 bit: embed .972, head .956, both .946 (n=429 base-correct, T1 probe; two miss the 95% per-kind gate). T1 dev .715 (Qwen3-8B reference .763). Rules: R1 FAIL by 0.19 pt unless a lossy step is taken, which then fails R2. **P: <0.3%.**

### C18. Ternary-Bonsai-2-27B (derived from a 27B parent)
- 26.87B ternary params; trit frequencies .3365/.3276/.3358, entropy 1.5848 bpw (= log2 3, zero entropy-coding slack); group scales (419.8 MB) cannot be dropped. **Minimum 5.743 GB = 10.67% of the 53.81 GB fp16**; best released file 11.05%. Vendor claim "98.2% of FP16 intelligence" is an average-benchmark metric, not item retention; item retention vs parent on T1 was NOT measured. Needs the Apple-MLX Hadamard runtime or a custom llama.cpp fork (custom kernels would count toward bytes). R1 FAIL. **P: <0.3%.** (NEXT_PLAN D2.)

### C19. Bonsai-8B 1-bit (PrismML), lossless repack  [CONDITIONAL on the maintainer's reading]
- **Base:** `prism-ml/Bonsai-8B-unpacked` (Apache-2.0), 8,188,548,096 params, every one of 254 matrices incl. embedding and untied lm_head is exactly +-scale per 128-group, fp16 checkpoint 16,377,141,728 B (identical to Qwen3-8B's fp16 size, which PrismML's NOTICE says it was built from).
- **Compression:** none by us: lossless pack (1 bit + fp16 scale = 1.125 bpw) + tokenizer/config + a loader; the loader unpacks to fp16 and runs the SAME HF forward as the reference.
- **Size:** body-only bit-exact pack 1,154,719,808 B = 7.05% (B_native); **shipped tree 1,168,021,073 B = 7.132%** real (`native_lowbit/bonsai_gate1_result.json`). Slack to cap about 2.9 pt (hosts fp16 patch rows).
- **Retention:** vs itself: Gate 1 PASS, **2000/2000 identical greedy token sequences** (600 T1 generations up to 300 tokens + 1400 TOX/INJ dev), logits bit-equal on 8 prompts, DEV only. Vs the parent Qwen3-8B (if the parent is "the model you started from"): **88.9% overall** (S1 96.4, S3 79.6, multi 84.7) = FAIL; ternary Bonsai 91.0%.
- **Base accuracy (rule 8):** T1 dev .700-.703 (S1 .823, S3 .583), Qwen3-8B reference .763 -> R8 OK for tool calls. TOX/INJ label-task probes (Gate 2) have NO result (v2/bonsai probe untracked and unfinished).
- **Feedback status:** the lm_head of a 1-bit net has no small step. `fb_legal` (3c): mnH + flip-only re-binarization, z-score greedy flips, hebb + binarization all fail to give a selective edit (teach 0.00 with neighbours 0.78-0.90, or teach 1.00 with shuffled also 1.00 and neighbours 0.00). The fp16-patch-row carrier (25 rows x 8 KB inside the 2.9% slack) was launched (`fb_legal/cfgs8.json: bonsai_fp16patch`) and has **no result in the repo yet**: UNTESTED.
- **Rules:** R1 PASS 7.132% real, but GRAY on the baseline reading (see Q1). R2 PASS vs itself by identity (100% on dev), FAIL vs parent (88.9%). R3 PASS* (nothing calibrated on any data; the lock is not built). R4 PASS* for the base; the feedback carrier is GRAY (F01). R5/R6 UNTESTED (FAIL for the sign-flip carrier). R7 PASS for D1a tool calls; D1b label tasks PASS only if the label is the product. R8 PASS on T1, UNTESTED on labels. R9 GRAY-to-risky: "the entrant did no compression; the 1.15 GB form is the authors' own release".
- **The three questions to the maintainer (from `research/B_native/REPORT.md`, NOT POSTED, needs user OK; it is a new comment on issue #1, not on the filed proof issue):**
  Setup: model M (>0.5B) released by its authors with an fp16 checkpoint F whose weights are exactly low-bit valued (every weight of every matrix, embeddings and lm_head included, is +-s or {-s,0,+s} per group of 128, so a lossless pack of F is bit-exact). M was trained by its authors (e.g. PrismML's Bonsai-8B, derived from Qwen3-8B), not by me. I would ship a lossless re-encoding of F at 7% of F's size, no extra parts, same forward pass.
  1. Is the size baseline for "at most 10% of its fp16 size" the fp16 checkpoint of M itself (F), even when M's authors also publish a packed form already below 10%? yes/no
  2. Is "the model you started from" (and the model whose score I must match) M, even though M was derived by its authors from another model (Qwen3-8B)? yes/no
  3. Does a lossless repack of a native low-bit model count as valid compression under the spirit rule, or must the entrant reduce a model that was not already <=10% in released form? yes (valid) / no (entrant must compress)
  Rule: if 1 and 2 are yes but 3 is no, do not submit this route; if any answer is no, drop it.
- **Remaining work:** (i) maintainer answers (the route is dead if any is no); (ii) G1 identity on the final deployment path (done on dev, 2000/2000); (iii) rule-8 probe on the label task chosen (about $2, 1 h); (iv) a feedback carrier that works on 1-bit (fp16 patch rows; $5-15, 1 day; kill if it does not pass by 11-09); (v) lock, stage runs, controls, proof (1 day). New money $15-30.
- **Blockers:** the base-identity/R9 reading (not measurable), and feedback on a 1-bit net (unproven; the only tested carrier family failed). NEXT_PLAN rated this 25-30% before FEEDBACK_LEGAL section 3c landed; I rate it **3%**: P(maintainer accepts base-identity and no-compression) about 0.3 x P(feedback passes R4/5/6 on this net and the maintainer accepts it) about 0.15 x other rules ok about 0.7.
- **Cheapest next experiment:** the fp16-patch carrier run on Bonsai TOX+selfharm / T1 (existing config, about $5), plus posting the three questions.

### C20. MDC-6 7B letter-head artifact (altpush) + lowbit_fb PCH feedback
- Qwen2.5-7B-Instruct, task MDC-6 (6 public classification benchmarks: ag_news, dbpedia, yelp, amazon, emotion, offensive) as lettered options; ECSQ c=1.15 dz=0.15, vocab 24k, 4-bit embedding, 14 label-letter rows, per-task logit bias (Adam on calib, <=14 floats). **Real size 1,482,962,994 B = 9.736%** (9.741% with feedback files). Dev n=2009: raw .8731 vs base .8721, with bias .8820 vs base-with-bias .8890, agreement with base .954. PCH feedback (746,644 B head fitted on 3600 calib rows; off at k=0): errors fixed 10-40% on ag_news/dbpedia/amazon, ~0 on emotion/yelp at large k; offensive loses retention (.90-.94); new-label dbpedia/Film recall .46-.93 at k=1-25.
- Rules: R4 FAIL (fitted head + count memory, same class the maintainer rejected on #2), R7 FAIL (letters), R2 FAIL (agreement .954 is a ceiling on retention), R9 FAIL (public benchmark classification stack). This is the DECISION.md fallback; keep as a reference. **P: ~0%.** BitNet-2B variant is C15.

### C26. GIANT_MOE: DeepSeek-R1/V3, Kimi-K2, Qwen3-235B/480B, GLM-4.5, Maverick, gpt-oss-120b at ~1.5 bit experts   (added by the giant-MoE lane, merged from commit 3d477b2)
- **Base / task:** various giant MoEs; no task fixed. **Compression:** experts must get 1.4-1.55 bit all-in. **Size:** byte math only: best community 1.58-bit file (R1 UD-IQ1_S) is 10.44%, all others 11.8-18%; needs 10.0. **Retention:** none measured. **Rules:** R1 FAIL on paper for every published file; others UNTESTED. **Cost:** no GPU run; probe about $130 exceeds the cap. **P(all 9):** <0.3% (V=3). **Verdict:** NO-GO on paper, $0 spent. Doc: `v2/GIANT_MOE.md`. **Cheapest next experiment:** none; see C13 for the measured MoE data.

## 3. Feedback mechanisms (sub-registry; combine with any compression candidate)

Common facts: no feedback mechanism has zero R4 risk (NEXT_PLAN ladder). The mechanism must also be run on the real compressed artifact, with the harness-v3 protocol (`harness/v3/PROOF_RUN.md`: real vs shuffled vs empty arms, teach >= 25 pre-locked, neighbours >= 400 verified). P here is P(R4+R5+R6 judged OK on a base where it applies), not P(all 9).

### F01. mnH: closed-form minimum-norm lm_head row edit in the calibration-Hessian metric (`fb_legal`)
- **Mechanism:** teacher-force the annotated answer; at each position where the argmax differs from the target token t, delta_t = M^-1 H^T (H M^-1 H^T + lam I)^-1 g, M = E[hh^T] of calibration inputs, margin m = 8; rows of the EXISTING lm_head rewritten in place; 0 bytes added; no gradient; no stored annotations.
- **Results (DEV, TOX with new label `selfharm`, <=25 annotations of the base's own errors, 56 teach items, 10 draws, mostly TF with greedy-agreement check 0.98-1.00):** Qwen3-8B teach .49 / .63 / .65 at n = 5 / 12 / 25 (real minus shuffled +.47 / +.63 / +.57), wrong-label 0.00, empty 0.00, neighbours kept .9996 / .9992 / .9990, worst draw .996, min cell .993. Qwen2.5-7B .65 / .78 / .75 (real minus shuffled +.62 / +.66 / +.68), neighbours .9992-.9981. Survives 3/4-bit lm_head re-quantization (teach .376 vs .401 fp). Strict mixed-label gate fails at n=5 (+.14) and by saturation at larger n.
- **Failures:** Qwen3-8B INJ + `Jailbreak`: teach .2-.45, shuffled .02, but one draw in ten collapses the model (neighbours 0.00; mean .82-.90) via an edited continuation/EOS row (fix proposed, not run). 1-bit Bonsai: no selective edit (C19). T1 argument-format mistakes (argstr/argkeys): gain +0.01 or non-specific (F02). Only gains of 0.5-0.8 absolute, not 1.0. It is a TAXONOMY EXTENSION (new label), not a natural argument-format fix.
- **Rules:** R4 GRAY (weights are NOT frozen: 2-3 rows change; "computation fitted to data" in the weak sense; M needs calibration texts shipped, <=0.5 MB, not yet counted). R5/R6 PASS on dev for TOX+selfharm; UNTESTED on any compressed artifact, harness-v3 only smoke-tested (commit dc3220a, Qwen3-1.7B, T1). MC UNTESTED.
- **Remaining:** ship as `feedback.py`, 10+ draws on the real compressed artifact, add Qwen2.5-7B INJ/SENT, fix the INJ collapse by adding continuation positions of all old-label outputs to M. About $10-20. **P(component): ~15-20%.** Cheapest next: the INJ-collapse fix on Qwen3-8B (existing `q3_inj_b` config + new M, about $3).

### F02. Class-mean / LDA / calibrated lm_head row edits (`feedback_v3`, `combo/*`, `tournament/*/feedback`)
- Variants: `lmhead_lda` (generic mu/Sigma from unlabeled text, dev-tuned alpha: GRAY), `lmhead_plain` (fixed alpha .03/.01, no covariance: cleanest, but no gain on multi-token or rare tokens), `lmhead_cal/cal2` (per-token alpha from annotation margins), `memit_down` (3 late MLP down_proj ridge updates: neighbours 0.57-0.94, FAIL).
- Results (Qwen3, DEV, 10-20 draws): synthetic `_v2` twin-tool pair on T1 passes (8B lda +0.27, neighbours .995; cal2 +0.57) but it is not a natural mistake (red-team). Natural T1 argument mistakes: FAIL (argstr gain +0.01; argkeys non-specific, random-tool control matches). New multi-token T1 tool: cal +0.86 gain but neighbours .536. TOX+selfharm: 4B lda +0.65 neighbours .981 (thin). TOOLSEL new-tool-as-label (name not in menu): 8B/14B natural-name `cal` passes G2 at n = 3-12 (teach .60 at n=8, .75 at n=25, N1 .994-.998), real-mixed fails at n=25 by saturation; `lda`/`plain` fail; 4B leaks onto `none`. INJ + new label: 8B `cal2` passes at n >= 8 (teach .34 / .37 / .48, neighbours .99), `cal` leaks (neighbours .71-.83), `lda` passes only without the mixed control; 4B 0.00; `tournament/INJ` first-token edit FAILS at 1.7B/4B (gain equals shuffled; neighbours .54-.63).
- Rules: R4 GRAY/FAIL for lda and cal (fitted statistics); `plain` is clean but does not teach. The lm_head covariance edit is "a few-shot LDA classifier written into output rows" (red-team finding 3); unruled by the maintainer. **P(component): ~8%.** Next: none new; F01 supersedes.

### F03. Annotation-aware requantization (feedback as extra GPTQ Hessian weight)
- Most defensible on R4 (a compression technique, maintainer q2) but the analysis in FEEDBACK_LEGAL 3a shows it only re-weights rounding error and cannot install a label the model ranks about 30 logits below; no fake version run. The honest variant is mnH followed by re-quantizing the touched rows at the same bit width (+0 bytes), which is F01 in disguise. **P(component alone): ~0%.** Cheapest next: none.

### F04. LDA/prototype head + running-mean memory (C2 `head.npz`+JSON; lowbit_fb PCH; feedback2 PCH)
- Works behaviourally (C2: new tool .855 -> .927 at k=1, 96 tools; confusable pair 92-100% of unseen pattern errors fixed; retention loss <=0.2 pt; MDC-6 7B new-label Film recall .46 -> .93). But it is exactly "side memory the system looks answers up from / classifier fitted to data": R4 FAIL as written (maintainer ruled on issue #2; GOAL_README R4). **P(component): ~0-2%.** Kept only as the reference upper bound on what unrestricted feedback can do.

### F05. Retrieval / kNN / lexical memory (`v2/feedback`, `trackB`, `trackL`)
- Gated kNN memory over frozen mid-layer hidden states (MMLU stand-in 0.5B), SST-2 lexical wordpiece perceptron memory (trackB), residual ridge heads, kNN of pool disagreements (trackL). They teach, some well (lexical memory works on BERT SST-2), but all are retrieval/side-memory; icl variants put text in the prompt (banned). R4 FAIL; maintainer q3 says no RAG. Dead. **P: 0.**

### F06. Taxonomy-extension feedback as a task design (INJ +Jailbreak, NEWS +Entertainment, TOX +selfharm, TOOLSEL new-name)
- Not a mechanism but the way to make a label task "have something to teach": prompt lists old labels only; the user's annotations introduce a new label word. Measured only with F01/F02 (above). NEWS Entertainment: data prepared (`task_search/NEWS/fb_annot_25`, `fb_teach_unseen_200`), nothing run. Risk: R5 wants "real mistakes the way a user would"; a new-label error is a real mistake, but "nearby exam items" definition is on us. **P(component): inherits F01.**

### F07. Sign-bit-flip edits on native 1-bit weights (Bonsai)
- Tested in FEEDBACK_LEGAL 3c (flip-only re-binarization of mnH, z-score greedy flips, hebb+binarize): no selective edit found. **P: ~0%**; the fp16 patch-row carrier is untested (C19).

## 4. Earlier, disqualified, or measured-negative routes (kept for completeness)

### C21. Qwen3-8B / 14B dense label tasks (`research/C_tasks`, Hadamard+GPTQ+ECSQ, TF)
- 14B at 9.6%: INJ 0.958 [0.938, 0.972] (easy safe-guard items), TOOLSEL 0.883, TOX 0.828, FIN 0.778, SENT 0.801, SPAM 0.708; 15%: TOOLSEL 0.992, FIN 0.987, SENT 0.986, INJ 0.980. 8B at 9.6%: 0.41-0.80. Margins are large (median 14-22 logits) but the quantisation noise std at 1.45 bpw (13 logits on 8B, 7 on 14B) cancels them. Not an entry; these numbers were later shown to be TF and easy-source. **P: ~0%.**

### C22. Legacy MMLU/BERT-era tracks (A-S, v2/engine, vq, prune, entropy, big)
- trackA-L: BERT-base SST-2, 109M params < 0.5B and fp32 base: DISQUALIFIED by R1 (trackA 9.76% of fp32 = 19.5% of fp16). trackH XLM-R-base sentiment 278M, trackI larger fine-tuned classifiers (DeBERTa-v3-large 435M etc.; GPTQ4 .125 ratio, agree .99) and trackR Qwen2.5-0.5B (494M, just under 0.5B; MMLU best ratio_bf16 .42 at acc .35 vs .466) are also below the size floor or off-target. Methods that carried over: GPTQ with Hadamard, vocab trimming, rANS (entropy coders beat plain histogram by <0.5%, `entropy/RESULTS.md`), calibration-bias fixes (trackJ), external correction (trackL).
- Qwen2.5-14B MMLU (base .8007, n=1987): uniform L=3 ratio .0968 acc .5209 (engine); VQ-D8 1.65 bpw .0961 acc .6195 (vq); layer drop + L5 .1172 acc .7031; removal-only prune never beats -0.4 pt for -4% bytes; delta coding dead (adjacent layers orthogonal); MoE Qwen3-30B-A3B GPTQ 2-bit .732 (C14). `v2/big` (32B/72B MMLU) produced no numbers. AI-vs-human (MAGE) base at chance; PII (task A, 14B u3 g256 .0935 acc .747 vs .747) letter-style span category task.
- trackQ (vision classifiers) and trackP/M (BERT artifacts, stand-in exams) are harness/method studies, not LLM >0.5B entries. All R1/R7/R9 FAIL for the actual challenge. **P: 0.**

### C23. Llama / Mistral / Gemma / GLM bases (`research/L1_large`)
- Llama-3.3-70B (fp8 proxy) 38%, Mistral-Small-24B, Gemma-3-27B, GLM-4-32B 14% under OUR scorer/template: probe-harness artefacts (tools not rendered, stringified-JSON multi-calls), not evidence about the models. No quantization run; R8 not established. A fair probe needs per-model prompt work. Yi-1.5-34B unprobed. **P: unknown, not tested; weight 0 until probed.** Cheapest: a per-model prompt probe on 300 items (about $5).

### C24. TOX content moderation (`tournament/TOX`)
- Lock exists (4520 items, 10 cells, S2 = Aegis 2.0). Base macro acc .45-.66; S1/sexual cell unusable (base .02-.165); several S3 safe cells fail R8. Compression: 14B 82.8% at 9.6%; 72B pure-label calibration 95.1% TF (worst cell .93). Feedback: first-token edit weak (1.7B teach .10, neighbours .82). Needs rebuild without the sexual cell. **P: <0.5%.**

### C25. NEXT_PLAN designs mapped to ids
- D1 -> C19. D2 (native low-bit student with >=98% parent retention) -> C18 (size NO-GO, parent retention unmeasured; cheapest check proposed there: T1 parent retention of the 27B, $8-15, but size already fails). D3 (provenance-clean from-scratch ternary + vocab trim) -> C15/C16. D4 (conventional PTQ of a big dense model on a label task, all levers stacked, feedback rung 3/4) -> C03/C04/C07/C11.

## 5. Cross-cutting facts that affect every row

- The only measured way retention reaches 98% is more bits: 32B tool calls 76 -> 89 -> 95% at 9.6 -> 12.0 -> 15.0% (fit: 98% at 19-25% of fp16). Label tasks on 32B reach >=98% at 12-15% and at 12.5% on 14B; none at <=10%.
- Numerics floor: re-running the unquantised bf16 base with different batching keeps only 99.3% of its own base-correct items (D_scale); to certify "Wilson lower bound >= 98%" at n=500-900 the observed retention must be >= 99.0-99.3%. 405B showed 0/1566 flips in INJ GEN, so the floor is task- and engine-dependent.
- Per-kind gates are underpowered at 300/kind: certifying >= 95% with Wilson needs about 97.7% observed.
- General-capability numbers (MMLU/ARC/HS/TQA) are NOT in the nine judging rules; they are an internal validity gate and a possible R9 argument. Every 1.6-1.9 bpw artifact measured loses 15-45% of MMLU accuracy (C01 .467, C02 .445, C11 .546 vs ~.68-.69). A Bonsai-8B-type native model is the only candidate whose general behaviour equals its base.
- The T1 (tool call) locked exam, INJ (2320), TOX (4520) and TOOLSEL/NEWS plans are intact and unscored; the C2 locked exam is burned.
- Already filed: issue #2 (C01). The maintainer's rulings so far: issue #1 (disk size vs fp16 equivalent OK; no weight retraining; no RAG/context editing; examples agentic/tool use, PII redaction, AI-vs-human detection) and issue #2 (head + prototype memory is a fitted part).

## 6. Add a candidate (template)

Copy this block, fill every field, keep honest status codes, append under section 2 (compression) or 3 (feedback), add a row to section 1 and 1b, and commit by path (`git add v2/SOLUTIONS.md`, run `scan_leaks.sh paths v2/SOLUTIONS.md` first). Do not edit other agents' blocks; add a dated "UPDATE" line under a block when a new result changes it.

```
### Cxx. <short name>   (added YYYY-MM-DD by <lane/agent>)
- **Base:** <HF id, params (>0.5B), fp16 bytes>.
- **Task / answer format:** <task, exact output (label word / generated JSON / letter), scorer>.
- **Compression method:** <recipe, no-gradient statement, calibration data and whether it overlaps any eval source>.
- **Feedback mechanism:** <F-id or description, bytes added, what changes, ruling risk>.
- **Size:** <bytes / fp16 bytes = x.xxx%; REAL written dir via size_check.py or EST; what is counted/excluded>.
- **Retention:** overall <k/n = x% [Wilson or bootstrap lo]>, worst class/cell <x% (n)>; DEV or LOCKED; TF or GEN; easy-source or real-source; base acc; link to per-item json.
- **General capability:** <MMLU/ARC/HS/TQA or "not measured">.
- **Rules:** R1 <S> <evidence + link> | R2 | R3 | R4 | R5 | R6 | R7 | R8 | R9 | MC   (S in PASS / FAIL / GRAY / UNTESTED; PASS* = on paper)
- **Remaining work and cost:** <steps, $ and days>.
- **Blockers:** <list>.
- **P(all 9 as written):** <x%> (V = 1/2/3), reasoning in one line.
- **Cheapest next experiment:** <what, cost>.
```

Update log (append): 2026-11-08 registry created; entries C01-C25, F01-F07 from repo state at commit eefb8f6 plus untracked `v2/bonsai/` (probe only, no result).
| GIANT_MOE research | Qwen3-235B own quantizer, experts ~1.5 b (literature + community PPL/KLD, no GPU) | needs 10.0 | MoE ~+3 pts of budget better than dense but 1.57 b MC-MoE -17% task acc, R1 IQ1_S_R4 10.4% PPL +52%, Kimi 11.5% PPL +73%; calibrated P(0.98 overall and 0.95/class) ~4% | NO-GO for $70-110 probe; optional <$10 Qwen3-30B-A3B vs 32B GGUF control | v2/GIANT_MOE_RESEARCH.md |
