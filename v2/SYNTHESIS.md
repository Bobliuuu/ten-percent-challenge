# SYNTHESIS: best end-to-end entry designs, dense stack plan, new ideas, 4-day schedule (2026-11-08)

Synthesis lane, $0 compute, nothing run, nothing posted. Deadline 2026-11-12 17:05 UTC (about 4.1 days; treat 11-12 09:00 UTC as the last filing time). Rules R1..R9 and MC (maintainer check) as in `GOAL_README.md`.
Read: GOAL_README, RULES, SOLUTIONS (C01-C26, F01-F07), TASK_SEARCH, NEXT_PLAN, NATIVE_LOWBIT, GIANT_MOE_RESEARCH, FEEDBACK_LEGAL, PRUNE_LEGAL, DECISION, research_papers/{CHALLENGE_INTEL,PRUNING,QUANT,CODING,FAITHFUL,EDITING}, live issues #1-#4 (read-only). Not present in the repo at write time (pulled and re-checked): FEEDBACK_EXPERIMENTS, ENTRY_BONSAI, ENTRY_32B_INJ, VQ_ROUTE, QUALITY_PUSH, NATIVE_SURVEY2, TASK_SEARCH2. Their numbers are taken from SOLUTIONS.md rows (C07, C11, C19). Re-check this file against them when they land.
Convention: numbers are DEV, generation-scored unless stated; probabilities are my subjective estimates; "x [lo,hi]" is a rough 80% interval, not a Wilson bound.

## 0. What changed since SOLUTIONS.md / NEXT_PLAN.md (read first)
1. **A rival measured the thing we called NO-GO.** Issue #4 (entrant #4, 2026-11-07, unanswered): Qwen3-30B-A3B-Instruct-2507, task-specific expert pruning (2,640 of 6,144 experts kept by routing mass on unlabeled calibration text) + RTN (experts 3-bit, attn 6, emb 5, head 4), 9.892% of fp16 (6,040,715,980 B), **99.36% of base-correct kept on 2,910 locked items (2,775/2,793), worst kind 98.56%, 600 items from an unseen source (CLINC150) at 99.12%**, free-generated skill name, 7 skills. Self-reported and unverified by us, but it is the only claimed >=98% at <=10% in the whole challenge. Our own MoE rows (C13/C14) used REAP on tool-call generation and are not comparable: the difference is task-specific pruning on a short-label routing task. The GIANT_MOE "NO-GO" holds for giant MoEs, not for a 30B-A3B with task-calibrated pruning.
2. **Feedback, not compression, is now the binding constraint.** The maintainer's rejections (#2 "a classifier fitted to 2000 labels"; #3 "a nearest-centroid classifier ... the grown model is never run on the new-intent items ... your answer command also only lists and parses the 10 old families ... the 10 overwritten rows are the same head as before, just kept inside the model file"; stated goal: "work around this problem ... without any external memory/parameters otherwise this step gets trivialised") are the only rulings we have. Every design below inherits one feedback factor: P(maintainer accepts a closed-form edit of existing tensors) = 15-20% (EDITING.md says <=20%). Issue #4's feedback (closed-form edit of existing output rows, new label `recipe` generated end to end) is the first one that answers two of #3's three objections; **its verdict is the best free signal we will get. Watch it until 11-10.**
3. Dense compression is at a wall: best honest numbers are 95-96% retention at 9.6-10.0% on 7B-72B (C03/C04/C07/C11). The 4.3% flip figure used in FAITHFUL.md was measured at 10.03% (0.03 pt OVER the cap). At the legal 9.6% I assume 4.8-5.2% flips. The stack in section 2 starts there.
4. NEXT_PLAN's 25-30% for Bonsai predates FEEDBACK_LEGAL 3c (no selective edit on a 1-bit head) and CHALLENGE_INTEL (rule-9 risk). SOLUTIONS' 3% is closer; I use 1-2%.

## 1. Decision table: best entry designs
Shared factor F = P(feedback judged legal by the maintainer AND works behaviourally) ~ 0.15. P(all 9) below already includes F.

| | A. Bonsai-8B 1-bit lossless repack | B. Dense Qwen2.5-32B (or 72B) VQ stack, INJ-real label | C. Qwen3-30B-A3B task-pruned MoE + RTN, routing/tool-name label | D. Guard-model start (Qwen3Guard-Gen-8B / Llama-Guard-3-8B) + dense stack |
|---|---|---|---|---|
| Task | tool call (T1) or label task with new-label feedback | prompt-injection Injection/Benign (real label word; lock of 2320 items exists, hash-locked, unscored) | skill/tool-name routing, free-generated word. Prefer a broader menu than the rival's 7 skills; needs a new lock | content-safety label (the model's native output); lock to build |
| Base | prism-ml Bonsai-8B (derived by its authors from Qwen3-8B) | Qwen2.5-32B-Instruct | Qwen3-30B-A3B-Instruct-2507 | released safety-tuned model; margins large by construction (UNMEASURED) |
| Compression | none by us (lossless 1-bit pack) | Hadamard + GPTAQ/Fisher LDLQ + 8-dim VQ or trellis + KL allocation + coded scales (section 2) | routing-mass expert prune (about 40-43% kept) + RTN 3-bit experts, attn 6, emb 5, head 4; optionally GPTQ instead of RTN | same stack as B at 8B |
| Feedback | fp16 patch rows + mnH (untested; sign-flip edits FAILED) | mnH closed-form lm_head row edit (F01) | mnH-style closed-form edit of existing rows, new word run end to end (same family as rival #4) | mnH |
| Format | Bonsai pack + loader unpacking to fp16 | rANS/VQ index stream + decoder | expert-sliced safetensors, RTN codes, tiny loader | as B |
| Expected overall retention [80%] | 100% vs Bonsai (2000/2000 identical greedy, dev); 88.9% vs parent Qwen3-8B (S1 96.4, S3 79.6, multi 84.7) | 97.6% [96.4, 98.3] after the full stack (today 95.7 at 10.03%) | 98.3% [95.0, 99.4] (rival self-report 99.36; lower prior because unverified and narrow menu) | 96.5% [93, 98.5] (pure prior) |
| Worst-cell retention | 100% vs self; 79.6% vs parent | 93 [85, 96] (S3_injection / S2 cells; today 81-94) | 96.5 [91, 99] (rival: 98.56) | 94 [88, 97] |
| Size (exact) | 1,168,021,073 B of 16,377,096,192 = **7.132%**; cap 1,637,709,619; slack 469.7 MB (25 fp16 rows = 0.2 MB, calibration texts <=0.5 MB) | fp16 about 65.5 GB (verify params), cap 6.55 GB, target <=6.29 GB (9.6%): index stream 88-91%, scales 2.7% (g256), emb+head 6.1%, other 0.2% (CODING). Estimate only, nothing written | fp16 61,064,245,248 B, cap 6,106,424,524; rival 6,040,715,980 = 9.892%. Target <=9.6% = 5.86 GB (about 2,450 experts or a 2.5-bit mix). Real bytes via `size_check.py` | fp16 about 16.4 GB, cap 1.64 GB; untied emb+head are 14-19% of bytes at 8B (CODING), so tighter than B |
| R1 size | PASS 7.13% real; GRAY on baseline reading (same fp16 size either way) | PASS* est | PASS* (rival real 9.892%) | PASS* est |
| R2 faithful | PASS vs itself, FAIL vs parent | FAIL today; P(pass) ~20% after stack | UNTESTED by us; P(reproduce) ~55% | UNTESTED; P ~12% |
| R3 exam | PASS* (nothing calibrated) | PASS* (lock 2320 built) | new lock needed (>=2000, >=300/kind, >=500 unseen source) | new lock needed |
| R4 / R5 / R6 | R4 PASS* base, feedback GRAY; R5/R6 UNTESTED, sign-flip carrier failed | GRAY (mnH). Behaviour passes on 8B TOX (teach +0.5-0.8, real minus shuffled +0.47-0.68, neighbours 99.8%) but INJ new-label had a 1-in-10 collapse (unfixed) | GRAY; rival #4: teach 18.97 to 64.87, shuffled 29.23, empty 18.97; one subset (music) shuffled > real | GRAY |
| R7 / R8 | R7 PASS for tool call; R8 PASS on T1 (.70), label tasks unprobed | R7 PASS; R8 GRAY (benign cells at 0.97-0.998 ceiling) | R7 PASS; R8 PASS (rival base 95.98, chance 14.3) but near ceiling | R7 PASS; R8 unknown |
| R9 / narrowness | HIGH: "entrant did no compression" | GRAY (benchmark-shaped) | MEDIUM: task-pruned experts are a narrow-distribution artifact; "nothing with a really narrow distribution" | LOW-MEDIUM |
| MC (own OOD data + own feedback) | retention safe (identical weights); feedback exposed | exposed on both | exposed: pruned experts may not cover his distribution; rival's unseen CLINC 99.1% is mild evidence for | exposed |
| **P(pass all 9)** | **1-2%** | **~1.5%** | **~2-3%** | **~0.5%** |
| Hours / $ to finish | 12-16 h, $15-30 (identity done; label probe, patch carrier, lock, stages) | 60-80 h, $120-220 (32B boxes about $3.4/h x draws; 72B variant +$40-100) | 30-45 h, $40-80 (60 GB model fits one 80-96 GB GPU; prune + RTN needs no Hessians) | 15-25 h, $25-50 |
| User must decide | OK to post the 3-question Bonsai query on #1 (new comment); accept R9 exposure if unanswered (I would not file) | spend approval; whether to run 72B; accept ~5% odds | spend approval; accept narrow-task R9 risk; accept rival #4 is ahead (if both pass, "best" decides) | whether a public fine-tuned base counts as "the model you started from" (I read yes) |

Reading the table: no design exceeds 3%. C has the best R2 evidence, B the cleanest rule story, A the cleanest R2 but a probable R9 reject, D is a cheap probe for a possible step change in margins. All four share F. If #4's feedback is accepted, F rises to about 0.45-0.5: C about 8-10%, B about 4%, A about 4-5% (R9 still binding). If #4 is rejected on feedback, F drops to about 0.05 and every P falls ~3x (and the right submission is "nothing", section 6).
Union over designs, honoring the common factor F: **P(any of our entries passes all 9 as the maintainer reads them) ~ 4-6%.** P(we win the prize) ~1-2%, because #4 sits ahead with 99.36% and a pending verdict.

### Feedback legality scored against the actual rejections
| objection (maintainer, #2/#3) | mnH row edit, run end to end on the compressed artifact | prototype head / memory (C2) | Bonsai sign flips |
|---|---|---|---|
| "classifier fitted to labels" / extra head or params | nothing added (rows exist), but a linear readout from <=25 labels is the closest thing; the reading decides | FAIL (ruled) | n/a |
| "nearest-centroid" | not a centroid: min-norm solve in the calibration-Hessian metric; still arguably ridge regression | FAIL | n/a |
| "grown model never run on new items" | satisfied if the harness generates with the grown artifact (FEEDBACK_LEGAL generation check 0.98-1.00) | n/a | n/a |
| "answer command lists only old labels" | satisfied only if the prompt does NOT enumerate the new label and the parser accepts free text | n/a | n/a |
| "same head, kept inside the model file" | NOT answered; the residual risk and the reason F is only 15-20% | FAIL | FAIL (no selective flips found) |
| "no weight retraining" | closed form, no gradient, 2-3 rows change; disclose weights are not frozen | n/a | n/a |

## 2. STACK plan for the dense route (design B; reused by D and the 72B variant)
Goal: move flips on base-correct items from about 5.0% (legal 9.6%) toward 2%, and lift the worst cell (S3_injection, S2_benign; NEWS Technology 0.90) above 95%.

### 2a. Order to try (cheapest, most independent first; each step re-scored on the same 1566-item INJ-real dev, 3 draws)
| # | lever | recipe detail | literature claim | my realized sigma reduction (haircut) | flips after (cumulative) |
|---|---|---|---|---|---|
| 0 | baseline at the LEGAL size | VQ-D8 K about 5300-14500 at <=9.6% real bytes | - | - | 5.0% |
| 1 | calibration: class-balanced, base-generated, in exam format | 300k tokens (7B: 300k beat 1M), balanced over labels and both sources, base's own greedy outputs as the stream, excludes dev/lock/unseen-source block; self-calibration (NAACL 2025) | V9-style label calibration gave +1 to +2 pt at 72B | 4% | 4.6% |
| 2 | GPTAQ asymmetric target | fix the failed `tgt` run (0% = numerics: damp the cross term, FP32 inverse, store FP activations per layer). 7B smoke first | 15-30% layer-error reduction at low bit | 8% (haircut: papers report ppl, we need flips) | 3.85% |
| 3 | GuidedQuant/Fisher Hessian on teacher-KL | per-output-group averaged Fisher of KL(base\|\|quant) on base-generated text, not gold labels; weight low-margin tokens | 10-25% on the decision direction | 5% | 3.5% |
| 4 | KL-sensitivity bit allocation | `quality/alloc.py` sens mode across tensors, incl. embedding/head (generic-frequency rows only) and last 4 layers; total fixed at the real written byte count | +2-5 pt claimed; the 7B alloc run was inside noise | 4% | 3.2% |
| 5 | trellis codes (QTIP-style, `vq/tcq.py`) | fixed computed code removes codebook bytes and the K coupling | 0.05-0.1 bit, "+2-4 pt" | 5% | 2.9% |
| 6 | coded scales + embed/head re-spent as body bits | 3-6% of bytes freed (CODING) at about 1.5-2 pt per +1 size pt | +0.5-1.0 pt | about 18% of flips directly | 2.4% |
| 7 | best-of-N Hadamard seeds on a DISJOINT selection set | N=8 (or 32 with a KL-proxy prefilter then top 4 on full eval); selection set >=3000 base-correct items disjoint from calibration, dev, lock; margin-weighted score; report only on an untouched second dev set | none; our spread is about +-2 pt at f=4.3 | -0.4 pt true after winner's-curse haircut | **2.0-2.2%** |

Model: flip rate ~ sigma^2.1 (halving needs a 25-30% sigma cut, FAITHFUL section 2). Levers overlap because they attack the same error, so I credit only the sigma factors in the table (raw product 0.65 of sigma, realized about 0.77), deliberately more pessimistic than adding gains. 80% interval for the final flip rate on unseen dev: **1.6-3.6%, median 2.4%**. P(point estimate >=98% on the locked exam) ~30%; P(also every kind >=95%) ~55% of that (kinds at 81-90% today are the real obstacle). P(certified at Wilson >=98) < 10%. Step 6 is the only deterministic one; steps 2, 3, 5 carry most of the variance. Learned rounding (AdaRound/BRECQ) is the strongest lever by effect (FAITHFUL ranks it 1) but gray under "can't train weights"; excluded.

### 2b. Cheapest decisive experiment (with kill gates)
1. 7B smoke (about 3 h, $4-8): steps 1+2+3 on Qwen2.5-7B K=14500, INJ-real + TOOLSEL dev. Gate: model not destroyed and flips below the 7B baseline; else fix numerics once, then drop GPTAQ.
2. **Decisive run (about 10-14 h incl. build, $40-70): Qwen2.5-32B, steps 1-4 at <=9.6% real bytes, 3 draws, INJ-real dev, generation scored.** Expected flips 3.2% (from 5.0). **Go if mean <=3.6% and worst cell >=92%; otherwise kill the dense route** (steps 5-7 cannot supply the remaining factor 1.8). Run the 72B config (`quality/cfgs_72.json`, +$40-100) only if 32B passes the gate.

## 3. NEW IDEAS nobody has tried (ranked by P x impact)
Impact 1-3 (3 = would change the verdict). P = probability the idea works as intended AND adds no rule problem.
| rank | idea | what and why | P | impact | rule flag |
|---|---|---|---|---|---|
| 1 | **Start from a RELEASED task-tuned base** (safety/guard model: Qwen3Guard-Gen, Llama-Guard-3-8B, gpt-oss-safeguard; released injection/PII/router fine-tunes) = margins larger by construction, without touching the exam | The wall is margin vs quantization noise (noise std 7-13 logits vs median base margin 10-12; flips come from the margin<10 tail, 64% of 32B failures). A classifier-tuned model has bimodal, much larger margins. Repo grep finds no guard-model probe. Probe: 300 dev items, margin histogram, one 7-8B VQ run, $10-20 | 0.15 | 3 | CLEAN: "any model is fine", real job. Check R8 (not at ceiling), R9 (not narrow); untied emb+head 14-19% of bytes at 8B |
| 2 | **Choose the task by FEWER, LARGER kinds**: binary or 3-4-way label, exactly 2 sources | Rule 2 fails on the worst kind, not the mean (S3_injection 81, Technology 0.90, S2_benign). Each extra kind is another chance to miss 95%. Zero compute | 0.5 | 1 | CLEAN; rule 3 needs >=500 from an unseen source so keep exactly 2 sources |
| 3 | Larger best-of-N (32-64 seeds) with a cheap KL proxy prefilter, then 4 full evals | Seed spread (+-2 pt) equals the gap; order statistics give +1.0-1.5 pt true. Cost is quantizer time only | 0.6 | 1 | low-gray ("hyper-parameter search"); selection set disjoint from lock AND unseen-source block |
| 4 | Task-specific expert pruning with a BROADER menu and two lock sources (design C) | The rival's saving is structural: 57% of experts removed nearly free because a routing task uses few. The only lever that beats the bits wall | 0.4 | 3 | R9/R1 narrowness; PRUNE_LEGAL "experts" row |
| 5 | Label-free norm/bias statistics correction folded into existing q/k/v biases and RMSNorm gains (DFQ-style mean-shift matching of quantized to base activations on calibration text) | Attacks mu (class-conditional drift) rather than sigma; GPTAQ subsumes part. Qwen2.5 has qkv biases already, so no new parts | 0.3 | 1 | low (not fitted to labels). NOT a per-class threshold: that is a bias table fitted to labels (R4) |
| 6 | Pareto "bigger base, same bits" from our data: 7B-VQ 94-96, 32B 95-96, 72B-scalar 95.2-95.6, 405B 90.2 (GEN) | Retention is flat at about 95-96 from 7B to 72B at 9.6-10%; the lever is the quantizer (VQ), not size. The one unmeasured corner is 72B WITH VQ+GPTAQ. Smaller base with more bits fails: 14B reaches 98-99 only at 12.5-15% | 0.1 | 2 | clean; $40-100 |
| 7 | Higher precision for a LABEL-FREE output slice (top-2k generic-frequency lm_head rows + last 2 layers); cut the rest of the head | Decision tokens sit in the head neighborhood (LFQ 2605.29756); frees bytes from 150k rarely used rows | 0.25 | 1 | PRUNE_LEGAL gray-legal: rule fixed before looking at labels; measure OOD (non-English) cost |
| 8 | Native ternary (Ternary-Bonsai-2-27B, min 10.67%) plus real compression on top: drop 4-5 tail layers (need -6.3% bytes; 14B: 4 tail layers -1.6 pt) | Fixes both Bonsai problems: we really compress (R9 better) and size goes <10%. Retention vs the 27B itself unmeasured; loader needs a Hadamard transform (counted) | 0.04 | 2 | base-identity still gray; custom runtime bytes count |
| 9 | Feedback as a DIFFUSE closed-form edit in the model's own MLP (ROME/MEMIT-style, several layers, same Hessian-metric min-norm solve) instead of lm_head rows | Dodges "same head, kept inside the model file"; the answer comes from the whole forward pass. Prior `memit_down` had neighbours 0.57-0.94 (FAIL), so it needs the mnH metric and old-label continuation protection | 0.2 | 3 | still "fitting weights to data"; unruled. Schedule lane L3 |
| 10 | Feedback by selection among pre-shipped alternatives (MoE: ship a few extra low-bit experts unused at stage 1; feedback re-enables those that fix annotated mistakes) | Changes behaviour by routing, no fitted real values | 0.05 | 2 | GRAY-to-bad: resembles "side memory the system looks answers up from"; bytes must fit. Flagged |
| 11 | Calibrate on the exam FORMAT only (template + base-generated answers, no gold) | Already step 1; V9 vs V3 shows +1-2 pt at 72B | 0.7 | 1 | clean |
| 12 | PII-redaction / AI-vs-human (the maintainer's named examples) | Safe on spirit; AI-vs-human base is at chance (R8 FAIL) and span-level exact match is harsher than one label | 0.05 | 2 | R8 |

**Do NOT do (gray or spiritually dubious; the user wants rules passed as written):**
(a) Fine-tune the START model on the task to sharpen margins and call it the base. It beats the wall by construction but trains the weights of the model you "started from" and defeats "the model already does the job"; the maintainer would read it as spirit-violating. Using a model someone else released (idea 1) is the clean version.
(b) Filter or weight the locked exam by base margin (cherry-picking easy items).
(c) "Larger fp16 denominator" via merging or fine-tuning: fp16 size is the shipped parameter count x2, merging does not inflate it, and fine-tuning is (a).
(d) Keep attention fp16 and quantize MLP harder: attention is about 13% of layer bytes, so fp16 attention alone exceeds the whole 10% budget. The sensible inverse (attention higher, MLP lower) is what KL allocation already finds, and it was within noise at 7B.
(e) Per-class thresholds or logit-bias tables (R4, the maintainer's "classifier").
(f) Task-selected lm_head rows (a classifier cut from the model; R4/R7/R9).
(g) Posting on #1/#4 or touching the filed issue without the user's approval.

## 4. Four-day schedule (parallel lanes, go/no-go gates), times UTC
Spend is NOT authorized by this document; every box needs the user's approval. Estimated total if all lanes run: $170-330. Minimal plan L0+L2+L3 about $70-130.

| lane | 11-08 (now) | 11-09 | 11-10 | 11-11 | 11-12 |
|---|---|---|---|---|---|
| L0 (free) intel + decisions | user answers section 5; watch #4 and README edits; draft the maintainer question | **G0 18:00**: has #4 been answered? | **G1 12:00 hard decision**: freeze ONE artifact | proof drafting | file by 09:00 or abandon |
| L1 dense stack (B, D reuse) | build disjoint selection set; 7B GPTAQ smoke | 32B steps 1-4 x3 draws ($40-70) | **G-B (by 10:00)**: mean flips <=3.6%, worst cell >=92% else KILL; if go, steps 5-7 | locked run only if artifact frozen | - |
| L2 task-pruned MoE (C) | broader-menu dev + lock plan (CPU, free); download base | prune + RTN, dev retention on 2 sources incl. an unseen block ($30-50) | **G-C (by 10:00)**: overall >=98.5%, worst kind >=96%, unseen-source block >=97% else KILL; real `size_check` <=9.6% | locked run, controls | - |
| L3 feedback (shared bottleneck) | package mnH as `feedback.py` with calibration texts counted in bytes; fix INJ/EOS collapse (continuation rows in M); diffuse MLP variant (idea 9) | harness-v3 arms on the compressed dev artifact, 10 draws, real vs shuffled vs empty, new word spelled by existing rows, end-to-end generation | **G-F**: teach gain >=+0.3, real minus shuffled >=+0.25, neighbours >=98%, empty byte-identical to stage 1, 0 collapsed draws, free-text answer command; else no entry | stage 2 on the frozen artifact | - |
| L4 Bonsai (A) | label-task rule-8 probe + fp16 patch carrier (config exists, $5-15) | **G-A (18:00)**: carrier selective AND maintainer reading obtained, else KILL | - | - | - |
| L5 guard probe (D) | margin histogram of a guard model on 300 items ($5) | one 8B VQ run if median margin > 1.5x generic ($10-20) | **G-D**: flips <=3% at 9.6% else KILL | - | - |
| L6 exam hygiene (free, CPU) | hash-lock plan for the chosen task(s); >=300 per kind, >=500 from an unseen source | commit hashes BEFORE any scored run | lock file chmod 444 | one-shot scoring | - |

Rules of the schedule: (1) a design reaches the locked exam only if its dev gate AND G-F pass; (2) one locked scoring run per artifact, no reuse (the C2 lock is burned); (3) every result goes into the repo, `scan_leaks.sh` before each commit; (4) if two designs pass, pick by the maintainer-risk column, not by retention.

## 5. What must be decided by the user (in order of value)
1. **Approve spending** about $170-330 across lanes, or the minimal plan (L0+L2+L3, about $70-130), which has the highest P per dollar.
2. **Post the Bonsai/base-identity question on #1?** A new comment, visible to rivals. If not posted, do not file A.
3. Confirm a failing entry is not filed as a pass (DECISION.md already says so).
4. Whether a public fine-tuned guard/classifier model counts as an acceptable "model you started from" (I read yes).

## 6. What to tell the user (honest)
- Probability that ANY of our entries passes all 9 rules as the maintainer reads them by 11-12 17:05 UTC: **about 5% (range 3-8%)**. Probability we win the prize: about 1-2%.
- Why so low, in order: (1) feedback: the maintainer rejected three variants of "a few output rows fitted to labels" and accepted none; our best mechanism is the same family, so F ~ 15-20%. (2) Dense compression stalls at 95-96% retention for 9.6-10% bytes; the full stack projects to about 97.6%, a coin flip against 98%, with per-kind cells as the real problem. (3) The one route with claimed >=98% (task-pruned 30B-A3B MoE) is already filed by someone else and unverified. (4) Native-low-bit repack is clean on R2 but likely rejected on R9.
- The highest-value information is free: the maintainer's answer on #4. If #4's feedback is accepted, raise our odds about 3x and go all-in on C plus L3. If rejected, the realistic outcome is no passing entry.
- If no design clears its gate by 11-10 12:00 UTC: **submit nothing claiming a pass.** Options in order: (a) nothing; (b) with the user's explicit approval, a clearly labelled negative-results issue (no pass claim) giving the retention-vs-bytes wall and the feedback-legality findings: honest, free, wins nothing; (c) never re-file the burned C2 candidate. I recommend (a) unless the user wants (b) for the record.
