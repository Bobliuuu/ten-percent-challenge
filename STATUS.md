# STATUS: where the 10% challenge entry stands

**Last updated: 2026-11-08 06:24 UTC.** Deadline: 2026-11-12 17:05 UTC (107 h away at this timestamp).

Every number below is copied from a committed result or report file in this repository, and each section links to the file it came from. Nothing here was re-run for this page. Where two documents disagree, both numbers are shown and the disagreement is listed in [section 8](#8-inconsistencies-between-documents-found-while-writing-this). Numbers from other entrants are their own self-reports and are **unverified by us**. Other entrants are referred to as "entrant #3" and "entrant #4" (issue numbers on the challenge repo), never by account name.

Reading order: [1 summary](#1-where-we-stand-one-paragraph) - [2 what the author said](#2-what-the-author-has-said) - [3 scoreboard](#3-rule-by-rule-scoreboard-and-closeness-to-best) - [4 everything we tried](#4-everything-we-tried) - [5 what is next](#5-what-is-next) - [6 can we do it in time](#6-can-we-do-it-in-time) - [7 open questions (not posted)](#7-open-questions-for-the-maintainer-not-posted) - [8 inconsistencies](#8-inconsistencies-between-documents-found-while-writing-this).

Authoritative companions: [`v2/RULES.md`](v2/RULES.md) (single source of truth for the rules and rulings), [`v2/SOLUTIONS.md`](v2/SOLUTIONS.md) (registry C01-C26, F01-F07), [`v2/SYNTHESIS.md`](v2/SYNTHESIS.md) (best designs, probabilities, schedule), [`v2/EXPERIMENT_QUEUE.md`](v2/EXPERIMENT_QUEUE.md) with [`v2/exp/queue.json`](v2/exp/queue.json) (pre-registered next experiments), and the upstream rules in [`GOAL_README.md`](GOAL_README.md).

---

## 1. Where we stand (one paragraph)

We have **one filed entry** ([issue #2](https://github.com/DeusExperiments/ten-percent-challenge/issues/2): Qwen2.5-7B-Instruct, 9.404% of bf16, tool selection among 4 tools plus "none", 5 option letters read from 5 head rows) and it is **not a passing entry** under the nine judging rules: on its locked exam it kept **93.9% [92.6] of base-correct items** (1441 of 1535; rule 2 needs 98%), it answers with letters (rule 7), and its feedback ran through a fitted head plus memory, which the maintainer ruled does not satisfy step 5. Its locked exam is burned and it will not be refiled. Since then we have measured, on **development data only**, how far a conventional training-free compression stack gets on real generation: best honest retention is **95-96% of base-correct items at 9.6-10.0% of fp16** (Qwen2.5-32B VQ 95.0% [94.2], Qwen2.5-72B 95.2% on injection detection and 95.6% on tool selection, Qwen2.5-7B VQ 96.2% as a best-of-12 on its own dev set), against a requirement of 98% overall and 95% in every kind; Llama-405B reached 90.2% on real generation (teacher-forced scoring had said 99.7%). Per-kind cells sit at 77-94% in several tasks. Feedback is the other wall: the maintainer has rejected a fitted head (#2) and a nearest-centroid overwrite of output rows (#3) and has accepted no feedback mechanism yet; our best closed-form mechanism (`mnH`) works on development data with a 7-8B model (teach gain +0.5 to +0.8, real minus shuffled +0.47 to +0.68, neighbours 99.8-99.96% kept) but has never been run on a compressed artifact, collapsed a model in 1 of 10 injection-task draws, and is still gray under rule 4. The only candidate that **retains 100% by construction** is a lossless repack of a maker-built 1-bit model (Bonsai-8B, 7.132% real bytes, identical greedy output on 2000/2000 dev items), but it keeps only 88.9% of its parent model's correct items, it involves no compression by us (rule 9 risk), and no feedback carrier has worked on it. A rival (entrant #4) self-reports 99.36% retention at 9.892% on a 30B MoE with pruned experts; that is unverified and its verdict from the maintainer is pending. Our own planning documents put the probability that **any** entry of ours passes all nine rules as the maintainer reads them at **about 5% (range 3-8%)** and the probability that we win the prize at 1-2%; the planned experiments in [section 5](#5-what-is-next) have no committed GPU result yet (every row of `queue.json` is still `designed` or `parked`; only CPU-side prerequisites are committed), and if none clears its gate by 2026-11-10 12:00 UTC the plan is to **file nothing that claims a pass**.

Headline numbers at a glance (all dev, real generation unless stated):

| quantity | value | source |
|---|---|---|
| Filed entry, locked exam, retention of base-correct | 93.9% [92.6] (1441/1535); raw accuracy 0.799 vs 0.7675 | [SOLUTIONS C01](v2/SOLUTIONS.md), [PROOF_C2](v2/PROOF_C2.md) |
| Filed entry, general capability | MMLU .467 vs .693, ARC .62 vs .895 | [SOLUTIONS C01](v2/SOLUTIONS.md) |
| 32B VQ injection detection, dev | 95.0% [94.2], n=3377 base-correct; worst cells S3_injection 81.3, S2_benign 85.8; 9.85% (estimate) | [SOLUTIONS C07](v2/SOLUTIONS.md) |
| 32B scalar GPTQ, real injection data | 0.957 / 0.958 / 0.942 at 10.0 / 11.3 / 12.5% (draw noise about +-2 pt) | [TASK_SEARCH](v2/TASK_SEARCH.md) |
| 72B, dev | injection 95.21%, tool selection 95.55%, 9.61% (estimate) | [SOLUTIONS C03/C04](v2/SOLUTIONS.md) |
| 405B, dev | 90.2% [88.5] real generation (99.7% teacher-forced) | [SOLUTIONS C12](v2/SOLUTIONS.md) |
| Bonsai-8B lossless repack | 7.132% real; 2000/2000 identical vs itself; 88.9% vs parent | [SOLUTIONS C19](v2/SOLUTIONS.md) |
| P(any of our entries passes all 9) | about 5% (3-8%); P(win) 1-2% | [SYNTHESIS s1, s6](v2/SYNTHESIS.md) |
| Experiments from the new queue | 0 of 15 have a committed result; `queue.json` still shows all `designed`/`parked`; CPU-side prerequisites are committed (see 5.1); budget plan $330 | [queue.json](v2/exp/queue.json), [queue s5](v2/EXPERIMENT_QUEUE.md) |

Status of sizes: almost every size in this repo for the 7B-72B-405B candidates is an **entropy-accounting estimate**, not a written artifact checked by `v2/harness/size_check.py`. Real written sizes exist only for the filed entry (9.404%), the MDC-6 7B artifact (9.736%), BitNet-2B (9.453%) and the Bonsai repack (7.132%).

---

## 2. What the author has said

The maintainer has written **three substantive comments** on the challenge repo (issues #1, #2, #3) and several README rewrites. Issue #4 has no maintainer reply as of the last check (read-only). Short verbatim quotes, with what each one implies for us. Full text and context: [`v2/RULES.md` section 4](v2/RULES.md), [`v2/research_papers/CHALLENGE_INTEL.md`](v2/research_papers/CHALLENGE_INTEL.md).

| where | what the maintainer wrote (verbatim, short) | what it implies for us |
|---|---|---|
| [#1](https://github.com/DeusExperiments/ten-percent-challenge/issues/1), 2026-11-06 05:57Z | "10% of size on disk is fine compared to fp16 equivalent weights." | Size is bytes on disk of what we ship against the fp16 size of the starting model. Count tokenizer, code, feedback files. |
| #1 | "can't train the weights means compression techniques are fine but no retraining." | Post-training quantization, pruning, rotations and coding are allowed. Any gradient/optimizer fit to weights is not. Learned rounding is gray. |
| #1 | "For user feedback section, no retraining of the weights or RAG methods because just context space being edited makes it less fun." | Feedback may not retrain weights, retrieve examples, or edit the prompt. Kills kNN/lexical memory (F05) and in-context feedback. |
| #1 | "agentic/tool use, PII redaction, AI vs Human detectors are few examples." | Named commercial-class tasks. We checked all three: PII is feasible but synthetic; AI-vs-human is at chance for the base (rule 8 fail); tool use is the strongest ([commercial](v2/commercial/RECOMMENDATION.md)). |
| [#2](https://github.com/DeusExperiments/ten-percent-challenge/issues/2) (ours), 2026-11-06 16:10Z | "steps 1-4 are fine as per relaxed submission criteria." | Compression, base choice, locked exam and matching were not objected to at that time (before the nine-rule list, which was committed 17:25Z). Do not read it as a pass of rules 2, 3 or 7 as later written. |
| #2 | "step 5 doesn't hold here though. the feedback only works through head.npz, which is a classifier fitted to 2000 labels, and without it your memory changes zero answers." | A fitted head or side memory is not feedback. The grown model must change behaviour through its own computation. |
| #2 | "the objective for the challenge is to work around this problem where i want to see what can be improved **without any external memory/parameters** otherwise this step gets trivialised." | The intent is the hard part of the challenge. Expect the feedback step to be read strictly. |
| [#3](https://github.com/DeusExperiments/ten-percent-challenge/issues/3) (entrant #3), 2026-11-07 14:25Z | "the 88.5% is computed in numpy ... from averages of your feedback features, which is **a nearest-centroid classifier**, and **the grown model is never run on the new-intent items**." | Any class-mean/LDA/centroid row overwrite (our F02) is in the rejected family. The shipped system must be run end to end on the new class, in real generation. |
| #3 | "your answer command also only lists and parses the 10 old families, so a new intent can't come out of the system you ship." | The answer command must be able to emit a class not in the original label set (free-text parse, prompt does not enumerate the new label). |
| #3 | "the 10 overwritten rows are **the same head as before, just kept inside the model file**." | Overwriting a few `lm_head` rows with something fitted to feedback is treated as a head. Closed-form edits of existing tensors are **not ruled on**: treat any few-row output edit as high risk (our `mnH`), prefer diffuse edits acting through the model's own forward pass. |
| #3 | "the compression side is genuinely nice, and you're welcome to try again in the new repo before the deadline." | Compression approach (3-bit-class quantization of a pruned 30B MoE) was not faulted; resubmission after rejection is allowed; the deadline is the only wall. |
| README, commit `645d161`, 2026-11-06 17:25Z | "Ship the artifact, the locked files, one command that answers questions and one that takes annotated items as feedback. Before any prize is paid, I'll run your system on my own held-out data from a different distribution and give it my own feedback." | The maintainer's check is a different-distribution, own-feedback run. Dev numbers on our sources do not transfer automatically. |
| README, same commit | "Among the entries that pass every rule, the best one wins the prize." | Pass-all first. "Best" is **never defined**. We assume size/retention as the natural axes (inference only). |

Not ruled (no maintainer statement anywhere): starting from an already-low-bit released model (Bonsai-type), closed-form or diffuse weight edits for feedback, vocabulary trimming and tokenizer changes, what "best" means, whether the nine rules apply to entries filed before they were published. See [`v2/RULES.md` section 5](v2/RULES.md) and [section 7 below](#7-open-questions-for-the-maintainer-not-posted).

Rivals (signals only, all unverified by us): entrant #3 was rejected on feedback only; entrant #4 (filed 2026-11-07 17:45Z, Qwen3-30B-A3B-Instruct-2507, 2,640 of 6,144 experts kept, RTN 3-bit experts, 9.892% of fp16) claims 99.36% of base-correct kept on 2,910 locked items including 600 from a source it never used, and a feedback that edits existing output rows but does run the new label end to end (18.97% to 64.87% on the taught skills, shuffled 29.23%, empty 18.97%). Its verdict is pending. Its feedback is the closest thing to a ruling on what counts as allowed.

---

## 3. Rule-by-rule scoreboard and closeness to best

**Definition used for "best":** the maintainer never defines it. We use the only defensible reading: first, **pass every rule as written** (fewest FAIL, then fewest GRAY); only then rank by size/retention. A row is "closer to best" when it has fewer measured FAILs and fewer GRAYs **and** more cells that are verified rather than assumed.

How to read the table:
- Columns R1-R9 are the nine judging rules (R1 size, R2 faithful, R3 exam, R4 no new trained parts, R5 feedback teaches, R6 controls, R7 real job, R8 base has something to keep, R9 spirit); **MC** = the maintainer's own-data-and-feedback check; **FB** = status against the maintainer's feedback rulings (#1 no retraining/RAG, #2 no external head/memory, #3 no centroid/same-head, new class run end to end).
- Cell codes: `P` measured pass; `P*` passes on paper or by construction only (this is **not** a verified pass); `F` fails; `G` gray, the maintainer decides; `U` untested; `-` not applicable.
- `FAIL`/`GRAY` count the F and G cells. `UNVERIFIED` counts `U` plus `P*`. **A low FAIL count with a high UNVERIFIED count means "nothing measured has failed yet", not "close to passing".** All measured compression candidates fail R2.
- P(all 9) is the subjective estimate in [SOLUTIONS](v2/SOLUTIONS.md)/[SYNTHESIS](v2/SYNTHESIS.md), not a measurement. Planned rows (B, C, D, X09) are designs that have not run.
- Evidence column links the file that holds the numbers behind the non-obvious cells. Per-cell evidence for the first 20 rows is in the matching block of [`v2/SOLUTIONS.md`](v2/SOLUTIONS.md) (sections 1b and 2).

| id | candidate / approach | R1 | R2 | R3 | R4 | R5 | R6 | R7 | R8 | R9 | MC | FB | FAIL | GRAY | UNVERIFIED | P(all 9) | evidence |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| L1 | Legacy A-S: BERT-base SST-2 / XLM-R / DeBERTa classifiers (109M-435M), GPTQ + vocab trim + memory feedback | F | F | F | F | U | U | G | G | F | F | F | 7 | 2 | 2 | 0% | [trackA](trackA/RESULTS.md) 9.76% of fp32 = 19.5% of fp16 and 109M params < 0.5B; agree .966; [trackB](trackB/RESULTS.md)/[trackL](trackL/RESULTS.md) feedback = lexical/kNN/ridge memory |
| L2 | Legacy MMLU-letter Qwen tracks: trackR 0.5B, engine/vq/prune/entropy/big/moe on 14B | P* | F | U | P* | U | U | F | P | G | F | F | 4 | 1 | 5 | 0% | [engine](v2/engine/RESULTS.md) L=3 ratio .0968 acc .52 vs .80; [vq](v2/vq/RESULTS.md) best <=10% agree .705; MMLU is a letter stand-in (R7); [big](v2/big/RESULTS.md) produced no numbers |
| C01 | Filed C2 entry (issue #2): Qwen2.5-7B, letter head A-E + LDA memory | P | F | F | F | F | F | F | P | F | F | F | 9 | 0 | 0 | 0% | [SOLUTIONS C01](v2/SOLUTIONS.md): 9.404% real, LOCKED retention 93.9% [92.6]; maintainer: step 5 fails ([PROOF_C2](v2/PROOF_C2.md), [REDESIGN](v2/REDESIGN.md)) |
| C20 | MDC-6 7B letter-head artifact + PCH feedback head | P | F | P* | F | G | U | F | P | F | F | F | 6 | 1 | 2 | 0% | [SOLUTIONS C20](v2/SOLUTIONS.md): 9.736% real, dev agree .954; fitted head + letters |
| C02 | fix_real: 7B real tool-call generation, no letter head | P* | F | U | G | U | U | P | P | G | U | U | 1 | 2 | 6 | 0% | [fix_real](v2/fix_real/RESULTS.md): dev retention .934 (irrelevant class .619), MMLU .445 vs .693 |
| C11 | Qwen2.5-7B VQ-D8 K=14500, INJ / TOOLSEL (quality push) | P* | F | P* | P* | U | U | P | G | G | U | U | 1 | 2 | 7 | 2% | [SOLUTIONS C11](v2/SOLUTIONS.md): INJ 96.2% (best of ~12 configs), TOOLSEL 94.3%, worst cells 77-94, 9.93% est |
| C21 | 7B/14B/8B label tasks (research C_tasks, task_search, Hadamard+GPTQ+ECSQ) | P* | F | U | P* | U | U | P | P | G | U | U | 1 | 1 | 7 | ~0% | [SOLUTIONS C21](v2/SOLUTIONS.md), [TASK_SEARCH](v2/TASK_SEARCH.md): 14B 0.88-0.96 at 9.6-10%, TF and easy-source numbers |
| C08 | Qwen2.5-32B NEWS 4-way router, scalar GPTQ | F | F | P* | P* | U | U | P | P | G | U | U | 2 | 1 | 6 | 0.8% | [TASK_SEARCH](v2/TASK_SEARCH.md): 0.961 [0.950] dev at 10.03% (over cap), Technology 0.90 |
| C09 | Qwen2.5-32B INJ-real, scalar GPTQ | F | F | P* | P* | U | U | P | G | G | U | U | 2 | 2 | 6 | 0.7% | [TASK_SEARCH](v2/TASK_SEARCH.md): 0.957 / 0.958 / 0.942 at 10.0 / 11.3 / 12.5%; S2_benign 0.78. The 0.981 INJ-easy figure is an easy-source probe, not evidence |
| C10 | Qwen2.5-32B TOOLSEL / FIN / ROUTE / TOX / SENT / LANGID / INTENT / EMOTION | F | F | F | P* | U | U | G | G | G | U | U | 3 | 3 | 5 | <0.5% | [SOLUTIONS C10](v2/SOLUTIONS.md): 0.83-0.95 at 10.03%; ROUTE 29 classes and INTENT 60 classes cannot meet 300 per kind (R3), LANGID base at ceiling (R8) |
| C07 | Qwen2.5-32B VQ-D8 K=5300, INJ (entry32b) | P* | F | P* | P* | U | U | P | G | G | U | U | 1 | 2 | 7 | 1.5% | [entry32b](v2/entry32b/results/ret_e32_k5300.txt): dev 95.0% [94.2] n=3377 base-correct, S3_injection 81.3, S2_benign 85.8, 9.85% est |
| C03 | Qwen2.5-72B TOOLSEL (tool name or none), label + self-distilled calibration | P* | F | P* | P* | U | U | G | P | G | U | U | 1 | 2 | 7 | 1.2% | [D_scale](v2/research/D_scale/REPORT.md): 95.55% GEN dev (TF said 98.9), S1/none 89.3, 9.61% est |
| C04 | Qwen2.5-72B INJ, label calibration | P* | F | P* | P* | U | U | P | G | G | U | U | 1 | 2 | 7 | 1.5% | [D_scale](v2/research/D_scale/REPORT.md): 95.21% GEN dev, S3_injection 93.6, S1_benign 93.0, 9.61% est |
| C05 | Qwen3-32B / Qwen2.5-72B T1 tool calls (generated JSON) | P* | F | P* | P* | F | U | P | P | P | U | U | 2 | 0 | 6 | <0.3% | [SOLUTIONS C05/C06](v2/SOLUTIONS.md): 85.8% / 84.2%; 98% would need 19-25% of fp16; natural T1 feedback gain +0.01 |
| C12 | Llama-3.1-405B INJ | G | F | P* | P* | U | U | P | P | P | U | U | 1 | 1 | 6 | <0.3% | [L3_405B](v2/research/L3_405B/REPORT.md): TF 99.7% but real GEN 90.2% [88.5]; injection class 78% |
| C13 | Qwen3-235B-A22B REAP prune + 3-bit experts | P* | F | U | P* | U | U | P | P | P | U | U | 1 | 0 | 7 | <0.3% | [L2_moe](v2/research/L2_moe/REPORT.md): 0.939 TF, n=231; no artifact |
| C14 | Qwen3-30B-A3B REAP50 + RTN4 (tool calls) | F | F | U | P* | F | U | P | P | P | U | U | 3 | 0 | 5 | <0.3% | [v2/moe](v2/moe/RESULTS.md): 0.944 GEN n=179 but ~12.4% size; MMLU best at <=10% 0.732 vs 0.808 |
| C26 | Giant MoE (R1/V3/K2/235B/480B/GLM-4.5/gpt-oss-120b) at ~1.5 bit experts | F | U | U | U | U | U | U | U | U | U | U | 1 | 0 | 10 | <0.3% | [GIANT_MOE](v2/GIANT_MOE.md), [GIANT_MOE_RESEARCH](v2/GIANT_MOE_RESEARCH.md): best public file 10.44%; P(pass) ~4%; NO-GO, $0 spent |
| MT | Qwen3-30B-A3B task-pruned MoE (routing-mass experts) + RTN3, our dev run (design C) | P* | F | U | P* | U | U | G | P | G | U | U | 1 | 2 | 7 | 2-3% | peer branch `rival-moe` `v2/moe_task/RESULTS.md` (not yet on main): fake-quant first-token eval at 9.90%: INJ .941 (S2 .851), NEWS .975 (Business .945), C2 .936; raw preds not saved |
| C15 | BitNet b1.58 2B ternary + 4-bit embedding + vocab trim, MDC-6 letters | P | G | U | P* | F | U | F | F | G | F | F | 5 | 2 | 3 | 0.8% | [SOLUTIONS C15](v2/SOLUTIONS.md): 9.453% real; MDC-6 agree .988; T1 base .267 (R8 fail); letters (R7) |
| C16-18 | Other native ternary: Falcon-E-3B, Ternary-Bonsai-8B, Ternary-Bonsai-2-27B (lossless) | F | U | U | P* | U | U | U | U | G | U | U | 1 | 1 | 9 | <0.3% | [NATIVE_LOWBIT](v2/NATIVE_LOWBIT.md): 13.8% / 10.19% / 10.67% minimum; every lossy step costs R2 |
| C19 | Bonsai-8B 1-bit lossless repack (maker-built from Qwen3-8B) | P | G | P* | P* | U | U | P | P | G | U | F | 1 | 2 | 5 | 1-3% | [SOLUTIONS C19](v2/SOLUTIONS.md): 7.132% real; identical to itself 2000/2000 (dev) but 88.9% of parent; base identity/R9 unasked; sign-flip feedback failed ([FEEDBACK_LEGAL 3c](v2/FEEDBACK_LEGAL.md)) |
| C27 | Ternary-Bonsai-2-27B + scale log-quant + vocab truncation to <=9.9% (NATIVE_SURVEY2) | U | U | U | P* | U | U | U | U | G | U | U | 0 | 1 | 10 | ~0.5% (no estimate yet) | [NATIVE_SURVEY2](v2/NATIVE_SURVEY2.md): byte plan only, RESULTS section empty at last pull; vocab truncation unruled |
| B | PLANNED: Qwen2.5-32B (or 72B) VQ + full stack, INJ/NEWS label | P* | U | P* | P* | U | U | P | G | G | U | G | 0 | 3 | 7 | ~1.5% | [SYNTHESIS design B](v2/SYNTHESIS.md): projected flips 2.4% (80% interval 1.6-3.6) vs 2% needed; queue X04 |
| C | PLANNED: Qwen3-30B-A3B task-pruned MoE + GPTQ/RTN, routing label | P* | U | U | P* | U | U | P | P | G | U | G | 0 | 2 | 7 | 2-3% | [SYNTHESIS design C](v2/SYNTHESIS.md), queue X13; rival #4's unverified claim is the reference |
| D | PLANNED: released guard/classifier model as the start + dense stack | P* | U | U | P* | U | U | P | U | G | U | G | 0 | 2 | 8 | ~0.5% | [SYNTHESIS design D](v2/SYNTHESIS.md), queue X14; margins UNMEASURED |
| F01 | Feedback mnH: closed-form min-norm lm_head row edit in the calibration-Hessian metric | - | - | - | G | P* | P* | P | P | - | U | G | 0 | 2 | 3 | (component) 15-20% | [FEEDBACK_LEGAL](v2/FEEDBACK_LEGAL.md): TOX+selfharm dev, teach +0.5 to +0.8, real minus shuffled +0.47 to +0.68, neighbours 99.8-99.96%; mostly teacher-forced; never run on a compressed artifact |
| F01b | mnH on INJ + new label Jailbreak (Qwen3-8B) | - | - | - | G | F | U | P | P | - | U | G | 1 | 2 | 2 | (component) | [FEEDBACK_LEGAL s4](v2/FEEDBACK_LEGAL.md): teach .2-.45 but 1 draw in 10 collapses the model (neighbours 0.00) |
| F02 | Class-mean / LDA / calibrated lm_head row edits, memit_down (feedback_v3, combo) | - | - | - | G | F | U | P | P | - | U | F | 2 | 1 | 2 | (component) ~8% | [SOLUTIONS F02](v2/SOLUTIONS.md): synthetic pair passes, natural T1 mistakes fail (+0.01), memit_down neighbours .57-.94; = 'nearest-centroid' family of issue #3 |
| F03 | Annotation-aware requantization (feedback as Hessian weight) | - | - | - | P | F | U | - | - | - | U | U | 1 | 0 | 3 | ~0% | [FEEDBACK_LEGAL 3a](v2/FEEDBACK_LEGAL.md): cannot install a label ranked ~30 logits down; no fake run made |
| F04 | LDA/prototype head + running-mean memory (filed C2, lowbit_fb PCH, feedback2) | - | - | - | F | P* | U | P | - | - | F | F | 3 | 0 | 2 | 0% | [SOLUTIONS F04](v2/SOLUTIONS.md): works on dev (new tool .855 to .927) but is exactly the issue #2 head |
| F05 | Retrieval / kNN / lexical memory (v2/feedback, trackB, trackL) | - | - | - | F | P* | U | P | - | - | F | F | 3 | 0 | 2 | 0% | [trackB](trackB/RESULTS.md) lexicon memory teaches on BERT; banned by issue #1 q3 and rule 4 |
| F07 | Sign-bit flips on a 1-bit net (Bonsai) | - | - | - | U | F | U | - | - | - | U | F | 2 | 0 | 3 | 0% | [FEEDBACK_LEGAL 3c](v2/FEEDBACK_LEGAL.md): teach 0.00 with neighbours .78-.90, or teach 1.00 with shuffled also 1.00 |
| X09 | PLANNED: diffuse (MEMIT-style, mnH metric) multi-layer edit on the compressed artifact | - | - | - | G | U | U | U | - | - | U | G | 0 | 2 | 4 | (component) ~20% | [SYNTHESIS idea 9](v2/SYNTHESIS.md), queue X09; the earlier memit_down variant failed on neighbours |

LEGEND: P = measured pass; P* = passes on paper / by construction only (not a verified pass); F = fails; G = gray (maintainer decides); U = untested; - = not applicable (feedback-only row or compression-only row)


### Closeness to best (summary of the table)

1. **No row has a verified pass on rule 2 except by identity.** Every measured conventional compression candidate (C02, C03, C04, C05/C06, C07, C08, C09, C10, C11, C12, C13, C14) fails R2: best 95-96% against 98% overall and 95% per kind.
2. **Fewest known failures, but mostly unmeasured:** planned designs C (task-pruned MoE, 0 F / 2 G) and B (32B stack, 0 F / 3 G), and the Bonsai repack C19 (1 F / 2 G, of which F is the feedback column). C19 is the only row with a measured R2 pass (against itself, dev), and it pays with R9/base-identity gray, a failed feedback carrier, and 88.9% against the parent.
3. **Feedback component:** only F01 (`mnH`) and the planned X09 have no FAIL; both are R4 gray and neither has been run on a compressed artifact.
4. **Dead or disqualified (4-9 FAIL, or off-target):** C01 (filed entry, 9 F), C20, L1 (legacy small BERT tracks: below the 0.5B floor, fp32 baseline), C15 (BitNet-2B), L2 (MMLU letters).
5. Probability ordering from our own estimates: C (2-3%), MT (2-3%, same design measured on dev only), C11 (2%, 7B, fails other gates), B / C04 / C07 (1.5%), C19 (1-3%, conditional on a maintainer answer). All are conditional on the feedback factor F (about 0.15). If entrant #4's feedback style is accepted, SYNTHESIS raises F to 0.45-0.5 and C to about 8-10%; if it is rejected, every probability falls by about 3x.

---

## 4. Everything we tried

Chronological, grouped by theme. For each: what, result with exact numbers, whether the number is dev or locked, teacher-forced (TF) or real generation (GEN), easy or real source, why it failed or stalled, and where the code/results live. Full per-candidate blocks are in [`v2/SOLUTIONS.md`](v2/SOLUTIONS.md); the live ops log is [`RUNS.md`](RUNS.md).

Conventions: **dev** = development data, **locked** = a hash-locked exam scored once. Only the filed C2 candidate was ever scored on a locked exam. **TF** overstated retention by 3-10 points wherever both were measured (72B tool selection 98.9 TF vs 95.55 GEN; 405B injection 99.7 TF vs 90.2 GEN), so TF is never pass evidence. Draw-to-draw quantizer noise is about +-2 points, and a best-of-N on one dev set carries 1-2 points of optimism.

### 4.1 Phase 0 (2026-11-05 to 11-06 early): small BERT-class tracks A-S (legacy)
Built on the wrong base: BERT-base SST-2 is 109M parameters (below the 0.5B floor) with an fp32 baseline, so 9.76% of fp32 is 19.5% of fp16. Reusable methods only. See [`PLAN.md`](PLAN.md) ("Gap vs what we built").

| track | what | result (dev) | why it stopped |
|---|---|---|---|
| [A](trackA/RESULTS.md) | GPTQ 2-4 bit + vocab/embedding recipe on BERT-base SST-2 | 42.73 MB = 9.76% of fp32; dev2 acc .907 vs .930, agree .966 | model < 0.5B, fp32 baseline |
| [B](trackB/RESULTS.md) | feedback via sparse lexical residual memory (wordpiece perceptron) | new-concept acc 61.9 to ~96.6; old dev 98.80 to 98.23; dense kNN/residual head failed | side memory / retrieval: banned (#1 q3, rule 4) |
| [C](trackC/README.md) | protocol harness: hash lock, single-use exam, size check | harness reused later | method only |
| [D](trackD/RESULTS.md), [E](trackE/sweep1.json), [J](trackJ/RESULTS.md), [K](trackK/RESULTS.md) | GPTQ sweeps; calibration-only recovery; embedding-only compression | rot256+awq agree .9775 (best); embedding 2-bit full-vocab agree .98 | small-model results; E stopped as off-goal 02:25 |
| [G](trackG/RESULTS.md) | un-memorized dev proxy (IMDB/Amazon/tweets) | base acc 93.0% on dev2 | proxy only |
| [H](trackH/RESULTS.md), [I](trackI/RESULTS.md) | other classifier bases (XLM-R 278M, DeBERTa-v3-large 435M, RoBERTa-large ...) | best agree .990 only at ratio .125; nothing >=99% at <=10% | bigger is not more robust at fixed bits; all < 0.5B or off-target |
| [L](trackL/RESULTS.md) | external correction on the quantized model | 2-byte bias +0.6 to +1.0 pt; ridge/lexicon/kNN no better | diffuse noise, no cheap fix |
| [N](trackN/RESULTS.md) | power analysis for the non-inferiority test | e.g. a 1% flip rate gives NI pass .57 at delta 0 | harness design |
| [M](trackM/build.py), [P](trackP/core.py), [Q](trackQ/launch.py), [R](trackR/r3.json), [S](trackS/sweep_bases.py) | BERT artifacts and stand-in exams; vision classifiers (Q); Qwen2.5-0.5B MMLU (R); base sweeps (S) | R: best ratio_bf16 .42 at acc .35 vs .466 | Q/S off-goal; R is 494M (just under 0.5B) |
| [F](trackF/RESEARCH.md) | first rules/landscape research | rules as of 11-05 | superseded by [RULES](v2/RULES.md) |

### 4.2 Phase 1 (11-06 02:10 EST pivot): Qwen2.5 7B-14B with MMLU as a stand-in task
Strategy v2 in [`PLAN.md`](PLAN.md). All numbers are dev, MMLU letter answers (a rule-7 stand-in), so they only measure how far a training-free quantizer gets.

| what | result | verdict | where |
|---|---|---|---|
| Hadamard + GPTQ + entropy coding on 14B (engine) | uniform L=3: ratio .0968, acc .5209 vs base .8007 (agree .583); rotation essential (L=8 no-rot .441 vs rot .764) | wall: ~28 pts below base at 10% | [engine](v2/engine/RESULTS.md) |
| Vector quantization 8-dim shared codebook, 14B (vq) | 1.654 bpw, ratio .0961: acc .6195, agree .705 | still 18 pts below base; slope ~+4 pts per +0.1 bit | [vq](v2/vq/RESULTS.md) |
| Layer drop / FFN prune / head prune (prune) | tail drop 2 layers -0.4 pt for -4% bytes; FFN-neuron prune -2.5 to -4 pt at keep .9 | break-even at best; quant+removal frontier never finished | [prune](v2/prune/RESULTS.md) |
| Entropy coding variants (entropy) | context modelling gains <0.5% of index bits (rotated) | dead end | [entropy](v2/entropy/RESULTS.md) |
| MoE Qwen3-30B-A3B prune+quant on MMLU (moe) | best at <=10% ratio: 0.7323 vs 0.8078 (GPTQ, 70% experts, 2-bit); 1-bit collapses | -7.5 pt; later reused with REAP on tool calls | [moe](v2/moe/RESULTS.md) |
| 32B/72B MMLU (big) | **no numbers**: compute blocked (GPU pool saturated, p4d no capacity, hook bug) | stopped by coordinator | [big](v2/big/RESULTS.md) |
| Feedback on a 0.5B/1.5B stand-in (feedback, feedback2) | kNN gated memory; prototype-calibrated head (PCH) | teach, but retrieval/head: banned | [feedback](v2/feedback/RESULTS.md), [feedback2](v2/feedback2/RESULTS.md) |

### 4.3 Task selection (11-06): what the model should do
[`v2/commercial`](v2/commercial/RECOMMENDATION.md): the maintainer's three named examples were probed on Qwen2.5 1.5B/3B/7B. PII span category (base .41/.52/.69, synthetic data), **AI-vs-human (MAGE) at chance** (base .50/.53/.50, AUROC .50-.58: rule 8 fail), and C2 tool selection among confusable tools (base .76/.75/.78 vs chance .23): chosen. A multi-domain classification hedge (MDC-6: ag_news, dbpedia, yelp, amazon, emotion, offensive) was built as a fallback ([`v2/alt`](v2/alt)). Pre-registered choice: [`v2/DECISION.md`](v2/DECISION.md).

### 4.4 The filed entry (C01, 11-06): what was submitted and why it is dead
Qwen2.5-7B-Instruct, block-128 Hadamard + GPTQ onto an entropy-constrained scalar grid + rANS, vocabulary cut to 24,000 tokens, 4-bit embedding, `lm_head` cut to 5 option-letter rows; feedback = `head.npz` (476,506 B PCA64+LDA head) + an 852-byte running-mean memory.
- **Locked, stage 1 (n=2000):** base .7675, candidate .799 (+3.15 pt, paired CI [+1.60, +4.70]); both right on 1441 of 1535 base-correct = **93.9% [92.6]** retention (recomputed in SOLUTIONS; [REDESIGN](v2/REDESIGN.md) quotes 93.5% on dev). Stage 2 (after feedback): .802. Size 9.404% real. Single source (xLAM/apigen hard-distractor build).
- Failures against the later rules: R2 (93.9%), R3 (no 500-item unseen-source block, kinds not enumerated at 300), R4 (fitted head + side memory, ruled in #2), R5/R6 (teach set not pre-locked, no shuffled control), R7 (letters), R9. General capability collapsed (MMLU .467 vs .693, ARC .62 vs .895, HellaSwag .497 vs .797, TruthfulQA .458 vs .627).
- The locked C2 file was consumed (single-use claim). Do not salvage ([REDESIGN s0](v2/REDESIGN.md)). Files: [`v2/PROOF_C2.md`](v2/PROOF_C2.md), [`v2/harness/receipts`](v2/harness/receipts), [`v2/c2_fb`](v2/c2_fb/RESULTS.md).
- Related: MDC-6 7B letter-head artifact (C20, 9.736% real, dev .8731 vs .8721 raw, agree .954) and BitNet-2B ternary (C15, 9.453% real, MDC-6 agree .988, tool-call base only .267): both letter-based with fitted heads, kept as reference only ([altpush](v2/altpush/ARTIFACT.md), [lowbit](v2/lowbit/ARTIFACT.md)).

### 4.5 Redesign against the nine rules (11-06 evening to 11-07)
[`v2/REDESIGN.md`](v2/REDESIGN.md) and [`v2/RULES.md`](v2/RULES.md): real generated output, full vocabulary, no head/adapter/memory, three-source exam with a never-used block, shuffled/empty controls. Locks built and hashed but **never scored**: injection detection 2,320 items ([`v2/tournament/INJ`](v2/tournament/INJ/REPORT.md)), tool calls ([`T1`](v2/tournament/T1)), content moderation 4,520 items ([`TOX`](v2/tournament/TOX)). Research lanes: [A_sota](v2/research/A_sota/REPORT.md) (no published training-free method reaches near-lossless at ~1.6 bpw on 14B-72B), [B_native](v2/research/B_native/REPORT.md), [C_tasks](v2/research/C_tasks/REPORT.md), [D_scale](v2/research/D_scale/REPORT.md), [L1_large](v2/research/L1_large/REPORT.md), [L2_moe](v2/research/L2_moe/REPORT.md), [L3_405B](v2/research/L3_405B/REPORT.md).

### 4.6 Scale and quantizer (11-07 to 11-08): dense conventional compression, real generation
| candidate | result (dev) | why it fails | where |
|---|---|---|---|
| Qwen2.5-72B tool selection (C03) | 95.55% GEN, 9.61% est; TF said 98.9; cell S1/none 89.3 | R2; levers tested were inside noise | [D_scale](v2/research/D_scale/REPORT.md) |
| Qwen2.5-72B injection (C04) | 95.21% GEN, S3_injection 93.6, S1_benign 93.0 | R2; benign cells near ceiling (R8 gray) | same |
| Qwen3-32B / Qwen2.5-72B tool calls (C05/C06) | 85.8% / 84.2%; 14B 56.1%; extrapolated 98% needs 19-25% of fp16 | R2, large margin | same |
| Llama-3.1-405B injection (C12) | **90.2% [88.5]** GEN; injection class 78%; TF 99.7% | R2; easy-source margins misled TF | [L3_405B](v2/research/L3_405B/REPORT.md) |
| Qwen3-235B-A22B REAP + 3-bit (C13) | 0.939 TF (n=231); frequency-based pruning collapses (0.459) | R2; no artifact | [L2_moe](v2/research/L2_moe/REPORT.md) |
| Qwen3-30B-A3B REAP50 + RTN4 (C14) | 0.944 GEN (n=179) but ~12.4% size | R1 and R2 | [moe](v2/moe/RESULTS.md) |
| 7B/14B/8B label tasks (C21) | 14B 0.88-0.96 at 9.6%; 15%: 0.98-0.99 | bits, not task, set retention; noise std 7-13 logits vs median margin 10-12 | [C_tasks](v2/research/C_tasks/REPORT.md) |
| 12-task search on 7B/14B/32B (C08, C09, C10) | 32B at 10.03% (over cap): NEWS 0.961, INJ-real 0.957, ROUTE 0.947, FIN 0.937, TOOLSEL 0.928; **INJ-easy 0.981 is an easy-source probe, not evidence** | R2, R1 (10.03%); EMOTION/INTENT/ROUTE cannot meet 300 per kind; LANGID at ceiling | [TASK_SEARCH](v2/TASK_SEARCH.md) |
| 32B VQ-D8, K=5300 (C07) | 95.0% [94.2], n=3377 base-correct; S3_injection 81.3, S2_benign 85.8; 9.85% est | R2, per-kind | [entry32b](v2/entry32b/results/ret_e32_k5300.txt) |
| 7B VQ-D8 quality push (C11) | INJ 96.2% (best of ~12 configs), TOOLSEL 94.3%; VQ buys ~20-25 pts over scalar at the same bytes; MMLU .546 vs .678; `tgt` (error-propagation target) run destroyed the model (0%) = numerics bug | R2; general capability | [quality](v2/quality) |
| fix_real 7B real tool-call generation (C02) | retention .934 (irrelevant class .619); MMLU .445 vs .693 | R2, 1.74 bpw body | [fix_real](v2/fix_real/RESULTS.md) |
| Other bases (Llama-3.3-70B, Mistral-24B, Gemma-3-27B, GLM-4-32B) (C23) | 14-38% under our scorer | probe-harness artefacts, not evidence; weight 0 | [L1_large](v2/research/L1_large/REPORT.md) |
| TOX moderation compression (C24) | 14B 82.8% at 9.6%; 72B 95.1% TF | sexual cell unusable, several safe cells fail R8 | [tournament/TOX](v2/tournament/TOX) |

Ambiguity audit ([peer branch `rival-e0`](https://github.com/Bobliuuu/ten-percent-challenge/tree/rival-e0/v2/ambiguity_floor), summarized in [`PLAN.md` section F](PLAN.md)): items that fail at every budget are only 0.6-1.4% of base-correct, so the gap is **quantizer noise**, not ambiguity: 22% of base-correct items have margin <10 logits and carry 63% of flips; rule 2 needs ~0.985 true retention in every cell (0.98 gives ~50%); estimated rule-2 pass chance for dense 32B at ~1.6 bpw: NEWS ~3%, INJ ~1.5%, TOOLSEL ~1%, tool calls <0.5%.

### 4.7 Native low-bit and already-small models
| candidate | result | status | where |
|---|---|---|---|
| Bonsai-8B 1-bit lossless repack (C19) | 7.132% real (shipped tree); 2000/2000 identical greedy tokens vs itself (dev); 88.9% of the parent Qwen3-8B (S1 96.4, S3 79.6, multi 84.7); T1 dev base .70 | clean on R2 only if the maker's model counts as "the model you started from"; R9 risk "the entrant did no compression"; **not asked of the maintainer** | [SOLUTIONS C19](v2/SOLUTIONS.md), [native_lowbit](v2/native_lowbit) |
| Ternary-Bonsai-8B / Bonsai-2-27B (C17/C18) | 10.19% lossless; 10.67% minimum; parent retention 91.0% on tool calls (8B) | size blocks; lossy steps cost R2 | [NATIVE_LOWBIT](v2/NATIVE_LOWBIT.md) |
| Falcon-E-3B (C16), BitNet-2B (C15) | 13.8% lossless; 9.453% real but base at .267 on tool calls | size or R8 | same |
| Ternary-Bonsai-2-27B plus scale log-quant, vocab truncation, 5 trits/byte (C27) | byte plan targets <=9.9% from 10.67%; **RESULTS section empty** at last pull | in flight, no numbers | [NATIVE_SURVEY2](v2/NATIVE_SURVEY2.md) |

### 4.7b Lane L1 (Bonsai-8B, 2026-11-08): new feedback mechanism + easy task
The Bonsai feedback is no longer an lm_head-row edit: `v2/bonsai/ship/feedback.py` now does a closed-form, column-sparse fp16 edit of the existing MLP `down_proj` of the last 6 layers (1024 columns each, 50.3 MB), generating the new label through the model's own forward pass (no head, no row edit). On the canonical easy task SKILL7 (voice-skill routing) with new skill `Recipe`: base .896, teach .876 (base 0), cross-source teach .867, neighbours .999, shuffled .002 / wrong .000 / empty 0 (10 draws, real generation), total 7.44% (7.13% base tree). NEWS stays secondary and does not meet 98% neighbours with a useful teach rate. Package, 9-rule table, byte accounting and the old-label annotation policy: `v2/lanes/L1/REPORT.md`. Weakest rules: base identity (gray), closed-form-edit legality (gray), and the per-kind locked exam for SKILL7 cannot reach 300/kind with the current hash bucket.

### 4.8 Feedback mechanisms
| mechanism | what | result | rule status | where |
|---|---|---|---|---|
| LDA/prototype head + memory (F04) | filed entry's | new tool .855 to .927 at k=1; confusable-pair errors 92-100% fixed; no loss | R4 fail, **ruled in #2** | [c2_fb](v2/c2_fb/RESULTS.md) |
| kNN / lexical / retrieval memory (F05) | tracks B, L, v2/feedback | works on BERT SST-2; kNN over mid-layer states; all lookup | R4 fail, **ruled out in #1 q3** | [trackB](trackB/RESULTS.md) |
| Class-mean / LDA / calibrated `lm_head` row edits, `memit_down` (F02) | feedback_v3, combo | synthetic twin-tool pair passes (+0.27 to +0.57); natural T1 mistakes +0.01; `memit_down` neighbours .57-.94; INJ first-token edit fails at 1.7B/4B | the "nearest-centroid" family of #3 | [feedback_v3](v2/feedback_v3/README.md) |
| Annotation-aware requantization (F03) | feedback as Hessian weight | cannot install a label ranked ~30 logits down; no fake run made | legal but cannot teach | [FEEDBACK_LEGAL 3a](v2/FEEDBACK_LEGAL.md) |
| `mnH` closed-form min-norm `lm_head` row edit (F01) | Hessian-metric solve, 0 bytes added, no gradient | Qwen3-8B TOX+selfharm n=5/12/25: teach .49/.63/.65, real minus shuffled +.47/+.63/+.57, neighbours .9996/.9992/.9990; Qwen2.5-7B .65/.78/.75; survives 3/4-bit requantization; **INJ + Jailbreak: 1 draw in 10 collapses (neighbours 0.00)**; mostly teacher-forced | R4 gray ("same head ... inside the model file" risk), never run on a compressed artifact | [FEEDBACK_LEGAL](v2/FEEDBACK_LEGAL.md) |
| Sign-bit flips on Bonsai (F07) | flip-only re-binarization, z-score greedy, hebb+binarize | teach 0.00 with neighbours .78-.90, or teach 1.00 with shuffled 1.00 | no selective edit found; fp16 patch-row carrier launched, no result in repo | [FEEDBACK_LEGAL 3c](v2/FEEDBACK_LEGAL.md) |
| MoE 30B REAP50 whitened `lm_head` edit | L2 | fixes 12% of held-out mistakes at alpha .02, shuffled 0%; larger alpha breaks neighbours | weak | [L2_moe](v2/research/L2_moe/REPORT.md) |
| Rival-style comparison (mnH vs rival LDA vs hebb vs MEMIT-style down_proj) | peer branch `rival-fbcmp` | Qwen3-8B TOX table committed; X09 consumes it | pending | [`rival-fbcmp`](https://github.com/Bobliuuu/ten-percent-challenge/tree/rival-fbcmp/v2/fb_legal/rival_cmp) |

### 4.9 Giant MoE (no run)
[`v2/GIANT_MOE.md`](v2/GIANT_MOE.md) / [`GIANT_MOE_RESEARCH.md`](v2/GIANT_MOE_RESEARCH.md) (C26): byte math says experts must hold 1.37-1.55 bits all-in; best public 1.58-bit file is 10.44%; community results at that size are perplexity +52% (R1 IQ1_S) to +73% (Kimi K2). Estimated P(0.98 overall and 0.95 every class) ~4%. NO-GO, $0 spent, optional <$10 Qwen3-30B-A3B vs 32B control.

### 4.10 Task-pruned 30B MoE on our side (peer branch, dev)
[`rival-moe`](https://github.com/Bobliuuu/ten-percent-challenge/tree/rival-moe/v2/moe_task/RESULTS.md) (not yet on main): routing-mass expert selection plus 3-bit RTN at 9.90% bytes, fake-quant first-token evaluation: NEWS faithful 0.975 (weakest label Business .945; quant-only at 20.5% is .976), INJ 0.941 (benign .892, unseen source S2 .851; quant-only .981), C2 0.936. REAP scoring and task-only calibration were clearly worse (INJ .870/.878). Raw per-item predictions were not saved. This is the single piece of our own evidence that pruning is nearly free on a broad four-way routing task.

---

## 5. What is next

### 5.1 Live experiment queue (pre-registered; source [`v2/EXPERIMENT_QUEUE.md`](v2/EXPERIMENT_QUEUE.md), [`v2/exp/queue.json`](v2/exp/queue.json))
Status column is **as committed in `queue.json`** (queue commit `37c1511`, 2026-11-08). Since then main also contains: the three-way split files and hashes for X01/X15-prerequisites ([`v2/exp/SPLITS.json`](v2/exp/SPLITS.json): CAL 8,000, SEL 3,839, REP 7,234, GEN 2,001 items, open-tail NEWS prompt; [`DISJOINT_CHECK.json`](v2/exp/DISJOINT_CHECK.json): zero overlap violations across 29 lock manifests), the X04a arm configurations at 9.90% ([`v2/entry32b`](v2/entry32b)), and a bounded residual-form GPTAQ target fix ([`v2/quality/qpush.py`](v2/quality/qpush.py), X02 code, not yet validated by a committed 7B result). No GPU result, no base-reference predictions and no `v2/exp/results/` directory are committed. No locked file has been read. The dispatcher may be ahead of the committed JSON; check `v2/exp/queue.json` and `v2/exp/results/` for live status. Byte target for every arm is **real 9.90% all-in** (changed from 9.60%). Primary selection task: NEWS 4-way routing (pools large enough for disjoint calibration/selection/report sets); injection detection is report-only. Budget: **$330** total, released by gate (T0 ungated $110, T1 $90 on X04a GO, T2 $85 per lane gate, final-run reserve $30, unallocated $15).

| id | what | status | trigger | decision rule (kill / go) | est. $ (Vast) | est. wall |
|---|---|---|---|---|---|---|
| X01 | splits, base references, noise floors, tooling | designed | now | PASS if base self-flip floor <=0.8 pt on SEL, enough base-correct items, rule-8 accuracy in range | 13 | 4 h |
| X02 | fix the failed GPTAQ error-propagation target; 7B smoke | designed | now (parallel to X01) | proceed if numerics acceptance passes and no harm (>= baseline -0.5 pt); drop GPTAQ after 2 failed fix attempts (3 h time box) | 6 | 5 h |
| X03 | 14B ablation ladder on NEWS (KL allocation, coded scales, GPTAQ, KL-weighted Hessian, self-distilled calibration, trellis) | designed | after X01, opportunistic | keep a piece if paired delta >= +0.5 pt; need >= +2.0 pt A_best vs A0 to claim the stack works at 14B | 32 | 12 h |
| X04a | **DECISIVE**: Qwen2.5-32B steps 1-4 at real <=9.9%, 3 draws vs 3 baseline draws | designed | X01 + X02 smoke (or 5 h) | **GO** if mean flips <=3.6% and worst source x class cell >=92% and gain >=1.0 pt; else KILL the dense route | 33 (phase a+b) | 12 h |
| X04b | steps 5-7, 8-seed best-of-N, real written pack, REP unseal once | designed | X04a GO | TIER-1: overall Wilson/cluster LB >=0.980, every class and source marginal LB >=0.950, worst gated cell >=0.95, real bytes <=9.90% | (in X04) | |
| X05 | 32B leave-one-out | designed | X04a GO + X03 table | informational | 20 | 6 h |
| X06 | 72B stack at 9.6-9.9% | designed | X04a GO (and X04b >=TIER-2 or SEL >=0.97) | must start by about h28; skipped if 32B SEL projection to 72B <0.98 | 60 | 18 h |
| X07 | prune-then-quantize frontier (tail-layer agreement-targeted removal, byte swap) | designed | L4 result or h12 | cancel if flips >3 pt/layer | 40 | 12 h |
| X08 | frozen-artifact cross-source proxy of the maintainer's check, task transfer, general capability | designed | each frozen winner | PASS if never-used-source retention >= overall -1.0 pt | 7 | 4 h |
| X09 | **diffuse (MEMIT-style, mnH metric) feedback edit on a compressed artifact**, consuming the peer `rival_cmp` table | designed | now on a 14B/8B dev artifact; 32B winner when it exists | cancel/escalate if neighbours <0.9 in both metrics; **if it returns NO-SHIPPABLE-FEEDBACK no compression result can pass rules 5/6, stop T1/T2 spending** | 16 | 8 h |
| X10 | Bonsai gate completion (label-task probe, fp16 patch carrier) | **parked** | maintainer answer or explicit user acceptance | G2 acc in [0.60, 0.92], G3 gain >= +0.15, real minus shuffled >= 0.10, neighbours >= 0.98; kill if fails by 11-09 evening | 16 | 8 h |
| X11 | native-base candidates, parent retention | **parked** | same as X10 | continue only if lossless size <=9.8% and retention vs parent >= 0.97, every kind >= 0.95 | 10 | 4 h |
| X12 | copy-style / margin-tournament tasks | designed | a task with 14B screen >=0.95 | cancel on no signal or X04 TIER-1 | 20 | 10 h |
| X13 | **task-pruned MoE probe**, Qwen3-30B-A3B, real generation, legal bytes, disciplined splits | designed | now | continue if M0 or M1 >=0.97 on a 300-item screen; KILL if best arm <0.975 or unseen-source block <0.95 | 16 | 8 h |
| X14 | **guard-model probe**: margin histogram of a released safety model, then one 8B VQ run | designed | now (step 1) | KILL after step 1 if median margin <=1.5x generic or accuracy outside [0.60, 0.92]; KILL after step 2 if mean flips >3% at 9.6% | 20 | 8 h |
| X15 | seal the NEWS locked-exam manifest (hashes only, no model run, 3,200 items, 8 cells x 400, 1,600 from a source never built on) | designed | after X01 | PASS if every cell has >=400 eligible (else 300 per cell) | 0 | 1 h |

Global gates (hours from an assumed T0 of 2026-11-08 20:00Z): G-A h4 (X01+X02), G-B ~h20 (X03: stack gain >= +2.0 pt at 14B), G-C ~h30 (X04 interim: S >= B0 + 1.0 pt), G-D ~h44 (REP unseal + X08: TIER-1 freeze / TIER-2 second wave / TIER-3 stop compression), **G-E h60 FREEZE**, G-F h72 last chance, locked-exam protocol h72-h93.

### 5.2 Ideas from the research notes not yet scheduled as experiments
From [`v2/SYNTHESIS.md` section 3](v2/SYNTHESIS.md) and [`v2/research_papers`](v2/research_papers): (ranked by P x impact there)
1. **Start from a released task-tuned model** (guard/classifier) so margins are large by construction: in the queue as X14 (step 1 only is cheap). P 0.15, impact 3.
2. **Choose the task by fewer, larger kinds** (binary or 3-4-way, exactly two sources): zero compute, helps the per-kind rule. P 0.5, impact 1.
3. Larger best-of-N (32-64 seeds) with a KL-proxy prefilter on a disjoint selection set: only partly in X04b (8 seeds). P 0.6, impact 1.
4. **Task-pruned MoE with a broader menu and two lock sources:** X13. P 0.4, impact 3.
5. Label-free norm/bias statistics correction folded into existing biases/RMSNorm gains (not a per-class threshold). P 0.3, impact 1.
6. 72B with VQ+GPTAQ (the only unmeasured corner of the bits-vs-size grid): X06. P 0.1, impact 2.
7. Higher precision for a label-free output-row slice (generic top-2k rows) and cutting the rest: P 0.25, impact 1; gray-legal (PRUNE_LEGAL).
8. Ternary-Bonsai-2-27B with real compression on top (C27): in flight in NATIVE_SURVEY2. P 0.04, impact 2.
9. **Feedback as a diffuse closed-form edit in the model's own MLP** instead of output rows (dodges "same head inside the model file"): X09. P 0.2, impact 3.
10. PII redaction / AI-vs-human (maintainer's named examples): base at chance for the latter; span-level exact match is harsher. P 0.05.
Literature scans: [QUANT](v2/research_papers/QUANT.md) (no published training-free method near-lossless at 1.5-1.6 bpw on 14B-72B; scale is the strongest lever), [CODING](v2/research_papers/CODING.md) (scale coding and embedding/head redesign free 3-6% of bytes = +0.5 to +1.0 pt), [PRUNING](v2/research_papers/PRUNING.md) (pruning only helps as a byte swap), [FAITHFUL](v2/research_papers/FAITHFUL.md) (optimise KL to the base, flips are generic approximation noise), [EDITING](v2/research_papers/EDITING.md) (anything writing a fitted linear readout into the output layer is caught by "classifier/centroid").

**Explicitly not doing** (gray or spirit-violating, from SYNTHESIS): fine-tuning the start model, filtering the exam by base margin, a larger fp16 denominator via merging, fp16 attention with harder MLP quantization, per-class thresholds/logit-bias tables, task-selected `lm_head` rows, learned rounding (AdaRound/BRECQ), Fisher via backward pass, label-free vocabulary-trim arm.

---

## 6. Can we do it in time?

Clock: the queue assumes **T0 = 2026-11-08 20:00Z**. Code and split prerequisites are already committed, but no GPU result is, so every hour of delay in launching comes straight off the 93-hour plan. Derived dates: G-E freeze (h60) **2026-11-11 08:00Z**; last quantization start (h72) **2026-11-11 20:00Z**; end of locked-exam protocol (h93) **2026-11-12 17:00Z**, with the deadline at 17:05Z. SYNTHESIS is stricter: hard decision **2026-11-10 12:00Z**, last filing time **2026-11-12 09:00Z** ([section 8](#8-inconsistencies-between-documents-found-while-writing-this) lists the difference).

| next step | hours (wall) | $ | depends on | on the critical path? | decision time | honest chance it helps | what happens if it fails |
|---|---|---|---|---|---|---|---|
| X01 splits and floors | 3-4 | 13 | nothing | yes (h1 splits, h2 base refs) | h4 | n/a (enabling step) | everything blocked; fix and rerun |
| X15 seal NEWS lock | 1 | 0 | X01 final prompt | yes: must precede any final artifact | h4 | n/a (enabling step; Technology pool may force 300 per cell) | no lock, no final run |
| X02 GPTAQ fix | 4-5 | 6 | nothing | no (S4 may start without it after 5 h) | h4 | no estimate; a numerics fix, benefit unproven (decided at 14B in X03) | drop GPTAQ; stack loses ~one lever |
| X04a 32B decisive | 10-12 | 33-35 | X01, X02 smoke | **yes** | ~h30 (2026-11-10 02:00Z) | no explicit estimate; steps 1-4 are projected to take flips from 5.0% to 3.2% against a 3.6% gate (SYNTHESIS, literature-based); the only test that can revive the dense route | KILL dense route; X06, T1 not released; freed $143 goes to X13>X14>X12 |
| X04b best-of-N, real pack, REP unseal | +10-12 | 28 | X04a GO | yes | h44 (2026-11-10 16:00Z) | P(TIER-1) about 5-8% (queue, thin); it needs about 0.985 true retention in every cell | TIER-2 second wave or stop |
| X06 72B | 14-18 | 55-60 | X04a GO | parallel; **must start by about h28 (2026-11-10 00:00Z)** | h44-h50 | P(point estimate >=98% on a locked exam) about 30% for the full stack at 32B, and about 55% of that for every kind >=95% (SYNTHESIS); 72B adds the only unmeasured corner (P 0.1) | skipped if 32B SEL projection <0.98 |
| X09 diffuse feedback | 8 + 8 on winner | 8 + 14 | a compressed dev artifact | **gates what may be frozen** | h26-h34 first read | P(shippable) ~20% for diffuse; mnH gray at ~15-20% | if NO-SHIPPABLE-FEEDBACK: stop spending, no entry can pass 5/6 |
| X13 task-pruned MoE | 6-8 (+6-8 phase b) | 12 (+12) | X01 splits for the final read | parallel | screen at about h10 | P(reproduce >=98%) ~55% on dev, lower for a broader menu and for the maintainer's own data | KILL if unseen-source block <0.95 |
| X14 guard probe | 6-8 | 5 (step 1) + 10-20 | nothing | no (cheapest information in the queue) | step 1 at h8 | P(margin >1.5x generic) ~15% | KILL after step 1 |
| X10/X11 Bonsai / native | 8 / 4 | 16 / 10 | **maintainer answer or user acceptance** (parked) | no | kill by 11-09 evening | P(Bonsai route passes all 9) 1-3% | do not file |
| X07 prune-then-quantize | 12 | 27-40 | L4 result; phase 2 after X04 | no | h30 | small | byte-swap not worth it |
| X08 transfer + general capability | 4 | 7 | a frozen winner | yes at h44-h48 | h48 | informational | |
| Final locked run, feedback controls, size check, proof write-up | 20 (h72-h93) | 30 reserve | freeze (h60) | **yes** | h93 | n/a | no filing if any gate failed |
| Maintainer question(s) | free, but a public comment | $0 | **user approval to post** (not posted) | no | maintainer replies took 2 h (#1), 3.6 h (#2), about 21 h (#3); #4 has had none for over 12 h | P(Bonsai base identity accepted) about 30% | Bonsai stays unfiled |

**Critical path:** X01 splits (h1) -> base references (h2) -> code readiness of the stack levers (h10) -> ladder gate (h20) -> 32B stack seeds (h30) -> real pack plus REP unseal (h44) -> freeze (h60) -> locked-exam protocol (h72-h93). Slip of more than about 6 h in code readiness pushes the 72B branch past its last-start gate.

**Honest probability that any entry of ours passes all nine rules** (from [`v2/SYNTHESIS.md` section 1](v2/SYNTHESIS.md); the common factor F, P(feedback judged legal and works) ~0.15, is already inside them):

| design | P(all 9) | main blockers |
|---|---|---|
| A Bonsai-8B lossless repack (C19) | 1-2% (SOLUTIONS 3%) | base identity, rule 9, no working feedback carrier |
| B dense Qwen2.5-32B/72B VQ stack, injection/news label | ~1.5% | R2 today fails; projected flips 2.4% vs 2% needed; per-kind cells; feedback |
| C task-pruned Qwen3-30B-A3B MoE | ~2-3% | R2 reproduction, narrow-distribution reading, feedback; entrant #4 already ahead |
| D guard-model start | ~0.5% | margins unmeasured |
| **union (any of ours)** | **about 5% (3-8%)** | feedback is shared: if entrant #4's feedback style is accepted raise to ~3x; if rejected divide by ~3 |
| we win the prize | 1-2% | entrant #4 sits ahead with a pending verdict |

**What we will do if no design clears its gate** ([`v2/SYNTHESIS.md` section 6](v2/SYNTHESIS.md), [`v2/DECISION.md`](v2/DECISION.md)): file nothing that claims a pass. Options in order: (a) file nothing; (b) with the user's explicit approval, a clearly labelled negative-results issue (no pass claim) documenting the retention-vs-bytes wall and the feedback-legality findings: honest, free, wins nothing; (c) never refile the burned C2 candidate. Issue #2 stays as filed; this repository documents that it does not meet the nine rules.

---

## 7. Open questions for the maintainer (NOT POSTED)

**Status: none of these has been posted anywhere. They are drafts for the owner to decide whether to ask.** They are phrased neutrally and do not name any other entrant.

1. **Starting from an already-low-bit release.** A model M above 0.5B is released by its authors with an fp16 checkpoint F whose weights are exactly low-bit valued per group (every matrix including embeddings and head, so a lossless pack of F is bit-exact). M was derived by its authors from another model. A submission would ship a lossless re-encoding of F at about 7% of F's size with the same forward pass. (a) Is the size baseline the fp16 checkpoint of M itself, even if its authors also publish a packed form already below 10%? (b) Is "the model you started from", and the score to match, M rather than the model M was derived from? (c) Does a lossless repack of a native low-bit model count as valid compression under the spirit rule, or must the entrant reduce a model that was not already at or below 10% in released form?
2. **Closed-form edits of existing tensors for feedback.** If the feedback step changes a few existing weights by a closed-form solve from the annotated items (no gradient, no optimizer, no new object, no stored annotations, nothing added to the file), is that "no weight training" and "no new computation fitted to data", or is it read as a head kept inside the model file? Does it matter whether the edited tensors are output rows or several MLP weights?
3. **What must the grown model demonstrate on a new class?** Is the intended test that the answer command, with a prompt that does not list the new label, generates the new label word in real generation on unseen items after the feedback command runs?
4. **Calibration data in the feedback command.** A closed-form edit that needs a covariance of ordinary hidden states can recompute it from a small unlabeled text file shipped with the artifact (counted in bytes, about 0.5 MB). Is such a file "feedback storage" or "side memory the system looks answers up from"?
5. **Vocabulary and tokenizer changes.** Is trimming the vocabulary (embedding and output rows) to tokens seen in unlabeled calibration text, with the full tokenizer behaviour otherwise unchanged, acceptable for size accounting and the "original model's own computation" rule?
6. **Pruned experts.** For a mixture-of-experts model, is dropping experts chosen by routing mass on unlabeled calibration text compression, and how does the "really narrow distribution" wording apply to a model whose kept experts were chosen for a four-way routing task?
7. **What does "best" rank on** among entries that pass every rule (size, retention, base size, feedback gain)?
8. **Do the nine judging rules apply to entries filed before they were published?** (The list was committed after the first maintainer replies.)
9. **Released task-tuned starting models.** Is a publicly released fine-tuned classifier or safety model an acceptable "model you started from"?
10. **Byte accounting for the runtime.** Are generic runtime libraries (torch, transformers, an entropy decoder library) excluded from "on-disk bytes of everything you ship", with only our loader and answer/feedback code counted?
11. **The maintainer's own check.** Will the answer and feedback commands be run as shipped, with the shipped prompt, on data from a different distribution, and are the label sets of that data expected to differ from the exam's?

---

## 8. Inconsistencies between documents (found while writing this)

These are conflicts in the committed documents. They were **not** resolved here (the owners of those files should reconcile them); this page uses the more conservative number.

| topic | document A | document B | used here |
|---|---|---|---|
| Probability that any of our entries passes | [SOLUTIONS s1](v2/SOLUTIONS.md): "the honest sum over disjoint routes is roughly 10%" | [SYNTHESIS s1, s6](v2/SYNTHESIS.md): about 5% (3-8%), union 4-6% | 5% |
| Bonsai route probability | [NEXT_PLAN](v2/NEXT_PLAN.md) rated it 25-30% (before FEEDBACK_LEGAL 3c) | SOLUTIONS 3%; SYNTHESIS 1-2% | 1-3% |
| Real byte target | SYNTHESIS and older queue text: 9.6% (baseline flips 5.0% measured at 10.03% and assumed 4.8-5.2% at 9.6%) | queue after commit `37c1511` and RULES section 6: 9.90% real; [`PLAN.md` section E](PLAN.md): aim <=9.5% | 9.90% (queue), 10.00% is the cap |
| Latest freeze/filing time | SYNTHESIS: hard decision 2026-11-10 12:00Z, last filing 2026-11-12 09:00Z | queue: freeze h60 = 2026-11-11 08:00Z, locked run h72-h93, buffer 4 h | both listed |
| Filed entry retention | SOLUTIONS C01: 93.9% [92.6] (recomputed from locked per-item data) | [REDESIGN](v2/REDESIGN.md), [DECISION](v2/DECISION.md): 93.5% | 93.9% locked; 93.5% was dev |
| Falcon-E-3B size | `lowbit/ARTIFACT.md`: about 9.97% ("tight") | [NATIVE_LOWBIT](v2/NATIVE_LOWBIT.md): 13.8% lossless | 13.8% (SOLUTIONS marks the old figure superseded) |
| 32B injection dev sets | C07 uses a 4,313-item set that includes a noisy audit slice | C03/C04/C11 use the 1,566-item set | not directly comparable; do not rank C07 against C11 |
| INJ 98.1% | [TASK_SEARCH](v2/TASK_SEARCH.md) "INJ-easy" 0.981 on 32B | real-source INJ on the same 32B: 0.957 / 0.958 / 0.942 | 0.957 (0.981 is an easy-source probe) |
| Bonsai size baseline reading | SOLUTIONS C19 marks R1 gray on the baseline reading | NEXT_PLAN: Bonsai fp16 and parent fp16 are the same size to the byte, so the reading cannot change the number | R1 shown as P (7.132% real); the open question is R2/R9 |
| README headline vs rulings | [README](README.md) headline: "scoring higher than the original" and "able to learn" | maintainer on #2: step 5 does not hold; locked retention 93.9% | README stays as the factual description of the filed entry; the new "Status and plan" section points here |
| Naming rule vs older notes | [`PLAN.md` section E](PLAN.md) and [`v2/SCOUT_LOG.md`](v2/SCOUT_LOG.md) mention an early competing repo by account name | RULES: never name other entrants | not repeated here; owners of those files may want to anonymize |

Companion documents not repeated here: [`v2/PRUNE_LEGAL.md`](v2/PRUNE_LEGAL.md), [`v2/NATIVE_SURVEY2.md`](v2/NATIVE_SURVEY2.md), [`v2/research/E_infra`](v2/research/E_infra), [`AGENTS.md`](AGENTS.md). Peer-session material: [`PLAN.md` section F](PLAN.md) (another session's ablation findings, linked and not edited here) and the peer branches `rival-e0`, `rival-moe`, `rival-fbcmp`, `margin-tournament` on the same repository.


**X04a RESULT (2026-11-08): KILL vs RULES.md gates.** 32B steps 1-4: S4 retention 96.5% (LB 96.0%) dev_inj, worst cell 79.1%; INJR subset 97.4%, worst 92.9%. Baseline B0 94.1%. See [v2/exp/X04a_REPORT.md](v2/exp/X04a_REPORT.md). Dense route = fallback (L5).


## L3 MoE lane (Qwen3-30B-A3B-Instruct-2507): measured on the shared locked exam (2026-11-08, `v2/lanes/L3/REPORT.md`)
| | SKILL7 (2,744) | BANK6 (2,237) |
|---|---|---|
| size, all shipped bytes | 9.896% (packed 9.676% + o_proj edit) | 9.896% |
| retention of base-correct [Wilson lo] | 99.32% [98.92] | 99.24% [98.76] |
| worst kind / never-used source | Calendar 96.7 / 99.65 (1,489 items) | Card 97.9 / Bitext 98.2 (600 items) |
| feedback (24 labels, closed-form o_proj edit) | teach 0 -> 79.9%, shuffled 0, wrong 0, empty 0, exam 99.20 after | teach 0 -> 16% (weak), neighbours 98.5 |
Recipe: 85/128 experts by routing mass, D4-K256 VQ 2.0 bpw with Hadamard + error feedback, attn 6 / emb 5 / head 4 bit. Weakest rules: closed-form weight edit legality (rule 4, gray), BANK6 teach (rule 5), per-cell 98.5 target not met. Near misses: `v2/NEAR_MISSES.md`.
