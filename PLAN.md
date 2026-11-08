# PLAN v2 (2026-11-06 02:10 EST) — SOURCE OF TRUTH: GOAL_README.md = github.com/DeusExperiments/ten-percent-challenge
Deadline per README: issue by **2026-11-12 17:05 UTC** (~6 days). 7am deadline REMOVED by user (2026-11-06 02:15). User authorized: open the submission issue the second a COMPLETE WORKING PROOF is done (never an incomplete/failing one).

## Real rules (README, updated after our early work)
1. ONE trained model **> 0.5B params** (before or after the cut); **commercial-class, non-toy, non-narrow task**; its score is the bar.
2. Keep ~**10% of fp16 size** (= ~1.6 bits/param average incl. everything). No weight training; every other technique OK.
3. **Match the base** score.
4. Locked held-out exam, never used to build/choose.
5. **Feedback** = annotation/labels, very few samples, like a real user. Learns new, keeps old, no weight training, still passes exam.
Prize needs a complete working proof as a GitHub issue.

## Gap vs what we built
- BERT-base SST-2 (109M): **does not qualify** (<0.5B, toy-shaped, fp32 baseline). trackA 9.76% of fp32 = 19.5% of fp16. Tracks A,B,D,E,G,K,L,J,Q,S built on it/small models: reusable METHODS only (GPTQ, vocab trim, entropy coding, memory-based feedback).
- trackR: Qwen2.5-0.5B (494M, just <0.5B!) MMLU: best ratio_bf16 0.42 at acc .35 vs base .466; 0.63 ratio at agree .99. Target 0.10 is far.
- Honest: 1.6 bpw training-free while matching a base is beyond known methods for small models. Large models quantize far better (QuIP#-class ~2-bit near-lossless at 7B+), plus vocab trim, entropy coding, layer drop.

## Strategy v2
1. Pick base: **Qwen2.5-7B/14B-class (or larger) on a commercial classification-style task** (output = few label tokens -> trim lm_head/vocab to label+prompt tokens; embedding/vocab trim as in trackR). Score = task accuracy.
2. Compress to <=10% fp16 using: vocab trim, GPTQ/QuIP#-style 2-bit + group scales counted, entropy coding, layer/head/FFN pruning, per-layer mixed bits via sensitivity (trackP/trackD tooling).
3. Measure by bytes (all files) and by accuracy vs base on dev; exam locked (hash-lock like trackC/LOCK.json).
4. Feedback: external memory/kNN/prompt-patch from few annotations; test learns-new/keeps-old/still-passes.
5. Submit issue before 11-12 17:05 UTC; leave 24h buffer.

## Next 24h
- A. trackR r3 (Qwen2.5-14B) results -> sets feasibility at big scale. Harvest it.
- B. Build 7B/14B sweep: bits 2-4, groups, pruning, entropy-coded bytes vs bf16, dev accuracy.
- C. Decide base+task by ~08:00 from data; then iterate.
- BERT-SST2 runs in flight (D,E,Q,S): let finish, no new launches for them.

## Rules: AGENTS.md (detach, results in repo, push every result).

## E. MAINTAINER RULINGS (issue #1 reply by tokenbender, 2026-11-06 05:57Z; verified) — see v2/RULES.md
1. 10% = size on disk vs fp16-equivalent weights. Count ALL inference files (weights, tokenizer, config, loader code, memory); aim <=9.5%.
2. "Can't train weights" = compression techniques OK, no retraining.
3. Feedback: NO retraining AND NO RAG methods ("just context space being edited"). => our kNN-answer-lookup memory (v2/feedback) is AMBIGUOUS and does not generalize. v2/feedback2 = non-retrieval, generalizing feedback (prototype heads, steering, closed-form calibration).
4. Commercial-class examples: agentic/tool use, PII redaction, AI-vs-human detectors. MMLU/MDC-6 may not count as commercial; v2/commercial builds frozen tasks for these three.
5. Natively ternary base (BitNet) = gray area (v2/lowbit explores anyway; prefer compressing a conventional model).
Competition (as of 06:00Z): one attempt (isHeSatoshi/tokenbender, Banking77, Qwen2.5-1.5B), no result; no submissions. Recheck issue list before submitting.

## F. RIVAL ENTRY + MARGIN FINDINGS (2026-11-08; read before choosing the final task/base) — private notes, not for the public README
Rival entry = issue #4 (Qwen3-30B-A3B at 9.892%, 7-way voice-skill router on MASSIVE+CLINC150, 99.36% faithful, exam opened once). Local notes (not in repo): ~/tenpct-rival/{AUDIT,COMPARE,FINDINGS_FOR_ENTRY_SESSION}.md. Evidence branches on origin: rival-e0 (v2/ambiguity_floor/REPORT.md), rival-moe (v2/moe_task/RESULTS.md), rival-tour (v2/margin_tournament/, running), rival-fbcmp (v2/fb_legal/rival_cmp/, running).
1. Their entry is complete and passes rules 1,2,3,7,8 on recomputation (size 6,040,715,980 B = 9.892%; 2775/2793 = 99.36%; worst source x label cell CLINC calendar 162/166 = 97.59%). Exposure: rule 4 (feedback = whitened class-mean direction added to existing lm_head rows from a shipped 1 MB covariance file: same family as what the maintainer called "the same head as before" on #3) and rule 6 (music: shuffled 88.3% > real 73.3%; only `recipe` is label-driven: real 61.1% vs shuffled 3.0%). Also: 3-4% near-duplicates between MASSIVE exam and dev/calib, unseen CLINC source has no email/news items, brand-new multi-token skill names untested (label_token = first token only).
2. We have NO artifact passing rule 2 today. Best real-exam numbers at ~9.6%: tool selection 95.5%, INJ 95.2% (72B). Dense 32B dev: 0.957-0.961 retention at 10.03% (over the cap).
3. E0: ambiguity floor on our tasks is small (0.6-1.4% of base-correct items fail at all budgets; rival ~0.8%). The limit is quantiser NOISE: only 15-25% of flips repeat across configs; 22% of base-correct items have margin < 10 logits and carry 63% of flips (margin < 2 flips 28%, 10-12 flips 3.2%). Estimated rule-2 pass chance, dense 32B at ~1.6 bpw: NEWS ~3%, INJ ~1.5%, TOOLSEL ~1%, T1/C2 tool calls < 0.5%. Uniform cell retention needed ~0.985 (0.98 -> ~50%, 0.99 -> ~100%). Exam sizing (Wilson LB > 95%, 80% power): 145 base-correct items per cell at true retention 0.99, 315 at 0.98, 780 at 0.97 -> planned INJ S2 (360) and NEWS Technology (400) cells are too small at 0.98.
4. MoE route (their recipe on OUR tasks, Qwen3-30B-A3B, 9.90% all-in, 3-bit experts/6-bit attn/5-bit emb/4-bit head, routing-mass pruning, broadened calibration): INJ 0.941, NEWS 0.975, C2 0.936 faithfulness; weakest labels benign 0.892 / Business 0.945 / "none fits" 0.862. Quantization-only (20.5%) is 0.981/0.976 so pruning costs ~4 pts on INJ. REAP score worse than routing mass (INJ 0.870/NEWS 0.808), task-only calibration worse (0.878/0.826), 2-bit with more experts no better. Verdict: does not beat the dense 32B plan; our earlier v2/moe failure was MMLU breadth + MMLU-only calibration + 4-bit attn/emb, not the metric.
5. LEVER = TASK CHOICE (margin decisiveness). The rival won on an easy, clearly separable task (base 95.98%), not on compression technique. Pick the task with the most decisive base-correct items (margin >= 8-10 logits for the large majority) subject to the rules (commercial, non-narrow, base well above chance and below ceiling, 2 sources, never-touched source with every cell >= 150 base-correct items). Do not curate locked items by base margin or compression outcome (breaks rule 4/3); curate SOURCES by label quality before any model run, hash-locked. The margin tournament (rival-tour) ranks candidates; see its README when it lands.
6. Feedback: mean-direction/LDA edits (ours: `hebb`/`swh`/feedback_v3; theirs: whitened mean + contrast) teach but are non-specific without a contrast term and are rule-4 gray; our `mnH` (Hessian-metric min-norm, FEEDBACK_LEGAL.md) keeps 99.8-99.96% of neighbours with every cell >= 98.6% on 7-8B dev but collapses in 1 of 10 INJ draws and is also gray. The head-to-head (rival-fbcmp) decides which to ship. Whichever ships: run it end to end through the real command on the real artifact (reload from disk), include a brand-new multi-token skill name, ship all calibration bytes in the size count, and never claim a control we haven't beaten.
