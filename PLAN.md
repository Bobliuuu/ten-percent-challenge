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

## F. ABLATION FINDINGS (2026-11-08): what we tried on the compression/task side
Branches on origin with the evidence: ambiguity-floor (v2/ambiguity_floor/REPORT.md), moe-task (v2/moe_task/RESULTS.md), margin-tournament (v2/margin_tournament/, running), fb-compare (v2/fb_legal/rival_cmp/, running; will be renamed).
1. Current standing: we have no artifact that passes rule 2 at <=10%. Best real-exam numbers at ~9.6%: tool selection 95.5%, INJ 95.2% (72B). Dense 32B dev: 0.957-0.961 retention at 10.03% (over the cap).
2. Ambiguity floor vs noise (E0, dev per-item rows): items failing at every budget are only 0.6-1.4% of base-correct items. The limit is quantiser NOISE: 15-25% of flips repeat across configs; 22% of base-correct items have margin < 10 logits and carry 63% of the flips (margin < 2 flips 28%, 10-12 flips 3.2%). Estimated rule-2 pass chance for dense 32B at ~1.6 bpw: NEWS ~3%, INJ ~1.5%, TOOLSEL ~1%, T1/C2 tool calls < 0.5%. Passing needs ~0.985 true retention in every cell (0.98 -> ~50%, 0.99 -> ~100%). Exam sizing (Wilson LB > 95%, 80% power): >=145 base-correct items per cell at true retention 0.99, 315 at 0.98, 780 at 0.97, so the planned INJ S2 (360) and NEWS Technology (400) cells are too small at 0.98.
3. Task-specific MoE pruning on Qwen3-30B-A3B (routing-mass expert selection with task + out-of-task + generic calibration, 3-bit RTN experts, 6-bit attn, 5-bit embeddings, 4-bit head, 9.90% all-in): INJ 0.941, NEWS 0.975, C2 0.936 faithfulness; weakest labels benign 0.892 / Business 0.945 / "none fits" 0.862. Quantization-only (20.5%) is 0.981/0.976, so pruning costs ~4 pts on INJ. REAP scoring worse than routing mass (INJ 0.870, NEWS 0.808), task-only calibration worse (0.878/0.826), uniform per-layer counts worse, 2-bit experts with more kept no better. It does not beat the dense 32B plan. Our earlier MMLU MoE failure came from task breadth, MMLU-only calibration and 4-bit attention/embeddings.
4. Lever = TASK CHOICE: pick the task whose base-correct items are most decisive (margin >= 8-10 logits for the large majority) subject to the rules (commercial, non-narrow, base well above chance and below ceiling, two sources, a never-touched source, every cell >= 150 base-correct items). Do not curate locked items by base margin or compression outcome; curate SOURCES by label quality before any model run, hash-locked.
5. Feedback mechanisms tried: mean-direction/LDA edits of existing lm_head rows (`hebb`, `swh`, feedback_v3) teach but leak and are non-specific without a contrast term; `mnH` (Hessian-metric min-norm edit, FEEDBACK_LEGAL.md) keeps 99.8-99.96% of neighbours with every cell >= 98.6% on 7-8B dev but collapses in 1 of 10 INJ draws; all are rule-4 gray. Whichever ships: run it through the real command on the real artifact (reload from disk), include a brand-new multi-token skill name, ship all calibration bytes in the size count, and never claim a control we have not beaten.
