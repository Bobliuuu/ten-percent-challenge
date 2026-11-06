# PLAN v2 (2026-11-06 02:10 EST) — SOURCE OF TRUTH: GOAL_README.md = github.com/DeusExperiments/ten-percent-challenge
Deadline per README: issue by **2026-11-12 17:05 UTC** (~6 days). User's own target: ASAP / "7am".

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
