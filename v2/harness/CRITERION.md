# Pass criterion (pre-registered; written before any build result)

README rule 3 says "same score". Exact equality is not a usable test (trackN: P(exact equal) <= ~20-35% even for a faithful model), so "same" is operationalised as statistical non-inferiority on the SAME locked exam, paired per item.

- Exam: v2/base/locked.jsonl, n = 2027 MMLU items. Score per item = 1 if predicted letter == gold. Base bar = Qwen2.5-14B-Instruct, exact prompt of TASK.md, scored by the harness on the same items, in the same run.
- theta = acc(candidate) - acc(base). H0: theta <= -margin. margin = 0.01 (1.0 accuracy point). alpha = 0.05.
- Test: exact unconditional one-sided test on the paired discordant counts (b = base-only-right, c = cand-only-right) with Berger-Boos nuisance handling (stats.py `ni_pvalue`, ported from trackC). PASS requires p < 0.05, i.e. the exact one-sided 95% lower confidence bound on theta is > -1.0 pt.
- Also reported (not gating): both accuracies with Clopper-Pearson CIs, McNemar p, the exact lower bound, discordant counts.
- Full PASS = size gate (total bytes of ALL files + memory <= 0.10 x base FP16 bytes = 2,954,006,733) AND NI at stage 1 AND NI at stage 2 AND sandbox evidence (network blocked) AND harness hashes unchanged since lock.

Why 1 point: (a) base-vs-itself noise: bf16 kernel/batch changes flip a few MMLU answers (~0.1-0.3 pt), so the margin must exceed this; (b) at n=2027 the standard error of a paired difference with 3% discordance is ~0.4 pt, so a 1-pt margin is the smallest round number a faithful model can pass with good power (see power_n2027_m1.csv); (c) 1 pt is below the gaps routinely called "same" in the quantization literature (typically 1-2 pts on MMLU). A tighter margin (0.5 pt) is infeasible at this n unless the candidate is near-lossless (discordance < ~1%). The margin is stored in LOCK_HARNESS.json and cannot change after locking. Honest limit: a passed NI test at 1 pt allows a true drop up to 1 pt; we disclose the point estimate and bound in the issue and do not claim exact equality.

## Power at the real n=2027, margin 1 pt (power_n2027_m1.csv, exact enumeration; d = fraction of items where base and candidate correctness differ)
P(pass) at true diff 0: d=2% -> 0.85, d=5% -> 0.52, d=10% -> 0.33. At true diff -0.5 pt: 0.29 / 0.17 / 0.12. At true diff -1.0 pt: <= 0.03 (the test controls false passes). Prior best on dev: agreement 0.88-0.89 (d ~ 11-12% in argmax; correctness-discordance is lower) and accuracy 4.5 pts below base, so today's artifacts would FAIL; a real pass needs near-lossless decisions (correctness-discordance <~3%) or a net-positive diff.
