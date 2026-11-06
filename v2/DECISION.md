# DECISION (pre-registered 2026-11-06 ~06:45 EST, BEFORE any real locked exam run)

Primary task: **C2 (tool selection among confusable tools)**, v2/commercial/C2 (locked exam hash-locked, never read/scored).
Primary base: Qwen/Qwen2.5-7B-Instruct (bf16 15,231,233,024 B; budget 1,523,123,302 B).
Primary artifact: v2/compress_C art_c120_dz15 + v2/c2_fb feedback code (size_check 9.404%).
Fallback (only if C2 cannot complete): MDC-6 7B (v2/altpush + v2/lowbit_fb), then BitNet-2B (v2/lowbit; rules gray area).

Evidence used (dev/pool only): C2 artifact dev .8025 raw vs base .7665; feedback S2 (new tool) .865->.927 at k=1, S3 (confusable pair) 94% of unseen pattern errors fixed at k=1, retention loss <=0.2pt; MDC-6 feedback partial (fix rates 17-39%, new-label retention loss). Both artifacts verified <10%. Rule: prefer C2 when its feedback works (it does for S2/S3; S1 weak).

Pass criterion (fixed now, harness CRITERION.md): paired exact non-inferiority at 1pt margin, alpha .05, on the locked exam for stage 1 and stage 2. The plain 'difference with 95% paired CI' is reported as well. A proof is filed ONLY if stage 1 and stage 2 both PASS, size_check passes, and the artifact tree hash is identical across stages. If the strict test fails the user decides; no issue is filed automatically.

Honest notes to include in the proof: (1) the k=0 head must not change stage-1 behaviour (head gated on memory presence, see c2_fb); (2) hyperparameters (c, dz, vocab) were selected on dev; (3) harness agent printed 600 bytes of v2/base/locked.jsonl (MMLU, not used); (4) C2 agreement with base is ~0.86-0.89.
