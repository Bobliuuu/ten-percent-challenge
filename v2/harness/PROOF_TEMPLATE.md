# GitHub issue body template: complete proof for the 10% challenge
(Fill every <...>. Link evidence by commit permalink in github.com/Bobliuuu/ten-percent-challenge. Do not open until stage 1 AND stage 2 receipts both say PASS and `REAL_RESULT: true`.)

**Title:** Proof: <BASE> compressed to <X.XX>% of FP16, MMLU <cand acc> vs base <base acc>, locked exam, feedback-grown

## Summary
| item | value | evidence |
|---|---|---|
| Base | Qwen/Qwen2.5-14B-Instruct (14,770,033,664 params), MMLU zero-shot | v2/TASK.md |
| Base FP16 bytes | 29,540,067,328 | LOCK_HARNESS.json `base_fp16_bytes` |
| Artifact bytes (ALL files + memory) | <N> = <X.XX>% (budget 2,954,006,733) | receipts `size.files` (every file listed) |
| Locked exam score, base / candidate (stage 1) | <..> / <..> (n=2027) | receipt_stage1 |
| Paired non-inferiority (margin 1.0 pt, alpha .05) | diff <..> pt, exact 95% lower bound <..> pt, p=<..> | receipt_stage1 `report` |
| After feedback (stage 2) | cand <..>, lower bound <..>, PASS | receipt_stage2 |

## Rule 1: one trained model, >0.5B, commercial-class task
- Base: <id@revision>, <params> params (> 0.5B). Task: MMLU, 57 subjects, general knowledge/reasoning QA (not toy, not narrow). Evidence: TASK.md, LOCK_HARNESS.json.
- The bar is the base's own score on the same locked items, computed by the harness in the same run (base function, `base_seconds`, `_base_pred` in the receipt).

## Rule 2: about 10% of FP16, no weight training
- Size: harness `size_check.py` counts every file in the artifact directory (weights, scales, codebooks, indices, tokenizer/vocab maps, code, config) plus the memory file. Listing and total in the receipt. Archive is a self-contained directory: <path/commit>, tree sha256 <..>.
- No weight training: list of every technique with whether it uses data (calibration set <name, from calib.jsonl/pool only>), and statement that no gradient step on task loss, no labels, no exam data was used. Include the disclosure table from trackN/RESULTS.md section 2 for any calibration-based step (GPTQ-style Hessian rounding etc.) and why it is not "training the weights".
- Code that built it: <commit>.

## Rule 3: matches the base ("same score")
- Operational definition (fixed before building): paired exact one-sided non-inferiority, margin 1.0 accuracy point, alpha 0.05, same locked items for base and candidate (v2/harness/CRITERION.md, hashed in LOCK_HARNESS.json at <UTC>).
- Result: base <..>% (CI <..>), candidate <..>% (CI <..>), discordant b=<..> c=<..>, McNemar p=<..>, exact lower bound on diff <..> pt > -1.0 pt. Honest statement: this shows the candidate is not worse than the base by more than 1 pt; it does not claim identical accuracy.

## Rule 4: unseen held-out exam, locked before building
- locked.jsonl sha256 <9165abc0...> (n=2027) recorded in v2/base/LOCK.json at 2026-11-06T06:57Z (commit <..>), BEFORE any compressed artifact existed (first artifact commit <..> at <UTC>). LOCK_HARNESS.json written <UTC>, read-only, pre-registers margin/alpha/budget.
- Split rule is deterministic hash of item content (TASK.md); dev/calib/pool disjoint; all selection used dev only (logs: <links>). Overlaps by question text only: locked-dev 4, locked-calib 6, locked-pool 34 (disclosed; distinct items).
- Single use: local O_EXCL claim, atomic Modal Dict claim, receipt in volume; refusal on second run tested (tests log <link>). The exam was run once per stage: Modal app ids <..>, `modal app logs` saved at <path>.
- Isolation: candidate ran in a Modal Sandbox with block_network=True, fresh image with no HF cache or base weights, only the artifact + memory mounted read-only, no labels (`sandbox_evidence`: network blocked, hf_cache_files=0). The base ran in a separate function, only to produce the bar score.
- Harness hashes unchanged since lock (`harness_unchanged_since_lock: true`).

## Rule 5: grows with few annotations, learns new, keeps old, still passes
- Feedback: <k> annotations (labelled items drawn from pool.jsonl, never from locked), simulating a user marking the model's wrong answers. Evidence: v2/feedback/<..>.
- Mechanism: external memory file only (<format>), size <bytes> counted in the budget. Artifact tree hash identical at both stages (<sha>); only memory changed (<sha0> -> <sha1>).
- Learns new: accuracy on held-out items of the fed-back kind (dev-side test, not locked): before <..> after <..>. Keeps old: dev accuracy on unrelated items before <..> after <..>.
- Still passes the locked exam at stage 2: cand <..>, lower bound <..> pt, `PASS_BOTH_STAGES: true`.

## Reproduce
`v2/harness/README.md`; commands, image/package versions <pinned>, GPU <..>, receipts: <links>.

## Limitations and disclosures
<what is not proven; e.g. margin 1 pt; sandbox is Modal gVisor; labels in the exam volume were not read before the end; any infra aborts (claims released, nothing scored) listed from /exam/aborts>.
