**Title:** Proof: Qwen2.5-7B-Instruct compressed to 9.40% of FP16, tool selection (C2) 0.799 vs base 0.7675 on a locked exam, grown with feedback to 0.802

Repo with everything (code, locks, receipts, logs): https://github.com/Bobliuuu/ten-percent-challenge (see `v2/`). Both single-use exam receipts: `v2/harness/receipts/REAL_final/stage1.json` and `stage2.json`.

## Summary
| item | value | evidence |
|---|---|---|
| Base | Qwen/Qwen2.5-7B-Instruct (7,615,616,512 params), zero-shot letter prompt | `v2/DECISION.md`, `v2/commercial/TASK_C.md` |
| Base bf16 bytes | 15,231,233,024 | `v2/harness/LOCK_HARNESS_C2.json` |
| Artifact bytes (all 9 files + memory) | 1,432,413,553 = **9.404%** (budget 1,523,123,302) | receipts, `size.files` |
| Locked exam, base / candidate, stage 1 (n=2000) | 0.7675 / **0.799** | `stage1.json` |
| Paired non-inferiority (margin 1.0 pt, alpha 0.05) | diff +3.15 pt, paired 95% CI [+1.60, +4.70], exact lower bound +1.69 pt, p=0.001 | `stage1.json` |
| After feedback, stage 2 | candidate **0.802**, diff +3.45 pt, CI [+1.85, +5.05], p=0.001, `PASS_BOTH_STAGES: true` | `stage2.json` |
| Artifact tree sha256 (identical in both stages) | `6290414d96f3119799516442cab5106243dede6a5b29256f24669f0581ee9661` | both receipts |

## Rule 1: one trained model, >0.5B, commercial-class task
Base: Qwen2.5-7B-Instruct, 7.6B parameters. Task C2: given a user query, pick the correct API tool among 4 confusable candidate tools or "none of these fit" (agentic / tool use, one of the maintainer's named examples). Items are from the public xLAM / apigen function-calling data, many distinct tools, with hard TF-IDF nearest-neighbour distractors (`v2/commercial/build_c2.py`, `TASK_C.md`). The bar is the base's own score on the same locked items, computed by the harness in the same run.

## Rule 2: about 10% of FP16, no weight training
- Size is on-disk bytes of the whole artifact directory (weights, scales, indices, embedding rows, vocabulary map, tokenizer, config, the 476 KB feedback head, and the loader `predict.py`) plus the memory file, counted by `v2/harness/size_check.py`. Python libraries (torch, transformers, constriction) are not counted.
- Compression is post-training, calibration-only: block-128 Hadamard rotation, Hessian-aware GPTQ rounding onto an entropy-constrained scalar quantizer with a dead zone (step 1.20 x group rms, dead zone 0.15, group 128, 8-bit log scales), rANS entropy coding of the indices, vocabulary trimmed to 24,000 tokens chosen from C2 pool+calib text only, 4-bit embedding rows, `lm_head` reduced to the 5 option-letter rows. Calibration data: 384 x 512 tokens from the C2 calibration split. No gradient step on any task loss, no weight update from labels, no exam data. Code: `v2/altpush/push.py`, `v2/compress_C/push_c2.py`, `pack_artifact.py`.

## Rule 3: matches the base ("same score")
Operational definition, fixed before building and hashed into the harness lock: paired exact one-sided non-inferiority at a 1.0 accuracy-point margin, alpha 0.05, on the same locked items (`v2/harness/CRITERION.md`). Stage 1: base 0.7675 (95% CI [0.748, 0.786]), candidate 0.799 ([0.781, 0.816]), discordant b=94 (base only right) and c=157 (candidate only right), McNemar p=8e-5, exact lower bound on the difference +1.69 pt, well above -1.0 pt. The candidate is not worse than the base by more than 1 pt (and scored higher on this exam). We do not claim bit-identical predictions: agreement with the base on dev is about 0.86-0.89.

## Rule 4: unseen exam, locked before building
- The C2 locked file (n=2000, sha256 `96aaab77d38cb6bc21f3b6cbb9d7f991954200efc7d2e3b92599ebc95367f33a`) was hash-locked in `v2/commercial/LOCK.json` before any model was scored or compressed; the split is a deterministic content hash (sha256(task|key) mod 7: 0 locked, 1 dev, 2 calib, rest pool). The harness lock `LOCK_HARNESS_C2.json` (margin, alpha, budget pre-registered; harness file hashes recorded) was committed before stage 1 (commit `0d963d1`).
- All selection used dev or pool only. The locked file was never read or scored by us before the final runs.
- Single use: a local O_EXCL claim, an atomic Modal Dict claim, and a receipt in a volume; refusal of a second run was tested (`v2/harness/tests/`).
- Isolation: the candidate ran in a Modal Sandbox with the network blocked, a fresh image with no Hugging Face cache and no base weights, only the artifact and memory mounted read-only, and no labels (`sandbox_evidence`: network blocked, hf_cache_files=0). The base ran in a separate function only to produce the bar score. The harness was unchanged since the lock (`harness_unchanged_since_lock: true`).

## Rule 5: grows with few annotations, no weight training, no RAG
- Feedback is an 852-byte external memory file built by `ingest()` (`v2/c2_fb/`) from 6 annotations drawn from the C2 pool (never from the exam): a new tool (5 annotated queries) and a confusable-pair error (1). Annotation = the correct option index. Memory holds only per-tool running prototype means and (true, predicted) tallies, folded in closed form. No annotated item is stored or looked up, no prompt tokens change, no weights change. The memory is counted in the budget.
- The artifact tree hash is identical at both stages (`6290414d…`); only the memory changed (stage 2 memory sha256 `3cc96ec2af0b43ad054868eb6a1b237c8895623bf9a37e880287820ce1793772`).
- Learns new, keeps old (dev/pool, not the exam; `v2/c2_fb/RESULTS.md`): a new tool's unseen-query accuracy rises .855 to .927 (k=1) and .934 (k=5) across 96 held-out tools, a confusable pair's unseen pattern errors are fixed 92-100%, and dev accuracy moves by +0.6 to +0.9 pt (never lower than plain).
- Still passes the exam at stage 2: candidate 0.802, diff +3.45 pt, exact lower bound +1.94 pt, p=0.001.

## Limitations and disclosures (read these)
1. The margin is 1 pt; this shows non-inferiority, not equality. The candidate scoring above the base is probably tied to prompt sensitivity of the base's "none fits" decisions under a frozen zero-shot letter prompt; we do not claim the compressed model is genuinely more capable.
2. Quantizer settings (step multiplier, dead zone, vocabulary size) were selected using dev accuracy among a handful of configurations, so dev scores are optimistic; the locked score is the unbiased number.
3. The feedback head is a closed-form linear (LDA) model fitted on labelled calibration items from frozen features; it is used only when a memory file is present. A strict reading of "no training" could object; the fallback is plain argmax with no feedback.
4. The memory's new-tool and pair examples were chosen with the help of dev outcomes (not exam data). Stage-2 changes about 6% of dev decisions relative to stage 1 (net gain), through the head's global term as well as the annotations.
5. The artifact needs a GPU and about 3 minutes to decode (rANS decoding of 6.5B symbols). It requires torch, transformers 4.51.3 and `constriction`.
6. Process incidents: two earlier stand-in (not real) harness runs crashed because a corrupt copy of `body.bin` on the Mac was used; the artifact was rebuilt from the verified Modal copy and re-hashed. During harness development an agent accidentally printed the first 600 bytes of the unrelated MMLU locked file (not the C2 exam; nothing was derived from it).
7. Stage-1 and stage-2 each used the exam once. Receipts and logs are in `v2/harness/receipts/REAL_final/` and `v2/harness/logs/`.

## Artifact and reproduction
Artifact directory (1,432,412,701 bytes, 9 files) and its SHA-256 manifest: `v2/c2_fb/artifact_MANIFEST.json`. Commands: `v2/harness/RUNBOOK.md`.
