# v2 TASK SPEC (frozen 2026-11-06) — base + task

**Base model:** `Qwen/Qwen2.5-14B-Instruct` (bf16/fp16), 14,770,033,664 params (untied embeddings; vocab 152064, 48 layers).
**Task:** MMLU (cais/mmlu "all", 57 subjects, 4-way multiple choice) — broad knowledge/reasoning QA, the standard commercial-class general-assistant benchmark; not toy, not narrow.
**Metric:** accuracy = argmax over the 4 letter logits; higher is better. "Match the base" = accuracy >= base accuracy on the same split.

## Numbers
| item | value |
|---|---|
| Base dev accuracy (n=1987, zero-shot, bf16, full vocab) | **0.8007** (1591/1987) — trackR r3, RESULT_JSON `base_acc` |
| Chance | 0.25 |
| fp16 base bytes (params x 2) | **29,540,067,328** (~29.54 GB) |
| 10% byte budget | **2,954,006,733 bytes** (~2.954 GB), counting EVERYTHING shipped (weights, scales, codebooks, indices, tokenizer/vocab map, config, any memory/feedback store) |
| Prior best on this base (trackR r3, w3 g128/64, vocab 32k, 4-bit emb) | ratio_bf16 0.179 / 0.167 entropy-coded, dev acc 0.754-0.756 (agree 0.88-0.89) — i.e. ~1.7x over budget and ~4.5 pts below base |

Bytes accounting reference: trackR/run.py. Dev base accuracy is evaluated on the dev split only; the locked exam is NOT yet scored for the base (do that once, at the very end, with the same harness).

## Splits (frozen; files in `v2/base/`, see `LOCK.json`)
Rule (as trackR): `h = int(sha256(subject|question|c0|c1|c2|c3)[:8],16)` over MMLU **test**; `h%7==0` -> LOCKED, `h%7==1` -> DEV, else -> POOL.
| file | n | use |
|---|---|---|
| `dev.jsonl` | 1987 | all model selection, sweeps, hyperparameters |
| `calib.jsonl` | 3816 | calibration for quantization/pruning (MMLU validation 1531 + MMLU dev 285 + 2000 seeded auxiliary_train). Disjoint from test. |
| `pool.jsonl` | 10028 | other test items (not locked, not dev): allowed for extra calibration, vocab stats, and as the source of simulated user-feedback annotations (feedback step: draw few labelled items from here, never from locked) |
| `locked.jsonl` | 2027 | **LOCKED EXAM.** Never read for building/choosing. Evaluate once at the very end. File is chmod 444; hashes in `LOCK.json`. |
Row schema: `{id, subject, question, choices[4], answer(0-3)}`. Verify: `sha256sum v2/base/*.jsonl` against `LOCK.json` `file_sha256`.
Caveat: a few question strings repeat across splits with different choices (locked∩dev 4, locked∩calib 6, locked∩pool 34 by question text only; generic stems like "Which of the following is true?"); rows are distinct items.

## Exact prompt (zero-shot; from trackR/run.py `fmt`)
```
The following are multiple choice questions (with answers) about {subject with _ -> space}.

{question}
A. {choice0}
B. {choice1}
C. {choice2}
D. {choice3}
Answer:
```
(no trailing space/newline after `Answer:`; no chat template; plain text tokenized with the Qwen2.5 tokenizer, truncated to the last 1024 tokens.) Prediction: final-position hidden state (after final norm) dotted with the lm_head rows for tokens `tok.encode(" A")[0]`..`" D"`; argmax of the 4 logits. Only those 4 lm_head rows are needed, so the full lm_head (~1.56 GB bf16) may be dropped; embedding vocab may be trimmed to tokens seen in non-locked text (trackR uses 32768 + single-char + specials, with greedy re-tokenization of OOV tokens).
Calibration text format: same prompt with answer appended as ` {letter}`.

## Rules for builders
- No weight training. Selection only on dev. Never touch `locked.jsonl` (don't even load it into a notebook). Final claim = locked accuracy vs base locked accuracy, run once.
- Always report: dev acc, agreement with base, total bytes of all shipped files vs 2,954,006,733.
- Modal rules per AGENTS.md (detach, RESULT_JSON, commit+push).

Regenerate/verify: `v2/base/build_splits.py` (refuses to run if LOCK.json exists).
