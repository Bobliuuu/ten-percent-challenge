# v2 proof harness (base >0.5B LLM, 10% of FP16, locked exam, single-use, remote GPU)

Adapted from trackC. Task-agnostic except `task_base.py` (base prompt/scoring; currently MMLU per v2/TASK.md).

| file | role |
|---|---|
| lock.py | writes read-only `LOCK_HARNESS.json`: exam file sha256 (bytes only, rows never parsed), match against v2/base/LOCK.json, base id, FP16 bytes, exact byte budget, margin/alpha, harness hashes, timestamp |
| size_check.py | every file in the artifact dir + memory file vs `budget_bytes` (`--lock`) |
| stats.py / CRITERION.md / power.py | paired exact non-inferiority test, the exact criterion and its power at n=2027 |
| exam_remote.py | Modal runner. Orchestrator (CPU, has labels) -> base function (own process, network on, sees no labels) -> candidate Sandbox (`block_network=True`, GPU, fresh image, only the cand volume mounted read-only, no labels, no base weights) |
| worker.py | runs inside the sandbox; records evidence (network blocked, HF cache empty) |
| standin_predict.py, tests/ | stand-in candidates: `tiny` (prior letter + memory), `cheat` (probes network/base weights), `fullcopy` (self-contained fp16 copy of the stand-in base) |

Artifact contract: dir with `predict.py`: `load(artifact_dir, memory_path|None) -> f(rows) -> letters`, rows = `{id,subject,question,choices}` (no answers). Memory file lives OUTSIDE the artifact dir; the artifact tree hash must be identical at stage 2.

Single use per (exam, tag, stage): (1) local O_EXCL claim `receipts/<tag>/stageN.claim`, (2) atomic `modal.Dict.put(skip_if_exists)`, (3) receipt dir in the exam volume. A crashed candidate consumes the run. Receipts are also printed as `RECEIPT_JSON:` in the app log; pull with `modal volume get tenpct-v2-exam receipts/ <dir>`.

Run (always detached):
```
python lock.py --exam-file ../base/locked.jsonl --task-lock ../base/LOCK.json --split locked --base-id Qwen/Qwen2.5-14B-Instruct --base-fp16-bytes 29540067328 --out LOCK_HARNESS.json
modal run --detach exam_remote.py --lock LOCK_HARNESS.json --tag final --stage 1 --exam-file ../base/locked.jsonl --artifact ART
modal run --detach exam_remote.py --lock LOCK_HARNESS.json --tag final --stage 2 --artifact ART --memory MEM1
```
Stage-2 (rule 5): same artifact hash, only the memory file changes, counted in size; the exam is re-run once (base predictions reused). TENPCT_GPU env selects the GPU (default A100-80GB; 14B bf16 needs >=40GB).

Known limits (disclosed): the sandbox is Modal's gVisor sandbox, not a formal proof of isolation; the runner and the candidate image are pinned only by package name (pin versions before the real run); labels are visible to the orchestrator (not the candidate) in the exam volume, so the operator must not read it before the end.

## General-capability / anti-overfit check (validity/general_remote.py)
Base vs artifact on held-out general MC (MMLU, ARC-C, HellaSwag, TruthfulQA-mc1, options shuffled) + WikiText-2 perplexity and token KL; positive controls RTN4/RTN3. Feed as `R["general"]={benchmarks,ppl,kl}` to scorecard.score (gates `general_*`; `require_general=True` makes absence a failure). Result q3_8b: artifact at chance, ppl 11.1 -> 643 (KL 4.6); RTN4 control passes (acc ratio >=.93, ppl x1.08, KL .15).
