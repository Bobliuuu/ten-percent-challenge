# RUNBOOK: MDC-6 (primary) with the BitNet-2B ternary artifact. Stage 1 then stage 2.
Run from `~/ten-percent/v2/harness`. ALWAYS `modal run --detach`. Never run these against the real exam until the artifact is final.
Values: base = `microsoft/bitnet-b1.58-2B-4T-bf16` (bf16 master), base bytes 4,825,641,380, budget = 482,564,138 B (10%). Exam = `../alt/locked.jsonl` (n=2008), task lock = `../alt/LOCK.json`.
Artifact dir `ART` must contain `predict.py` with `load(artifact_dir, memory_path|None) -> f(rows) -> list of letters "A".."N"` where rows = `{id,task,text,names,inst}` (no answers; format the prompt exactly as `../alt/base_eval.py` `fmt`: predict argmax over the first len(names) letters). It may use only torch/transformers/numpy/safetensors/scipy (candidate image) and no network. Memory (feedback) file lives OUTSIDE ART.

## 0. Pre-checks (Mac, light)
```
python size_check.py ART [--memory MEM1] --base-fp16-bytes 4825641380      # must say PASS; every file counts
```
## 1. Lock the harness (once, BEFORE the final artifact run; read-only, never overwritten)
```
python lock.py --task mdc6 --exam-file ../alt/locked.jsonl --task-lock ../alt/LOCK.json --split locked \
  --base-id microsoft/bitnet-b1.58-2B-4T-bf16 --base-fp16-bytes 4825641380 --out LOCK_HARNESS_MDC6.json
git add LOCK_HARNESS_MDC6.json && git commit -m "lock harness MDC-6" && git push
```
(hashes the exam file bytes only; margin 1.0 pt and alpha .05 are pre-registered in the lock.)
## 2. Stage 1 (single use)
```
TENPCT_GPU=A100-40GB modal run --detach exam_remote.py --lock LOCK_HARNESS_MDC6.json --tag final --stage 1 \
  --exam-file ../alt/locked.jsonl --artifact ART --cand-gpu A10G
```
`TENPCT_GPU` = GPU for the base (bar score); `--cand-gpu` = GPU of the no-network candidate sandbox (omit for CPU). Receipt: `receipts/REAL_final/stage1.json` (local, written by the entrypoint) and in the Modal volume (`modal volume get tenpct-v2-exam receipts/ receipts_pulled/ --force`); the full receipt is also in the app log as `RECEIPT_JSON:`. If the Mac dies, pull from the volume/log.
## 3. Feedback (not the exam)
Build MEM1 from <= k annotations drawn from `../alt/pool.jsonl` only; artifact tree must not change (weights, code, tokenizer identical).
## 4. Stage 2 (single use; artifact tree hash must equal stage 1, memory must differ, memory counted in size)
```
TENPCT_GPU=A100-40GB modal run --detach exam_remote.py --lock LOCK_HARNESS_MDC6.json --tag final --stage 2 \
  --artifact ART --memory MEM1 --cand-gpu A10G
```
The exam file is already in the volume; base predictions are reused from stage 1.
## 5. Read the verdict
Receipt fields: `PASS` (size ok AND exact NI at 1 pt AND network blocked AND harness unchanged), `PASS_BOTH_STAGES` (stage 2), `REAL_RESULT`, `size`, `report`:
- `base_acc/cand_acc` with `base_ci95/cand_ci95` (Clopper-Pearson), `both_right/neither_right`, `base_only_right_b`, `cand_only_right_c` (paired discordant counts), `mcnemar_exact_two_sided_p`.
- Strict gate: `ni_exact_unconditional_p`, `exact_one_sided_lower_bound_on_diff`, `non_inferior` (margin 1 pt).
- Plain statement: `plain_statement` = difference in points with paired 95% Wald CI, plus `per_task_n_basecorrect_candcorrect`. Report BOTH in the issue: the strict NI result decides PASS; the CI statement is disclosure when NI is borderline (strict NI passes only ~36% at 2% disagreement, trackN/power for n=2008 in CRITERION.md).
## Failure handling
- Candidate crash (nonzero exit / no output): run CONSUMED, receipt `candidate_crashed`.
- Harness/infra exception before scoring: claim auto-released, record in volume `aborts/` (`labels_scored:false`); just rerun the same command. If the app is hard-killed (`modal app stop`) the claim stays: check the volume receipt says `status: started` (nothing scored), then release manually: `python -c "import modal;modal.Dict.from_name('tenpct-v2-claims').pop('<key>')"`, `modal volume rm tenpct-v2-exam receipts/<dir> -r`, `rm receipts/REAL_final/stageN.claim`, and log it in the proof. Never release a `completed` receipt.
- Pre-emption restarts of the orchestrator are tolerated (same nonce, status still `started`).
## Stand-in verification (done on DEV with Qwen2.5-0.5B-Instruct, see tests/run_tiny_e2e.sh and tests/E2E_LOG.md)
