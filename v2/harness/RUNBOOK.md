# RUNBOOK: MDC-6 (primary) with the altpush Qwen2.5-7B-Instruct packed artifact. Stage 1 then stage 2.
(BitNet-2B backup: same commands with `--base-id microsoft/bitnet-b1.58-2B-4T-bf16 --base-fp16-bytes 4825641380`, and `TENPCT_TF=4.57.0` or any version >= 4.52 for both images.)
Run from `~/ten-percent/v2/harness`. ALWAYS `modal run --detach`. Do not run against the real exam until the artifact is final.

Values: base = `Qwen/Qwen2.5-7B-Instruct` (bf16). Denominator CONFIRMED from the safetensors headers: 7,615,616,512 params, `total_size` 15,231,233,024 bytes = bf16 bytes. Budget (10%) = 1,523,123,302 B. Artifact `ART=/Users/jhome/ten-percent-artifacts/art7b_c115`: 1,482,962,994 B = 9.736% -> PASS. Exam = `../alt/locked.jsonl` (n=2008), task lock `../alt/LOCK.json`.
Note: ART had a stray `__pycache__/` (counted by size_check and by the tree hash); it was removed. Always `rm -rf ART/__pycache__` and never import `predict.py` from inside ART before the run.
Artifact contract: `predict.py: load(artifact_dir, memory_path|None) -> f(rows) -> list of letters or 0-based class indices`; rows = `{id,task,text,names,inst}`, no answers. Candidate image: torch, transformers==4.51.3 (TENPCT_TF), constriction, numpy, safetensors, scipy; no network; GPU `--cand-gpu L40S` (load ~3 min, 7B bf16 body). The memory/feedback file lives OUTSIDE ART; the current predict.py ignores `memory`, so the feedback agent must ship a predict.py that consumes it BEFORE stage 1 (stage 2 requires an identical artifact tree).
Base bar score = raw bf16 Qwen2.5-7B-Instruct, zero-shot letter argmax, NO per-task bias (conservative: the candidate's shipped bias is part of the candidate).

## 0. Pre-check (Mac, light)
```
rm -rf ART/__pycache__; python size_check.py ART [--memory MEM1] --base-fp16-bytes 15231233024     # must PASS
```
## 1. Lock the harness (once, before the final run; read-only, never overwritten)
```
python lock.py --task mdc6 --exam-file ../alt/locked.jsonl --task-lock ../alt/LOCK.json --split locked \
  --base-id Qwen/Qwen2.5-7B-Instruct --base-fp16-bytes 15231233024 --out LOCK_HARNESS_MDC6_7B.json
git add LOCK_HARNESS_MDC6_7B.json && git commit -m "lock harness MDC-6 7B" && git push
```
(hashes the exam file bytes only; margin 1.0 pt and alpha .05 are pre-registered; harness file hashes are recorded and compared inside the container at run time, so do not edit harness files after this step.)
## 2. Stage 1 (single use)
```
TENPCT_GPU=A100-40GB modal run --detach exam_remote.py --lock LOCK_HARNESS_MDC6_7B.json --tag final --stage 1 \
  --exam-file ../alt/locked.jsonl --artifact ART --cand-gpu L40S
```
`TENPCT_GPU` = GPU for the base (bar score); `--cand-gpu` = GPU of the no-network candidate sandbox (omit for CPU). The artifact dir is uploaded from the Mac to the sandbox volume (1.5 GB); its tree sha256 is recorded as `artifact_tree_sha256` in the receipt (stage 2 refuses if it differs). Receipt: `receipts/REAL_final/stage1.json` (local, written by the entrypoint) and in the Modal volume (`modal volume get tenpct-v2-exam receipts/ receipts_pulled/ --force`); the full receipt is also in the app log as `RECEIPT_JSON:`. If the Mac dies, pull from the volume/log.
## 3. Feedback (not the exam)
Build MEM1 from <= k annotations drawn from `../alt/pool.jsonl` only; artifact tree must not change (weights, code, tokenizer identical).
## 4. Stage 2 (single use; artifact tree hash must equal stage 1, memory must differ, memory counted in size)
```
TENPCT_GPU=A100-40GB modal run --detach exam_remote.py --lock LOCK_HARNESS_MDC6_7B.json --tag final --stage 2 \
  --artifact ART --memory MEM1 --cand-gpu L40S
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
