# Track C: protocol harness for the 10% challenge

Base: `textattack/bert-base-uncased-SST-2` (fp32 safetensors 437,958,644 B; 109,483,778 params). Locked exam: SST-2 `validation` (872).
All harness code is stdlib/numpy/scipy; the exam is touched only by `lock.py` (hash) and `final_exam.py` (once).

## Files
- `lock.py` : writes read-only `LOCK.json` (sha256 of every row `idx,label,sentence`, row count, dataset fingerprint, UTC timestamp, sha256 of splits.json, base weights and the harness files, and a hash-only count of exam sentences that also appear in train). Never overwrites. `--split dev --out X` is the stand-in.
- `final_exam.py` : the single-use exam (details below).
- `size_check.py ARTIFACT [--memory F]` : per-file true bytes (st_size, plus allocated blocks), total vs 10% of base fp32 safetensors, param ratio (stored tensor elements in .safetensors/.npz/.npy/.pt vs 109.48M). Exit 1 if over 10%. Bytes of code, tokenizer, metadata and memory all count.
- `tests/` : stand-in artifact generator (`make_toy.py`) and a base-weight-peeking artifact (`cheat/`) used to test the sandbox.

## Artifact contract
A directory containing `predict.py` with `load(artifact_dir, memory_path_or_None) -> f(list[str]) -> list[int]` (0/1). It must be fully self-contained: everything needed at inference (weights, tokenizer, code) lives in the dir, and growth after deployment lives only in the memory file.

## Procedure
1. Freeze splits (`splits.json`, done) and run `python lock.py` BEFORE building. Commit `LOCK.json`; its timestamp predates every artifact.
2. Build and choose using dev/feedback only. Run `size_check.py` freely.
3. Final exam, stage 1: `python final_exam.py --artifact A [--memory MEM0]`.
4. Feedback phase: candidate ingests feedback into the memory file only (no weight training), producing MEM1.
5. Stage 2: `python final_exam.py --artifact A --memory MEM1 --stage 2`.
Stand-in test: `--split dev --lock <dev LOCK.json> --receipt-dir <scratch>`.

## What final_exam.py does
- Pre-flight without touching the exam: artifact tree sha256, memory sha256, size gate. Stage 2 additionally refuses unless a completed stage-1 receipt exists, the artifact tree hash is identical (no weight/code change) and the memory differs.
- Claims `receipts/receipt_stageN.json` with `O_EXCL` before labels are read. A second run refuses. A crash or aborted run still consumes the stage (status `candidate_crashed` / `aborted_lock_mismatch`). Final receipts are read-only.
- Recomputes the exam content hash and requires it to equal LOCK.json. Warns (`harness_changed_since_lock`) if harness files differ from the hashes in the lock.
- Orchestrator holds labels; workers get texts only. Base model runs in its own process (same tokenization as `base_eval.py`). The candidate runs on a scratch copy of the artifact in a macOS `sandbox-exec` with network denied, `~/.cache/huggingface` unreadable (so it cannot load base weights it did not pack), HOME redirected and writes limited to scratch. Verified: the cheat artifact crashes.
- Reports: base acc, candidate acc (Clopper-Pearson 95%), discordant counts b (base-only right) and c (cand-only right), exact two-sided McNemar, and non-inferiority at margin 1 pt, alpha 0.05, H0: acc_cand - acc_base <= -0.01. The NI test is an exact multinomial test on D=c-b with the nuisance discordance rate handled by Berger-Boos (sup over a 99.9% Clopper-Pearson interval, +0.001), so it is exactly valid and not just asymptotic. Also the exact one-sided 95% lower bound on the accuracy difference (test inversion). Monte Carlo at the boundary gave type-I 1.4-2.7% (<5%). PASS needs size within 10% AND NI p<0.05 (lower bound > -1 pt).
- Stage 2 receipt also stores the stage-1 report and `PASS_BOTH_STAGES`. Stage 2 reuses stage-1 base predictions.
- Note the strict NI criterion needs about b+c <= ~10 on 872 examples; a candidate must be nearly identical to base.

## Evidence per challenge rule
1. One trained model: `LOCK.json` pins base weights sha256 and id; the base is evaluated by the harness on the same exam; receipt holds base acc/preds.
2. Keep ~10%, no weight training: `size_check` output is embedded in each receipt (`size`); the sandbox forces the artifact to carry everything; stage 2 verifies the artifact tree hash is unchanged so only memory moved. "No training of the weights" of the kept part is attested by the artifact tree hash plus the build log (outside this harness; honor-system for how weights were produced).
3. Match the base: paired report with margin 1 pt and exact bound in the receipt.
4. Unseen exam, locked until the end, never used to build or choose: `LOCK.json` timestamp and content hash taken before building; single-use O_EXCL receipts; exam hash re-verified at run; train-overlap count recorded by `lock.py` (hash-only); builders use only `splits.json` dev/feedback. Not enforceable by code: that nobody peeked; hence the lock-first ordering and receipt chain.
5. Grow with feedback, no weight training, still passes: stage 2 with changed memory only, same artifact hash, memory counted in size, and a fresh NI test on the locked exam. "Learns the new / keeps the old" on feedback data is not measured here (stand-in: evaluate on `splits.json` feedback with the same `predict.py` contract using the dev/feedback splits).

## Test log (stand-in = dev, 2000 rows)
Toy lexicon artifact (66 KB, 0.015% of base): lock written once and refused on overwrite; stage 1 and 2 ran, second run of each refused, stage 2 refused without stage 1 / on changed artifact / on unchanged memory; cheat artifact crashed in sandbox and consumed its receipt; toy fails NI as expected (75% vs 98.8% base on dev, since base trained on train).
