# Final checklist: per-lane fresh-clone checks

Run on a fresh sparse clone of the pushed `origin/main` (tools + exam + lanes only, blobs fetched lazily, a few hundred MB at most; temp dir deleted afterwards).

## Usage

```
v2/tools/fresh_clone_check.sh --all                          # all lanes (L9b L1 L5 L3 L9 L4 L10 L2)
v2/tools/fresh_clone_check.sh --lane L9b                     # one lane (repeatable)
v2/tools/fresh_clone_check.sh --lane L1 --artifact-dir DIR   # artifact present (GPU box): real bytes_audit + smoke
v2/tools/fresh_clone_check.sh --all --artifact-root ROOT     # ROOT/<lane> is each lane's artifact dir
options: --ref origin/main  --no-fetch  --keep (keep clone)  --no-checklist  --out FILE  --smoke-cmd skip|"CMD"
```

Push your lane first: the clone is of `origin/main`, not the working tree. Results: `v2/tools/fresh_clone_results.json` (machine readable, merged per lane, so re-running one lane keeps the others) and the table below (rewritten between the markers; do not edit by hand). Exit code 1 if any check is FAIL. Re-running is idempotent. Lane config (fp16 denominator, declared bytes, smoke commands) lives in `LANES` in `v2/tools/fresh_clone_check.py`; add or fix a lane there when it lands.

## Checks

1. **bytes**: `v2/tools/bytes_audit.py` on the artifact, cap 10% of the start model fp16 bytes. Artifact absent (all lanes today: 1-46 GB, not in git) -> SKIPPED, with the MANIFEST-declared total/denominator printed; FAIL only if the declared ratio already exceeds 10%. A declared ratio is NOT a PASS.
2. **leak**: `v2/tools/final_leak_check.py` (exact exam-text overlap with annotation/teach/calibration/selection files, repo-wide on the clone); FAIL if any hit file is under `v2/lanes/<lane>/`; repo-wide hit files from other places are printed in the detail.
3. **sha256**: every hash in the lane MANIFEST.json (`*sha256*` dicts) and SHA256SUMS files. Repo files are hashed (mismatch = FAIL). Entries for artifacts not in the clone are reported `artifact not in clone: needs rebuild` and the status is SKIPPED (not PASS). Lanes with no hashes listed are SKIPPED with that gap named.
4. **smoke**: one `answer.py` and one `feedback.py` call on a one-item input. Needs the artifact and a CUDA device; otherwise SKIPPED-NEEDS-GPU (never PASS). Lanes without a standalone feedback entry (L4 stub, L5 CFG-driven) stay SKIPPED even if answer works.

## Status table

<!-- FCC-TABLE-BEGIN -->
| lane | bytes | leak | sha256 | smoke | date | commit |
|---|---|---|---|---|---|---|
| L12 | PASS | PASS | PASS | SKIPPED | 2026-11-10 | f33ef51 |
| L9b | SKIPPED | PASS | SKIPPED | SKIPPED | 2026-11-09 | 30ed2e9 |
| L1 | SKIPPED | PASS | SKIPPED | SKIPPED | 2026-11-09 | 30ed2e9 |
| L5 | SKIPPED | PASS | SKIPPED | SKIPPED | 2026-11-09 | 30ed2e9 |
| L3 | SKIPPED | PASS | SKIPPED | SKIPPED | 2026-11-09 | 30ed2e9 |
| L9 | SKIPPED | PASS | SKIPPED | SKIPPED | 2026-11-09 | 30ed2e9 |
| L4 | SKIPPED | PASS | SKIPPED | SKIPPED | 2026-11-09 | 30ed2e9 |
| L10 | SKIPPED | PASS | SKIPPED | SKIPPED | 2026-11-09 | 30ed2e9 |
| L2 | SKIPPED | PASS | SKIPPED | SKIPPED | 2026-11-09 | 30ed2e9 |

Details (per lane, from its last run):

- **L12**: bytes: bytes_audit.py on artifact rc=0:        104,858,748   0.1513%  layer14.safetensors / leak: 0 hits under lanes/L12/ (exam texts: 4981 from ['bank6_exam.jsonl', 'skill7_exam.jsonl']; files scanned: 280 rows: 167264; repo-wide hit files: 0) / sha: 61 verified OK, 0 MISMATCH, 0 not in clone / smoke: --smoke-cmd skip (SKIPPED-NEEDS-GPU: not run)
- **L9b**: bytes: artifact not in clone; MANIFEST-declared 6,726,881,085 B = 9.704% of fp16 69,321,221,376 B (cap 10.0%) [MANIFEST total_with_feedback; denominator text-only fp16 (stricter than full 71.9 GB)]; real bytes_audit needs artifact / leak: 0 hits under lanes/L9b/ (exam texts: 4981 from ['bank6_exam.jsonl', 'skill7_exam.jsonl']; files scanned: 275 rows: 165058; repo-wide hit files: 0) / sha: 11 verified OK, 0 MISMATCH, 5 not in clone: artifact not in clone: needs rebuild [body_bin, chat_template_jinja, manifest_json, tokenizer_config_json, tokenizer_json] / smoke: SKIPPED-NEEDS-GPU: artifact not in clone (pass --artifact-root DIR/<lane> on a GPU box); not run
- **L1**: bytes: artifact not in clone; MANIFEST-declared 1,218,467,357 B = 7.438% of fp16 16,381,470,720 B (cap 10.0%) [MANIFEST total_bytes_with_feedback_patch; denominator Qwen3-8B fp16 (MANIFEST 7.438%)]; real bytes_audit needs artifact / leak: 0 hits under lanes/L1/ (exam texts: 4981 from ['bank6_exam.jsonl', 'skill7_exam.jsonl']; files scanned: 275 rows: 165058; repo-wide hit files: 0) / sha: 1 verified OK, 0 MISMATCH, 33 not in clone: artifact not in clone: needs rebuild [bits.bin, build/LICENSE, build/NOTICE.txt, build/bits.bin, build/chat_template.jinja, build/config.json ...] / smoke: SKIPPED-NEEDS-GPU: artifact not in clone (pass --artifact-root DIR/<lane> on a GPU box); not run
- **L5**: bytes: artifact not in clone; MANIFEST-declared 6,503,994,225 B = 9.926% of fp16 65,527,752,704 B (cap 10.0%) [MANIFEST bytes.packed_body_real+carrier+code]; real bytes_audit needs artifact / leak: 0 hits under lanes/L5/ (exam texts: 4981 from ['bank6_exam.jsonl', 'skill7_exam.jsonl']; files scanned: 275 rows: 165058; repo-wide hit files: 0) / sha: no sha256 entries in MANIFEST.json and no SHA256SUMS: nothing to verify (gap: lane lists no hashes) / smoke: SKIPPED-NEEDS-GPU: artifact not in clone (pass --artifact-root DIR/<lane> on a GPU box); not run
- **L3**: bytes: artifact not in clone; MANIFEST-declared 6,042,759,448 B = 9.896% of fp16 61,064,245,248 B (cap 10.0%) [MANIFEST skill7x.total_bytes]; real bytes_audit needs artifact / leak: 0 hits under lanes/L3/ (exam texts: 4981 from ['bank6_exam.jsonl', 'skill7_exam.jsonl']; files scanned: 275 rows: 165058; repo-wide hit files: 0) / sha: no sha256 entries in MANIFEST.json and no SHA256SUMS: nothing to verify (gap: lane lists no hashes) / smoke: SKIPPED-NEEDS-GPU: artifact not in clone (pass --artifact-root DIR/<lane> on a GPU box); not run
- **L9**: bytes: artifact not in clone; MANIFEST-declared 46,527,928,070 B = 9.896% of fp16 470,187,269,120 B (cap 10.0%) [MANIFEST shipped_bytes + feedback carrier (537,001,984, counted conservatively)]; real bytes_audit needs artifact / leak: 0 hits under lanes/L9/ (exam texts: 4981 from ['bank6_exam.jsonl', 'skill7_exam.jsonl']; files scanned: 275 rows: 165058; repo-wide hit files: 0) / sha: no sha256 entries in MANIFEST.json and no SHA256SUMS: nothing to verify (gap: lane lists no hashes) / smoke: SKIPPED-NEEDS-GPU: artifact not in clone (pass --artifact-root DIR/<lane> on a GPU box); not run
- **L4**: bytes: artifact not in clone; MANIFEST-declared 6,042,513,585 B = 9.895% of fp16 61,064,245,248 B (cap 10.0%) [MANIFEST bytes_SKILL7_BANK6 total (ESTIMATE from bit counts; no packed exporter yet)]; real bytes_audit needs artifact / leak: 0 hits under lanes/L4/ (exam texts: 4981 from ['bank6_exam.jsonl', 'skill7_exam.jsonl']; files scanned: 275 rows: 165058; repo-wide hit files: 0) / sha: no sha256 entries in MANIFEST.json and no SHA256SUMS: nothing to verify (gap: lane lists no hashes) / smoke: SKIPPED-NEEDS-GPU: artifact not in clone (pass --artifact-root DIR/<lane> on a GPU box); not run
- **L10**: bytes: artifact not in clone; MANIFEST-declared 2,131,462,551 B = 9.930% of fp16 21,464,450,880 B (cap 10.0%) [MANIFEST total_bytes (accounting estimate from entropy; packed file not written)]; real bytes_audit needs artifact / leak: 0 hits under lanes/L10/ (exam texts: 4981 from ['bank6_exam.jsonl', 'skill7_exam.jsonl']; files scanned: 275 rows: 165058; repo-wide hit files: 0) / sha: no sha256 entries in MANIFEST.json and no SHA256SUMS: nothing to verify (gap: lane lists no hashes) / smoke: SKIPPED-NEEDS-GPU: artifact not in clone (pass --artifact-root DIR/<lane> on a GPU box); not run
- **L2**: bytes: artifact not in clone; MANIFEST-declared 5,346,244,844 B = 9.936% of fp16 53,808,408,928 B (cap 10.0%) [MANIFEST shipped_total_bytes (denominator back-computed from artifact report; o4 m16 final numbers in lanes/L2/MANIFEST.json after the 2026-11-09 margin run)]; real bytes_audit needs artifact / leak: 0 hits under lanes/L2/ (exam texts: 4981 from ['bank6_exam.jsonl', 'skill7_exam.jsonl']; files scanned: 275 rows: 165058; repo-wide hit files: 0) / sha: 2 verified OK, 0 MISMATCH, 24 not in clone: artifact not in clone: needs rebuild [chat_template.jinja, config.json, generation_config.json, hadamard.json, index.json, rowbase.bin ...] / smoke: SKIPPED-NEEDS-GPU: artifact not in clone (pass --artifact-root DIR/<lane> on a GPU box); not run
<!-- FCC-TABLE-END -->
