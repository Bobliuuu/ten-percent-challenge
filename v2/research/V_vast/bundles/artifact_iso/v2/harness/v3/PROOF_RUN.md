# How the final proof run works (one page)

Inputs, all fixed BEFORE the run and hashed: base id + `base_arch.json` (tensor names/shapes/dtypes, base fp16 bytes = 2 x params); the artifact dir (`answer.py`, `feedback.py`, weights); the locked exam
`tournament/T1/data/locked.jsonl` (sha256 in `data/LOCK.json`); `ann.jsonl` (<=25 corrected items), `teach.jsonl` (>=25 unseen items the feedback is about), `nb.jsonl` (neighbors); harness file hashes (printed in every receipt).
Pre-flight (no labels, no GPU): `pytest -q` green; `checks.size_check` + `no_new_parts` on the artifact; `FEEDBACK` sets pass `validate_sets`.

```
export TENPCT_FINAL=1            # the only way the locked file is accepted
python final_proof.py --exam .../locked.jsonl --allow-locked --artifact ART --base-id ID --base-arch ART.base_arch.json \
   --annotations ann.jsonl --teach teach.jsonl --neighbor nb.jsonl --tag FINAL --out-dir RUN/ --isolation netns
```
1. Guard + claim. Locked exam accepted only with the flag + env. `receipts/FINAL/stage1.claim` is created O_EXCL; a second attempt is refused.
2. Checks on the artifact as shipped: size (all bytes vs 10% of base fp16), no-new-parts (files, tensors, config, grep), leak scans on the view. Any failure is recorded in the receipt and `all_checks_ok` is false.
3. Stage 1. A per-run CANARY is attached to the label side only. The base runs in its own process on `{id,query,tools}`. The artifact's `answer` runs under `unshare -mn` (no network, exam directory masked) on the same view; ART tree hash must be unchanged afterwards; outputs/prompts scanned for the canary and for gold calls. T1 scorer -> `base_correct`, `cand_correct`, `agree`, per-kind, per-source arrays. Receipt = hashes of exam, view/rows, scorer, harness, artifact tree, base/candidate outputs, isolation evidence.
4. Feedback lock. `FEEDBACK_LOCK.json` (read-only) stores the sha256 + ids of annotations / teach / neighbor and the artifact tree hash. Nothing is edited yet.
5. Three fresh copies of the artifact: REAL (`feedback` with the annotations), SHUFFLED (same prompts, corrected labels permuted, seeded derangement), EMPTY (zero annotations). Each runs `feedback` in the same sandbox. Recorded per copy: files changed, bytes added, size check and no-new-parts check on the edited copy. EMPTY must be byte-identical to the original (tree sha256 equality) - `empty_byte_identical`.
6. Evaluation: teach-set for base / REAL / SHUFFLED / EMPTY; neighbor-set before vs after REAL; the full locked exam again with REAL (stage 2, own claim; base outputs reused from stage 1).
7. `scorecard.score()` on the assembled input -> gates (size <=10%, retention 98%/95% per kind, exam >=2000 / kinds >=300 / OOD block >=500, base in 15-97% band, real output, no new parts, feedback <=25, teach >=25, teach gain CI, controls clearly worse, neighbors >=98%, exam after feedback >=98%, size after feedback) and the soft score vector. Output: `FINAL.scorecard_input.json`, `FINAL.scorecard.json`, `FINAL.PROOF_SUMMARY.json` (read-only; hashes of every receipt, `REAL_RESULT` = locked exam AND all checks AND netns isolation AND empty-identical).
8. Publish: RUN/ (receipts, arrays with raw model texts, lock, summary) goes in the repo with the harness commit hash; anyone can re-score the arrays from the raw texts with `scorer.py`.

Failure policy: a harness/infra failure before scoring releases the claim; a candidate crash or bad output consumes it (reported, not retried). A failed gate is reported as FAIL; nothing is re-run on the locked exam to fish for a pass.
