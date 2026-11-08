# X15: NEWS locked-exam manifest SEAL

Sealed (UTC): **2026-11-08T08:06:36Z**, before any final artifact exists. Nothing was run on the model; no item was scored; no item text is stored (ids and sha256 only).

| field | value |
|---|---|
| manifest | `v2/exp/LOCK_NEWS_MANIFEST.json` (chmod 444) |
| manifest file sha256 | `8df842b3290604d48f7a1381ae45dc1a5f1b77061b2d0e611b801a179e6fb2fd` |
| sha256 of items list | `09c7e103eaffe9107f2428e08520660b0fceffb6d4f8ae592b076d451f9ebd89` |
| sha256 of sorted `id\|answer` list | `c0ae9df0a656ecee0c8f9e77cf7ff86afdbfc69186f71549d24ab14608fcc04a` |
| builder | `v2/exp/seal_news_lock.py` (sha256 `80082e5a48654ad74908f5ead06b82baf4169badd923fd86ac0d169e19b91b9c`) |
| prompt (X01-fixed open tail) sha256 | `ed4c93b547979c48951212afc19d9795857430b2e7426a2afbe79dce841bd1e9` |

## Composition
- 3200 items = 8 cells (AG news, HuffPost) x (World, Sports, Business, Technology) x 400.
- Per kind: 800 each (>= 300 required). Per source: AG 1600, **HuffPost 1600 (>= 500 required)**.
- Eligible before taking 400: AG 6141/6185/6168/6227 (World/Sports/Business/Technology, after repo-use exclusion); HuffPost 6561/3301/4112/1555 (all >= 400, so no cell reduction).
- Group-safe: candidates are AG bucket [80,100) and HuffPost bucket [20,100) of `build_news.bucket` (sha256 of the first 12 normalised words, mod 100), the same function as X01 (`build_splits.py`); a reprint group has one bucket, so it cannot straddle splits. The builder re-verifies that no locked group key appears in any non-locked row.
- Selection: per cell, sort by `sha256('news-lock-order|'+id)`, take the first 400. No label or text look beyond the bucket rule and the cell (class) definition.
- Exclusion: any candidate whose normalised text appears in ANY jsonl in the repo (earlier dev/calib/pool/feedback/other-design files) was dropped before selection (AG: 480 candidates dropped; HuffPost: 0).

## Never-used checks
- HuffPost: locked buckets [20,100) are used by no split (CAL, SEL, REP, GEN, feedback) nor by any repo jsonl: 0 hits in 385 files / 1,963,849 rows (`results/X15/repo_wide_overlap_check.json`). REP uses HuffPost buckets [0,20) only. So HuffPost is a source never built, tuned or calibrated on (rule 3 needs >= 500; we have 1600).
- `check_disjoint.py` after the seal: 0 violations (text hash, id, manifest-hash collisions, locked-bucket structural check) (`results/X15/disjoint_check_after_seal.json`).
- Caveat stated plainly: HuffPost buckets [0,20) (REP, spare, probe) are the same source as the locked HuffPost items, so the "never used" claim is about the locked items and buckets, not the source as a whole. The probe slice [0,10) was only used for the rule-8 health check; REP is report-only. If a reading of rule 3 demands that the whole source be untouched, REP's HuffPost part must be treated as burned for that purpose; the AG-only and BBC data then cover REP. (Flagged, not resolved here.)

## Single-run protocol (final exam; run once, after the freeze gate, outside this queue)
1. Freeze gate first: the artifact (bytes, config, answer command, feedback command, prompt) is frozen and its hash recorded. Only a frozen TIER-1 artifact (EXPERIMENT_QUEUE 1.4) reaches the exam. Nothing about the artifact may change after the exam is opened; if it does, the result is void and the exam is declared burned for that artifact.
2. Materialise: rebuild the 3200 items from the public datasets with `seal_news_lock.py`'s loader; accept only if the item list hashes equal the sealed `items_sha256`, `id_answer_sorted_sha256`, every `text_sha256`, and `prompt_sha256` equals the sealed values. Texts are written to a local file chmod 444, never committed.
3. Run order: base first, then artifact, in the same session on the SAME vLLM build and flags (chat template, temperature 0, max_new_tokens 8, strict exact-string scorer from `entry32b/scorer.py`, no constrained decoding, full lm_head), with the sealed open-tail prompt. The baseline run is identical to the candidate run in everything but the weights (same items, same batch order, same flags). Each model is run exactly once; no reruns, no resampling; a crash is resumed from the same deterministic order without changing any setting.
4. Base floor: a second base pass with a different batch order is allowed only as the pre-registered noise floor and is reported next to the result; it is never used to pick which pass to quote (the first pass is the one quoted).
5. Metrics: base accuracy per cell/kind/source (rule 8: well above chance, below ceiling). Primary: retention = P(candidate correct | base correct) over the base-correct items, overall (need >= 98%) and per kind (4 classes), per source (AG, HuffPost) and per cell (8), need >= 95% in each kind; Wilson 95% and cluster-bootstrap (group key) bounds on the lower end; the 8 cells are reported with n and Clopper-Pearson LB. Pass/fail is on the point estimate against the rule thresholds and the bound is shown beside it. Candidate accuracy and agreement are secondary.
6. Reporting: publish the per-item `id -> base pred, candidate pred` file with the result; report the retention on the unseen source (HuffPost, 1600 items) separately as the rule-3 / maintainer-check proxy. If the outcome is a fail, it is filed as a fail; no tuning on the exam; no second artifact scored on the same exam (a second artifact would need a new sealed exam).
7. Feedback exam (rule 5/6) is separate (its own sealed >= 25 unseen teach items, X09); the NEWS exam here is the "keeps the old" check and is run on the post-feedback artifact too only if the freeze gate says the feedback state is part of the shipped artifact, still once.
