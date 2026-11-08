# LOCK_PLAN_C27: locked-exam plan for the C27 entry (design only, nothing built or scored here)
Entry: Ternary-Bonsai-2-27B + train-free compression (scale log-quantisation 16 steps/octave, vocab prefix 140k+33, 5-trit packing), 9.835% real. Plan follows `v2/RULES.md` section 6 and the X15 pattern (`v2/EXPERIMENT_QUEUE.md`). No locked file is opened, hashed-for-selection or scored by this document.

## 1. What is measured on the exam
- Primary metric: R2 retention = P(shipped artifact correct | base correct), base = the released ternary model reconstructed exactly (`unfold_box2.py 0 0`) AND, reported separately and honestly, the grand-parent Qwen3.8-27B (94.6% on dev, expected to fail; the maintainer decides the reading, RULES section 5).
- Both outputs from real greedy generation of the task word; the shipped artifact is run through `art_load.py` bytes, never a developer-side variant.
- Gates: overall >= 98% and every kind >= 95% (kind = class label x source, cells with >= 300 items), Wilson 95% lower bounds reported; R8 band check (base well above chance, below ceiling) from the same run.

## 2. Task and lock composition (choose ONE primary task, others as supporting blocks)
Primary: NEWS-open-tail (4 old topics + free word; feedback protocol Entertainment already built in `task_search/NEWS`).
| block | source | cells | items | note |
|---|---|---|---|---|
| in-distribution | AG News | World, Sports, Business, Technology x 400 | 1600 | buckets disjoint from CAL/SEL/REP and from every dev-large id |
| unseen source (R3 >= 500) | HuffPost (2012-2022 headlines+short desc) | same 4 classes x 400 (Technology may reduce to 300) | 1600 (>= 1500) | never used to build, tune, calibrate or probe; our dev-large used HuffPost REP items, so the lock draws only from the other id buckets and `check_disjoint.py` must pass |
| total | | 8 cells | 3200 (>= 2000, >= 300 per kind, 1600 >= 500 unseen) | |
Supporting blocks (separate manifests, same procedure, optional): TOX locked slice + Aegis S2 (tournament/TOX LOCK), INJ locked slice + S2 (tournament/INJ LOCK), T1 locked (tournament/T1). These already exist and are sealed by their lanes; C27 would only add a hash of its own run, never reselect.

## 3. Sealing procedure (X15 copy)
1. Build with `task_search/NEWS/build_news.py` locked mode by one dedicated step, items chosen by sha256 order of id inside each cell bucket, no label look beyond the bucket rule, open-tail prompt from `v2/exp/SPLITS.json`.
2. Write `manifest.jsonl` (id|kind|source|bucket) and the exam file; record sha256 of both and of the sorted `id|answer` list in `v2/exp/LOCK_C27_MANIFEST.json` (hashes only), chmod 444, exam content kept off-repo on a volume; commit the hash file BEFORE the shipped-artifact exam run.
3. `check_disjoint.py` extended with `devlarge.jsonl` ids (sha256 `a5f11a5f...3d045e`, built by `build_devlarge.py`), the 25 annotated + 200 teach + neighbour feedback ids, and all calibration ids: intersection must be empty.
4. The C27 compression uses NO data (no calibration, no Hessian, no tuning); this is recorded in the manifest as `calibration_ids: []` so "never used to build or tune" is true by construction for all sources, not only the 500.
5. Run once: base (exact reconstruction) and shipped artifact on the same prompts in one session, vLLM bf16 greedy, same batch settings; also run base twice with different batch settings to report the numerics floor (flip rate of base vs base).
6. Report: per-cell retention with Wilson bounds, overall, parse-failure rate, R8 numbers, size from `size_check.py` over the shipped directory (all files incl. tokenizer, loader, hadamard.json, any feedback delta).

## 4. Feedback block (R5/R6; only after the mechanism passes its dev gate)
- <= 25 annotated real mistakes (new word `Entertainment`, base-wrong), >= 25 unseen locked teach items (we have 200 hash-locked: `fb_teach_unseen_200.jsonl`), neighbours: the main exam classes (a feedback edit must keep the exam items; the exam run is repeated AFTER feedback).
- Controls: real vs shuffled-label vs empty; empty must change zero bytes and zero outputs.
- The shipped `answer` command must be able to emit the new word end to end (free-text parser, open-tail prompt).

## 5. Open exposures to state in the final report
- Base-identity reading (RULES section 5 not ruled), vocabulary trimming and tokenizer change (not ruled), multilingual/code behaviour after trimming (measured on proxies in NATIVE_SURVEY2 section E, not on the exam).
