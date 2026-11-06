# PLAN (written 2026-11-06 01:50 EST)

DEADLINE AMBIGUOUS: user said "7am tomorrow" at ~01:30 on 11-06. Plan to the STRICT reading = 07:00 EST TODAY 11-06 (~5h left). If it is 11-07, extra time goes to step 4/6 polish.

Challenge (challenge.png): keep ~10% of a trained model, no weight training, match base score, pass a locked unseen exam, and grow from user feedback (learn new, keep old) without weight training.

## State at 01:46 EST 11-06
- Best artifact so far: trackA/model.bin 42.73 MB = 9.76% of fp32 BERT-SST2; dev2 acc ~0.907 vs base ~0.93 (gap to close).
- Feedback memory: trackB (frozen base + memory). External correction: trackL (~+0.6-1.0pt).
- Rules/power analysis: trackC (final_exam, NI test, 1pt margin), trackN (n=872).
- Insurance base: trackH (xlm-roberta); bigger bases: trackI.

## A. Runs in flight (all `modal run --detach`) and ETA
| run | state | ETA |
|---|---|---|
| trackQ vitl_c100, in_vitl | ~10/10 cfgs logged | done now; merge with trackQ/salvage.py |
| trackD c3,c4 | 12 results each | done; c5 running ~10-20 min |
| trackR r3 | on layer 10-11 | ~10 min |
| trackS sweep_bases | RES lines streaming | ~15-30 min |
| trackE sweep (74 plans, 40 L4) | just started | ~20-40 min |
Expected: all done by ~03:00 EST. Rate limit hit -> retry; still blocked -> AWS GPU box.

## B. Next steps (strict 07:00 today deadline)
1. 01:50-03:00 Harvest D/E/R/S/Q results (modal app logs RESULT_JSON, salvage.py); commit+push each.
2. 03:00-04:00 Pick recipe: merge best of I (GPTQ bases), J (calibration recovery), K (embedding), D (mixed-bit), E (layer/head plan), S. Target <10% bytes, agreement >=99% vs base. Gap now: trackA dev2 0.907 vs 0.93.
3. 04:00-05:00 Add trackL correction + trackB feedback memory; rebuild artifact (trackM/build.py), verify (trackM/verify.py: reload matches, bytes <=10%).
4. 05:00-05:30 Feedback test (new learned, old kept) per trackB protocol.
5. 05:30-06:15 Locked exam ONCE via trackC final_exam; never use it to choose. Fallback: trackH insurance base.
6. 06:15-07:00 README/final report; push code, results, modal-logs, scrubbed traces.
Contingency: if recipe merge misses target by 04:30, ship trackA model.bin + trackL + trackB (already works) and report the gap honestly.

## C. Rules
Follow AGENTS.md: detach everything, results in repo, push after every result, log runs in RUNS.md.
