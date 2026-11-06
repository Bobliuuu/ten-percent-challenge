# PLAN (written 2026-11-06 01:50 EST) — HARD DEADLINE 07:00 EST 2026-11-07 (~29h left; aim to finish by ~02:00 on 11-07 and leave buffer)

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

## B. Next steps (ordered)
1. 02:00-03:00 Harvest: pull `modal app logs` RESULT_JSON for D/E/R/S, run salvage.py, commit+push each result (AGENTS.md rule 4).
2. 03:00-04:30 Pick recipe: combine best of I (GPTQ bases), J (calibration recovery), K (embedding), D (mixed-bit), E (layer/head pruning plan), S (bases). Target: <10% bytes with agreement >= 99% vs base on dev2/hard. Gap to close: trackA dev2 0.907 vs 0.93.
3. 04:30-06:00 Add trackL correction + trackB feedback memory on the chosen model; rebuild artifact via trackM/build.py; verify with trackM/verify.py (reload matches fake-quant, bytes <= 10%).
4. 06:00-08:00 Feedback test (step 5): new concepts learned, old dev unchanged (trackB protocol).
5. Leave exam LAST: run trackC final_exam ONCE on the locked split; never use it to choose. Fallback: trackH insurance base if SST-2 path misses.
6. Write README/final report, push everything (code, results, modal-logs, scrubbed traces).
Contingency: if recipe merge fails to reach target by 05:00, ship trackA model.bin + trackL + trackB (already works) and report the gap honestly.

## C. Rules
Follow AGENTS.md: detach everything, results in repo, push after every result, log runs in RUNS.md.
