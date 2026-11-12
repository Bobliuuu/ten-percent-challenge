# L1 deletion carrier, variant B: gradient-free saliency deletion (2026-11-12)

Feedback only ZEROES existing numbers: units = (layer, output row i, 128-group g) of the 1-bit `down_proj` of the last N layers (their fp16 scale is set to 0; the sign bits stay). No value is fitted, no gradient, no optimiser, no new tensor. The deletion set is a zlib-compressed bitmap (4096 x 96 bits per layer, row-major). Empty feedback = no bitmap = byte-identical tree (nothing is written; not re-tested through a CLI, because there is no CLI for this carrier yet).

## Method (code: `delcore.py`, `dev_sweep.py`, `run_locked_del.py`, launchers `modal_dev.py`, `modal_locked.py`)
Closed-form first-order saliency. Zeroing unit (i,g) changes row i of the layer output by dy = -s_ig * a_ig(k), a_ig = sum over the group of sign*k (k = down_proj input). With the final norm held fixed, the margin (label token minus competitor) changes by dm = -v_i s_ig a_ig, v = gamma (W_U[t]-W_U[c]) / rms(x).
- gainA = mean over the <=25 annotations of dm / gap (annotated sites where the original argmax != label token).
- gainC = the same direction (annotated label vs the model's own answer) on the 182 shipped UNLABELED calibration prompts (first answer position); dmg = calibration damage (squared row change weighted by ||gamma_i W_U[:,i]||^2, plus squared change of the own-answer margin); wanda = s_ig * ||k_g|| (activation x weight).
- Rules tried: gain (no contrast), ratio, wanda, wandaratio, **contrast = z(gainA) - mu * z(relu(gainC)) - nu * z(sqrt(dmg))** (z = divide by RMS over all units). gainC is computed per label token (weights = label frequency among the annotations), so wrong-label annotations are penalised toward THEIR labels.
- Top-k global over the last N layers. **k is chosen per draw by leave-one-out over the annotations**: held-out annotation's first answer token must be its label when the set is selected from the other 24, subject to <=5% changed first answer tokens on the unlabeled calibration prompts; ties -> smaller k. No exam, teach or dev eval data is used for k.

## DEV results (never touched locked data; seeds 0-2, 25 annotations from MASSIVE+CLINC150 cooking pools, 150 held-out Recipe items, 250 base-correct old items, shuffled/wrong arms on 100 items; `results/dev_*.json`, `agg.py`)
- Rules without contrast LEAK: gain / ratio / wanda / wandaratio reach teach .93-.95 only at k >= 4000 where the SHUFFLED-label control also teaches Recipe (.92-.95) and neighbours collapse (.0-.8): the first-order gain of "raise Recipe" is mostly generic, not text specific. Wanda-style (activation x weight) cost alone does not fix it.
- The calibration contrast (mu >= 1) is what makes it selective: contrast mu=3, nu=1, last 10 layers, 3 seeds: k=30000 teach .91 / neighbours .994; **k=50000 teach .956, neigh .991, shuffled .000, wrong-label teach .000, wrong-label neighbours .98, Alarm retention 1.0**; k=80000 teach .973 / .988. mu=1 is too weak (neigh .94 or worse at teach .95+), mu>=10 lowers teach (.80-.87 at k=30000). 6, 8, 10 layers are similar (L10 slightly best). Larger k only matters up to ~80000 units (<2% of the 3.9M candidates, bitmap 60-110 KB).

## LOCKED run (ONE run; config fixed from dev in `locked_cfg.json`: contrast mu 3, nu 1, last 10 layers, k by LOO from {1e4,2e4,3e4,5e4,8e4,1.2e5}; 8 draws x 25 annotations from `ann_pool`; teach set 164; 352 base-correct neighbours; exam 2744 re-scored after feedback for real draws 0-2 and shuffled/wrong draw 0; `results/locked_d*.json`, `results/locked_summary.txt`). Base outputs reused from `results/locked_L1_scale.json` (single bs16 reference run, not the 3-run intersection).
| arm | teach (164) | neighbours kept (352) | note |
|---|---|---|---|
| real, 8 draws | **0.942** (draws .84-.99; massive .857, hwu64 .833, mtop .964, clinc .943) | **0.9595** (draws .940-.989) | k chosen by LOO 20k-80k units, bitmap 59 KB mean |
| shuffled (Recipe on 25 unrelated items) | **0.000** all 8 | | LOO picks only 10k units |
| wrong-label | 0.000 all 8 | 0.968 | |
| empty | 0 | byte-identical (no patch written) | |
Exam after feedback (2744 items, retention of base-correct): real 0.9912 / 0.9908 / 0.9912 (Wilson lower .986-.987; gate 98% overall PASS), per-kind: Lists .944-.956 (the 95% per-kind gate fails in 2 of 3 draws), Alarm .983-.993, all others .99-1.0; Recipe leaks into the exam 3-9 items; shuffled draw 0 retention .998, wrong-label .992.
Comparison with the shipped scale carrier (locked, lam-fitted values): teach .899-.909, neighbours .9936, shuffled .001-.005. The deletion carrier teaches MORE (.942) with a cleaner shuffled control (0.000) but keeps FEWER neighbours (.9595 vs .994).

## Verdict (honest)
- Teach .942 (clear gain over 0), shuffled 0.000, wrong 0.000: **PASS** on those.
- Neighbours kept .9595 on average (worst draw .940): **FAIL** the 98% bar on the locked neighbour set (it fails because of Lists: .956; Alarm 1.0). Exam-wide retention after feedback .991 PASSes the 98% gate (Lists .944-.956 is the per-kind weak cell).
- Dev predicted neighbours .99, the locked neighbours (harder, 4 kinds incl. Lists vs Weather/Transport) .96: dev under-estimated the damage; mu/nu were not retuned on locked data. A follow-up should raise nu (calibration damage) or mu and restrict k to <=50000 (draws with k=20-50k kept .95-.99).
- Rule 4 reading (our best-evidence reading, not settled): this is the cleanest "no fitted value" carrier (zeros only, bitmap 59 KB, no new tensor); selection is closed-form first-order saliency, no gradient. GRAY->closer to PASS than the lam-fitted scale carrier, but neighbours-kept is the open weakness. Overall: **WEAK-PASS (rule 6 clean, teach better than scale, neighbours below bar)**.
- Not done: CLI (`feedback.py` integration, loader overlay of the bitmap), continual feedback, BANK6, a second locked run with retuned mu/nu.
