# v2/lowbit_fb: feedback stage (no weight training, no RAG/context editing) on the MDC-6 artifacts

PRIMARY: Qwen2.5-7B packed artifact (v2/altpush, 9.736% -> 9.741% with feedback files). SECONDARY: BitNet-b1.58-2B ternary (v2/lowbit, 9.453% -> 9.465%).
Only MDC-6 calib (fit), pool (tuning, annotation source), dev (reporting) were used. The locked exam was never read or scored. Nothing in v2/altpush or v2/lowbit was modified.

## What was built (all inside the artifact format; code `artifact_7b/predict.py`, `artifact/predict.py`)
- `head.npz` (counted in bytes): features of the DECODED model = concat(last-token hidden at 3/4 depth, mean-pooled hidden at 1/2 depth); the shipped head uses the mean-pooled half only (cols in meta). 96-d PCA, one shared shrunk within-class inverse covariance (alpha .1), per-(task,class) mean + count, per-task scalars (lam0, lam_fb, w, eta, n0). Fitted in closed form on MDC-6 calib (3600 rows). 7B: 746,644 B. BitNet: 547,986 B.
- Score = log-softmax(letter logits [+ altpush per-task bias for 7B]) + lam * LDA_c(P x) (+ count-ratio label bias after feedback).
- `ingest(artifact_dir, annotations, memory_in, memory_out)`: annotation = (task row, correct label name), 1-25 items. Each folds into its class's running mean/count (Bayesian posterior mean, item weight w, prior count = calib count) and into (true,predicted) tallies for a label bias eta*log((t+1)/(p+1)). A label not among the head's classes creates a new class (prior = mean of old prototypes, weight n0) and the row answers with the "Other" option when the geometry says "new" (gated: base decision kept otherwise). No gradients, no stored items, no retrieval, no prompt edit; `load(artifact_dir, memory_path)` applies it. Memory = tiny JSON of class statistics: 0.6-4.1 KB.
- Strict gate: head is used at k=0 only if total dev accuracy improves by >= +0.3 pt over without head, per task lam0 tuned on pool (0 = off for that task).
- Selection note: feature columns (both / mean-only / last-only) were chosen after looking at dev and pool numbers of the three variants (mean-only: best new-label recall and retention); disclose this as mild dev-selection. Hyperparameters (lam0, w, lam_fb, eta, n0) were tuned on pool.

## 1. k=0 head: with vs without (dev n=2009, MDC-6)
| artifact | head OFF | head ON (per-task lam0 from pool) | gate | shipped |
|---|---|---|---|---|
| 7B (with altpush per-task bias) | .8815 (real code path .8820; raw logits .8731) | .8820 (+0.05 pt, 1 item) | FAIL (< +0.3 pt) | **OFF at k=0** (lam0=0 for all tasks); head state kept only as the feedback substrate |
| BitNet-2B | .8522 | .8686 numpy / .8676 real code path (+1.6 pt: dbpedia .907->.960, offensive .734->.771, ag_news .828->.848; yelp/amazon/emotion off) | PASS | ON for ag_news, dbpedia, offensive |
Honest statement: for the 7B artifact the calibrated head gives no k=0 gain (its per-task bias already calibrates), so k=0 behaviour is byte-identical in decisions to the altpush artifact (real-path run: raw .8731, with bias .8820, equal to ARTIFACT.md; head ON == head OFF). For BitNet the head is a supervised calibration probe fitted on 3.6k labelled calib items and does raise dev score (disclose).

## 2. Size (v2/harness/size_check.py, base fp16 bytes)
| artifact | bytes | base fp16 | ratio | budget (10%) | PASS |
|---|---|---|---|---|---|
| 7B artifact_7b (+head.npz 746,644 B, predict.py 11,435 B vs 5,353) | 1,483,715,720 | 15,231,233,020 | 9.741% (<= 9.9% target) | 1,523,123,302 | yes |
| BitNet artifact | 456,731,417 | 4,825,641,380 | 9.465% (<= .095) | 482,564,138 | yes |
Memory file adds <= 4.2 KB on top (counted by size_check --memory). Large files (body.bin/emb.bin/ternary.bin/emb.bin) are NOT in git (hashes in `artifact_MANIFEST.json`); copies: ~/ten-percent-artifacts/art7b_c115_fb, Modal volume lowbit-data:/art7b, lowbit-data:/art/bitnet2b_v42k_e4.

## 3. Stage-2 protocol on DEV/POOL, 7B artifact (primary)
Metrics: fix = accuracy on UNSEEN dev items the shipped system got wrong (same kind of error); acc = same-task dev accuracy; ok = retention = accuracy on same-task dev items the shipped system had right (old dev); rec = accuracy on the annotated (corrected) items re-asked; other tasks' dev predictions are unchanged by construction (verified True in the real code path). Annotations = k pool items the shipped system got wrong (10 seeds, numpy simulation, `results/ana_fb_7b_mean.log`, `results_fb_7b_mean.json`; k0 = shipped system).
### B. Fixing systematic errors (7B, "combo" = shipped; bias-only = fallback)
| task (k0 dev acc) | k | fix | acc | ok (retention) | rec | bias-only fix / acc / ok |
|---|---|---|---|---|---|---|
| ag_news (.875) | 1/5/10/25 | .11/.14/.15/.17 | .878/.882/.879/.883 | .988/.987/.983/.985 | .60/.34/.22/.21 | .08/.11/.14/.15, acc .879-.891, ok >= .994 |
| dbpedia (.978) | 1/5/10/25 | .22/.21/.28/.39 | .981/.981/.983/.986 | .998/.999/.999/1.00 | .80/.56/.58/.54 | .04/.18/.32/.44, ok >= .999 |
| yelp (.963) | 1/5/10/25 | .10/.11/.10/.05 | .964/.959/.959/.956 | .997/.991/.992/.991 | .50/.30/.20/.23 | .09/.11/.10/.05 |
| amazon (.963) | 1/5/10/25 | .14/.17/.15/.24 | .959/.962/.963/.971 | .990/.992/.995/.999 | .40/.28/.16/.15 | .12/.11/.04/.04 |
| emotion (.557) | 1..25 | .005-.008 | .558-.559 | .998-1.00 | .00-.04 | same (nothing to learn: errors idiosyncratic) |
| offensive (.853, n=109) | 1/5/10/25 | .12/.09/.10/.11 | .820/.801/.804/.781 | .941/.923/.925/.897 | .70/.32/.37/.32 | bias-only fix .08/.02/.02/.02, acc .850/.837/.835/.820, ok .983..958 |
Real code path (ingest -> memory file -> predict, 3 seeds, 7B on L40S, `results/verify7b.json`): ag_news k=1/5/10/25 fix .13/.17/.17/.18, task acc .882/.884/.879/.882, ok .990/.987/.981/.983, memory 616/1782/1782/1784 B; dbpedia fix .22/.11/.11/.11, acc .980/.978/.978/.979, ok .997/.998/.998/.999, 638-4142 B; yelp fix .11/.02/.07/.00; offensive fix .15/.12/.17/.06, ok .935/.925/.907/.900. Total dev stays .8779-.8839 vs .8820 (<= +/-0.4 pt; the only real loss is offensive, dev n=109).
Reading: honest, modest. Errors are fixed on ~10-40% of unseen same-kind errors (ag_news, dbpedia, amazon), ~0 where errors are idiosyncratic (emotion, yelp at large k); retention stays >= .98 except offensive (small, hard dev: .90-.94, i.e. net harm; recommend leaving that task's feedback to the bias-only fallback, ok >= .96).
### A. NEW label (7B; label absent from options, base can only say "Other"; head refit without the class; k annotated items; new = recall on UNSEEN dev items of the class; old = accuracy on the other classes, base before feedback in brackets)
| task | k=1 | k=5 | k=10 | k=25 |
|---|---|---|---|---|
| dbpedia/Film (base new 0, old .976) | new .48 old .951 | .80 / .965 | .88 / .963 | .93 / .966 |
| ag_news/Sci-Tech (base new .44, old .872) | .53 / .868 | .71 / .855 | .79 / .845 | .84 / .831 |
| emotion/Love (base new .60, old .271) | .60 / .270 | .66 / .246 | .71 / .238 | .75 / .226 (fa .67-.72: new label is confusable with Joy; fails retention) |
Comparator "bias" (just raising Other until the k items pass): old collapses (.83->.05). Real code path, dbpedia Film via ingest with a fresh class name, 5 seeds (`results/verify7bA.json`): k=1/5/10/25 new .46/.71/.81/.92, old .928/.960/.950/.963 (base old .973), memory 578 B constant.

## 4. Stage-2 on BitNet-2B (secondary), same protocol (`results/ana_fb_bitnet_mean.log`)
- k=0: ON for 3 tasks (above). Fixing errors (combo; fix at k=1/5/10/25 with ok): ag_news .24/.22/.21/.23 (ok .97); dbpedia .21/.36/.38/.44 (ok .96 -> .93), bias-only fix .08/.46/.53/.68 with ok .996-.974 (better trade-off here); yelp .37/.46/.45/.55 (ok .99 -> .95); amazon .37/.47/.46/.53 (ok .98 -> .94); emotion ~.05-.07; offensive .15-.22 (ok .94 -> .93).
- New label: dbpedia/Film k=1/5/10/25 new .42/.69/.82/.89 old .891/.897/.897/.896 (base .903); ag_news .38/.49/.59/.69 old .885->.848 (base .899); emotion fails retention.
- Real path (dbpedia, errors ingested, `results/verifybit.json`): task acc .925/.938/.900/.918 at k=1/5/10/25 vs .960 at k=0 (retention costs 2-6 pts on dbpedia at large k; bias-only fallback keeps .96); memory 625-4110 B; other tasks unchanged.
Using the both-halves feature (last-token + mean) instead of mean-only gave much worse new-label recall for BitNet (dbpedia new .50 vs .86 at k=10, pool), `ana_fb_bitnet_both.log`.

## 5. Why this is not RAG and not weight training
- No weights change: artifact files are byte-identical; feedback only changes the memory JSON (class means, counts, tallies). The tree hash of the artifact is the same at stage 1 and 2 (head.npz is part of the tree and fixed).
- Nothing retrieved: the memory holds per-class sufficient statistics (a mean vector and a count per class, 96 floats), not items. Annotations are consumed and discarded; the state after k=25 has the same size as after k=1 (bytes grow only with the number of distinct classes touched, max 14 classes x ~380 B). Predictions on unseen items come from class geometry; there is no similarity search, top-k, nor stored-answer lookup (contrast: v2/feedback kNN).
- No context editing: the model input tokens are identical before/after feedback; adjustment happens on logits via a fixed-size head over frozen features.
- Residual risk (disclose): a maintainer who reads "no retraining" as "no learned parameter changing at all" could object to the running means; and the k=0 head (BitNet) is fitted from labelled calib data in closed form (not gradient-based).
## 6. Fallbacks
(a) Frozen head only / head OFF at k=0 and feedback = count-fitted per-task label bias (<= 14 floats/task): the `bias-only` columns above (7B dbpedia fix .04-.44 at ok 1.00; ag_news .08-.15 at ok >= .994; amazon .04-.12; yelp .05-.09; offensive ~0.02-.08). It satisfies retention more safely but gives smaller fixes and cannot add a new label (the bias-only new-label comparator trades old accuracy away: old .83->.05).
(b) BitNet: k=0 head fixed (gate passed), feedback = bias only.
Limitations: dev numbers on n=100-400 per task (SE 1-4 pt); 7B real-path checks use 3-5 seeds; emotion/Love new-label and idiosyncratic errors are negative results; the locked exam has not been run with feedback; the stage-2 memory must be built from <= 25 annotations drawn from pool.jsonl only.

## Repro (all detached/box)
`feat_box.py` (features, AWS g6e L40S; Modal `feat_fb.py`/`feat7b.py` for BitNet), `ana_fb.py` (`COLS=mean OUT=.. HEADOUT=.. python ana_fb.py feats7b export`), `verify_fb.py`, `verify_fbA.py`, `verify_bit.py` (real code path), logs in `results/`. AWS box i-0aad61d1dd9c5c95e terminated; Modal app tenpct-lowbit-fb-feat (BitNet features) done; tenpct-lowbit-fb-feat7b stopped unused.
