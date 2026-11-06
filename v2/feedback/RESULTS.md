# v2 feedback without weight training (MMLU, stand-in Qwen2.5-0.5B-Instruct, d=896)

Code: `fb.py` (Modal, app ap-MFVbuvP2PA7PZBG1ZF36dh, log `run05.log`, full numbers `results_qwen05.json`). `fb_sst_old.py` = discarded first SST-2 attempt.
Base/task per v2/TASK.md (14B MMLU); development done on the 0.5B stand-in (base dev acc 0.4645, n=1987). Locked exam never touched.
Data: feedback pool = v2/base/pool.jsonl (3500 shuffled; 1000 = tuning pool, 2500 = final pool); old-knowledge dev = dev.jsonl (all 1987); tuning-old = 500 calib items; 200 calib items = anchors (only used by bias/ridge). All hyperparameters were chosen on the tuning pool + tuning-old, never dev.

## Protocol
User marks k in {1,5,10,25} items the base got WRONG (random from pool) with the right answer. No weight training. Metrics (mean of 5 seeds):
- rec: accuracy on the corrected questions when re-asked (base = 0 by construction).
- rec_shuf: same questions with choices rotated (letters change; tests that memory keys on answer text, not letter).
- rel: accuracy on up to 300 other pool questions of the same subjects (related, no repeats) vs `rel_base` (generalisation of the new knowledge).
- dev: accuracy on the full old dev set (must not drop below 0.4645).
- bytes: memory payload, d=896 (stand-in) and extrapolated to the 14B hidden size d=5120 (bytes14b); budget is 2,954,006,733.

## Methods
knn = gated retrieval memory: per correction store int8 mean-pooled mid-layer hidden state (unit-norm, centred) + correct answer string; if cosine >= tau, vote for the choice whose text equals the stored answer (override), else base untouched. knnsoft = same but additive logit boost. bias = 4-vector bias on letter logits (fit on corrections + 200 anchors). ridge = closed-form residual linear head on last hidden (int8) with anchors. icl = retrieve top-4 corrections, gated by sim, prepend as solved examples. (Tuned: knn HM tau .95; knnsoft tau .85 alpha 5; ICL HM tau .6; last-token hidden `HL` is much worse than mid-layer mean `HM` for retrieval: rec_shuf .18-.78 vs .98.)

## Results (k = 1 / 5 / 10 / 25)
| method | rec | rec_shuf | dev (base .4645) | rel gain vs base | bytes d=896 | bytes d=5120 (k=25) |
|---|---|---|---|---|---|---|
| base | 0 | .20/.28/.24/.18 | .4645 | 0 | 0 | 0 |
| **knn (gated kNN, answer-text vote)** | **1.00 all k** | **1.00 all k** | .465/.465/.465/.467 | 0/+.9/+.5/+.4pt | 2.7K/6.5K/11K/25K | 139 KB |
| knnsoft | 1.0/.96/.94/.94 | 1.0/.96/.96/.96 | .465/.465/.465/.466 | ~+.1 | same | same |
| bias (4 floats) | 0/0/0/.05 | ~base | .465/.465/.465/.463 | 0 | 16 | 16 |
| ridge head | .80/.84/.76/.78 | .00/.08/.04/.07 (below base: letter-bound) | .466/.464/.462/.460 | -.8/-.6/+.1/-.2pt | 185K | 1.06 MB (anchors dominate) |
| icl (4 shots) | 1.0/.80/.68/.65 | .40/.32/.22/.19 | .474/.476/.479/.477 (few-shot effect on 0.5B, not feedback) | -1.9/+.5/-.3/+.4pt | 2.9K..38K | 152 KB |

## Findings
1. Best: **gated kNN over frozen mid-layer mean hidden states, keyed to answer text.** 100% recall of every correction, also under reordered choices, with ZERO change in dev (0.4645 -> 0.465/0.467, within seed sd .004; never below base), at 25 corrections = 25 KB (d=896) / 139 KB (d=5120, i.e. 0.005% of the 10% budget). Corrections never hurt old knowledge because the override only fires at cosine >= .95 to a stored item.
2. Limits (honest): the memory teaches the *specific* corrected facts (and near-duplicates/reformattings). Generalisation to different questions of the same subject is ~0 (+0.4 to +0.9pt, noise level ~ same size) for every method; MMLU questions are not lexically/semantically redundant enough, unlike the lexical shift in trackB. A user "teaching" new facts is therefore guaranteed only on re-asked or paraphrased-choice items.
3. Dense/logit-space updates (bias, ridge) do not work for 4-way MCQ: bias cannot encode item-specific answers; ridge learns letters, fails under choice reorder and drifts dev down (-0.9pt at k=25). ICL is weaker and not safe (cost = longer prompts; recall decays as k grows since only 4 shots are used).
4. Retrieval key matters: last-token hidden (post-"Answer:") is dominated by the answer-letter state; mid-layer mean pooling over the prompt is the right key. Note the `HM` features need the layer-24-of-48 hidden states of the 14B quantized model (extra pass is free; same forward).
5. Integration for 14B: store per correction d=5120 int8 (+2B scale) + answer string; keep the centring mean (10 KB). Cosine gate tau ~.95 tuned on tune pool of the stand-in -- RE-TUNE tau on 14B-quant features using pool (not dev/locked). The quantized model will produce different hidden states, so the memory must be built from the shipped (quantised) model's features. Exam safety: memory empty at exam time => identical to the compressed model; or exam-time memory built only from non-locked pool corrections (fires only at sim>=.95; dev showed no change).
6. Caveat: the "dev" regression test here only checks that an unrelated-question set is untouched; a real test of "keeps the old" on 14B should additionally confirm that dev items with question stems repeated across splits (generic stems) do not hit the gate (cosine .95 on mid-layer features should be conservative; verify).
