# FEEDBACK_EXPERIMENTS: diffuse closed-form feedback edits vs the maintainer's actual words (2026-11-08)

Code, configs, raw results: `v2/fb_edit/` (`edit.py` harness, `run_local.py`, `analyze_edit.py`, `cfgs_*.json`, `results/*.json`, `data_news/`). Dev data only (TOX dev/pool + new class `selfharm`; NEWS `fb_annot_25` / `fb_teach_unseen_200` / `items_dev`); locked exams, the filed issue and the rival's issue untouched, nothing posted. Compute: one Vast A100-80GB box (label tenpct-x09, destroyed at the end; about 1.4 USD/h), Modal queue was saturated by other jobs (the Modal job was cancelled). Companion docs: `FEEDBACK_LEGAL.md` (mnH), `research_papers/EDITING.md` (E1-E5 plan), `RULES.md` section 4/4d (the rulings).

## 0. What the maintainer actually said (RULES.md 4a-4d, short)
- #2 (ours): feedback "only works through head.npz, which is a classifier fitted to 2000 labels"; goal: improve "without any external memory/parameters".
- #3 (entrant #3): overwriting 10 lm_head rows with feedback centroids is "a nearest-centroid classifier", "the grown model is never run on the new-intent items", the answer command only lists/parses the old families, and "the 10 overwritten rows are the same head as before, just kept inside the model file".
- #1: "no retraining of the weights or RAG methods".
- Issue #4 (entrant #4, closed-form edit of existing output rows + free generation of the new skill): still OPEN with NO maintainer comment as of 2026-11-08 (checked with `gh api`, read-only). Nothing accepted yet by anyone.

Tests used below (coordinator's, + RULES 4d):
(a) lives entirely in existing weight tensors: no new tensor/file/head/memory;
(b) the model's own forward pass yields the new behaviour (diffuse, multi-layer), not one fitted readout;
(c) the new class is generated end to end in REAL greedy generation by the unmodified answer path;
(d) shuffled / wrong / empty controls clearly worse.
Plus the neighbours-kept requirement (X09: aggregate >= 0.98, every class/cell >= 0.95, no collapsed draw).

## 1. Diff against `rival_cmp` (peer branch rival-fbcmp) so nothing is duplicated
`v2/fb_legal/rival_cmp/` (legal_cmp.py, cfg_q3/q25 tox/inj/route) compares only lm_head ROW arms (mnH, hebb, the rival's LDA reference, quantised-row survival) on TOX/INJ/route. It contains no down_proj arm; the earlier `memit_down` (3 late down_proj ridge updates, neighbours 0.57-0.94, FAIL) is recorded in SOLUTIONS F02 and lacked the answer-position metric. This file adds exactly what was missing: the closed-form diffuse multi-layer arm WITH the mnH-style protection (calibration covariance of the keys, plus a heavily weighted covariance of the keys at the old-label / EOS positions), the column-sparse (1-bit body) and scale-only carriers, the gap-closing init, the Bonsai fp16 carrier, the NEWS+Entertainment protocol, and a fake-quantised-body proxy for E4. The mnH row edit is re-run inside the same harness only as the unshippable reference.

## 2. Mechanism (all closed form, no optimiser, no stored annotation, empty feedback = byte-identical weights)
For every annotated item (the user marks a base error and gives the correct label; the new label's tokens are existing vocabulary tokens): teacher-force the corrected answer through the ORIGINAL model; at each answer position where argmax != annotated token t, take the final residual x and the competitor c = best other token. Closed-form residual target `dx = rms(x) * (gap + m) * u / (u.u)`, `u = gamma * (W_U[t] - W_U[c])` (final-norm gain times the output-row difference; m = 16 or 8 logits). Over the last k layers (6 or 8), ascending: layer l delivers 1/(layers left) of the CURRENT remaining dx through `down_proj`: `dW = R K^T (lam (C + awt C_ans + ridge I) + K K^T)^-1` via Woodbury (K = the layer's MLP intermediate activations at the site positions, 12288-d; C = mean outer product of the same activations over all tokens of 300 task-format calibration prompts + the model's own greedy answers; C_ans = same over the answer / continuation(EOS) positions only, i.e. the "old-label and EOS protection in the metric"). No gradient anywhere. Edited tensors are rewritten in place (fp32 compute, stored back as the model dtype).

Best configs (chosen on the 3-draw n=12 sweeps, `results/sweep1-3.json`): F1 = last 8 layers, lam 1000, awt 10, m 16; F2 = last 6 layers, lam 1000, awt 10, m 16. Reference: mnH lm_head row edit (m 8).

## 3. Main result: Qwen3-8B bf16, TOX + new label `selfharm` (10 annotation draws per n, annotations = the base's own errors, 56 unseen teach items, 489 base-correct neighbours in 7 source x class cells, exact generated string, base teach 0.000)
Teacher-forced on cached tail layers, cross-checked by REAL greedy generation (below). Bootstrap CI over draws. "wrong" = annotation carries a wrong old label; "mixed" = half wrong; "shuffled" = the same labels moved onto unrelated ordinary inputs; "empty" = base (byte-identical; teach 0.000).
| method | n | teach real [CI] | shuffled | wrong | mixed | real - shuf [CI] | unseen-bucket (ood) | paraphrase of annotated | neigh kept | worst draw | min cell |
|---|---|---|---|---|---|---|---|---|---|---|---|
| F1 L8 | 5 | 0.480 [.43,.53] | 0.007 | 0.000 | 0.202 | +0.473 [.42,.53] | 0.535 | 0.340 | 0.9928 | 0.988 | 0.945 |
| F1 L8 | 12 | 0.664 [.63,.70] | 0.068 | 0.000 | 0.291 | +0.596 [.53,.65] | 0.682 | 0.600 | 0.9877 | 0.986 | 0.923 |
| F1 L8 | 25 | 0.768 [.74,.79] | 0.216 | 0.000 | 0.441 | +0.552 [.48,.64] | 0.718 | 0.628 | 0.9816 | 0.975 | 0.916 |
| F2 L6 | 5 | 0.430 [.38,.47] | 0.002 | 0.000 | 0.171 | +0.429 [.38,.47] | 0.500 | 0.320 | 0.9922 | 0.988 | 0.950 |
| F2 L6 | 12 | 0.618 [.58,.66] | 0.020 | 0.000 | 0.252 | +0.598 [.55,.64] | 0.647 | 0.508 | 0.9896 | 0.986 | 0.927 |
| F2 L6 | 25 | 0.734 [.71,.76] | 0.111 | 0.000 | 0.402 | +0.623 [.56,.68] | 0.706 | 0.604 | 0.9818 | 0.975 | 0.918 |
| mnH row (reference, UNSHIPPABLE) | 5 | 0.512 [.43,.59] | 0.041 | 0.000 | 0.311 | +0.471 | 0.529 | 0.420 | 0.9990 | 0.998 | 0.993 |
| mnH row | 12 | 0.659 [.60,.70] | 0.070 | 0.000 | 0.157 | +0.589 | 0.606 | 0.642 | 0.9986 | 0.996 | 0.986 |
| mnH row | 25 | 0.571 [.42,.69] | 0.021 | 0.000 | 0.050 | +0.550 | 0.559 | 0.528 | 0.9992 | 0.998 | 0.993 |

Bytes added: 0 in a bf16 body (same tensors). In a QUANTISED body the edited down_proj matrices must be fp16 overrides: 100.7 MB per 8B layer (F2 6 layers = 604 MB, F1 8 layers = 805 MB; for 32B per layer about 283 MB), so on a 9.6%-sized artifact this carrier does not fit unless it is column-sparse (section 5). Collapsed draws (neighbours < 0.9): 0 of 30 per method.
Real generation check (2 draws x {100 teach-or-neighbour items}, actual `generate`, greedy): teacher-forced vs generated agreement 1.00 (F1), 0.98-1.00 (F2); generated teach acc 0.625 / 0.571-0.607, generated neighbour acc 1.00 / 0.99-1.00. So the new class `selfharm` really is emitted by the model's ordinary greedy decoding with the unmodified prompt (which lists only the 5 old labels).
Which cell fails the per-class bar: always `S1dev|violence` (44 items) at 0.92-0.93 and, at n=25, `S3dev|safe` at 0.93-0.94; i.e. the new class leaks onto its semantic neighbour. All other cells >= 0.97. mnH has no such leak (min cell >= 0.986).

Honest read of the numbers:
- Teach is partial (0.43-0.77), not 1.0; it grows with n; the gain over base is +0.43 to +0.77 (>= +0.15 with CI > 0 everywhere; strong tier teach >= 0.3 and real - shuffled >= 0.25 passed in every row).
- Shuffled is near zero at n=5/12 but rises at n=25 (F1 0.216): the edit partly learns "emit selfharm for this kind of input" from any 25 inputs. wrong-label control is exactly 0.00; mixed (half correct) is naturally high (0.17-0.44) because half the annotations are real.
- Paraphrases of the annotated items are taught 0.32-0.63 and the unseen hash-bucket is 0.50-0.72, i.e. it generalises, but there was no second SOURCE for self-harm text (all from one dataset): the OOD-source test is NOT done for TOX (the NEWS run below has the three-source neighbour cells).
- X09 decision rule status on this protocol: teach gain, real - shuffled, empty = 0, aggregate neighbours >= 0.98 (n=25 is borderline 0.9816-0.9818), no collapsed draw: PASS. "Every class/cell >= 0.95": FAIL for F1/F2 (min cell 0.916-0.950), PASS for mnH. So the diffuse arm as tuned does NOT yet meet the per-class neighbour bar; higher awt/lam fixes the cells (sweep2: lam 1000 awt 100 gives min cell 0.97) but then teach collapses to 0.06-0.46.

### 3.1 Sweep summary (n=12, 3 draws, Qwen3-8B; `results/sweep1-3.json`)
- Without answer-position protection (awt 0): teach 0.54-0.84 but neighbours only 0.93-0.97 and shuffled 0.15-0.46 (no selectivity) = the earlier `memit_down` failure reproduced.
- With protection (awt 10-30, lam 300-1000): neighbours 0.984-0.994, shuffled 0.00-0.13, teach 0.38-0.66 (m 12-16 best). Fewer layers (L4) loses teach; L8 > L6 by about 0.05. m 4 is too weak (0.39-0.44), m 16 best.
- Ridge 0.5 or column-sparse (256 of 12288 columns, bf16 body) destroys selectivity (shuffled 0.65-0.77 > real).

## 4. E3 gap closing (init the new row from the model's own neighbouring token rows, no annotations)
Qwen3-8B, n=12, 3 draws, on top of the diffuse edit: `oldmean` (move the new label's token rows 50% toward the mean of the old labels' first-token rows): teach 0.250, neighbours 0.772. `nn` (50% toward the mean of the 8 nearest lm_head rows by cosine): teach 0.464, shuffled 0.036, neighbours 0.768. Both FAIL (neighbours < 0.8): a fixed prior that lifts the new token everywhere cannot be made selective without annotations. The gap is closed by the annotations plus the diffuse edit, not by priors. Not pursued further.

## 5. 1-bit Bonsai-8B (prism-ml/Bonsai-8B-unpacked, fp16 view of the 1-bit weights; n=12, 3 draws; base-correct neighbours 521)
| carrier | teach real | shuffled | real - shuf | neigh kept | min cell | bytes added (counted) |
|---|---|---|---|---|---|---|
| mnH row edit written as fp16 rows (reference) | 0.774 | 0.095 | +0.679 | 0.9974 | 0.986 | 3 rows x 8 KB |
| E2 scales-only (32 fp16 group scales of the target rows), m 8 | 0.845 | 0.762 | +0.083 | 0.898 | 0.713 | 0 |
| E2 scales-only, m 16 | 0.708 | 0.881 | -0.173 | 0.761 | 0.442 | 0 |
| diffuse L6 dense fp16 override, m 8 / m 16 | 0.643 / 0.786 | 0.000 / 0.024 | +0.643 / +0.762 | 0.9968 / 0.9936 | 0.992 / 0.980 | 604 MB (too big) |
| diffuse L5 dense fp16 override, m 16 | 0.762 | 0.006 | +0.756 | 0.9936 | 0.980 | 503 MB |
| diffuse L4 dense fp16 override, m 16 | 0.762 | 0.000 | +0.762 | 0.9936 | 0.985 | 403 MB |
| diffuse L6 column-sparse, top 16 / 64 / 256 columns | 0.000 / 0.000 / 0.077 | 0 | ~0 | 0.998-0.999 | 0.99 | 0.8 / 3.1 / 12.6 MB |
| diffuse L6 column-sparse, top 1024 columns | 0.589 | 0.000 | +0.589 | 0.9987 | 0.993 | 50.3 MB |
| diffuse L6 column-sparse, top 2048 columns | 0.708 | 0.000 | +0.708 | 0.9968 | 0.990 | 100.7 MB |
E2 (scales only, 0 bytes) is dead: it teaches but is not selective. E5 (rank-1 sparse columns of 16-256) is dead: the MLP key activations at the answer position are not sparse enough (teach 0.0-0.08). Needing 1024-2048 columns of each of 6 down_proj layers (50-101 MB, 0.3-0.6% of the fp16 size) works: teach 0.59-0.71 with shuffled 0.00 and neighbours 0.997-0.999. Against Bonsai's own 7.13% pack this keeps the total at about 7.4-7.7% (L4 dense: 9.6%). All of this is conditional on the unruled Bonsai base-identity reading (RULES section 5) and uses fp16 override tensors of existing weights, which are bytes that must be counted. The 3-draw/n=12 sizes are sweep-grade: no 10-draw table, no real-generation check yet on Bonsai.

## 6. NEWS + Entertainment (X09 protocol) on Qwen3-8B bf16
Setup: `task_search/NEWS` (`items_dev` AG + BBC + HuffPost, `fb_annot_25` = 25 HuffPost Entertainment items, `fb_teach_unseen_200` hash-locked teach set, `calib` as calibration and shuffle pool). Prompt = the real task prompt, menu lists only World/Sports/Business/Technology, free-text parse (exact string `Entertainment<|im_end|>`), 1237 base-correct neighbours in 12 source x class cells (AG, BBC, HuffPost), annotations = the base's own errors (18 of 25 annotated items are natural errors; n=25 therefore draws 18 distinct items). IMPORTANT: Qwen3-8B already says "Entertainment" itself on 24% of the teach items and on 6 of 25 annotated items, so base teach = 0.24, not 0 (the label is not new to the model, which makes the shuffled control stronger than on TOX).
| method | n | teach real [CI] (base 0.240) | shuffled | wrong | mixed | real - shuf [CI] | paraphrase of annotated | neigh kept | worst draw | min cell |
|---|---|---|---|---|---|---|---|---|---|---|
| F1 L8 | 5 | 0.733 [.71,.76] | 0.415 | 0.050 | 0.522 | +0.318 [.28,.35] | 0.860 | 0.9893 | 0.985 | 0.938 |
| F1 L8 | 12 | 0.796 [.79,.81] | 0.537 | 0.018 | 0.430 | +0.259 [.21,.32] | 0.892 | 0.9852 | 0.982 | 0.929 |
| F1 L8 | 25 | 0.832 [.83,.83] | 0.598 | 0.004 | 0.481 | +0.235 [.18,.29] | 0.928 | 0.9840 | 0.983 | 0.928 |
| F2 L6 | 5 | 0.715 [.69,.74] | 0.374 | 0.056 | 0.517 | +0.341 [.30,.38] | 0.780 | 0.9906 | 0.987 | 0.953 |
| F2 L6 | 12 | 0.782 [.77,.79] | 0.487 | 0.015 | 0.408 | +0.295 [.25,.34] | 0.850 | 0.9855 | 0.984 | 0.932 |
| F2 L6 | 25 | 0.814 [.81,.81] | 0.538 | 0.003 | 0.491 | +0.275 [.22,.33] | 0.889 | 0.9842 | 0.983 | 0.926 |
| mnH row (reference, UNSHIPPABLE) | 5 / 12 / 25 | 0.730 / 0.773 / 0.830 | 0.482 / 0.628 / 0.606 | 0.00 | 0.51 / 0.50 / 0.34 | +0.249 / +0.145 / +0.224 | 0.92-0.95 | 0.9922 / 0.9895 / 0.9871 | 0.986 | 0.926-0.951 |
Real generation (2 draws, 100 items): teacher-forced vs generated agreement 1.00 on teach, 0.98-0.99 on neighbours; generated teach 0.72-0.77. Bytes: 0 in a bf16 body (F2/F1 as fp16 override on a quantised 8B body: 604 / 805 MB).
Per source: neighbours kept AG 0.992-0.993, BBC 0.984-0.989 (the unseen-source cells of the maintainer-style check), HuffPost 0.968-0.969 (the source of the new class). Failing cells (< 0.95 for F1/F2): `HuffPost|Technology` 0.926-0.932 (n=68) and `HuffPost|Sports` 0.951-0.952 (n=84, borderline). The row edit (mnH) leaks onto exactly the same HuffPost cells (0.926-0.949), so this is genuine label ambiguity (celebrity-tech / sports-star stories), not an artefact of diffuse editing.
Read: teach gain +0.48 to +0.59 over base (CI > 0 everywhere); real - shuffled +0.24 to +0.34 (weak tier >= 0.10 passes everywhere; strong tier >= 0.25 passes except F1 n=25 at +0.235), but the shuffled control is large (0.37-0.60) because the model already knows the word and ANY 5-25 inputs labelled Entertainment push its prior; only the wrong-label control is clearly below base (0.00-0.06). Aggregate neighbours 0.984-0.991 (>= 0.98); per-class cell bar (>= 0.95) FAILS for the HuffPost cells. The cross-source arm (annotations from AG errors, teach from BBC/HuffPost) was NOT run: all annotations and the teach set are HuffPost Entertainment; only the neighbour cells are cross-source. A per-source teach number does not exist because no second source of Entertainment items exists in the prepared data.

## 7. Qwen2.5-7B-Instruct, TOX + selfharm
PENDING_Q25

## 8. E4 proxy: edit on a fake-quantised body (RTN 4-bit / 3-bit, group 128, all attention+MLP linears of Qwen3-8B; edited tensors fp16; lm_head fp)
PENDING_FQ
E4 proper (the real 32B 1.6 bpw compressed body) was NOT run: no compressed 32B artifact exists yet (X03/X04 are the producers) and the dense-override byte cost (about 283 MB per 32B layer) is the binding constraint, not the numerics.

## 9. Legality read, mechanism by mechanism (against the maintainer's words and tests (a)-(d))
| mechanism | (a) existing tensors only | (b) own forward pass, diffuse | (c) new class in real generation | (d) controls | maintainer-wording read | verdict |
|---|---|---|---|---|---|---|
| mnH lm_head row edit (F01) | YES (rows rewritten in place) | NO: 2-3 output rows = a linear readout fitted to <=25 hidden states | YES (agreement 0.98-1.0) | YES (+0.47..+0.68) | Exactly "the same head as before, just kept inside the model file" and "a classifier fitted to labels" (4b/4a). Caught. | UNSHIPPABLE reference only |
| Diffuse closed-form multi-layer MLP edit, bf16 body (F1/F2) | YES (down_proj of the last 6-8 layers, rewritten in place; 0 bytes) | PARTLY: the answer comes out of the whole forward pass through 6-8 changed layers, but the target is derived from the label's own output row and the solve is a ridge fit to <=25 annotated keys (an associative memory in down_proj) | YES (agreement 0.98-1.00; prompt lists old labels only) | YES on wrong (0.00) and shuffled (0.00-0.22); mixed is naturally partial | Not a head, not a centroid, not an added tensor, not RAG/prompt. Still "fitting weights to annotations" in closed form; "no retraining of the weights" is unruled for closed-form edits (RULES 4d). A hostile reader can call down_proj a key-value memory fitted to the feedback keys. | BEST AVAILABLE, GRAY; fails the per-cell >= 0.95 bar (violence cell 0.92) |
| Same, on a QUANTISED body (dense fp16 override) | YES for the tensor, but +100 MB per 8B layer / +283 MB per 32B layer | as above | as above | as above | Does not fit a 9.6% artifact except via the column-sparse carrier | NOT SHIPPABLE as dense |
| Column-sparse fp16 patch (1024-2048 columns x 6 layers), Bonsai | YES (override of existing tensors, bytes counted: 50-101 MB) | as diffuse | untested in generation on Bonsai | YES (shuffled 0.00) | Adds stored patch bytes = "new parameters" in the maintainer's eyes? It replaces columns of existing tensors, so no new tensor, but it is 0.3-0.6% extra bytes that exist only because of feedback. | GRAY, only under the Bonsai base reading |
| Scales-only (E2) | YES, 0 bytes | NO: 32 numbers of one row | YES | NO (shuffled 0.76-0.88) | Fails on behaviour (not selective) | DEAD |
| Rank-1 16-256 column patch (E5) | YES | diffuse | n/a | teach 0.0-0.08 (bf16 256 cols: shuffled > real) | Fails on behaviour | DEAD |
| Gap-closing prior (E3) | YES | n/a | n/a | neighbours 0.77 | Fails on behaviour | DEAD |
| Feedback in the requantisation (F03) | YES | YES | n/a | cannot teach | Cleanest but cannot install a label | NOT A MECHANISM |

Plain statement: no mechanism clears all of (a)-(d) AND is unambiguously outside the maintainer's wording. The diffuse edit is the only one that satisfies (b) in a defensible way and passes (a), (c), (d) behaviourally on Qwen3-8B TOX, but (i) it is closed-form weight fitting to the annotations ("no retraining of the weights" is unruled for that), (ii) it leaks 8% onto the nearest neighbour class (fails the per-class >= 0.95 bar as tuned), and (iii) it is proven only on bf16 and on fp16-view Bonsai, not on a real compressed 32B body. Everything that is a few-row output edit is caught by his #3 sentence. If the maintainer's verdict on entrant #4 (open, no comment yet) comes back "classifier", the row arms are dead and only this diffuse arm remains; if he rules that any closed-form weight edit is "retraining", every mechanism here is caught and the honest outcome is NO-SHIPPABLE-FEEDBACK.

## 10. Pass / kill (X09 rules, applied verbatim)
PENDING_RULES
