# feedback_v3 (round 2): in-place closed-form feedback edits, strict red-team protocol
Round-1 write-up (proxies A/B/C, MEMIT sweep) is README_round1.md. Everything here is Qwen3, greedy decoding, HF generate, dev data only (T1 S1/S3 dev, TOX/INJ/SENT dev or public non-locked sources), never locked/S2 rows. No commits.
Tables are generated: `analyze5.py` -> `table5.md` (cluster bootstrap, controls), `analyze4.py` -> `table4.md` (round-2a, older item-bootstrap protocol). Code: `fb4.py`, `fb5.py`, `fb6.py` (+ `fb6c.py` copy of fb6 with the codec fix); raw per-item results `results/*.json`.

## 0. READ FIRST: data caveat
The T1 dev files were re-sliced/re-locked at 2026-11-07 00:36Z while my first T1 cells were running. 177 of the OLD dev queries are inside the NEW locked set. All runs on the old files (T1pairS/T1pairP/T1new/last-format cells, first natural-mistake attempt) are in `results_olddata/` (see OLD_DATA_NOTE.txt) and are SUPERSEDED; nothing below cites them as dev-only except where marked. Everything in section 2-5 uses the new dev files (md5 bb0286be...).
Also: the earlier "fmt=last" (name-last call format) cells were dropped from anything shippable (they only ever existed on old data).

## 1. What each edit is, and what is FITTED to data (rule 4 audit)
All edits are in place on EXISTING tensors, 0 bytes added, no gradient/optimizer. `lmhead_*` edit rows of lm_head (the unembedding of the correct output token): row[t] += alpha * ||row[t]|| * d, with t the correct token at each position where the model's greedy token != the annotated token (teacher-forced on the annotation), d = unit direction computed from the final-normed hidden states h at those positions.
| variant | direction d | fitted/derived quantities |
|---|---|---|
| lmhead_lda ("gray") | Sigma^-1 (mean_h(annotations of token t) - mu) | (1) mu, Sigma = mean/covariance of final-normed hidden states over UNLABELLED text (wikitext + 30 generic chat prompts + their base responses + 120 unlabeled calib prompts rendered in the task format); (2) per-token class mean from <=25 annotations; (3) alpha and ridge CHOSEN ON A DEV SET with labels (5 alphas, ridge 0.1) = a hyperparameter tuned on held-out labelled data. This is a few-shot LDA discriminant written into lm_head rows: gray on rule 4 (no new object, but a fitted classifier-like statistic + label-tuned alpha). |
| lmhead_plain / plain_xs ("clean-ish") | mean_h(annotations of token t), normalised; NO mu, NO covariance | only the annotation hidden-state mean; alpha is a FIXED constant declared in advance (0.03 and 0.01; chosen before seeing results, not tuned on labels). No fitted generic statistic, no label-tuned hyperparameter. (0.2 was first tried by mistake as a 'fixed' value and destroys the model: neighbours 0.00; reported as such.) |
| lmhead_cal / cal2 | as lda but alpha self-calibrated per token to flip the annotated sites by margin 1 (forward-free) ; cal2 uses mu/Sigma from the model's own output positions on calib prompts | still needs the generic stats; no dev tuning of alpha |
| memit_down | ridge-regularised least squares update of 3 late-layer MLP down_proj matrices (lam=1000), covariance from generic text | fitted generic covariance + label-driven least squares: gray/fail |
Controls: empty = untouched weights (byte-identical; gain vs empty is the teach gain); mixed = 50/50 (each annotation keeps its right label or carries the base's own wrong output); randtool = random wrong label/tool replaces the right one. Gate used: gain >= +15 pts (cluster CI lo > 0), real minus each control >= +10 pts (cluster CI lo > 0), neighbours >= 98% (point estimate; Wilson lo also shown), 0 bytes. Cluster bootstrap = resample annotation draws, then items within draw; 10 draws (fewer where noted). Neighbours = 400 base-verified-correct items re-verified under the same engine/batching (not assumed).
IMPORTANT property of the 50/50 mixed control: half of its annotations are still correct, so when a method saturates with ~12 annotations (TOX selfharm) real == mixed and the gate real-mixed>=10 cannot pass even though randtool (all wrong labels) gives 0. I report it as FAIL on the strict gate and say why.

## 2. Synthetic confusable pair on REAL T1 queries (T1pairS: a '_v2' twin of the gold tool is the correct one; distinguishing token is LATER in the name) — NEW dev data
Not a natural mistake (red-team: a cloned schema, installs a naming preference). 10 annotation draws of 25.
| model | method | base teach | teach gain (lo) | real-mixed (lo) | real-randtool (lo) | neighbours (n=400) | G2 |
|---|---|---|---|---|---|---|---|
| 8B | lmhead_lda a=0.1 (dev-chosen) | 0.12 | +0.27 (+0.24) | +0.25 (+0.19) | +0.28 (+0.25) | 0.995 | PASS (gray: fitted stats) |
| 8B | lmhead_plain a=0.03 FIXED | 0.12 | +0.22 (+0.14) | +0.21 (+0.12) | +0.24 (+0.15) | 0.994 | PASS (no fitted stat; clean on rule 4 data-fitting, still a label-driven row edit) |
| 8B | lmhead_cal2 | 0.12 | +0.57 (+0.53) | +0.18 (+0.06) | +0.61 (+0.56) | 0.996 | PASS (gray) |
| 8B | lmhead_cal | 0.12 | +0.56 | +0.20 | +0.60 | 0.954 | FAIL (neighbours) |
| 8B | memit_down | 0.12 | +0.51 | +0.13 | +0.51 | 0.935 | FAIL (neighbours) |
| 8B | lmhead_plain a=0.2 (mistaken fixed alpha) | 0.12 | -0.12 | 0.00 | 0.00 | 0.000 | FAIL (destroys model) |
| 14B | lmhead_lda a=0.2 | 0.29 | +0.48 (+0.45) | +0.44 (+0.40) | +0.51 (+0.46) | 0.987 | PASS (gray) |
| 14B | lmhead_plain a=0.03 FIXED (4 draws only, killed for time) | 0.29 | +0.18 (+0.13) | +0.15 (+0.09) | +0.21 (+0.17) | 0.994 | PASS (4 draws; below the 10-draw requirement) |
First-token vs later-token (OLD data, superseded, indicative only; results_olddata/): prefix twin (first token differs) 8B lmhead_lda +0.59, neighbours 0.99; suffix twin +0.43; memit_down failed on the prefix twin (neighbours 0.57).

## 3. NATURAL mistakes (the base's own failures on T1 dev, tools disjoint between annotation and teach) — NEW data, Qwen3-8B
Base mined on 2367 single-call dev items (1737 correct). Failure classes (n, distinct gold tools): argkeys 249 (162), argstr 154 (114), parsefail 62, argnum 56, abstain 47, sibling-tool only 14 (too few for a sibling-tool test), multi 21. Teach/annotation tools disjoint by tool-name hash; neighbours 400 base-verified, mostly on the same tools.
| class | method | base teach | gain (lo) | real-mixed | real-randtool | neighbours | G2 |
|---|---|---|---|---|---|---|---|
| argstr (string-valued argument differs) | lmhead_lda a=0.1 | 0.00 | +0.01 (+0.00) | +0.01 | +0.01 | 0.992 | FAIL (no gain) |
| argstr | lmhead_cal | 0.00 | +0.01 | +0.00 | +0.00 | 0.979 | FAIL |
| argstr | memit_down (8 draws) | 0.00 | +0.03 | +0.01 | +0.00 | 0.978 | FAIL |
argkeys (the largest class) was only run on the OLD data (indicative, superseded): lmhead_lda gain +0.08..0.35 per draw, but the random-wrong-tool control matched or beat the real labels (e.g. real 0.15 vs rand 0.25; real 0.35 vs rand 0.33): the gain was a non-specific format shift, FAIL controls; larger alpha broke neighbours (0.46 and below). The new-data argkeys run was not completed.
Plain statement: for argument-format/extra-argument mistakes the first-token lm_head edit does not work, and nothing else I tried does. No natural-mistake pass exists.

## 4. NEW TOOL (an unlisted multi-token tool 'escalate_to_specialist' for nocall queries) — NEW data, Qwen3-8B, 5 draws
| method | base | gain (lo) | real-mixed | real-randtool | neighbours | G2 |
|---|---|---|---|---|---|---|
| lmhead_lda (dev-chosen a=0.45) | 0.00 | +0.02 (+0.00) | +0.01 | +0.02 | 0.966 | FAIL (no gain) |
| lmhead_plain a=0.2 | 0.00 | +0.00 | +0.00 | +0.00 | 0.002 | FAIL (destroys) |
| lmhead_cal (per-token calibrated alpha) | 0.00 | **+0.86 (+0.77)** | +0.72 | +0.86 | **0.536** | FAIL (neighbours: also fires on ordinary calls) |
| lmhead_cal2 (contrast vs own output positions) | 0.00 | +0.08 (+0.01) | +0.04 | +0.08 | 0.659 | FAIL |
| memit_down | 0.00 | +0.45 (+0.36) | +0.39 | +0.45 | 0.584 | FAIL |
So: the method that writes the name (cal) really teaches the new multi-token call (controls ~0) but it is not selective; nothing passes. **We cannot claim a new-tool story from this mechanism.** Multi-token note: plain lda fails on multi-token names because one global alpha cannot move a rare multi-token path; per-token calibration fixes the gain but not the leakage.

## 5. NEW LABEL on the real label tasks (taxonomy extension; prompt lists the OLD labels only; <=25 annotations of new-class items the base answered with an old label; >=25 unseen new-class teach items fixed by hash before feedback; 400 verified old-class neighbours)
TOX + 'selfharm' (Aegis train 'Suicide and Self Harm', unsafe, non-locked label; neighbours from TOX s1/s3 dev+pool old classes, base ~0.6 accurate so 400 verified from 1100 candidates; teach 73, annotation pool 58 base-wrong):
| model | method | base teach | gain (lo) | real-mixed (lo) | real-randtool | neighbours | G2 |
|---|---|---|---|---|---|---|---|
| 4B-Instruct-2507 | lmhead_lda a=0.7 (10 draws) | 0.00 | +0.65 (+0.61) | +0.19 (+0.05) | +0.65 (+0.61) | 0.981 (Wilson lo 0.976) | PASS (gray: fitted stats; neighbour margin thin) |
| 4B-Instruct-2507 | lmhead_plain a=0.03 / 0.01 FIXED | 0.00 | +0.00 / +0.00 | 0 | 0 | 0.985 / 0.999 | FAIL (no gain) |
| 4B-Instruct-2507 | lmhead_cal (4 draws) | 0.00 | +0.02 | -0.08 | +0.02 | 0.917 | FAIL |
| 8B | lmhead_lda a=0.45 (10 draws) | 0.00 | +0.62 (+0.58) | -0.00 (-0.02) | +0.62 (+0.58) | 0.988 | FAIL on strict mixed gate (real == mixed: saturation, see section 1); empty and random-wrong-label controls are clearly worse |
| 8B | lmhead_plain a=0.03 / 0.01 FIXED | 0.00 | +0.03 / +0.00 | 0 | +0.03 | 0.944 / 0.977 | FAIL (fixed alpha too small to move the new token; larger fixed alpha not tried) |
SENT +'Neutral' (zeroshot/twitter-financial-news-sentiment train; prompt lists Bullish/Bearish only), Qwen3-8B, lmhead_lda a=0.45, 5 draws: base teach 0.18, gain +0.37 (lo +0.31), real-mixed +0.06 (lo 0.00), real-randtool +0.55, neighbours 0.984 (lo 0.977): FAIL on the strict mixed gate (same saturation effect), empty/random controls clearly worse. INJ +'Jailbreak' (jackhhao) did not complete (8B job killed for the box deadline; 4B Modal job stopped).

## 6. Real-codec size check (item 6)
Stand-in for the real rotation+ECSQ+rANS path (code taken from compress_A/sweep_A.py primitives: sign-diag x block-Hadamard-128 rotation, L=8 symmetric grid, per-128-group fp16 scale, ideal-rANS size = entropy of the level histogram), lm_head only, plain RTN (no GPTQ). The edit is applied to the dequantised artifact, edited rows are re-encoded (refit scales, or original scales), the histogram-based size is recomputed. The codec step crashed in all earlier runs (a variable-shadowing bug, `dev`), fixed in fb6c and re-run on 8B pairS (lmhead_lda a=0.2, 3 annotation draws). RESULT (negative): the quantised lm_head artifact is 225,358,389 bytes (2.897 bits/weight; fp16 would be 1,244,659,712). Re-encoding only the ~16 edited rows changes the size by +8..+15 bytes (about 0 bytes, ideal-rANS entropy estimate), neighbours stay 0.995, BUT the teach gain does NOT survive: +0.01 (refit scales) and +0.00 (original scales) vs the quantised base (fp edit gave +0.27..+0.40). At ~3 bits/weight the per-element edit (alpha*||row||*d spread over 4096 coordinates after rotation, a fraction of one grid step) is rounded away. So with this stand-in codec the lm_head edit is size-neutral but ineffective; keeping the edited rows in fp16 would cost about 16 rows x 4096 x 2 B = 131 KB (not 0 bytes). The RTN-4bit result from round 1 (gain survives at 4 bit g128) used finer steps and the non-rotated grid; the real codec at <=3 bits is the harsher test and it fails. Not run on the real GPTQ/ECSQ artifact. The earlier RTN-4bit lm_head codec tests (round 1/2a, `table4.md`) showed the gain survives re-quantising only the edited rows with 0 bytes added (8B pairS old data: +0.44 vs +0.43 fp16-row; 14B +0.57/+0.57/+0.57), but those were not the real codec.

## 7. What did NOT finish (stated plainly)
- INJ 'Jailbreak' new-label: not completed (no result). SENT: 8B only (5 draws, lmhead_lda only), 4B SENT not completed. 8B new-label for TOX has 10 draws for lda/plain/plain_xs, lmhead_cal 1 draw (killed for the deadline).
- The AWS box SSM credentials expired once (~04:08Z) and the user re-logged-in; one 8B nl job crashed on a diag bug and was relaunched, so some cells started late.
- Not run: 1.7B and 14B new-label; argkeys/argnum on new data; sibling-tool class (14 items); 10 draws for 4B TOX (6) and 14B plain (4).
- Single base run per cell (no noise-floor / multi-run base verification, red-team finding 12).

## 8. Verdict by method (rule 4 status | what passed)
- lmhead_plain with a small FIXED alpha (no covariance, no mean, no tuned hyperparameter): rule-4-clean in the sense of no fitted statistic. Passes G2 on the synthetic twin pair (8B 10 draws, 14B 4 draws), fails on TOX new-label and new-tool (gain too small / alpha-sensitive; 0.2 destroys the model). Alpha window is narrow (0.01 too small, 0.2 too large on 8B), so a "fixed without labels" alpha is not robust across tasks.
- lmhead_lda: GRAY on rule 4 (generic-text covariance + label-tuned alpha + class means). Passes: task B/A proxies, T1 twin pair (8B, 14B), TOX selfharm on 4B. Fails natural mistakes, new tool, and the strict mixed-control gate on TOX 8B.
- lmhead_cal / cal2: gray; cal2 passes the twin pair, cal installs the new tool but leaks.
- memit_down (late-layer least squares): fails neighbours on every real scenario here (pair 0.935, new tool 0.58, natural 0.98 with no gain), and is the most clearly "fitted weight update".
- Honest bottom line: closed-form lm_head edits teach NEW OUTPUT TOKENS/LABELS that the model can already rank (new classification label, naming preference) and do not teach argument-format/natural mistakes or new multi-token tools selectively. The only scenarios that pass are the synthetic twin and (partly) the new-label taxonomy extension; whether either counts as "what the maintainer will test" is a judgement the controls cannot make.
