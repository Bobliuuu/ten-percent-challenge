# v2/c2_fb: feedback for the C2 tool-use artifact (Qwen2.5-7B, art_c120_dz15), no weight training, no retrieval, no prompt edits

Everything below is measured on C2 dev (n=2000) with annotations drawn from C2 calib+pool only. The locked file was never read or scored. Code: `artifact_code/predict.py` (+ `head.npz`), analysis `ana.py exp.py run.py`, features `feat_c2.py` (Volume tenpct-c2fb:/f), end-to-end check `verify_e2e.py`, raw results `res_s*_{tune,dev}.json`, logs `tune_*.log fin_*.log`.

## Design (closed form, all counted in bytes)
Frozen features of the DECODED artifact (one forward pass, same prompt as the shipped zero-shot letter prompt): per-option mean-pooled hidden state at 2/3 depth (layer 18/28; options follow the query in the prompt so each option vector is conditioned on the query). PCA r=64 of option vectors, shared shrunk within-class covariance (LDA), class means for "correct option" vs "wrong option" (tool options A-D and the none option E separately). Shipped fixed part = `head.npz` (476,506 B, fit on C2 calib only; inside the artifact tree).
Score per option = log-softmax(base letter logits) + lam0 * LDA_relevance(option) + memory terms. Memory (JSON, outside the artifact tree; a few hundred bytes to a few KB):
- per tool name: running prototype mean of the option vector when that tool was the correct answer (n, mu[64]); term = lam_t * LDA score toward that prototype (this is what generalises to UNSEEN queries for a new tool);
- per tool (true count, predicted count) tallies over annotated items; term = eta*log((true+a)/(pred+a)) (S3 confusable pair, and the none option via a reserved key);
- a prototype for the none option itself (optional, lam_n; ships 0).
`ingest(art, annotations, mem_in, mem_out)` does one forward pass over the annotated rows, then folds each into those statistics (no gradients). An annotation = the correct option index (4 = none fits); right/wrong is inferred by comparing with the system's own answer before the update.
Hyper-parameters (lam0=1.0, lam_t=2, n0t=2, eta=1, a=0.5, eta_n=1) were tuned only on pool (annotations from calib+pool-half-A, evaluation on pool-half-B), then run once on dev.

## k=0 gate (the shipped head is ON at k=0, it helps)
Chosen on pool: lam0=1.0 (pool acc .7788 -> .7952, irrelevance recall .594 -> .627). On dev: raw bf16-faithful artifact .804 (harness-style batch 32 re-extraction; the packed-artifact run gave .8025) -> .812 with the head (irrelevance recall .591 -> .621, relevant .852 -> .855). If the maintainer wants the exact reproduction of the .8025 headline, `TENPCT_NOHEAD=1` (or setting meta lam0=0) gives the plain argmax. The head is fit on calib, so it is part of the artifact (bytes counted), not feedback.

## S2: NEW tool, k annotated queries for it (96 held-out tools, 4 folds; the head used for a tool's fold was refit WITHOUT any calib item showing a fold tool), eval = UNSEEN dev items whose gold is that tool; retention = full dev acc; 12 seeds
| k | tools | unseen dev items | new-tool acc before -> after | dev acc before -> after | annotated items acc after | memory bytes |
|---|---|---|---|---|---|---|
| 1 | 96 | n=519 | .865 -> .927 | .8123 -> .8120 | .989 | 436 |
| 5 | 96 | n=519 | .865 -> .934 | .8123 -> .8110 | .990 | 436 |
| 10 | 63 | n=380 (tools with >=11 annotation items) | .874 -> .927 | .8122 -> .8107 | .992 | 440 |
| 25 | 5 | n=67 | .552 -> .583 | .8116 -> .8096 | .974 | 446 |
Hard subset (tools the base mostly gets wrong, base acc <= .6): k=1 13 tools (82 items) .256 -> .538; k=5 .256 -> .586; k=10 6 tools .196 -> .454. Note: k=25 only exists for 5 tools in C2 (annotations are scarce per tool; 'search' dominates), so the k=25 row is low-n and not representative. Memory per tool ~ 440 B (64 fp32 + count), 10 new tools ~ 4.4 KB.
Retention cost: -0.03 to -0.2 pt dev (mostly from the added tool-bias terms); never an improvement.

## S3: confusable pair (tool A wrongly picked for B), k annotated pattern errors; eval = unseen dev items with the same A/B error pattern (the pattern errors of the base) and dev items offering both with gold in {A,B}
| k | patterns | unseen pattern errors fixed | group acc before -> after | dev before -> after | bytes |
|---|---|---|---|---|---|
| 1 | 15 (35 err items) | 94% | .357 -> .634 | .8120 -> .8115 | ~456 |
| 5 | 6 (12 err items) | 100% | .381 -> .692 | .8120 -> .8099 | ~458 |
(Only pairs with >=2 dev error items are scored; k=10/25 do not exist for pairs in C2.)

## S1: systematic 'none fits' miscalibration, errors-only annotation (none-related errors in pool), unseen = dev items with none-related errors
Honest result: weak. Natural dev none-related errors are two-sided (over- and under-calling none), so a one-dimensional none bias cannot fix them: k=1..25 changes the fixed-fraction of those errors by +1 to +2 pt and dev by -0.5 pt (k=1) to 0 pt (k=25). With an injected systematic shift of the none logit (+3 / -3 / +6 added to the deployed model): +3 dev .802 -> .798/.802/.807/.805 (k=1/5/10/25); -3 .806 -> .809/.808/.808/.808; +6 .7625 -> .769/.789/.786/.778 (recovers up to +2.6 pt of the 4.8 pt lost). Irrelevance accuracy moves +1..+3 pt. The only real S1 gain is the k=0 head (+3 pt irrelevance recall). The prototype for 'none' (lam_n) was tested on pool and did not beat the count bias, so it ships off. Say S1 is a limited fix, not a headline.

## Why this is not RAG and not weight training
- No weights change (artifact tree hash identical before/after feedback; only the memory JSON differs). No gradients, no optimiser.
- No retrieval: annotated items are not stored. Memory holds per-tool sufficient statistics (a running mean and two counts). A prediction never looks up a stored case or compares a test item to stored items; the prototype score is a linear function of the item's own features, like a naive-Bayes/LDA class. 25 annotations of one tool and 1 annotation produce a memory of the same size.
- No prompt/context edits: the token ids fed to the model are byte-identical with and without memory (memory acts only on the final 5 option scores).
- Residual honesty: the memory is a learned parameter set (class means); a strict reading of "no training" could object to closed-form estimation. It is count-based statistics, the same class as the lowbit PCH head accepted in v2/lowbit_fb.
- Fallback if the head is disallowed: `TENPCT_NOHEAD=1`/lam0=0 and no head.npz: plain argmax, feedback inert (no new-tool learning). The prior-art alternative (prompt few-shot) is excluded by RULES.

## Artifact candidate
`/Users/jhome/ten-percent-artifacts/artC2_fb` = art_c120_dz15 + new `predict.py` (load(art_dir, memory_path), ingest) + `head.npz`. size_check: 1,432,412,419 B = 9.404% (PASS, <=9.9%). Memory counts against the budget (<=~10 KB used). Not committed (1.4 GB); rebuild: copy artC2_c120, delete __pycache__, copy `artifact_code/predict.py head.npz`.
