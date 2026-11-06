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

## Gate / stage-1 contract (updated on coordinator request)
`predict.load(art, None)` (stage 1, no memory) NEVER consults the head: decisions are the plain letter-logit argmax of art_c120_dz15 (verified by `verify_final.py`, see below). `head.npz` (476,506 B) sits in the artifact tree and is counted, but is only used when a non-empty memory file is supplied (lam0=1.0 term + memory terms). Reason: the head alone adds +0.8 pt on dev (.804 -> .812, pool .7788 -> .7952, lam0 chosen on pool), but that gain is only claimed together with feedback, so stage 1 stays identical to the plain artifact. The "before" columns below are therefore the PLAIN artifact (dev .804 in the feature-extraction run, .8025 in the packed-artifact run; 3 items differ by batch size) and the "after" columns are with memory (head on). Part of every "after" gain on dev total (+0.7 pt) is the head's lam0 term, not the annotation; retention rows should be read with that in mind.

## S2: NEW tool, k annotated queries for it (96 held-out tools, 4 folds; the head used for a tool's fold was refit WITHOUT any calib item showing a fold tool), eval = UNSEEN dev items whose gold is that tool; retention = full dev acc; 12 seeds
| k | tools | unseen dev items | new-tool acc before -> after | dev acc before -> after | annotated items acc after | memory bytes (one tool) |
|---|---|---|---|---|---|---|
| 1 | 96 | 519 | .855 -> .927 | .804 -> .812 | .989 | 436 |
| 5 | 96 | 519 | .855 -> .934 | .804 -> .811 | .990 | 436 |
| 10 | 63 (>=11 annotation items) | 380 | .861 -> .927 | .804 -> .8107 | .992 | 440 |
| 25 | 5 (only 5 tools have 26+ items) | 67 | .537 -> .583 | .804 -> .8096 | .973 | 446 |
Hard subset (tools the plain artifact mostly gets wrong, acc <= .6): k=1 14 tools / 87 items .230 -> .564; k=5 .230 -> .610; k=10 7 tools / 56 items .161 -> .503; k=25 1 tool .0 -> .037. k=25 is low-n (annotations are scarce per tool in C2). 10 new tools ~ 4.4 KB.

## S3: confusable pair (tool A wrongly picked for B), k annotated pattern errors from pool; eval = unseen dev items with the same A->B error pattern of the plain artifact, and dev items offering both tools with gold in {A,B}
| k | patterns | unseen pattern errors fixed | group acc before -> after | dev before -> after | bytes |
|---|---|---|---|---|---|
| 1 | 14 (35 err items) | 92% | .337 -> .651 | .804 -> .8117 | ~450 |
| 5 | 5 (13 err items) | 100% | .249 -> .817 | .804 -> .8115 | ~455 |
(Only patterns with >=2 dev error items are scored; k=10/25 do not exist for pairs in C2.)

## S1: systematic 'none fits' miscalibration, annotations = none-related errors from pool; unseen = dev items with none-related errors
Natural (no shift), plain .804, irrelevance recall .591: after memory k=1/5/10/25 dev .8083/.8100/.8094/.8109, irrelevance .597/.631/.630/.626, fraction of the plain none-related errors fixed 11-14% (mostly the head term; the count bias adds little because natural none errors are two-sided).
Injected systematic shift of the deployed none logit (added to the raw E logit for deployed model, annotations and eval): shift +3: plain dev .739 -> .8016/.8036/.8053/.8064 (k=1/5/10/25), unseen none-errors fixed 39-40%; shift -3: .8005 -> .807/.808/.808/.809; shift +6: .492 -> .781/.783/.788/.787 (66-68% fixed). So a strongly miscalibrated none option is repaired by 1 annotation (log-count-ratio bias, 1 scalar), and a mild natural miscalibration is not a one-dimensional problem. The none prototype (lam_n) did not beat the count bias on pool; it ships off.

## Why this is not RAG and not weight training
- No weights change (artifact tree hash identical before/after feedback; only the memory JSON differs). No gradients, no optimiser.
- No retrieval: annotated items are not stored. Memory holds per-tool sufficient statistics (a running mean and two counts). A prediction never looks up a stored case or compares a test item to stored items; the prototype score is a linear function of the item's own features, like a naive-Bayes/LDA class. 25 annotations of one tool and 1 annotation produce a memory of the same size.
- No prompt/context edits: the token ids fed to the model are byte-identical with and without memory (memory acts only on the final 5 option scores).
- Residual honesty: the memory is a learned parameter set (class means); a strict reading of "no training" could object to closed-form estimation. It is count-based statistics, the same class as the lowbit PCH head accepted in v2/lowbit_fb.
- Fallback if the head is disallowed: `TENPCT_NOHEAD=1`/lam0=0 and no head.npz: plain argmax, feedback inert (no new-tool learning). The prior-art alternative (prompt few-shot) is excluded by RULES.

## Artifact candidate
`/Users/jhome/ten-percent-artifacts/artC2_fb` = art_c120_dz15 + new `predict.py` (load(art_dir, memory_path), ingest) + `head.npz`. size_check: 1,432,412,419 B = 9.404% (PASS, <=9.9%). Memory counts against the budget (<=~10 KB used). Not committed (1.4 GB); rebuild: copy artC2_c120, delete __pycache__, copy `artifact_code/predict.py head.npz`.
