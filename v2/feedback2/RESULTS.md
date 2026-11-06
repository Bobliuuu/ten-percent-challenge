# v2/feedback2: feedback with no retrieval, no context edit, no weight training (stand-in Qwen2.5-1.5B-Instruct)

Code: `feat.py` (Modal A10G, app ap-SJ6DvDkqMKmI47PMMZ3bPL; frozen-LM features -> Volume tenpct-fb2:/f2), `ana.py` (numpy, local; all numbers below from `python ana.py feats/f2 q15 8`, raw in `results_q15_L8.json`, log `ana_q15_L8.log`). 0.5B run was queued on Modal (no A10G capacity) and cancelled; 1.5B only. Locked exam never touched; hyperparameters tuned on pool, reported on dev. Features L8 = concat(last-token hidden at layer 3/4, mean-pooled mid-layer hidden), d=2x1536. 10 seeds per k.

## Chosen design: PCH = Prototype-Calibrated Head (closed-form class statistics, parametric, no stored items)
Shipped state (all counted in bytes): PCA projection P (r=96 x d), shrunk shared within-class inverse covariance (96x96), per-class mean (96 floats) + count, per-label bias vector, scalars (lambda, gate). Built ONCE from the labelled calib split (disjoint from locked).
Inference: score_c = base_logit_c + lambda * LDA_c(P x) + b_c (task B); new label: LDA argmax==new -> output new label else base/old head (task A).
Feedback update (annotation = right/wrong mark + true label, or a label for a new class): per annotated item add its projected feature into the running mean of its class (Bayesian posterior mean with prior count n0 / item weight w) and update b_c by count log-ratio of (true label counts)/(predicted-wrong counts). O(1) state per class, nothing per item is retained: after k annotations the state is the same size as after 0. No gradients, base weights untouched, prompt untouched.

## Why this is not RAG (ruling: no retraining; no RAG-style/context-editing)
- Nothing is retrieved: there is no store of items, no similarity search, no top-k, no lookup of a stored answer. Annotated examples are consumed into sufficient statistics (running class mean + counts) and discarded; they cannot be recovered.
- Context is never edited: no text, example, or label is added to the prompt; the model's input tokens are byte-identical before and after feedback.
- It acts on a fixed-size classifier head over frozen features (like naive-Bayes/LDA counting, "learn the class prior by counting"), not on memory of cases. Behavior on an unseen item comes from generalising class geometry, not from matching a stored case. Contrast v2/feedback kNN which stores items+answers and fires on cosine>=tau (lookup).
- Honest residual risk: it is a trained-from-labels head on calib (closed-form, not gradient, not on base weights). If the maintainer reads "no retraining" as "no learned parameters changing", running means still change. Fallback below.

## Task 1: MDC-6 (v2/alt), Qwen2.5-1.5B, accuracy on dev
### A. NEW label added by user with k annotated examples (label never in prompt; base can only say "Other")
new = recall of the new label on UNSEEN dev items of that class (generalisation), old = accuracy on the old-class dev items (retention; base = shipped system before feedback), fa = old items wrongly sent to new label. "bias" = just raising the Other logit until the k annotations are labelled new (comparator), "steer" = gated diff-of-means steering toward annotated items (comparator).
| task (new label) | k | PCH new | PCH old | PCH fa | gated PCH (base kept unless head says new) new/old | bias new/old | steer new/old |
|---|---|---|---|---|---|---|---|
| dbpedia (Film), base old .943 | 1 | .62 | .989 | .00 | .62/.94 | .54/.87 | .02/.94 |
| | 5 | .94 | .985 | .01 | .94/.94 | .93/.80 | .03/.94 |
| | 10 | .99 | .982 | .01 | .99/.93 | .99/.77 | .03/.94 |
| | 25 | .99 | .982 | .01 | .99/.93 | 1.0/.76 | .03/.94 |
| ag_news (Sci/Tech), base old .902 | 1 | .20 | .916 | .02 | .28/.89 | .50/.81 | .22/.90 |
| | 5 | .59 | .873 | .08 | .60/.85 | .79/.57 | .26/.90 |
| | 10 | .71 | .848 | .11 | .71/.82 | .93/.38 | .28/.90 |
| | 25 | .83 | .81 | .16 | .83/.79 | .95/.33 | .30/.90 |
| emotion (Love), base old .582 (base says Other .45) | 1 | .12 | .64 | .04 | .55/.55 | .81/.21 | .47/.56 |
| | 5 | .42 | .54 | .18 | .70/.47 | .89/.07 | .47/.56 |
| | 10 | .58 | .45 | .32 | .75/.39 | .91/.05 | .47/.56 |
| | 25 | .83 | .34 | .51 | .91/.30 | .93/.03 | .48/.56 |
Reading: works when the new class is separable (dbpedia: 94% recall of unseen Film items from 5 annotations, old 0.985 > base .943 because the calib-fitted head also helps old classes). Partial for ag_news. FAILS the retention requirement for emotion/Love (confusable with Joy: new recall comes with old collapse). Bias-only and steering are strictly worse trade-offs.

### B. Correcting systematic errors, k annotated base+head errors (shipped system = base + k=0 head). fix = accuracy on UNSEEN dev items that the shipped system got wrong (0 by construction at k=0); acc = full-task dev accuracy
| task | base acc / k=0 acc | k=1 | k=5 | k=10 | k=25 |
|---|---|---|---|---|---|
| offensive | .440 / .49 | fix .14 acc .55 | .39 / .66 | .55 / .72 | .75 / .77 |
| dbpedia | .948 / .988 | .08 / .988 | .26 / .986 | .44 / .988 | n/a (too few errors) |
| ag_news | .838 / .860 | .04 / .859 | .11 / .863 | .10 / .863 | .07 / .862 |
| amazon | .958 / .96 | .17 / .96 | .19 / .96 | .12 / .96 | .16 / .96 |
| yelp | .960 / .96 | .06 / .96 | .05 / .96 | .10 / .96 | .12 / .96 |
| emotion | .597 / .60 | ~0 | .01 | .01 | .01 |
Retention (corr_ok = accuracy on items shipped system had right) stays .98-1.00 everywhere. Lookup-kNN reference (old design, tau .9 untuned) buys fix only by collapsing accuracy on yelp/amazon (.56-.77 at k=25). Generalisation is real where errors are systematic (offensive threshold shift, dbpedia confusions) and ~0 where they are idiosyncratic (emotion).

## Task 2: MMLU (v2/base), Qwen2.5-1.5B, letter-class head
Base dev acc .582. PCH combo (tuned lambda=.1): dev .583 at every k; rec (re-asked corrected items) 0-.02; same-subject unseen pool items ('rel') +0 to +1pt (.557 -> .599 at k=25 is the subject-mix of the annotated items, base .582 on dev); fix ~0. Lookup-kNN reference: rec 1.0 but dev .581/.577/.573/.567 (k=1/5/10/25, tau .9 untuned) and rel only partially up. Conclusion: NEW FACTUAL knowledge cannot be added without a lookup of the specific item; non-retrieval, non-training heads can only fix systematic/calibration errors, and MMLU has few of those.

## Bytes (stand-in d=1536 L8 features dim 3072; counted in the 10% budget)
P 96x3072 fp16 = 590 KB, Ci 18 KB, mu0 6 KB, class means 192 B/class (~42 classes across MDC-6 = 8 KB), counts+bias+scalars < 1 KB: ~ 0.62 MB total, constant in k (feedback changes values, adds 0 bytes). 14B (d=5120, concat 10240): P int8 ~1 MB (fp16 2 MB) + same small rest = ~1-2 MB = 0.03-0.07% of the 2.95 GB budget.

## Recommendation
Primary: PCH (prototype/count updates over frozen features) restricted to classification-style tasks (new category or systematic mistakes), gated mode (override only when head fires for new/corrected label) for retention. Fallback if even running-mean updates are ruled "learning": ship PCH frozen at k=0 (calibration-only, no feedback state) and make feedback a single per-label bias vector fitted by counting (16-64 bytes; works for offensive: .44 -> .66 at k=5, but poor retention for new labels); if retrieval-like memory is judged acceptable, v2/feedback gated kNN remains the only method that adds specific facts.
## Caveats
Single model/seed family; 1.5B stand-in; PCH k=0 head is trained on 3.6k labelled calib items (disclose; could also raise the locked-exam score vs base, i.e. this is also a supervised calibration probe); hyperparameters at grid edges for task B (w=30, eta=2); emotion/Love and MMLU are negative results; 0.5B not run.
