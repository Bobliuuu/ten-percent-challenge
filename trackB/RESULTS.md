# Track B: feedback memory on frozen base (no weight training)

Base textattack/bert-base-uncased-SST-2 (frozen). Data: SST-2 *train* only (dev 2000 = "old" regression set, feedback 2000 = anchors/real errors). Validation/test never loaded.
Reproduce: `python embed.py` (caches logits/features, writes cache.npz) then `python run3.py`; log in final_log.txt. API: `lex.py` -> `Memory`.

## Scenarios
- "New" (b): synthetic new-concept domain: 24 new/invented slang words (bussin, slaps, zorbly, mid, cringe, blenky ...), templated sentences, label set by the word. Base acc 61.9% (chance-ish). 4 sequential feedback batches of 6 words (8 examples/word given as feedback); eval on held-out sentences of the words of each batch (32/word).
- (a) base's own errors: base gets 24/2000 feedback-split sentences wrong (it has memorised train: 98.8%). Half used as feedback, half held out.
- "Old": dev 2000, base 98.80%.

## Findings
1. Dense memories over frozen BERT pooled/CLS/mean embeddings (kNN with sim-gated vote, int8; and a residual linear head) FAIL: kNN over embeddings trades dev for new (dev 0.78-0.95 for syn ~0.65-0.75; pooler sims are class-collapsed). Residual dense head keeps dev (0.98) but learns only ~0.5-0.9 on new and forgets. Code in negative/. Sentence embeddings of the (already overconfident) classifier carry little word-level polarity for unseen words.
2. WORKS: sparse **lexical residual memory** over wordpiece ids (token -> int8 weight in 1/8 steps, sum/sqrt(n) added to the logit difference), trained by a mistake-driven hinge perceptron (margin 6 in logit-diff units) on pending feedback plus a replay buffer of 1000 old anchor sentences (token-id lists) from the feedback split; tokens with anchor document frequency >30 are frozen (so new knowledge lives in rare/new tokens; keeps old behaviour). 5 seeds averaged:

| step | dev (old) | b0 | b1 | b2 | b3 | err-heldout |
|---|---|---|---|---|---|---|
| base | 98.80 | .710 | .599 | .555 | .619 | 0/12 |
| after batch0 | 98.38 | .988 | .605 | .557 | .593 | .083 |
| after batch1 | 98.38 | .947 | .990 | .576 | .625 | .083 |
| after batch2 | 98.20 | .933 | .943 | .974 | .578 | .083 |
| after batch3 | 98.23 | .960 | .976 | .949 | .978 | .083 |

New-concept acc 61.9 -> ~96.6 (mean over 4 held-out sets); old dev 98.80 -> 98.23 (-0.6pt; -0.36pt, 98.44, if base-error items are excluded from feedback). Forgetting curve: earlier batches drop ~3-5pt after later batches (0.99 -> 0.95) but stay far above base. Real base errors (train label noise) are essentially not learnable for held-out ones (idiosyncratic); memorising the given ones works.
3. Size: ~1.7k int8 weights (3B each ~5KB) + 1000 anchors (~22KB) = ~28KB total (<<1MB). Anchors are the bulk; 300 anchors (~10KB) give dev ~97.9 / new ~94. Eviction/dedup: weights are pruned (|w|<1/8) and quantised on each consolidation; anchors are capped (cap by count); dedup via token-set.
4. Caveats: dev is SST-2 train, which the base memorised, so any residual can only lose there; real held-out (locked exam) retention should be measured by caller once, via `Memory.predict` with the memory frozen. The synthetic shift is lexical (word->label), exactly what a lexicon memory fits; kNN/dense memories did not. Hyperparams (margin 6, max_df 30, 1000 anchors) were tuned on 3-5 seeds of this same scenario (seeds differ in word order/sample).
5. Integration: the memory needs the sentence's wordpiece ids (tokenizer is already in the pipeline) and the base logits; inference cost is negligible. Does not require the exam model to change otherwise; to make it safe for the exam, ship an empty-new-memory or anchors-only memory and verify exam acc is unchanged (not done here).
