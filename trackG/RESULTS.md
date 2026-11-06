# trackG: un-memorized dev proxy
Never touched SST-2 validation/test. Excluded rotten_tomatoes / SST / SetFit-sst2 / glue copies (RT snippets are SST's source, overlap with the exam is possible).
Sources (independent of SST): IMDB train sentences (stanfordnlp/imdb, sentence label = review label, noisy), Amazon polarity review titles (mteb/amazon_polarity), tweet_eval sentiment pos/neg tweets.
Base (textattack bert SST-2) raw acc on full pool of 15,936 deduped: imdb 71.9%, amazon 82.2%, tweet 81.3% (much label noise + domain shift), vs 98.8% on memorized SST-2 train.

- dev2.json: 3000 ex (1200 imdb/1000 amazon/800 tweet), 50/50 labels, base acc exactly 93.0% (by sampling 93% base-correct / 7% base-wrong per source, label-balanced). Fields: text,label,source,base_margin,base_pred. Margin pctl (10/25/50/75/90): 1.76/3.71/5.52/6.82/7.66; 5.8% have margin<1, 18.7% <3.
  Caveat: the 7% "wrong" are partly label noise, so use dev2 mainly for relative comparison / agreement with base, not absolute acc.
- dev2_hard.json: 1500 lowest-|logit margin| pool examples disjoint from dev2 (margin 0.0-1.4, median 0.81; base acc 48%, so labels ~coin flip). Use for agreement-with-base / flip rate, not accuracy.
API: `from dev2 import load_dev2` (run with trackG on sys.path); `load_dev2()` -> (texts, labels); `load_dev2(hard=True)`; `with_meta=True` returns dicts.
Recommended metrics: acc on dev2, agreement-with-base on dev2 and dev2_hard, plus SST-2 train dev (existing).
Files: build.py/make.py regenerate (seeds fixed); pool_logits.pkl = base logits for the pool.
