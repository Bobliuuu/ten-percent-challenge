# TASK B: AI-vs-human text detection

Maps to maintainer example "AI vs Human detectors". Data: yaful/MAGE (open; 10 human domains: cmv, eli5, hswag, roct, sci_gen, squad, tldr, wp, xsum, yelp; ~27 generators incl. GPT-3.5/davinci, OPT, LLaMA 7-65B, T0, FLAN-T5, GLM, BLOOM...). Files `B/*.jsonl`, `LOCK.json`.
**Item**: text (200-1500 chars after whitespace normalisation); output A=Human, B=AI. Metric: accuracy of argmax of 2 letter logits (+AUROC of logit(B)-logit(A)).
**Splits**: h=int(sha256("B|"+text)[:8],16) over MAGE `test.csv`; h%7==0 LOCKED (2000), ==1 DEV (2000), else POOL (4000); CALIB 2000 from MAGE `train.csv`. Dev: 51% AI; domains 110-265 each.

## Base dev scores (n=2000)
| base | zs acc / AUROC | 4-shot acc / AUROC |
|---|---|---|
| Qwen2.5-1.5B-Instruct | 0.504 / 0.499 | 0.499 / 0.500 |
| Qwen2.5-3B-Instruct | 0.509 / 0.577 | 0.527 / 0.532 |
| Qwen2.5-7B-Instruct | 0.487 / 0.504 | 0.496 / 0.560 |
Majority baseline 0.513. **All three bases are at chance** (predictions collapse to one class). Zero-shot "ask an instruct LLM if text is AI" does not work on MAGE.
Binoculars-style zero-shot score (Qwen2.5-1.5B base vs instruct, no training, 1000 dev): AUROC 0.648 (log-ppl alone 0.610), per-domain 0.39 (squad) to 0.77 (eli5); threshold fit on calib gave only 0.518 acc on dev (calib class mix 69% AI vs dev 51% shifts the threshold). Qwen2.5-7B pair: AUROC 0.588 (log-ppl 0.547), acc 0.517 -- even weaker. Raw: `bino_results.log`.
**Verdict**: no usable base score. A real detector needs a *trained* classifier (public ones are all <0.5B: roberta-large-openai-detector 355M, desklib deberta-v3-large 435M, Hello-SimpleAI roberta 125M), which fails rule 1 (>0.5B) unless one is found, and the task is also not a "small instruct model does the job" setting. MAGE is also known-hard for OOD generators.
