# TASK A: PII detection / redaction (span-category classification)

Maps to maintainer example "PII redaction". Data: ai4privacy/pii-masking-300k (open, synthetic, 6 languages: en/de/fr/it/nl/es; document types: forms, JSON, HTML, emails, chat, logs). Files `A/{locked,dev,pool,calib}.jsonl`, hashes in `LOCK.json`, built by `build_data.py` (Modal ap-SPmnKvzT9zzcRJjlIrrKSF) before any scoring; locked file chmod 444.

**Item**: a +-350-char window of a document with one candidate span wrapped `<<...>>`. Output = one letter, 8 options: A name, B address/location, C contact (email/phone), D gov/document ID, E date/time, F online id/credential (username/password/IP), G gender/sex, H not personal data (negatives = random 1-3 token spans not overlapping any annotated PII, 60% capitalised/digit-bearing hard negatives). 27 fine labels map to 7 PII classes. Up to 3 positive + 1 negative span per document.
**Metric**: 8-way accuracy of argmax over the 8 letter logits (also reported: binary PII-vs-not accuracy). Redaction view: a document redaction pipeline = run this classifier over candidate spans; entity-level F1 would need span proposal + generation and is NOT used because the output space is large.
**Splits**: key = source document text (all spans of one doc stay in one split); h=int(sha256("A|"+source_text)[:8],16); h%7==0 LOCKED (2000 items), ==1 DEV (2000), else POOL (4000); CALIB 2000 from ai4privacy *train* (disjoint). Class mix dev: NOT-PII 25%, address 15%, ID 15%, name 11%, date 14%, credential 11%, contact 8%, gender 3%; majority-class chance 0.246; languages balanced (~330 each).
**Prompt**: `{inst}\n\nText: {window}\n\nOptions:\nA. ...\n...\nAnswer:` (plain text, no chat template); prediction = argmax over logits of tokens " A".." H". Few-shot = one calib example per class prepended in the same format.

## Base dev scores (n=2000, bf16, letter-logit)
| base | zero-shot | 8-shot (1/class) | binary PII acc zs / fs |
|---|---|---|---|
| Qwen2.5-1.5B-Instruct | 0.373 | 0.412 | 0.553 / 0.726 |
| Qwen2.5-3B-Instruct | 0.431 | 0.521 | 0.598 / 0.764 |
| Qwen2.5-7B-Instruct | 0.657 | 0.691 | 0.763 / 0.801 |
Chance 0.246 (majority) / 0.125 (uniform). Per-language 3B fs: 0.48-0.55 (uniform, so not a single-language artifact). Recalls (3B fs) strong on address 0.91, ID 0.78, name 0.68; weak on NOT-PII 0.10 and date/credential 0.37-0.43 (over-calls PII).
Raw: `base_results.log`, logits in Modal Volume commercial-data:/base/.
**Caveats**: scores are far from ceiling with zero/few-shot, which gives headroom, but the labels are synthetic and partly ambiguous (e.g. title-vs-name, date vs time), so a ~0.8-0.9 ceiling is plausible; headroom is also a risk: the compressed model must reach a *low* base score exactly, noisy at 0.66-0.69 (CI +-2 pts at n=2000).
