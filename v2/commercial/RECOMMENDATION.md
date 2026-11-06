# v2/commercial: ranking and recommendation (dev numbers, 3 Qwen2.5-Instruct bases; locked files untouched)

| | A PII span-category | B AI-vs-human (MAGE) | C2 tool selection (hard) | (C easy / C gen) |
|---|---|---|---|---|
| output space | 8 letters | 2 letters | 5 letters | 5 / open JSON |
| base dev (1.5B/3B/7B), best of zs/fs | 0.41 / 0.52 / 0.69 | 0.50 / 0.53 / 0.50 (AUROC 0.50-0.58) | 0.76 / 0.75 / 0.78 | C 0.96/0.95/0.98; gen exact 0.61/0.66/0.78 |
| chance (majority) | 0.25 | 0.51 | 0.23 | 0.23 |
| headroom over chance | large for 3B/7B | none (at chance) | large | C near ceiling |
| commercial clarity | clear (redaction pipelines, GDPR) but synthetic data, span-labelled reformulation of redaction | clear but base cannot do it | clearest (agent tool routing/function calling) | |

Ranking: (1) **C2 tool use** -- strong base score (0.76-0.78 vs 0.23), 5-letter output, same family as "agentic/tool use"; caveat: irrelevance items unstable w.r.t. prompt, and the weaker 1.5B is ~equal to 7B zero-shot (so compression has less headroom from base size). (2) **A PII** -- headroom and small output space, but base scores are only 0.41-0.69 and depend on a synthetic corpus; the classification reformulation is a proxy for redaction, not the full entity-level F1 redaction. (3) **B AI-vs-human** -- reject for LLM-base use: all bases at chance zero/few-shot, Binoculars zero-shot AUROC 0.59-0.65, so "the model already does the job" fails; only a >0.5B trained detector would qualify and none is public.

Recommended adoption: **C2 (tool selection among confusable tools + 'none fits') as the primary task, with a 7B or 3B Instruct base**; keep A as the second skill for the feedback stage (new-label learning: e.g. annotate 'credential' vs 'ID' examples) or as an alternative primary if a PII claim is preferred. Open risks: pick one frozen prompt; report CIs (n=2000 -> +-2 pts); 1.5B is likely too close to chance-calibrated to be a meaningful "bar" if few-shot.
