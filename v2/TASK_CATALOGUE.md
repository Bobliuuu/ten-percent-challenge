# TASK_CATALOGUE: what is the EASIEST commercial task that can pass the rules? (T3 lane, 2026-11-08)

DEV DATA ONLY. No locked/sealed exam, X15 file or GitHub post touched. Every number is REAL greedy generation of the label word (`<|im_end|>` stop, exact string, zero-shot prompt, no letters/options), bf16 base vs the repo quantizer at 9.85% of fp16 (`dense_core.py`, target_ratio 0.0985, Hadamard + GPTQ + entropy-coded grids, ~1.5 bpw body) on Qwen2.5-14B/32B-Instruct. Retention = P(quantized output correct | base output correct) on identical items. Items: `task_search/T3/items_dev_all.jsonl` (round 1, 26 tasks, 7.2k items), `items_r2.jsonl` (7 refined tasks, 2.6k), `items_r3.jsonl` (6 cleaner-label tasks, 2.5k); class-balanced, sha256-ordered, deduplicated, 160 items per source (2 or 3 sources per task), builder `build_t3.py`, licences `LICENCES_T3.md`, pool sizes `PLAN*.json`. Quantizer calibration = the mixture of ALL tasks' calibration rows of that item file (hash-disjoint), not task-matched, so retention is a little pessimistic. Compute: 5 Vast boxes over the day (RTX PRO 6000 96GB x2, A100-80GB, A40 x1, one unreachable box destroyed), all destroyed at the end (RUNS.md). Two quantization protocols were run: (a) one quantization per model calibrated on a MIXTURE of every task in the item file (`mixture_retention.md`), (b) per-task calibration with 400 hash-disjoint rows of that task only (`per_task_retention.md`, the comparable protocol: EASY lane also calibrates on its own tasks). Protocol (a) is pessimistic (class-prior flips, e.g. PHISH2 Legitimate 0.00), so only (b) is used for ranking.
The other agent's lane (`EASY_TASKS.md`: SKILL7, SKILL10, BANK6/8, SPAM3, SENT, LANG8, QTOPIC6, ...) is NOT duplicated here; where it matters its numbers are quoted.

## 0. The filters used (ROADMAP_LANES TASK OBJECTIVE)
R1 commercial/not toy/not narrow; R3 >=2 real sources, >=2000 locked, >=300 per kind, >=500 from a source never tuned on; R7 real label word; R8 base .94-.98 target (drop >.985); R5 teachable NEW label; then rank by predicted robustness (few classes, separable labels, big margin, clean labels, short inputs).

## 1. RESULT: nothing probed here beats SKILL7; best new find is NEWS4C (still ~4-7 pts short)
Retention at 9.85% of fp16 (real generation, per-task calibration, one quantizer draw, DEV, Wilson lo in brackets). EASY-lane rows quoted from EASY_TASKS.md for comparison (same quantizer, own-task calibration).
| rank | task | classes | sources | base 14B / 32B | retention 14B | retention 32B | worst kind (14B) | rule gate (TASK_RULE_AUDIT.md) |
|---|---|---|---|---|---|---|---|---|
| 1 | **SKILL7** (EASY lane) | 7 | MASSIVE + CLINC150 | .957 / .969 | .977 [.962] | **.985 [.972]** | Music .95 | no FAIL; GRAY R1, R2 pool depth, R3 (seal CLINC, MASSIVE-only dev) |
| 2 | BANK6 (EASY lane) | 6 | Banking77 | .905 | .962 | .986 | | FAIL R3 (single source); BANK8/BANKFAM with sealed CLINC = GRAY |
| 3 | SKILL10 (EASY lane) | 10 | MASSIVE | .93-.94 | .946 | .976 | | FAIL R3 (single source), Cooking <300 |
| 4 | **NEWS4C** newsroom desk (T3) | 4 | HuffPost 2012-22 + BBC 2004 | .928 / .928 | .916 [.879] | .949 [.918] | Business .85 | no FAIL; GRAY R1, R3 (BBC ~1k rows, NC licence), BBC cells <300 |
| 5 | PII2C sensitive-identifier yes/no (T3) | 2 | ai4privacy + gretel (both synthetic) | .849 / .919 | .957 [.924] | pending | Harmless .92 | base <.88, both sources synthetic: R1/R9 GRAY-FAIL |
| 6 | SENT2C sentiment-2 (T3) | 2 | Yelp + IMDB + tweets | .902 / .881 | .841 [.803] | pending | tweets Positive .39 | R8 ok, retention far below gate |
| 7 | TOX2C toxicity (T3) | 2 | civil_comments + Davidson | .841 / .838 | .796 [.743] | pending | Fine .60 | label noise |
| 8 | PHISH2B email phishing, definition prompt (T3) | 2 | zefang + seven-corpora (not independent) | .90 / .956 | pending | .807 [.759] | | R3 GRAY (overlapping upstream corpora) |
| 9 | COMM4 forum-post community routing (T3) | 4 | StackExchange titles + Reddit titles | .875 / pending | .750 [.696] | pending | Astronomy .21 | collapses on one class |
| 10 | PHISH2C Scam/Genuine (T3) | 2 | same two phishing corpora | .931 / .912 | .508 [.452] | pending | Genuine .06 (class flip) | collapse |
Ceiling controls (base >.985, R8 FAIL, not ranked): LANG12B language ID (2 real sources) .997; CODEPROSE 1.000 (mixture-calibrated retention .991); CODELANG .984.
**Answer to the objective: the easiest task that passes the rules on today's evidence is still SKILL7 (voice-skill routing) with MASSIVE as tuning source and CLINC150 sealed. NEWS4C is the best independent alternative (two sources a decade apart, 4 coarse classes, base .928) but loses 6-8 points of retention to HuffPost label noise (Business/Sports boundary).** Why the new finds retain worse: their per-class base accuracy is limited by dataset label noise (borderline items have small margins), and 1.5 bpw noise flips the borderline items; SKILL7's labels are clean (utterance -> scenario is nearly deterministic).

## 2. Per-task-calibrated retention (full table, `per_task_retention.md`)
| model | task | classes | base acc | base-correct | margin>=6 | retention [Wilson lo] | worst kind (n) | worst src x kind (n) | ratio | body bpw |
|---|---|---|---|---|---|---|---|---|---|---|
| Qwen2.5-14B-Instruct | COMM4 | 4 | 0.875 | 280 | 0.95 | 0.750 [0.696] | 0.21 Astronomy (72) | 0.08 stackexchange_titles/Astronomy (36) | 9.88% | 1.53 |
| Qwen2.5-14B-Instruct | NEWS4C | 4 | 0.928 | 297 | 0.95 | 0.916 [0.879] | 0.85 Business (74) | 0.83 huffpost/Business (36) | 9.88% | 1.53 |
| Qwen2.5-14B-Instruct | PHISH2C | 2 | 0.931 | 295 | 0.94 | 0.508 [0.452] | 0.06 Genuine (155) | 0.04 zefang_phish/Genuine (77) | 9.88% | 1.53 |
| Qwen2.5-14B-Instruct | PII2C | 2 | 0.849 | 254 | 0.92 | 0.957 [0.924] | 0.92 Harmless (137) | 0.85 ai4privacy/Harmless (59) | 9.88% | 1.53 |
| Qwen2.5-14B-Instruct | SENT2C | 2 | 0.902 | 433 | 0.95 | 0.841 [0.803] | 0.67 Positive (207) | 0.39 tweets/Positive (71) | 9.88% | 1.53 |
| Qwen2.5-14B-Instruct | TOX2C | 2 | 0.841 | 269 | 0.90 | 0.796 [0.743] | 0.60 Fine (132) | 0.51 civil_comments/Fine (69) | 9.88% | 1.53 |
| Qwen2.5-32B-Instruct | NEWS4C | 4 | 0.928 | 297 | 0.95 | 0.949 [0.918] | 0.91 Business (65) | 0.87 huffpost/Business (31) | 9.87% | 1.56 |
| Qwen2.5-32B-Instruct | PHISH2B | 2 | 0.956 | 305 | 0.94 | 0.807 [0.759] | 0.63 Legitimate (159) | 0.59 zefang_phish/Legitimate (80) | 9.87% | 1.56 |

Not finished at the time of writing (jobs were cut for time): PII2C 32B, SENT2C 32B, TOX2C 32B, COMM4 32B, PHISH2C 32B, PHISH2B 14B; their configs/items are in `research/V_vast/bundles/t3`.

## 3. Base-accuracy tables (zero-shot, real generation; margin = min-token top1-top2 logit gap of the gold answer)
### Qwen2.5-14B round 1 (26 tasks)
| task | n | acc | worst class | worst src | m_min>=6 of correct | median m_min |
|---|---|---|---|---|---|---|
| ADE | 160 | 0.863 | 0.86 No | 0.86 ade_corpus | 0.93 | 10.8 |
| ANSWERABLE | 160 | 0.650 | 0.61 Answerable | 0.65 squad2 | 0.23 | 4.2 |
| BOOLQ | 160 | 0.806 | 0.64 Yes | 0.81 boolq | 0.96 | 10.1 |
| CODELANG | 312 | 0.984 | 0.92 Ruby | 0.98 codesearchnet | 0.97 | 10.4 |
| CODEPROSE | 320 | 1.000 | 1.00 Code | 1.00 csn_vs_agnews | 1.00 | 10.9 |
| DEFECT | 160 | 0.419 | 0.29 Safe | 0.42 devign | 0.30 | 4.4 |
| DEPT3 | 318 | 0.654 | 0.36 Sales | 0.60 tobi_queue | 0.89 | 10.1 |
| ENTAIL2 | 320 | 0.894 | 0.86 Yes | 0.87 scitail | 0.96 | 10.8 |
| FAKENEWS | 160 | 0.719 | 0.51 Fake | 0.72 gonzaloa | 0.87 | 11.0 |
| FINSENT | 159 | 0.836 | 0.66 Neutral | 0.84 twitterfin | 0.78 | 10.2 |
| HATE2 | 320 | 0.853 | 0.81 Fine | 0.81 tweet_eval_hate | 0.93 | 10.6 |
| INJ2 | 320 | 0.809 | 0.64 Injection | 0.72 deepset | 0.85 | 10.2 |
| LANG20 | 400 | 0.955 | 0.50 Portuguese | 0.95 papluca | 0.95 | 10.8 |
| NLI3 | 477 | 0.843 | 0.79 Neutral | 0.83 anli | 0.92 | 11.1 |
| OFFEN2 | 320 | 0.812 | 0.77 Clean | 0.76 tweet_eval_off | 0.48 | 5.8 |
| PARA2 | 320 | 0.784 | 0.64 Yes | 0.78 mrpc | 0.91 | 10.8 |
| PHISH2 | 320 | 0.903 | 0.89 Legitimate | 0.89 zefang_phish | 0.84 | 10.8 |
| PIIFLAG | 319 | 0.696 | 0.51 Yes | 0.55 gretel_fin | 0.95 | 11.2 |
| PSAFE | 480 | 0.787 | 0.74 Safe | 0.66 beavertails_prompt | 0.94 | 10.2 |
| RCT5 | 320 | 0.672 | 0.45 Conclusions | 0.66 pubmed_rct | 0.86 | 10.4 |
| SARC2 | 320 | 0.681 | 0.54 Sincere | 0.62 tweet_irony | 0.87 | 10.2 |
| TICKETPRIO | 159 | 0.365 | 0.00 Low | 0.36 tobi_tickets | 0.71 | 9.8 |
| TICKETTYPE | 160 | 0.556 | 0.00 Change | 0.56 tobi_tickets | 0.79 | 10.5 |
| TOPIC5 | 320 | 0.819 | 0.53 Business | 0.76 yahoo_answers | 0.91 | 10.1 |
| TOX2 | 320 | 0.791 | 0.68 Toxic | 0.77 toxic_chat | 0.85 | 10.2 |
| UNFAIR | 160 | 0.713 | 0.60 Fair | 0.71 lexglue_unfairtos | 0.66 | 8.9 |

### Qwen2.5-14B round 2 (refined prompts)
| task | n | acc | worst class | worst src | m_min>=6 of correct | median m_min |
|---|---|---|---|---|---|---|
| LANG12 | 360 | 0.997 | 0.97 Turkish | 1.00 papluca | 1.00 | 10.8 |
| NEWS5 | 320 | 0.887 | 0.72 Technology | 0.82 huffpost | 0.93 | 10.1 |
| PHISH2B | 320 | 0.897 | 0.79 Phishing | 0.89 zefang_phish | 0.90 | 10.9 |
| PII2 | 320 | 0.887 | 0.86 Clean | 0.82 ai4privacy | 0.90 | 10.5 |
| SENT2B | 480 | 0.919 | 0.89 Positive | 0.88 sst2_movie | 0.93 | 10.6 |
| TOPIC4 | 320 | 0.869 | 0.82 Sports | 0.84 huffpost | 0.97 | 10.0 |
| TOX2B | 480 | 0.790 | 0.69 Toxic | 0.76 civil_comments | 0.89 | 10.5 |

### Qwen2.5-14B round 3 (cleaner label sets)
| task | n | acc | worst class | worst src | m_min>=6 of correct | median m_min |
|---|---|---|---|---|---|---|
| LANG12B | 720 | 0.997 | 0.97 Arabic | 0.99 massive_utterances | 1.00 | 10.8 |
| NEWS4C | 320 | 0.928 | 0.89 Sports | 0.91 huffpost | 0.95 | 10.0 |
| PHISH2C | 320 | 0.928 | 0.88 Scam | 0.92 zefang_phish | 0.94 | 11.1 |
| PII2C | 320 | 0.859 | 0.86 Harmless | 0.83 ai4privacy | 0.93 | 10.6 |
| SENT2C | 480 | 0.902 | 0.86 Positive | 0.84 imdb_movies | 0.95 | 10.8 |
| TOX2C | 320 | 0.841 | 0.82 Fine | 0.79 civil_comments | 0.90 | 10.4 |

### Qwen2.5-14B COMM4
| task | n | acc | worst class | worst src | m_min>=6 of correct | median m_min |
|---|---|---|---|---|---|---|
| COMM4 | 320 | 0.875 | 0.78 Brewing | 0.81 reddit_titles | 0.95 | 10.5 |

### Qwen2.5-32B round 1
| task | n | acc | worst class | worst src | m_min>=6 of correct | median m_min |
|---|---|---|---|---|---|---|
| ADE | 160 | 0.844 | 0.76 Yes | 0.84 ade_corpus | 0.87 | 10.2 |
| ANSWERABLE | 160 | 0.894 | 0.89 Unanswerable | 0.89 squad2 | 0.94 | 11.6 |
| BOOLQ | 160 | 0.856 | 0.72 Yes | 0.86 boolq | 0.93 | 12.0 |
| CODELANG | 312 | 1.000 | 1.00 Go | 1.00 codesearchnet | 1.00 | 11.8 |
| CODEPROSE | 320 | 1.000 | 1.00 Code | 1.00 csn_vs_agnews | 1.00 | 11.9 |
| DEFECT | 160 | 0.350 | 0.29 Safe | 0.35 devign | 0.16 | 4.1 |
| DEPT3 | 318 | 0.673 | 0.56 Technical | 0.64 tobi_queue | 0.94 | 11.6 |
| ENTAIL2 | 320 | 0.881 | 0.86 No | 0.86 scitail | 0.52 | 6.2 |
| FAKENEWS | 160 | 0.600 | 0.23 Fake | 0.60 gonzaloa | 0.91 | 10.6 |
| FINSENT | 159 | 0.855 | 0.68 Neutral | 0.86 twitterfin | 0.85 | 10.9 |
| HATE2 | 320 | 0.828 | 0.81 Hate | 0.79 tweet_eval_hate | 0.91 | 11.9 |
| INJ2 | 320 | 0.853 | 0.71 Injection | 0.78 deepset | 0.82 | 9.2 |
| LANG20 | 400 | 0.995 | 0.95 Hindi | 0.99 papluca | 0.97 | 11.4 |
| NLI3 | 477 | 0.866 | 0.83 Neutral | 0.84 anli | 0.89 | 11.8 |
| OFFEN2 | 320 | 0.806 | 0.78 Clean | 0.74 tweet_eval_off | 0.51 | 6.0 |
| PARA2 | 320 | 0.328 | 0.00 No | 0.29 qqp | 0.73 | 7.4 |
| PHISH2 | 320 | 0.916 | 0.89 Legitimate | 0.91 sevenphish | 0.87 | 11.5 |
| PIIFLAG | 319 | 0.715 | 0.55 Yes | 0.58 gretel_fin | 0.92 | 12.1 |
| PSAFE | 480 | 0.804 | 0.74 Safe | 0.66 beavertails_prompt | 0.94 | 11.2 |
| RCT5 | 320 | 0.688 | 0.27 Conclusions | 0.68 pubmed_rct | 0.91 | 12.1 |
| SARC2 | 320 | 0.662 | 0.40 Sincere | 0.58 tweet_irony | 0.83 | 11.5 |
| TICKETPRIO | 159 | 0.358 | 0.00 Low | 0.36 tobi_tickets | 0.86 | 11.8 |
| TICKETTYPE | 160 | 0.544 | 0.00 Change | 0.54 tobi_tickets | 0.78 | 11.8 |
| TOPIC5 | 320 | 0.809 | 0.52 Business | 0.78 yahoo_answers | 0.92 | 11.1 |
| TOX2 | 320 | 0.759 | 0.56 Toxic | 0.72 toxic_chat | 0.86 | 11.9 |
| UNFAIR | 160 | 0.731 | 0.72 Unfair | 0.73 lexglue_unfairtos | 0.68 | 9.0 |

### Qwen2.5-32B rounds 2 and 3
| task | n | acc | worst class | worst src | m_min>=6 of correct | median m_min |
|---|---|---|---|---|---|---|
| LANG12 | 360 | 0.997 | 0.97 Turkish | 1.00 papluca | 1.00 | 11.5 |
| NEWS5 | 320 | 0.881 | 0.72 Technology | 0.84 huffpost | 0.94 | 11.0 |
| PHISH2B | 320 | 0.953 | 0.91 Phishing | 0.94 sevenphish | 0.94 | 11.6 |
| PII2 | 320 | 0.912 | 0.85 Clean | 0.86 ai4privacy | 0.94 | 12.1 |
| SENT2B | 480 | 0.881 | 0.85 Positive | 0.84 sst2_movie | 0.94 | 11.9 |
| TOPIC4 | 320 | 0.859 | 0.76 Health | 0.81 huffpost | 0.96 | 11.1 |
| TOX2B | 480 | 0.752 | 0.60 Toxic | 0.68 civil_comments | 0.90 | 11.8 |

| task | n | acc | worst class | worst src | m_min>=6 of correct | median m_min |
|---|---|---|---|---|---|---|
| LANG12B | 720 | 0.996 | 0.97 Arabic | 0.99 massive_utterances | 1.00 | 11.5 |
| NEWS4C | 320 | 0.928 | 0.81 Business | 0.91 huffpost | 0.95 | 11.1 |
| PHISH2C | 320 | 0.912 | 0.87 Genuine | 0.91 sevenphish | 0.87 | 10.9 |
| PII2C | 320 | 0.919 | 0.91 Harmless | 0.89 ai4privacy | 0.96 | 11.8 |
| SENT2C | 480 | 0.881 | 0.85 Positive | 0.84 imdb_movies | 0.93 | 11.8 |
| TOX2C | 320 | 0.838 | 0.77 Toxic | 0.78 civil_comments | 0.93 | 11.6 |

### Qwen2.5-7B round 1
| task | n | acc | worst class | worst src | m_min>=6 of correct | median m_min |
|---|---|---|---|---|---|---|
| ADE | 160 | 0.825 | 0.72 Yes | 0.82 ade_corpus | 0.90 | 12.8 |
| ANSWERABLE | 160 | 0.644 | 0.49 Answerable | 0.64 squad2 | 0.76 | 10.0 |
| BOOLQ | 160 | 0.812 | 0.70 Yes | 0.81 boolq | 0.95 | 13.0 |
| CODELANG | 312 | 1.000 | 1.00 Go | 1.00 codesearchnet | 0.98 | 12.6 |
| CODEPROSE | 320 | 0.997 | 0.99 Prose | 0.99 devign_c_vs_imdb | 0.86 | 13.3 |
| DEFECT | 160 | 0.494 | 0.25 Safe | 0.49 devign | 0.35 | 4.2 |
| DEPT3 | 318 | 0.619 | 0.52 Technical | 0.59 bitext | 0.84 | 8.9 |
| ENTAIL2 | 320 | 0.897 | 0.87 No | 0.87 scitail | 0.93 | 13.2 |
| FAKENEWS | 160 | 0.606 | 0.24 Fake | 0.61 gonzaloa | 0.85 | 14.2 |
| FINSENT | 159 | 0.811 | 0.68 Neutral | 0.81 twitterfin | 0.63 | 8.2 |
| HATE2 | 320 | 0.828 | 0.79 Fine | 0.79 tweet_eval_hate | 0.55 | 6.5 |
| INJ2 | 320 | 0.856 | 0.78 Injection | 0.82 deepset | 0.31 | 5.2 |
| LANG20 | 400 | 0.925 | 0.35 Russian | 0.93 papluca | 0.85 | 9.6 |
| NLI3 | 477 | 0.839 | 0.73 Neutral | 0.84 anli | 0.86 | 9.1 |
| OFFEN2 | 320 | 0.762 | 0.67 Clean | 0.70 tweet_eval_off | 0.32 | 5.1 |
| PARA2 | 320 | 0.756 | 0.54 Yes | 0.75 mrpc | 0.82 | 9.5 |
| PHISH2 | 320 | 0.900 | 0.88 Phishing | 0.87 zefang_phish | 0.92 | 10.4 |
| PIIFLAG | 319 | 0.674 | 0.40 Yes | 0.57 gretel_fin | 0.87 | 13.1 |
| PSAFE | 480 | 0.773 | 0.72 Safe | 0.66 beavertails_prompt | 0.44 | 4.4 |
| RCT5 | 320 | 0.613 | 0.09 Conclusions | 0.59 pubmed_rct | 0.86 | 9.2 |
| SARC2 | 320 | 0.537 | 0.10 Sincere | 0.52 tweet_irony | 0.24 | 5.6 |
| TICKETPRIO | 159 | 0.358 | 0.00 Low | 0.36 tobi_tickets | 0.65 | 9.2 |
| TICKETTYPE | 160 | 0.550 | 0.07 Change | 0.55 tobi_tickets | 0.73 | 8.2 |
| TOPIC5 | 320 | 0.759 | 0.48 Business | 0.72 yahoo_answers | 0.81 | 8.0 |
| TOX2 | 320 | 0.828 | 0.77 Toxic | 0.77 civil_comments | 0.73 | 8.2 |
| UNFAIR | 160 | 0.606 | 0.47 Fair | 0.61 lexglue_unfairtos | 0.67 | 9.0 |

Pattern: zero-shot base accuracy is capped by DATASET LABEL NOISE and fuzzy class definitions, not by the model: only coarse, cleanly defined label sets (language, code-vs-prose, voice skills, desk routing) reach .93+. Too noisy/hard (base .65-.85): toxicity, hate, offensive, PII flag, injection (also lock-limited: deepset Injection 203 rows), NLI, paraphrase, BoolQ, FINSENT, DEPT3, UNFAIR, RCT5, SARC2, FAKENEWS (artefact), TICKETTYPE/PRIO, ANSWERABLE, DEFECT (chance).

## 4. Desk catalogue (45+ candidate tasks: public sources, licence, lockability, estimate, R1 risk, new-label story)
| ID | task (real output: one label word unless noted) | classes | public sources, >=2 distinct (licence) | lockable (>=2000, >=300/kind, >=500 other source) | est. base (14B) | R1 risk | new-label feedback story | status |
|---|---|---|---|---|---|---|---|---|
| SKILL7/10 | voice-assistant skill routing | 7-10 | MASSIVE (CC-BY-4.0), CLINC150 (CC-BY-3.0) | pooled yes; CLINC Alarm/Weather thin (200/100) | .93-.95 | low | new skill (Email/SmartHome) | EASY lane, not re-probed here |
| BANK6/8 | banking77 grouped into queues | 6-8 | banking77 (CC-BY-4.0), CLINC banking (CC-BY-3.0) | yes for Card/Transfer/TopUp; CLINC small | .90 | low | new queue (Exchange/Identity) | EASY lane |
| SPAM3 | email/SMS spam vs ham | 2 | Enron-spam (public), SMS Spam (CC-BY-4.0), +phishing set | yes (Enron 28k, SMS 5k, SMS spam only 631) | .90-.95 | low-mid | 3rd class Phishing/Scam | EASY lane |
| PHISH2 | email phishing vs legitimate | 2 | zefang-liu phishing-email, puyang2025 seven-phishing-email (mixed public corpora; licence per upstream) | yes (6.4k/11k and large) | .9 | low | Scam (3rd) | PROBED |
| TOX2 | comment toxicity (Toxic/Safe) | 2 | civil_comments (CC0), toxic-chat (CC-BY-NC-4.0) | yes (toxic-chat Toxic 372 in 5k: tight) | .8-.9 | low | Threat/Harassment label | PROBED |
| HATE2 | hate speech vs fine | 2 | tweet_eval hate (CC-BY-NC), Davidson (MIT) | yes | .7-.8 | mid (label noise) | Offensive-not-hate | PROBED |
| OFFEN2 | offensive language | 2 | tweet_eval offensive, Davidson | yes | .8 | mid | - | PROBED |
| PSAFE | is the user request unsafe (guardrail) | 2 | Aegis-2.0 (CC-BY-4.0), toxic-chat, BeaverTails prompts (CC-BY-NC-4.0) | yes | .8-.9 | low | new harm class | PROBED |
| INJ2 | prompt injection / jailbreak | 2 | deepset (Apache), jackhhao (CC-BY-4.0) | NO: deepset injection 203, jackhhao 482 (small) | .95 | low | - | PROBED (lock-limited; INJ real lock exists in tournament/INJ) |
| PIIFLAG | text contains PII yes/no | 2 | ai4privacy (open, synthetic), gretel synthetic_pii_finance (Apache-2.0) | yes | .9 | mid (synthetic) | PII-type label | PROBED |
| TOPIC5 | coarse topic routing (Sports/Business/Health/Tech/Politics) | 5 | HuffPost news-category (CC0), Yahoo Answers topics (research) | HuffPost Tech 2055 only; ok | .9 | low | Entertainment/Travel | PROBED |
| NEWS4 | news topic (AG/BBC/HuffPost) | 4 | AG, BBC, HuffPost | yes (earlier lane plan) | .87 | low | Entertainment | EARLIER LANE (TASK_SEARCH) |
| DEPT3 | email routing Billing/Technical/Sales | 3 | Tobi-Bueck helpdesk tickets (CC-BY-NC-4.0), Bitext (CDLA-Sharing) | Sales in Tobi only 513 | .8-.9 | mid (synthetic) | Returns | PROBED |
| LANG20 | language ID, 20 languages | 20 | papluca (CC-BY-NC-SA? check), MASSIVE | pooled yes | .98+ | low | new language | PROBED (single source) |
| CODELANG | programming-language ID | 6 | CodeSearchNet (MIT per repo), CodeXGLUE c2t (C-UDA) | yes (3000/2000 per lang) | .9-.97 | low-mid | new language (Rust/C#) | PROBED |
| CODEPROSE | code vs prose | 2 | CSN+AG News, Devign C + IMDB | yes | .99 | HIGH (toy, ceiling) | - | PROBED (control: ceiling) |
| NLI3 | entailment/neutral/contradiction | 3 | SNLI (CC-BY-SA), ANLI (CC-BY-NC), MNLI (OANC mix) | yes | .85-.9 | mid (benchmark-shaped) | - | PROBED |
| ENTAIL2 | does hypothesis follow (Yes/No) | 2 | RTE (GLUE), SciTail (Apache) | RTE only 2.5k; SciTail large | .85-.9 | mid | - | PROBED |
| PARA2 | duplicate/paraphrase | 2 | QQP, MRPC | MRPC 3.7k only | .8 | mid | - | PROBED |
| BOOLQ | yes/no QA from passage | 2 | BoolQ (CC-BY-SA) | single source | .85-.9 | low-mid | - | PROBED |
| ANSWERABLE | is question answerable from passage | 2 | SQuAD2 (CC-BY-SA) | single source | .8 | mid | - | PROBED |
| SARC2 | sarcasm | 2 | news-headlines (CC0?), tweet_eval irony | headlines big, irony 2.8k | .8-.9 | HIGH (narrow) | - | PROBED |
| RCT5 | sentence role in medical abstract | 5 | PubMed RCT20k (CC0 text) | single family | .8 | mid | - | PROBED |
| FINSENT | financial headline sentiment | 3 | twitter-financial-news (MIT), phrasebank (NC) | tight | .87 | mid (narrow domain) | - | PROBED + earlier lane |
| ADE | adverse drug event mention | 2 | ADE corpus v2 (CC0/open) | single | .9 | mid (medical narrow) | - | PROBED |
| FAKENEWS | fake vs real | 2 | GonzaloA/fake_news (ISOT-like, leaky) | single | .98+ | HIGH (artefact-prone) | - | PROBED (control) |
| UNFAIR | unfair ToS clause | 2 | LexGLUE unfair_tos (CC-BY-4.0) | Unfair 619 only | .8 | mid legal | - | PROBED |
| TICKETTYPE / TICKETPRIO | helpdesk type / priority | 4 / 3 | Tobi-Bueck tickets (single) | single, noisy | .4-.5 | mid | - | PROBED (hard control) |
| DEFECT | C vulnerability yes/no | 2 | Devign | single | ~.55 | HIGH | - | PROBED (hard control; near chance) |
| SENT2/3, LANG8, QTOPIC6, TICKETQ, SUPPORT7 | sentiment, lang, topic, queues | - | see EASY_TASKS.md | - | - | - | - | EASY lane (not duplicated) |
| INJ real | prompt-injection (tournament lock) | 2 | PromptShield, deepset, gandalf, JailbreakHub | locked 2320 | .81 | low | - | EARLIER LANE |
| TOOLSEL | tool/function selection | 2+ | xLAM-style | yes | .86 | low | - | EARLIER LANE |
| HALU | hallucination/grounding check | 2 | HaluEval, RAGTruth | yes | .8 | mid | - | EARLIER LANE |
| TOX/BTSAFE | toxicity, response safety | 2 | civil, toxic-chat, BeaverTails, Aegis | yes | .82 | low | - | EARLIER LANE |
| REVRATE | review star bucket (Neg/Neu/Pos) | 3 | Amazon reviews, Yelp (research licences) | yes | .6-.7 (Neutral) | low | - | DESK (SENT3 in EASY lane shows .66: not easy) |
| STANCE | tweet stance (abortion etc.) | 3 | tweet_eval stance, SemEval | small per topic | .6 | HIGH narrow | - | DESK |
| EMOTION | 6-way emotion | 6 | dair-ai/emotion, tweet_eval | surprise/fear <300 | .6 | mid | - | DESK (dead per earlier lane) |
| CLICKBAIT | clickbait headline | 2 | Webis17 / Chakraborty (no parquet mirror found) | n/a | .85 | mid | - | DESK (no loadable licensed source) |
| JOBFAKE | fake job posting | 2 | EMSCAD (single, 800 fakes) | NO (<2000) | .9 | mid | - | DESK dead on R3 |
| RESUME | resume category | 25-43 | resume-atlas (single) | NO (many classes) | .7 | mid | - | DESK dead on R3 |
| LEGALCLAUSE | contract clause type | 100 | LEDGAR | NO (100 classes) | .6 | mid | - | DESK dead |
| ICD | medical coding | 1000s | MIMIC (credentialed) | NO | - | - | - | DESK dead (licence) |
| CHURN | churn text | 2 | no public text corpus | NO | - | - | - | DESK dead |
| INVOICE / RECEIPT | field extraction / doc-type | JSON | CORD (1k), SROIE (1k) | NO (<2000) | - | - | - | DESK dead (TASK_SEARCH2 also rejected) |
| NER / PIIRED / PIIDET / TOOLARG / LOGTPL / SQL | structured generation | JSON/text | see TASK_SEARCH2.md | partly | .3-.9 | - | - | EARLIER LANE (copy tasks retain worse) |

## 5. Feedback (R5) feasibility of the leading finds
- NEWS4C: new label **Technology** (HuffPost TECH 2055 rows + BBC tech ~400; base cannot emit a label absent from its menu) or **Health** (HuffPost WELLNESS 17.9k). 25 annotated + >=25 unseen teach items + neighbours Business (gadget/earnings stories). Weak boundary: Technology vs Business.
- SKILL7: new skill Email/SmartHome (MASSIVE 949/768 rows). BANK6: new queue.
- PHISH2B/C: new label **Spam** (Enron/SMS spam, large pools).
- PII2C: new type label (e.g. Credential) hard to separate. COMM4: new community (e.g. Photography, SE photo 3.8k rows).
- LANG12B (ceiling): new language trivially teachable but R8 FAIL.

## 6. Caveats
Single zero-shot prompt per task (rounds 2/3 are prompt/label refinements chosen on the same dev items: ~1-2 pt selection optimism on base accuracy). 160 items per (task, source), per-class cells n=40-80. One quantizer draw (+-2 pt draw noise, larger for 2-class tasks with a class flip). Calibration for the per-task protocol includes ~400 rows from every source of the task, i.e. it uses the future sealed source (BBC for NEWS4C), which a legal artifact must not do: retention for a legal NEWS4C build (BBC sealed) would be a little lower. Committed item files drop rows that the repo leak scanner flagged (IP/token-like strings; up to ~20/320 rows in PII tasks). No locked data used, nothing sealed. Licences: `LICENCES_T3.md`.
