# TASK_CATALOGUE: what is the EASIEST commercial task that can pass the rules? (T3 lane, 2026-11-08)

DEV DATA ONLY. No locked/sealed exam, X15 file or GitHub post touched. Every number is REAL greedy generation of the label word (`<|im_end|>` stop, exact string, zero-shot prompt, no letters/options), bf16 base vs the repo quantizer at 9.85% of fp16 (`dense_core.py`, target_ratio 0.0985, Hadamard + GPTQ + entropy-coded grids, ~1.5 bpw body) on Qwen2.5-14B/32B-Instruct. Retention = P(quantized output correct | base output correct) on identical items. Items: `task_search/T3/items_dev_all.jsonl` (round 1, 26 tasks, 7.2k items), `items_r2.jsonl` (7 refined tasks, 2.6k), `items_r3.jsonl` (6 cleaner-label tasks, 2.5k); class-balanced, sha256-ordered, deduplicated, 160 items per source (2 or 3 sources per task), builder `build_t3.py`, licences `LICENCES_T3.md`, pool sizes `PLAN*.json`. Quantizer calibration = the mixture of ALL tasks' calibration rows of that item file (hash-disjoint), not task-matched, so retention is a little pessimistic. Compute: 3 Vast boxes (RTX PRO 6000 96GB, A100-80GB, A40), all destroyed at the end (RUNS.md).
The other agent's lane (`EASY_TASKS.md`: SKILL7, SKILL10, BANK6/8, SPAM3, SENT, LANG8, QTOPIC6, ...) is NOT duplicated here; where it matters its numbers are quoted.

## 0. The filters used (ROADMAP_LANES TASK OBJECTIVE)
R1 commercial/not toy/not narrow; R3 >=2 real sources, >=2000 locked, >=300 per kind, >=500 from a source never tuned on; R7 real label word; R8 base .94-.98 target (drop >.985); R5 teachable NEW label; then rank by predicted robustness (few classes, separable labels, big margin, clean labels, short inputs).

## 1. INTERIM RESULT (base accuracy, margins; quantized retention at 9.85% still running, see section 2 when filled)
Best new finds that satisfy the filters on paper and have base in or near the .93 band (Qwen2.5-14B bf16, real generation):
| task | classes | sources (period) | base 14B | worst class | margin>=6 of correct | why it is easy | rule risks |
|---|---|---|---|---|---|---|---|
| **NEWS4C** newsroom desk: Business/Entertainment/Politics/Sports | 4 | HuffPost 2012-22 (CC0) + BBC 2004-05 (non-commercial research licence) | .928 | .89 Sports | .95 | coarse separable topics; BBC cells .93-.97 | HuffPost Sports/Business label noise; BBC licence NC (sealed-source licence GRAY); BBC pool only ~1k (>=500 OK, per-class cells inside BBC <300 but pooled >=300 OK); topic routing is a benchmark-shaped job (R1/R9 GRAY) |
| **PHISH2C** email Scam/Genuine | 2 | zefang-liu phishing-email + puyang seven-phishing-email (both Kaggle-style aggregations of Nazario/Enron/SpamAssassin: NOT independent, R3 GRAY) | .928 | .88 Scam | .94 | binary, strong lexical cues | the two sources overlap in upstream corpora; mixed-era; licence per upstream |
| SPAM3 / SKILL7 (EASY lane) | 2 / 7 | see EASY_TASKS.md | .953 (Q2.5-14B) / .949 (Q3-14B) | | | | see TASK_RULE_AUDIT.md |
Ceiling controls (rule 8 FAIL, base >.985): LANG12B language ID 12 languages .997 (papluca + MASSIVE utterances, two real sources), CODEPROSE 1.000, CODELANG .984 (.92 Ruby), LANG20 .955 (Portuguese .50 label noise). Near misses (base .88-.92): SENT2B .919 (SST2 Positive .80), SENT2C .902 (IMDB Positive .75), ENTAIL2 .894, PHISH2 .903, PHISH2B .897, NEWS5 .887 (HuffPost Technology .53 label noise), TOPIC4 .869. Too hard (base .65-.85, label noise or ambiguous definitions): TOX2/TOX2B/TOX2C .79-.84, HATE2 .85, OFFEN2 .81, PSAFE .79, PII2/PII2C .86-.89, INJ2 .81 (also lock-limited: deepset Injection 203 rows), NLI3 .84, PARA2 .78, BOOLQ .81, FINSENT .84, DEPT3 .65, TOPIC5 .82, UNFAIR .71, RCT5 .67, SARC2 .68, FAKENEWS .72 (artefact), TICKETTYPE .56, TICKETPRIO .37, ANSWERABLE .65, DEFECT .42 (chance). Pattern: zero-shot base accuracy is capped by DATASET LABEL NOISE and fuzzy class definitions, not by the model; only coarse, cleanly-defined label sets (language, code-vs-prose, voice skills, desk routing) reach .93+.

## 2. Per-task base tables (zero-shot, real generation)
### Qwen2.5-14B round 1
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


## 3. Quantized retention at 9.85% of fp16 (real generation)
PENDING (14B/32B quantization jobs running on Vast; 7B finished: 7B retention collapses to class priors, e.g. CODELANG Go 0.00, PHISH2 Legitimate 0.02; 7B is not a candidate). Section is regenerated by `rank_t3.py` when files land.

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
- NEWS4C: new label **Technology** (HuffPost TECH 2055 rows + BBC tech ~400; base cannot emit a label absent from its menu) or **Health** (HuffPost WELLNESS 17.9k). 25 annotated + >=25 unseen teach items + neighbours Business (gadget/earnings stories). Mismatch risk: Technology vs Business is the base's weakest boundary.
- PHISH2C: new label **Spam** (junk advertising, Enron/SMS spam, large pools) or Fraud-by-SMS; neighbours = Scam/Genuine mail that mentions offers. 
- SPAM3 (EASY lane): new label Phishing (zefang-liu 6.4k phishing rows).
- SKILL7: new skill Email/SmartHome (MASSIVE).
- LANG12B (ceiling): new language (e.g. Vietnamese) is trivially teachable but base is at ceiling, R8 FAIL.

## 6. Caveats
Single zero-shot prompt per task (round 2/3 are prompt refinements of round 1; they are NOT hidden-set tuned: all numbers are on the dev items used to choose, so base accuracies carry 1-2 pt selection optimism). 160 items per (task, source) so per-class cells are n=40-80. Quantizer calibration is a cross-task mixture. No locked data used; nothing sealed. Full per-task rule gate: TASK_RULE_AUDIT.md.
