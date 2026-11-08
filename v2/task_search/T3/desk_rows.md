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
