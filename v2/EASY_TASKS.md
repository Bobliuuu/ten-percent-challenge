# EASY_TASKS: narrow routing tasks for every lane (PROVISIONAL, 2026-11-08, evidence pending)

Status: 9 easy tasks built (dev/calib only, `task_search/EASY/build_easy.py`, `items_dev.jsonl` 5360 items, `calib.jsonl`, `PLAN.json` with locked-candidate pool counts from hash bucket [70,100) never used). Sweep launched on Modal (--detach): Qwen2.5-14B, Qwen2.5-32B, Qwen3-14B, bf16 base + one quantized point at 9.85% (target_ratio 0.0985). Real greedy generation, exact label word. Numbers will replace this section.

## Base accuracy (Qwen3-14B bf16, real greedy generation, zero-shot, dev items; Qwen2.5-14B/32B bases pending)
| task | n | base acc | verdict (rule 8: above chance, below ceiling) |
|---|---|---|---|
| SKILL7 | 682 | 0.949 | GOOD (worst cells MASSIVE Calendar .83, CLINC Alarm .83) |
| LANG8 | 360 | 0.997 | ceiling (rule 8 risk), single source |
| SUPPORT7 | 364 | 0.926 | GOOD but synthetic/templated, single source |
| SENT2 | 720 | 0.892 | ok-ish (Amazon Positive .79) |
| SPAM2 | 718 | 0.848 | SMS Legit only .58 (prompt/ham ambiguity), needs prompt tweak |
| QTOPIC6 | 720 | 0.714 | label ambiguity (Business .30, 20NG Science .30) |
| BANKFAM | 716 | 0.700 | family boundaries fuzzy (CLINC Transfer .26, Cash .54): redesign |
| SENT3 | 720 | 0.664 | Neutral hard: not easy |
| TICKETQ | 360 | 0.378 | queue labels noisy (HR .03, Returns .05): dead |
Conclusion so far: SKILL7 is the only multi-source task near the 93-98% target at base; BANKFAM as built is NOT easy (use banking77 grouped differently or with few-shot-free clearer label names). Quantized retention results pending.

## Provisional recommendation (GUESS, no retention data yet)
1. **SKILL7** voice-skill routing (MASSIVE + CLINC150, Alarm/Weather/Music/Calendar/Lists/News/Transport). Two real sources, narrow label word, high margin. Caveat: News exists only in MASSIVE, Alarm in CLINC is thin (200) so per-kind 300 in the exam is tight for CLINC.
2. **BANKFAM** banking77 grouped to 7 families (Card/Transfer/TopUp/Payment/Cash/Account/Exchange) + CLINC banking intents as second source (5 families). Commercial; some family boundaries (Payment vs Cash vs Card) are fuzzy, so expect base ~90-95%.
3. (downgraded, see base table) **SPAM2** Spam/Legit, Enron email + SMS spam (two real sources, 2 classes, huge pools). Likely base ~97-99%: check rule 8 ceiling.

Others built: SUPPORT7 (Bitext ticket queues, templated/synthetic), TICKETQ (helpdesk queues, noisy labels), SENT3 / SENT2 (tweet_eval + Amazon reviews), LANG8 (language ID, probably too easy, single source), QTOPIC6 (Yahoo Answers + 20NG). Prompts and label sets in `task_search/EASY/PLAN.json`.

Use the dev items for configuration only; build your own locked exam from hash bucket [70,100) (`easy-split-v1|task|normalised text`), never from [0,70).
