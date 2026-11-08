# EASY_TASKS: narrow routing tasks for every lane (PROVISIONAL, 2026-11-08, evidence pending)

Status: 9 easy tasks built (dev/calib only, `task_search/EASY/build_easy.py`, `items_dev.jsonl` 5360 items, `calib.jsonl`, `PLAN.json` with locked-candidate pool counts from hash bucket [70,100) never used). Sweep launched on Modal (--detach): Qwen2.5-14B, Qwen2.5-32B, Qwen3-14B, bf16 base + one quantized point at 9.85% (target_ratio 0.0985). Real greedy generation, exact label word. Numbers will replace this section.

## Provisional recommendation (GUESS, no retention data yet)
1. **SKILL7** voice-skill routing (MASSIVE + CLINC150, Alarm/Weather/Music/Calendar/Lists/News/Transport). Two real sources, narrow label word, high margin. Caveat: News exists only in MASSIVE, Alarm in CLINC is thin (200) so per-kind 300 in the exam is tight for CLINC.
2. **BANKFAM** banking77 grouped to 7 families (Card/Transfer/TopUp/Payment/Cash/Account/Exchange) + CLINC banking intents as second source (5 families). Commercial; some family boundaries (Payment vs Cash vs Card) are fuzzy, so expect base ~90-95%.
3. **SPAM2** Spam/Legit, Enron email + SMS spam (two real sources, 2 classes, huge pools). Likely base ~97-99%: check rule 8 ceiling.

Others built: SUPPORT7 (Bitext ticket queues, templated/synthetic), TICKETQ (helpdesk queues, noisy labels), SENT3 / SENT2 (tweet_eval + Amazon reviews), LANG8 (language ID, probably too easy, single source), QTOPIC6 (Yahoo Answers + 20NG). Prompts and label sets in `task_search/EASY/PLAN.json`.

Use the dev items for configuration only; build your own locked exam from hash bucket [70,100) (`easy-split-v1|task|normalised text`), never from [0,70).
