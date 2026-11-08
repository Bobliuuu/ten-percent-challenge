# EASY_TASKS: narrow routing tasks for every lane (PROVISIONAL, 2026-11-08, evidence pending)

Status: 9 easy tasks built (dev/calib only, `task_search/EASY/build_easy.py`, `items_dev.jsonl` 5360 items, `calib.jsonl`, `PLAN.json` with locked-candidate pool counts from hash bucket [70,100) never used). Sweep launched on Modal (--detach): Qwen2.5-14B, Qwen2.5-32B, Qwen3-14B, bf16 base + one quantized point at 9.85% (target_ratio 0.0985). Real greedy generation, exact label word. Numbers will replace this section.

## RESULTS at 9.86-9.88% of fp16 (batch 1; real greedy generation; DEV; 1 quantizer draw; retention of base-correct, Wilson lo in brackets)
| rank | task | base (14B/32B/Q3-14B) | Qwen2.5-14B | Qwen2.5-32B | Qwen3-14B | worst pooled kind (32B) |
|---|---|---|---|---|---|---|
| 1 | **SKILL7** | .957/.969/.949 | .977 [.962] | **.985 [.972]** | .980 [.966] | Music .95; worst src x kind .92 (n49) |
| 2 | SUPPORT7 (single synthetic source) | .893/.907/.926 | .975 [.952] | .973 [.949] | .976 [.954] | Order .88 |
| 3 | LANG8 (base at ceiling, 1 source) | .997/1.0/.997 | .869 | .978 [.957] | .978 | Portuguese .91 |
| 4 | BANKFAM (base .70, too hard) | .69/.70/.70 | .933 | .976 [.959] | .968 | Payment .93 |
| 5 | SENT2 | .91/.88/.89 | .867 | .975 [.959] | .871 | Positive .95 |
| 6 | TICKETQ (base .38), QTOPIC6, SENT3 | low | .75-.85 | .88-.97 | .81-.91 | dead/hard |
| 7 | SPAM2 (prompt flaw: Legit/ham read as Spam) | .85-.92 | .545 | .757 | .627 | Legit .50: use SPAM3 |
Verdict: SKILL7 is the only task that passes overall >= 98% with the base in the .94-.98 window and two real sources (Qwen2.5-32B .985 [.972], Qwen3-14B .980 [.966], Qwen2.5-14B .977 [.962]). Per-kind: pooled kinds >= .94-.95; per-cell (src x kind, n 43-50) minima .91-.94 are the only item below the 95% line and only because cells are small (rule is per kind, pooled). 14B/32B models both hit the 9.9% cap (9.86-9.88%). **RECOMMENDED PRIMARY: SKILL7 on Qwen2.5-32B (or Qwen3-14B for the smaller base).** SECONDARY: the redesigned banking task (BANK6/BANK5, batch 2, retention pending) or SUPPORT7 (single synthetic source: R3 weakness). Caveats: one draw (+-2 pt noise), dev items, per-cell n small.

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

## Batch 2 (cleaner label sets; `task_search/EASY/build_easy2.py`, items2_dev.jsonl 2330 items) base accuracy, bf16 real generation
| task | Qwen2.5-14B | Qwen3-14B | note |
|---|---|---|---|
| BANK6 (banking77: Card/Transfer/TopUp/Cash/Identity/Exchange; REDESIGNED EASY-B candidate) | .905 | .896 | weakest cell Cash (.72-.78); merging Cash away (BANK5) should lift base >= .92 |
| SKILL10 (MASSIVE 10 skills) | .932 | .938 | SmartHome .80, News .85 |
| SKILLB (7 other MASSIVE skills) | .889 | .899 | fuzzier skills |
| SPAM3 (Enron email + SMS, Spam/Ham) | .953 | .897 | two real sources, 2 classes |
| DOMAIN6 (CLINC domains) | .781 | .841 | Dining/Work ambiguous: not easy |

## Provisional recommendation (GUESS, no retention data yet)
1. **SKILL7** voice-skill routing (MASSIVE + CLINC150, Alarm/Weather/Music/Calendar/Lists/News/Transport). Two real sources, narrow label word, high margin. Caveat: News exists only in MASSIVE, Alarm in CLINC is thin (200) so per-kind 300 in the exam is tight for CLINC.
2. **BANKFAM** banking77 grouped to 7 families (Card/Transfer/TopUp/Payment/Cash/Account/Exchange) + CLINC banking intents as second source (5 families). Commercial; some family boundaries (Payment vs Cash vs Card) are fuzzy, so expect base ~90-95%.
3. (downgraded, see base table) **SPAM2** Spam/Legit, Enron email + SMS spam (two real sources, 2 classes, huge pools). Likely base ~97-99%: check rule 8 ceiling.

Others built: SUPPORT7 (Bitext ticket queues, templated/synthetic), TICKETQ (helpdesk queues, noisy labels), SENT3 / SENT2 (tweet_eval + Amazon reviews), LANG8 (language ID, probably too easy, single source), QTOPIC6 (Yahoo Answers + 20NG). Prompts and label sets in `task_search/EASY/PLAN.json`.

Use the dev items for configuration only; build your own locked exam from hash bucket [70,100) (`easy-split-v1|task|normalised text`), never from [0,70).

## How to finish (runs in flight, Modal detached)
- Apps `ap-1ijOLCu4ciOBrFYZBkR0fJ` (batch 1: Qwen2.5-14B/32B, Qwen3-14B at 9.85%, files `ct_<model>_e_r985.json`) and `ap-Af7mjo8pozdtrnn25aLxp0` (batch 2: `ct_<model>_f_r985.json`) write to Modal volume `tenpct-tasksearch:/dense/`.
- Pull: `modal volume get tenpct-tasksearch dense/ct_<model>_<e|f>_r985.json task_search/results/easy/ --force`; then `python3 task_search/analyze_easy.py task_search/results/easy` gives per-task retention + Wilson lower bound + worst (src x kind) and worst pooled kind cells.
- Rule-8 filter: base .94-.98. Currently SKILL7 (.95-.97) is the only built task in that window with two real sources; BANK6 .90, SPAM3 .90-.95, SKILL10 .93-.94 are close; LANG8 too easy (.997), others too hard.
