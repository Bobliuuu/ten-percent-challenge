# SKILL7 big exam (UNSCORED, run once)

Built by `v2/lanes/L5/build_s7big.py`. Task SKILL7 (voice-skill routing), prompt identical to `task_search/EASY/items_dev.jsonl`.
Sources: MASSIVE en-US (`mteb/amazon_massive_scenario` parquet, train+validation+test) and CLINC150 plus (`DeepPavlov/clinc_oos` parquet, all splits, same intent->skill map as build_easy.py). No other locales.

Split rule (same as build_easy.py): bucket = sha256("easy-split-v1|SKILL7|" + normalised text) mod 100. Texts deduplicated by normalised text across both sources.
- dev.jsonl and calib.jsonl: bucket [0,70) only. calib = first 30 items per (src,kind) cell by hash; dev = the rest of the cell. Both exclude the 182 texts of the old `task_search/EASY/calib.jsonl`.
- exam.jsonl: bucket [70,100) only; all available items. Asserted: no exam text appears in dev, calib or old calib.
- exam.sha256 = sha256 of exam.jsonl, sealed at build time. The exam has NOT been scored or looked at; it must be run exactly once at the end and never used to choose anything.

Totals: dev 5913, calib 390, exam 2806. Exam by source: {'clinc150': 848, 'massive': 1958}. Exam by kind: {'Alarm': 256, 'Calendar': 901, 'Lists': 417, 'Music': 317, 'Transport': 405, 'Weather': 291, 'News': 219}.

| cell (src|kind) | dev | calib | exam |
|---|---|---|---|
| clinc150|Alarm | 169 | 30 | 83 |
| clinc150|Calendar | 368 | 30 | 187 |
| clinc150|Lists | 362 | 30 | 180 |
| clinc150|Music | 354 | 30 | 179 |
| clinc150|Transport | 376 | 30 | 177 |
| clinc150|Weather | 43 | 30 | 42 |
| massive|Alarm | 332 | 30 | 173 |
| massive|Calendar | 1601 | 30 | 714 |
| massive|Lists | 508 | 30 | 237 |
| massive|Music | 286 | 30 | 138 |
| massive|News | 443 | 30 | 219 |
| massive|Transport | 529 | 30 | 228 |
| massive|Weather | 542 | 30 | 249 |

## Shortfalls (honest)
Rule 3 wants >=300 per kind: Alarm (256), Weather (291) and News (219) are BELOW 300 (News exists only in MASSIVE; CLINC Weather has only 42 exam items). Other kinds pass. Total >=2000 and >=500 from the non-main source (clinc150: 848) pass, but the source must also be one never built/tuned/calibrated on: CLINC rows are fine for that only if the system was not built with CLINC data. Dev CLINC Weather has 43 items (<150); CLINC Alarm 169. Enlarging with other MASSIVE locales is not allowed.
