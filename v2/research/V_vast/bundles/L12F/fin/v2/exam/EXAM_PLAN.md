# EXAM_PLAN: the shared locked exam for the easy tasks (SKILL7, BANK6)

Built 2026-11-08 by `build_exam.py` (+ `pools.py`, `leakscan.py`), sealed by sha256 in `MANIFEST.json` and committed BEFORE any model run. No model, no lane result and no sealed/locked file of any lane was read to build it. Public data only. Loader and usage: `README.md`, `loader.py`. Prompts are the lanes' current prompts (`task_search/EASY/PLAN.json` SKILL7, `PLAN2.json` BANK6), 900-char truncation, labels unchanged, so existing dev numbers stay comparable.

## 1. Audit: what pools exist and what was already used
Raw pools after exact-text dedupe (all splits, rows mappable to the task labels):
- SKILL7 kinds. MASSIVE-en (mteb scenario mirror; train+val+test = 17.5k rows -> 5.8k mapped), CLINC150 plus (train+val+test, oos not mapped), **HWU64** (DeepPavlov mirror, 64 intents / 18 scenarios, 10.0k rows; no lane ever touched it), **MTOP-en** (domains alarm, timer, weather, music, news, recipes; 19.7k rows, CC-BY-SA; no lane ever touched it). SLURP: audio-heavy and its transcripts are the same utterances as MASSIVE en-US (MASSIVE is derived from it), so it is NOT an independent source and was not used. Snips (7 intents, no alarm/lists/news) and ATIS (airline-only) do not map to these skills.
- BANK6 kinds. Banking77 train+test (13.1k), **Bitext retail-banking** (25.5k synthetic, 26 intents; maps to Card/Transfer/Cash only; never used by any lane). The CLINC banking domain was calibrated/probed in earlier banking tasks (BANKFAM, banking8), so it is not a never-used source. MINDS-14 en (563-1.7k transcripts) is too small. Banking77 itself has only 637 Exchange rows and 909 Identity rows in total.
- Hash-bucket [70,100) alone (the lanes' original plan) gives, all splits: Alarm 173+83, Weather 249+42, Music 138+179 (MASSIVE+CLINC), News 219 (MASSIVE only), Lists 417, Transport 405, Calendar 901; i.e. Alarm 256, Weather 291, News 219 < 300. That is why SKILL7 as hash-bucketed failed. Widening to [40,100) is NOT possible: [0,70) of MASSIVE and CLINC is almost completely used (dev, calibration, selection, other lanes' bigger caps and the older MoE voice / TOX routing / banking10 / banking8 tasks used different hash schemes over the same texts). `leakscan.py` scans every non-locked json/jsonl in the repo (2.7k files, skips exam|lock|seal|x15|never paths and raw public-dataset dumps); only about 15% of the MASSIVE+CLINC pool survives as clean (nearly all of it inside [70,100)).
- Consequence: the exam is built from rows that occur NOWHERE else in the repo (tier `clean`), plus a flagged tier `seen_*` used only to fill kinds that would otherwise be < 300: `seen_tox` = text occurs only in the TF-IDF-baseline/probe files of the old queue-routing task (no entry, calibration or lane config was fitted on them), `seen_l5dev` = text occurs only in L5's banking10 *measurement-only* dev items (never calibrated on). Each exam row carries `tier`; `load(..., clean_only=True)` drops them.

## 2. Design decisions
- SKILL7 keeps its 7 kinds/labels/prompt. Sources: tuned-on (MASSIVE, CLINC150) plus two NEVER-USED sources (HWU64, MTOP). Never-used = sealed completely: no lane may use HWU64 / MTOP / Bitext for dev, calibration, selection, feedback annotation or config choice. MTOP timer/alarm -> Alarm (same mapping as CLINC), recipes -> teach only.
- BANK6 keeps its 6 kinds/labels/prompt. Sources: Banking77 (tuned-on) plus Bitext retail-banking (never used; Card/Transfer/Cash only). Exchange and Identity have no second source: they are Banking77-only and can only reach 300 with the `seen_l5dev` tier.
- NEW label (feedback target) never in the prompt menu: SKILL7 -> `Recipe` (MASSIVE cooking + CLINC recipe/ingredients_list/ingredient_substitution/cook_time + HWU cooking_recipe + MTOP recipes); BANK6 -> `Security` (Banking77 compromised_card, lost_or_stolen_phone, passcode_forgotten, pin_blocked, change_pin + Bitext block_card, get_password, set_up_password). Neither set of items is in any exam kind.
- Locked teach sets (rule 5, >= 25 unseen, gold = the new label): SKILL7 164 items ({'clinc150': 42, 'hwu64': 15, 'massive': 7, 'mtop': 100}); BANK6 200 ({'banking77': 100, 'bitext_bank': 100}). Same-source and cross-source halves; cross-source (MTOP/HWU64/Bitext) is also the maintainer-style own-data proxy.
- Annotation pools (feedback candidates, pick <= 25 real mistakes; tuned-on sources ONLY, never HWU/MTOP/Bitext): `*_ann_pool` (SKILL7 300 Recipe rows from MASSIVE+CLINC [0,70); BANK6 150 Security rows from Banking77). Disjoint from teach and exam (asserted at build).
- Neighbours (rule 5 "nearby exam items the feedback could affect"): `skill7_neighbours.json` (379 exam ids: all Lists items + food/shopping/cooking keyword matches); `bank6_neighbours.json` (804 ids: all Card + Identity items + lost/stolen/pin/password/fraud keyword matches). Keyword rule is fixed text matching, no model.
- Dev / calibration / selection (all disjoint from exam, teach, ann, by construction: eligible exam rows exclude any text present in any existing split file; verified again at build against every EASY and skill7_* file): SKILL7 dev = EASY `items_dev` SKILL7 rows (682), cal = EASY `calib` (182) / `moe_quant/data/skill7_cal` (2031, `cal_big`), sel = `moe_quant/data/skill7_sel` (1802): all UNCHANGED files. BANK6 dev = EASY batch-2 `items2_dev` (402), cal = `calib2` (84) + new `bank6_cal` (698), sel = new `bank6_sel` (342).

## 3. Counts (rule 3 gates: >= 2000 items, >= 300 per kind, >= 500 from a never-used source)
### SKILL7 exam (2744 items)
| kind | clinc150 | hwu64 | massive | mtop | total | clean-only | seen-in-other-task-dev |
|---|---|---|---|---|---|---|---|
| Alarm | 60 | 89 | 71 | 200 | 420 | 358 | 62 |
| Calendar | 76 | 100 | 184 | 0 | 360 | 337 | 23 |
| Lists | 123 | 112 | 109 | 0 | 344 | 223 | 121 |
| Music | 122 | 100 | 54 | 144 | 420 | 335 | 85 |
| News | 0 | 64 | 91 | 265 | 420 | 379 | 41 |
| Transport | 125 | 136 | 99 | 0 | 360 | 236 | 124 |
| Weather | 32 | 37 | 109 | 242 | 420 | 344 | 76 |
| **all** | 538 | 638 | 717 | 851 | 2744 | 2212 | 532 |

Never-used source items (HWU64 + MTOP): 1489 (gate 500). Tuned-on sources (MASSIVE, CLINC150): 1255.
### BANK6 exam (2237 items)
| kind | banking77 | bitext_bank | total | clean-only | seen-in-other-task-dev |
|---|---|---|---|---|---|
| Card | 200 | 200 | 400 | 400 | 0 |
| Cash | 200 | 200 | 400 | 400 | 0 |
| Exchange | 347 | 0 | 347 | 142 | 205 |
| Identity | 330 | 0 | 330 | 187 | 143 |
| TopUp | 360 | 0 | 360 | 360 | 0 |
| Transfer | 200 | 200 | 400 | 400 | 0 |
| **all** | 1637 | 600 | 2237 | 1889 | 348 |

Never-used source items (Bitext retail-banking): 600 (gate 500).
### Teach / annotation / selection sets
| file | n | per source |
|---|---|---|
| bank6_ann_pool | 150 | {'banking77': 150} |
| bank6_cal | 698 | {'banking77': 698} |
| bank6_sel | 342 | {'banking77': 342} |
| bank6_teach | 200 | {'banking77': 100, 'bitext_bank': 100} |
| skill7_ann_pool | 300 | {'clinc150': 196, 'massive': 104} |
| skill7_teach | 164 | {'clinc150': 42, 'hwu64': 15, 'massive': 7, 'mtop': 100} |

## 4. Rule-3 verdicts (honest)
- **SKILL7: PASS** on all three counts under the stated policy: 2744 items, every kind 344-420, 1489 items (54%) from sources never used (HWU64, MTOP). Honest caveats: (a) 532 exam items are tier `seen_tox` (Alarm 62, Calendar 23, Lists 121, Music 85, News 41, Transport 124, Weather 76: text occurs in the old queue-routing task's baseline/probe files only). On `clean_only` every kind is still >= 300 EXCEPT **Lists 223 and Transport 236** (MASSIVE+CLINC+HWU64 simply do not contain more unused Lists/Transport utterances). (b) per-kind source mix differs: Alarm/Weather/Music/News are 58-78% HWU64+MTOP, Calendar/Lists/Transport are MASSIVE/CLINC/HWU64 only. (c) MTOP domain labels are noisy (some 'alarm' rows are reminders); noisy rows are base-wrong and simply drop out of the retention denominator.
- **Closest strictly-clean variant if the seen_tox tier is refused: SKILL5 = Alarm, Weather, Music, News, Calendar** (drop Lists and Transport; clean counts 358/344/335/379/337). The menu/prompt changes, so dev results are no longer comparable. A second option is to merge Lists into Calendar as `Planner` (337+223 clean). Not built; build only if the maintainer-risk is judged real.
- **BANK6: PASS on counts, GRAY on cleanliness.** 2237 items, kinds 330-400, 600 Bitext items from a never-used source (Card/Transfer/Cash only, synthetic). Exchange (347) and Identity (330) reach 300 only with 205 / 143 `seen_l5dev` rows (clean: Exchange 142, Identity 187). Strictly-clean variant: BANK4 = Card, Transfer, TopUp, Cash (all >= 360 clean) or add a real second banking source for Exchange/Identity (none exists in open data we could find).
- Not read by this build: every file whose path contains exam|lock|seal|x15|never (L4 rival3 exam/never, X15, L9 sealed, ...). Overlap with those is therefore unchecked by design; they are other tasks' exams.

## 5. What lanes must switch to
1. Evaluate the final model ONCE on `load('SKILL7','exam')` (or BANK6) with the prompt in the item (`user`), greedy, exact label match; report with `report(...)` (overall retention >= 98%, per kind >= 95%, Wilson bounds, never-used-source retention, `clean_only` retention as a sensitivity row).
2. Dev results already obtained on EASY dev files stay as they are (same files via `load(task,'dev')`). Calibration and selection ONLY from `load(task,'cal'|'cal_big'|'sel')`. Never build splits by the hash-bucket rule again (it overlaps the exam for BANK6).
3. Never touch HWU64, MTOP or Bitext rows for anything but the exam/teach; never annotate from teach or exam (feedback annotation <= 25 from `ann_pool`).
4. Rule 5: score the new label on `teach` before and after feedback, neighbours via `load(task,'neighbours')` ids inside the exam, plus shuffled-label and empty-feedback controls.
5. Run `python3 -I v2/exam/loader.py` before any run: it fails if a sealed file changed.
