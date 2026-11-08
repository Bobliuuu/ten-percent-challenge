# E0: ambiguity-floor audit (DEV per-item rows only, Mac/CPU, no locked file opened)

Code: `analyze.py` (reproduces everything; `python3 analyze.py` ~2 s). Raw numbers: `tables.json`.
Concept ("ambiguity sets a floor"): faithfulness = kept / base-correct is capped by base-correct items that flip under *any* compression. A floor of ~0.8% (about 11 of ~1,340 base-correct items) is typical for a clean 7-way router.

## 0. Bottom line
1. The ambiguity floor itself is NOT our binding problem. Items that fail at all three budgets (1.57, 1.79, 1.96 bpw) are 0.6-1.4% of base-correct at 32B for the candidate tasks, about the 0.8% typical of a clean router. So a 98% pass is not ruled out by ambiguity; it is ruled out by quantization noise: single-config flips at 10% are 4-7% (32B) and 9-12% (14B), and 80-90% of them do not repeat in a second config (section 2).
2. Most fragile items are not at the label boundary only: 22% of 32B base-correct items have margin <10 logits and carry 63% of the 10% flips; margin <4 items (6% of INJ, 3.5% of NEWS) flip 21-28% of the time (section 3).
3. Per-task ceilings at 10% from our data (32B dev, single draw): NEWS 0.961, INJ 0.957, TOOLSEL 0.928. 72B real generation 0.94-0.95. Rule 2 needs ~0.985 true retention in EVERY cell to pass with ~95% probability (0.98 gives ~50%). Estimated pass probabilities are 1-4% for the best (section 5).
4. Exam sizing: at the 95% per-cell lower-bound target, a cell needs 75 to 3,400 base-correct items depending on its true retention (section 6). Our planned cells (330-800 items, base accuracy 0.5-0.99) are enough only if the true cell retention is >= ~0.99; at 0.98 the INJ S2 cells (360 items, base acc 0.71-0.81 => 255-290 base-correct) are too small.

## 1. Data used (all DEV; nothing locked, no S2-lock rows)
| what | file |
|---|---|
| 12 tasks x (S1 + S3 source) ~600 items; configs bf16, r096 (14B/7B only), r100 (1.57 bpw, 10.0%), r1125 (1.79 bpw, 11.3%), r125 (1.96 bpw, 12.5%); fields correct, m_min (min base logit gap over answer tokens), pred_first, src, kind | `v2/task_search/results/Qwen2.5-{7B,14B,32B}-Instruct_r1.json` |
| INJR exact build (promptshield S1, deepset+gandalf S3, JailbreakHub audit S2audit) | `v2/task_search/results/Qwen2.5-{14B,32B}-Instruct_i.json` |
| NEWS 4-menu exact build (AG S1, BBC S3, HuffPost probe S2) | `v2/task_search/results/Qwen2.5-32B-Instruct_n.json` (14B: bf16 only, so no 14B NEWS flips) |
| 72B teacher-forced label tasks, 7 configs at 9.6%, base bf16 | `v2/research/D_scale/results/ct_Qwen2.5-72B_*.json` |
| 72B real greedy generation INJ / TOOLSEL, fp8 base vs artifacts V3, V7 | `v2/research/D_scale/results/real_{base72,art72_V3,art72_V7}_{inj,toolsel}.json` |
| lock cell structure (counts only) | `v2/tournament/INJ/LOCK.json` + `v2/tournament/INJ/REPORT.md`, `v2/task_search/NEWS/HASH_PLAN.json`, `v2/TASK_SEARCH.md` s5 |
| `v2/entry32b/` | contains code only (no per-item outputs in the repo; the notes say results sit in the Modal volume). Not used. |
| `v2/compress*/`, `v2/quality`, `v2/harness` | no per-item label-task rows with margins; T1 tool-call rows only through D_scale; not used |

Stand-in for margin: `m_min` is the top1-top2 logit gap at the weakest answer token under the bf16 base (logits, not probabilities).

## 2. (1) Flip fraction vs bits, and persistent (ambiguity-like) fraction
Retention = P(candidate correct | base correct). `floor3` = fraction of base-correct failing in ALL of r100, r1125, r125. `indep3` = product of the three single-config fail rates (what pure independent noise would give). `persist` = floor3 / fail rate at r100 (share of 10% flips that also fail at 12.5%). `agree` = fraction of ALL items whose correct/incorrect status is identical across the three configs. Source `analyze.py` -> `tables.json:T1`.

| model | task | base acc | n base-correct | ret 1.57 | ret 1.79 | ret 1.96 | floor3 | indep3 | persist | agree |
|---|---|---|---|---|---|---|---|---|---|---|
| 32B | NEWS (4-menu, AG+BBC+HuffPost) | 0.868 | 1389 | 0.961 | 0.962 | 0.979 | 0.6% | 0.1% | 17% | 0.92 |
| 32B | INJR (tournament dev) | 0.809 | 1590 | 0.957 | 0.958 | 0.942 | 1.1% | 0.2% | 25% | 0.90 |
| 32B | TOOLSEL | 0.862 | 517 | 0.928 | 0.967 | 0.979 | 1.4% | 0.2% | 19% | 0.93 |
| 32B | FIN | 0.872 | 523 | 0.937 | 0.971 | 0.975 | 1.0% | 0.2% | 15% | 0.91 |
| 32B | ROUTE (29 cls) | 0.774 | 475 | 0.947 | 0.958 | 0.981 | 0.8% | 0.2% | 16% | 0.90 |
| 32B | INJ-easy | 0.952 | 571 | 0.981 | 0.989 | 0.989 | 0.0% | 0.0% | 0% | 0.95 |
| 14B | INJR | 0.791 | 1555 | 0.912 | 0.954 | 0.968 | 1.8% | 0.4% | 20% | 0.87 |
| 14B | TOOLSEL | 0.843 | 506 | 0.881 | 0.945 | 0.958 | 2.8% | 0.7% | 23% | 0.90 |
| 14B | NEWS-6 menu | 0.908 | 545 | 0.850 | 0.917 | 0.947 | 3.3% | 1.2% | 22% | 0.84 |
| 7B | INJ-easy | 0.913 | 548 | 0.901 | 0.953 | 0.958 | 1.5% | 0.5% | 15% | 0.85 |
| 7B | TOOLSEL | 0.848 | 509 | 0.819 | 0.939 | 0.963 | 1.8% | 1.1% | 10% | 0.81 |
| 7B | NEWS-6 menu | 0.847 | 508 | 0.687 | 0.831 | 0.872 | 8.1% | 5.3% | 26% | 0.72 |

Full 12-task x 3-size table: `tables.json:T1`. Reading:
- Bits dependence (32B): +0.1 to +5 pts from 1.57 to 1.96 bpw (NEWS +1.8, TOOLSEL +5.1, INJR -1.5, non-monotone = draw noise ~2 pt). 14B: +5 to +10 pts. 7B: +7 to +18 pts. Bits matter more than the floor.
- floor3 is 3-8x larger than the independent expectation, so a real persistent component exists, of size 0.6-1.4% at 32B. That is the best available estimate of the ambiguity floor; it is an UPPER bound for the true floor because some of those items fail at 1.96 bpw from noise too. It is in line with the ~0.8% typical of a clean router.
- Only 15-25% of the 10% flips are persistent. The other 75-85% are noise flips that a better quantiser or higher bits could in principle remove. That is where the 4-7% gap to 98% lives.
- 72B teacher-forced, 7 configs at 9.6% (`tables.json:T72`): fail in all 7 configs: INJ 0.18%, TOOLSEL 0.57%, SENT 1.8%, TOX 1.3%, ROUTE 2.2%, FIN 4.2% (SPAM 20%, NLI 8.8%, AIDET 51% are class-collapse tasks). Mean retention across the 7 configs: INJ 0.932 (best 0.972), TOOLSEL 0.974 (best 0.989). The 72B real-generation numbers are lower (below).
- 72B REAL generation, 2 artifacts (V3, V7) vs fp8 base, dev: INJ 0.938 / 0.944 (fail in both: 3.9%), TOOLSEL 0.943 / 0.954 (both: 3.0%). Binding cells: INJ S1_benign 0.886-0.917, S1_injection 0.92, S3_injection 0.91-0.95; TOOLSEL S1/none 0.80-0.88. `tables.json:T72R`. Real generation is clearly harder than the teacher-forced proxy (D_scale REPORT s0 says the same).

## 3. (2) Margin-vs-flip curve and the fragile fraction
Pooled over 12 tasks, 32B, base-correct items, flip rate by base m_min bin (n in brackets). `flip_all` = fails at all three budgets. Source `tables.json:margin`.

| m_min | n | flip 1.57 | flip 1.79 | flip 1.96 | flip at all 3 |
|---|---|---|---|---|---|
| [0,2) | 243 | 27.6% | 27.2% | 33.3% | 14.0% |
| [2,4) | 349 | 21.2% | 20.9% | 18.9% | 8.6% |
| [4,6) | 409 | 16.9% | 12.0% | 14.2% | 3.7% |
| [6,8) | 458 | 18.6% | 12.9% | 10.0% | 4.8% |
| [8,10) | 574 | 17.9% | 11.0% | 5.4% | 2.8% |
| [10,12) | 6172 | 3.2% | 2.1% | 0.9% | 0.4% |
| [12,16) | 870 | 3.9% | 1.5% | 0.1% | 0.1% |

14B same curve: [0,2) 52% / 45% / 39%; [10,12) 14.8% / 4.7% / 1.7%.
- Fragile definition (margin <10): 22% of 32B base-correct items; they carry 63% of the 1.57 bpw flips. Persistent flips concentrate even more: margin <4 is 6.5% of the items but 38% of the all-budget failures.
- Margin-<4 share of base-correct items (32B, `tables.json:small_margin`): INJR 6.5%, NEWS 3.5%, TOOLSEL 2.9%, ROUTE 5.7%, SENT 2.2%, TOX 6.7%, HALU 9.8%, FIN 33% (FIN is the smallest-margin task; why it flips most).
- Task-specific curves (32B): NEWS [0,2) flips 48%, [2,4) 31%, [4,10) 12-30%, [10,12) 1.2% (n=1234); INJR [0,2) 29%, [10,12) 2.8% (n=1206). For NEWS and INJR the confident bulk is already quiet at 1.57 bpw (1.2% and 2.8%); the failures are in the margin <10 tail (NEWS 3.5+% of items at margin <4, but 12-48% flip).
- Implication: if noise were removed, flips would fall to ~persistent floor3. At current noise, ~(fragile share) x (flip rate) already costs INJR 6.5% x ~20% = 1.3 pts and NEWS 3.5% x ~35% = 1.2 pts for margin <4 alone, plus 2-3 pts from margin 4-10 and 1-3% from confident items.

## 4. (3) Label noise / ambiguity of gold, base accuracy, confusable pairs
Base accuracy and confusions come from the bf16 rows (`tables.json:base_cells`, `confusions`). Confusion pairs map the base's first predicted token to a label via base-correct rows.

INJR (32B bf16): overall 0.809. Base errors are 309 injection->benign vs 62 benign->injection (83% of errors miss attacks).
| cell | base acc | 32B ret 1.57 | ret 1.96 |
|---|---|---|---|
| S1 benign | 0.995 | 0.995 | 0.987 |
| S1 injection | 0.593 | 0.945 | 0.882 |
| S3 injection (deepset+gandalf) | 0.702 | 0.952 | 0.888 |
| S3 benign | 0.995 | 0.979 | 0.987 |
| S2 injection (JailbreakHub, community labels) | 0.815 | 0.988 | 0.926 |
| S2 benign (JailbreakHub) | 0.710 | 0.782 | 0.908 |
Reading: S2-benign is a noisy-gold cell (community labels, base only 71%); half of the base-correct items there are probably borderline. S1-injection (PromptShield inserts attacks into benign templates) is hard for the base (59%) even though flips there are small.

NEWS (32B bf16): overall 0.868; Technology is the low-accuracy class in every source (AG 0.64, BBC 0.81, HuffPost 0.69), World in BBC 0.76, Business in HuffPost 0.84. Confusable pairs (count of base errors): Technology->Business 73, World->Business 45, Technology->World 36, Business->World 23, Business->Technology 15. Tech-vs-Business and World-vs-Business are boundary labels (gold is a single topic for multi-topic articles). Dev cell retention at 1.57 bpw: AG Tech 0.906 (Wilson lo 0.83), BBC Tech 0.893, BBC World 0.877, HuffPost Tech 0.913 / Business 0.94; all other cells 0.98-1.0.

TOOLSEL (32B): S3/none base acc 0.493 (the base calls a tool when none applies; 27 none->tool), S3/tool retention 0.786 at 1.57 bpw (Wilson lo 0.71), 0.945 at 1.96 bpw. FIN: Neutral<->Bullish/Bearish dominates (44+25 errors).

Upper bound on faithfulness at 10% per task (estimates; "ambiguity-only" = 1 - floor3; "observed" = best of our dev measures at <=10%):
| task | ambiguity-only ceiling | observed 32B 1.57 bpw | observed 72B 9.6% real gen | verdict |
|---|---|---|---|---|
| NEWS 4-menu | ~99.4% | 96.1% | not run | 2.0 pts short at 32B before cell effects; Tech cells 0.89-0.91 |
| INJ (tournament) | ~98.9% | 95.7% | 93.8-94.4% (V3/V7), 95.2% (V9 per D_scale REPORT) | tight ceiling; S2 labels noisy |
| TOOLSEL label | ~98.6% | 92.8% | 94.3-95.4% | S1/none and S3/tool cells fail |
| T1/C2 tool calls | not measurable here (no per-item margins) | n/a | D_scale: best 84-86% at 9.6% | far; S1/nocall 45-57% |
So ambiguity leaves < 1.5 pts of headroom to lose at most; the other ~3-7 pts are noise that has to be removed by the compressor, not by task curation.

## 5. (4) Ranking by expected probability of passing rule 2 (>=98% overall, >=95% in every kind and every source x kind cell)
Monte Carlo (`analyze.py`, 4,000 draws): exam sampled with the planned cell sizes (INJ lock: S1/S2 x inj/benign = 800/800/360/360; NEWS plan: AG, HuffPost x 4 classes x 400), base-correct counts at the dev base accuracy, retention drawn from Beta posteriors of the dev cell retention. Source `tables.json:pass_prob`, `sweep_uniform`.
| task | config | P(overall>=98%) | P(all cells>=95%) | P(both) |
|---|---|---|---|---|
| NEWS 32B | 1.57 bpw | 0.3% | 0.2% | 0.05% |
| NEWS 32B | 1.96 bpw (12.5%, NOT legal) | 11.9% | 1.9% | 1.5% |
| INJ 32B | 1.57 / 1.79 / 1.96 | 0% | 0% | 0% |
| INJ 14B | any | <=0.1% | 0% | 0% |
Uniform-retention sweep (all cells share one true retention): P(pass) = 0 at 0.96, 0.2% at 0.97, ~50% at 0.98, 95-98% at 0.985, 100% at >=0.99 (both INJ and NEWS layouts).

Ranking (judgement, not measured; confidence low-medium; single draw per config, quantiser draw noise ~2 pts, dev sets ~100-400 base-correct per cell):
1. NEWS-topic router on a 32B-class dense base: ~3% (range 1-8%). Highest retention, huge pools, ceiling 99.4%, 4 classes x 2 sources. Binding cells: Technology in both sources.
2. INJ: ~1.5% (0.5-4%). Ceiling 98.9% but S2-benign noisy gold and S1/S3 injection cells at 0.88-0.95 at 1.96 bpw already.
3. TOOLSEL as a label (single tool name / none): ~1% (0-3%). Base errors concentrate in S3/none (acc 0.49); 72B real generation 0.943-0.954.
4. T1 / C2 multi-call tool calls: <0.5%. D_scale: best 84-86% at 9.6%, 98% extrapolated at ~19-25% of fp16.
Same caveats for all: 72B is fp8-referenced; 32B 1.57 bpw is 10.03% (0.03 pt over the cap); legal 9.6% points exist only for 7B/14B.

## 6. (5) Cells and locked-exam size needed
Rule 3 asks >=300 items per kind, >=2,000 items, >=500 from a never-seen source. "Every kind" in rule 2 is each answer class, each source, and (our reading, `PLAN.md` strict) each source x class cell.
Cell counts: INJ lock 4 cells (S1, S2 x inj/benign; `v2/tournament/INJ/LOCK.json` note + REPORT.md; the dev table above has an extra S3 cell not in the lock). NEWS plan: 8 cells (AG, HuffPost x 4 classes; BBC S3 dev-only). TOOLSEL: 4 cells (S1, S3 x tool/none). T1: 6 cells.

Base-correct items needed per cell so the Wilson 95% lower bound exceeds 95%, when the observed cell retention equals the true one (`analyze.py`, `tables.json:sizing`, `sizing_exact`):
| true cell retention | n base-correct (observed=true; ~50% chance of certifying) | n base-correct for 80% chance of certifying |
|---|---|---|
| 0.96 | 1,760 | 3,420 |
| 0.97 | 415 | 780 |
| 0.975 | 260 | n/a |
| 0.98 | 175 | 315 |
| 0.985 | 145 | n/a |
| 0.99 | 110 | 145 |
| 0.995 / 1.00 | 75 | 110 |
Note: the rule text is "at least 95% within every kind", a point-estimate test, which is far easier to satisfy than a Wilson lower bound; the lower bound is the conservative reading of a judge who asks for statistical evidence.

Locked items per cell = base-correct needed / cell base accuracy (32B dev base accuracy from `tables.json:celltab`). 80% certification target, cell true retention 0.99 | 0.98:
| task / cell | base acc | items @0.99 | items @0.98 | planned |
|---|---|---|---|---|
| INJ S1 benign | 0.995 | 146 | 317 | 800 |
| INJ S1 injection | 0.593 | 245 | 532 | 800 |
| INJ S2 injection | 0.815 | 178 | 387 | 360 |
| INJ S2 benign | 0.710 | 205 | 444 | 360 |
| NEWS AG Technology | 0.64 | 227 | 493 | 400 |
| NEWS HuffPost Technology | 0.69 | 211 | 457 | 400 |
| NEWS HuffPost Business | 0.84 | 173 | 375 | 400 |
| NEWS other 5 cells | 0.91-0.99 | 146-159 | 318-345 | 400 |
| TOOLSEL S3 none | 0.493 | 294 | 639 | n/a |
Conclusion: at true retention 0.99 every planned cell is big enough (and rule 3's 300/class floor dominates). At 0.98 the INJ S2 cells (360) and NEWS Technology cells (400) are under-sized; at 0.97 nothing we can lock suffices (780 base-correct per cell, i.e. ~1,100-1,600 items in noisy cells). The needed size is not the binding constraint: true cell retention of 0.985+ is. If a cell's true retention is < 0.95 no exam size helps.

## 7. Recommended actions
1. Fix the compressor budget, not the exam. The audit says floor3 is ~1% but r100 flips are 4-7% (32B). The gain must come from reducing noise on margin <10 items (22% of items, 63% of flips): per-token/ per-module bit allocation guided by those items, bits above 1.57 where the answer token's logit gap is small, seed/draw selection on dev (+-2 pt). Task choice alone does not change this (TASK_SEARCH s0 agrees).
2. Curate the SOURCE, not the exam items. Rules: the locked exam is chosen before building and never used to build or choose (`GOAL_README.md` rule 4; `RULES.md`). It is therefore not allowed to filter locked items by base margin or by any compression outcome. It is fine to define the exam from clean sources chosen up front on label quality alone (e.g., items with inter-annotator agreement >= 2/3, NEWS classes merged or pruned to clearly separable ones such as dropping the Tech/Business boundary by redefining the class definitions, INJ sources with expert labels instead of JailbreakHub community labels). Write the filtering rule (annotator agreement, dataset-provided quality flags, dedupe) into the pre-registration and hash it before any compression run; apply it identically to base and candidate. Do NOT filter by "base margin" or "base confident": that is selection by the base's behaviour and a maintainer can call it gaming (strict-rules memo: no gray readings). Expected gain: it only removes the 0.6-1.4% persistent floor and shifts base accuracy up; it does not remove the 75-85% noise flips.
3. Before any more GPU spend, decide the per-cell exam layout: for every planned cell, require expected true retention >= 0.985 on dev AND >= ~150 base-correct items (items = base-correct / base acc). Drop or merge cells that cannot meet it by task design (NEWS: merge Technology+Business or add nothing; INJ: replace S2 community labels or accept a smaller number of kinds only if the rules allow it). Re-run the 72B real-generation INJ/TOOLSEL on the re-defined cells to see whether the ~95% real-generation ceiling moves.

## 8. Caveats
- Single quantiser draw per (model, ratio) config; Wilson and Beta intervals ignore draw noise (~2 pts). The cells flip rates at n~70-140 base-correct are wide (e.g. NEWS Tech 0.906 [0.83, 0.95]).
- Dev tasks are Qwen2.5 dense bases and a teacher-forced proxy (answer tokens argmax). The 72B real-generation check shows real generation is 3-4 pts lower than the proxy for INJ/TOOLSEL.
- "floor3" mixes true ambiguity and noise that happened to hit 3 configs; it is an upper bound on the ambiguity floor. Independent configs share the same Hadamard/GPTQ recipe, so their errors are positively correlated for recipe-level reasons, not only ambiguity.
- INJR S1 uses benign-biased prompt v1 (see tournament REPORT); base errors are mostly missed injections.
- Margins are logit gaps under the bf16 base and are not calibrated probabilities; no probability margins are stored.
- No entry32b, compress*, quality or harness rows could be used: they hold either code only or aggregate scores. If entry32b per-item outputs exist in the Modal volume, rerun `analyze.py` with them as a fourth config set.
