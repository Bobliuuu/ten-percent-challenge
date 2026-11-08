# TASK_SEARCH2: do copy / derive tasks retain better under extreme quantization? (2026-11-08) DEV DATA ONLY

**Verdict: NO-GO. The hypothesis is refuted.** Tasks whose output is copied from or deterministically derived from the input (PII redaction, PII detection to JSON, NER to JSON, log-template parsing, SQL-from-schema, tool-call argument filling) do NOT retain better than the label tasks of TASK_SEARCH.md; most retain far worse, because strict exact match of a multi-token output multiplies per-token flips. Best cell in the whole lane: tool-call argument filling on Qwen2.5-32B at 12.53% of fp16, **97.9% [Wilson lo 96.3]** (worst kind 96%, n=48) -- that is the same ~98% at ~12.5% that the label tasks already reach, and it is above the 10% cap. At the legal point (10.0%) the best is TOOLARG 95.0% [92.9] on 32B and 81.1% on 14B. Nothing clears 98% overall and 95% per kind at <=10%.

All numbers are REAL greedy generation of the whole answer (HF `generate`, bf16 dequantised weights, full lm_head, `<|im_end|>` stop), scored strictly on the whole output; retention = P(candidate output correct | base output correct) on identical items. No locked exam, S2 row or GitHub issue was touched; dev splits of public datasets only, sha256-ordered, deduplicated on input text.

## 1. Setup
- Quantizer: the repo pipeline, unchanged (`task_search/dense_core.py`: random-sign Hadamard + GPTQ + entropy-coded L-level grids, Lagrangian per-module allocation, `target_ratio` allocator, no optimizer, no backward). Added only a `gen` flag (real generation eval, `run_gen`). Calibration = 480 rows of the SAME six tasks (40 per task x source, hash-disjoint from the eval items). Allocation tables (`sens_*`) were reused from the earlier TASK_SEARCH runs of the same model.
- Operating points (file ratio / body bpw): 14B 10.03% / 1.56, 11.29% / 1.78, 12.54% / 1.94; 32B 10.02% / 1.58, 11.28% / 1.79, 12.53% / 1.98 (the lane's 1.6 / 1.8 / 2.0 bpw). The 10.0% rows are within 0.03 pt of the cap (optimistic by that amount, as in TASK_SEARCH.md).
- Items: 6 tasks x 2 sources x 300 = 3600 (`task_search/T2/items_dev.jsonl`, builder `T2/build_t2.py`, licences `T2/LICENCES.json`). Two ids' scanner-tripping synthetic key strings are redacted in the committed jsonl (see `T2/NOTE.txt`); the Vast runs used the committed (redacted) file, the early Modal attempt (cancelled before it started) would have used the unredacted one.
- Scoring: JSON-output tasks (PIIDET, NER, TOOLARG) compare `json.loads(output) == json.loads(gold)` (whitespace-insensitive, content and order strict; a first pass with raw string match gave 1-15% base accuracy purely from `", "` vs `","` spacing and was discarded); all other tasks exact string after strip. SQL is strict (quote style, aliases and semicolons all count), which is why SQL base accuracy is 4-14%.
- Kind of item: PII = rarest PII type present; NER = rarest entity type; LOGTPL = log system; SQL = query shape; TOOLARG = number of arguments. "worst cell" below = worst (source x kind) cell with >=8 base-correct items (stricter than the rule's per-kind pooling).
- Compute: 2 owned Vast boxes (A100-80GB for 14B and the small bases, H100-80GB for 32B), both destroyed (verified); about 8-9 h each, est. $30 total; Modal jobs were queued behind the 10-GPU cap for >1 h with 0 tasks started and were cancelled. Per-item rows: `task_search/results/raw3/ct_<model>_t_<cfg>.json` (outputs + gold + teacher-forced margins). Tables: `task_search/results/table3_*.md`; analysis `task_search/analyze3.py`.

## 2. Datasets (all dev/train splits, licences recorded)
| task | output | S1 source (licence) | S2 source (licence) | exam feasibility (>=2000, >=300/kind, >=500 other source) |
|---|---|---|---|---|
| PIIRED redact PII to [TAG] | full rewritten text | ai4privacy pii-masking-200k (open custom, synthetic) | gretel synthetic_pii_finance_multilingual (Apache-2.0) | S1 17k eligible, S2 3.1k paragraphs (phone-heavy); kinds >=300 feasible for S1, tight for S2 (SSN 16, IBAN 12 of 300 probe) |
| PIIDET detect PII to JSON | JSON list of [type,value] | same | same | same pools |
| NER entities to JSON | JSON list of [text,type] | CoNLL-2003 train (Reuters, research-only) | WNUT-17 train (CC-BY-4.0) | S2 pool 1187 sentences: below 2000; PER/ORG fine, creative-work/product thin |
| LOGTPL log line to event template | template string | loghub 2k: Apache, HDFS, Linux, OpenSSH, Hadoop, Zookeeper, Spark, HealthApp | loghub 2k: Android, BGL, HPC, Mac, OpenStack, Proxifier, Thunderbird, Windows (free for research) | only 2000 lines and 6-120 templates per system: heavy template repetition (narrow distribution, rule 1/9 risk) |
| SQL schema+question to SQL | one-line SQL | sql-create-context (CC-BY-4.0) | gretel synthetic_text_to_sql (Apache-2.0) | pools ample; but base accuracy 4-14% strict |
| TOOLARG function call with copied args | `{"name":..,"arguments":{..}}` JSON | glaive-function-calling-v2, function-name hash half A (Apache-2.0) | same, half B (disjoint function names) | pool 7.9k + 8.8k single-turn calls, kinds 1/2/3+ args >=300 feasible; one dataset, synthetic, narrow function vocabulary |
CORD receipts were rejected before building (800/100/100 rows total: cannot reach 2000 locked items). Not built: invoice-field JSON from OCR text (no licensed >=2000 text corpus found), few-NERD (CC-BY-SA, would only repeat NER).

## 3. Base accuracy (strict whole-output exact match; sources in the order listed in section 2)
| task (S1 / S2) | Qwen2.5-7B | Qwen2.5-14B | Qwen2.5-32B | Qwen3-8B | Qwen3-14B |
|---|---|---|---|---|---|
| PIIRED | .38 / .23 | .64 / .47 | .52 / .41 | .51 / .19 | .55 / .46 |
| PIIDET | .13 / .17 | .74 / .51 | .76 / .58 | .56 / .50 | .72 / .48 |
| NER | .24 / .01 | .39 / .18 | .50 / .18 | .30 / .01 | .42 / .08 |
| LOGTPL | .15 / .22 | .57 / .33 | .56 / .36 | .43 / .15 | .59 / .31 |
| SQL (gretel / create-context) | .01 / .05 | .04 / .05 | .14 / .06 | .18 / .00 | .17 / .00 |
| TOOLARG | .92 / .94 | .93 / .93 | .93 / .93 | .90 / .94 | .91 / .96 |
(PIIRED/PIIDET/NER/LOGTPL S1/S2 order = ai4privacy/gretel, ai4privacy/gretel, conll/wnut, loghub_a/loghub_b; PIIDET 7B is format-sensitive. Rule 8 is satisfied only by TOOLARG (0.93, near its ceiling), PIIRED, PIIDET, LOGTPL; SQL and NER-WNUT are too low to teach or to lock 300/kind of base-correct items.)

## 4. Retention vs base-correct items, real generation (overall, Wilson 95% lower bound in brackets, n = base-correct items; worst source x kind cell)
**Qwen2.5-32B-Instruct** (n per task = 600 items):
| task | n | base | 10.02% / 1.58 bpw | 11.28% / 1.79 bpw | 12.53% / 1.98 bpw |
|---|---|---|---|---|---|
| TOOLARG | 559 | .932 | .950 [.929], worst 3+args .83 | .946 [.924], worst .88 | **.979 [.963]**, worst 3+args .96 (48) |
| PIIDET | 403 | .672 | .861 [.824], worst CARD .67 (9) | .931 [.901], worst CARD .78 (9) | .878 [.843], worst PASSWORD .76 (34) |
| PIIRED | 281 | .468 | .701 [.645], IP .40 (25) | .804 [.754], CARD .50 (12) | .804 [.754], CARD .50 (12) |
| LOGTPL | 275 | .458 | .793 [.741], Zookeeper .25 (24) | .807 [.757], Zookeeper .00 (24) | .862 [.816], OpenSSH .38 (24) |
| NER | 204 | .340 | .299 [.240], ORG .12 (50) | .436 [.370], ORG .22 | .515 [.446], ORG .34 |
| SQL | 59 | .098 | .237 | .288 | .407 |
| all six | 1781 | .495 | .768 [.748] | .819 [.800] | .838 [.820] |

**Qwen2.5-14B-Instruct**:
| task | n | base | 10.03% / 1.56 bpw | 11.29% / 1.78 bpw | 12.54% / 1.94 bpw |
|---|---|---|---|---|---|
| TOOLARG | 556 | .927 | .811 [.777], 3+args .57 (49) | .874 [.844], .82 | .917 [.891], 2args .87 (131) |
| PIIDET | 374 | .623 | .690 [.641] | .840 [.799] | .906 [.873], EMAIL .76 (25) |
| PIIRED | 333 | .555 | .273 [.228], EMAIL .06 (17) | .679 [.627] | .790 [.743], SSN .50 (14) |
| LOGTPL | 272 | .453 | .331 [.278], Apache .00 (31) | .783 [.730] | .761 [.707] |
| NER | 171 | .285 | .082 [.049] | .298 [.235] | .339 [.272] |
| all six | 1734 | .482 | .529 [.506] | .752 [.731] | .805 [.786] |
Quantizer-draw noise is large (14B r100 -> r1125 jumps 53 -> 75% overall; 32B PIIDET is non-monotone .93 -> .88; LOGTPL 14B .78 -> .76), consistent with the +-2 pt draw noise in TASK_SEARCH.md and bigger for long outputs.

## 5. Comparison with the label tasks (TASK_SEARCH.md, same quantizer, TF label word)
| model, ratio | label tasks (NEWS / INJ-real) | best copy task (TOOLARG) | copy tasks pooled |
|---|---|---|---|
| 32B, 10.0% | .961 / .957 | .950 | .768 |
| 32B, 11.3% | .962 / .958 | .946 | .819 |
| 32B, 12.5% | .979 / .942 | .979 | .838 |
| 14B, 10.0% | .850 (6-menu) / .912 | .811 | .529 |
| 14B, 12.5% | .947 (6-menu) / .968 | .917 | .805 |
So the best copy task equals the best label task at best, and every other copy task is 10-60 points worse. Why: the margin picture is the same as before (the noise in TOOLARG items is concentrated in the ~10% of base-correct items whose minimum gold-token margin is <6: retention 53% / 77% for margin <2 / [2,6) vs 96% / 99% for [6,10) / >=10 at 32B 10.0%), but a 20-100-token output has 20-100 chances for a flip, copy spans are not "easier" for a perturbed model (a flipped digit or quote breaks the exact match), and the base itself is borderline on the formatting/ontology choices (what counts as a PII span, SQL quoting, template granularity), so most base-correct items are low-margin by construction (PIIRED median min-margin 5-6 vs 10-12 for label tasks).

## 6. Ranking and recommendation
1. **TOOLARG (tool-call argument filling)**: only copy task that is both commercial (agentic tool use is named in maintainer issue #1) and robust. 32B: 95.0 / 94.6 / 97.9% at 10.0 / 11.3 / 12.5% -> still NO-GO at <=10% (Wilson lo 92.9). It is the same family as TOOLSEL / T1 in SOLUTIONS.md (C03, C05, C06, C11), now with real argument generation; the nearest-to-pass cell is 32B at 12.5% and it fails the cap. Single synthetic source (glaive), narrow function vocabulary: R3 (500 other-source items) would need a second real source.
2. PIIDET (PII detection to JSON): 86-93% at 10-11% on 32B, base only .67; per-type cells 0.67-0.78.
3. LOGTPL: 79-86% on 32B; narrow (2k lines per system) -> R1/R9 risk.
4. PIIRED (redaction, the maintainer-named PII example): 70-80% on 32B, CARD/IP/IBAN cells 0.4-0.5.
5. NER (30-52%) and SQL (24-41%, base <=14%): dead.
No task here beats the earlier top-2 (NEWS, INJ) at <=10%; none reaches 98% overall / 95% per kind at <=10% on 14B or 32B. The only route consistent with these data is the same as before: more bits per weight or a bigger base (72B+), or a native-low-bit parent; the task choice does not remove the compression problem, and "copy tasks are robust" should not be assumed.

## 7. Caveats
- One quantizer draw per (model, ratio); no repeat seeds; Wilson bounds ignore draw noise and item dependence. The 10.0% points are 0.02-0.03 pt over the cap.
- 300 items per source, so per-cell n is small (many cells n<30); this is a ranking probe, not an exam. Base-correct counts are low for NER/SQL (n=59-204), which makes their retention numbers noisy.
- Prompts are zero-shot, one per task; JSON scoring is whitespace-tolerant (a deliberate relaxation, stated above); a stricter string match would lower every JSON-task number.
- Calibration used the same six tasks (hash-disjoint items, same sources); retention would be somewhat lower with no task-matched calibration.
- Qwen3-8B/14B and Qwen2.5-7B were measured for base accuracy only (not quantized).
- The 32B r100 first attempt OOM-ed at eval (leftover GPU state); it was rerun with smaller eval batches (gen batch 12, teacher-forced batch 2) and `empty_cache`; results above are from the rerun. Teacher-forced margins in the raw rows are on the formatted gold string and are only meaningful for non-JSON tasks.
