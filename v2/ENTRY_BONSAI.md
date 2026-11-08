# ENTRY_BONSAI: Bonsai-8B 1-bit lossless repack, built out as one complete candidate (2026-11-08)

Order from the user: ASSUME the maintainer accepts a lossless repack of the natively 1-bit Bonsai-8B (prism-ml/Bonsai-8B-unpacked) as "the model you started from"; nothing was posted to GitHub; no locked or sealed exam was read, run or scored (only DEV data; the sealed NEWS manifest `v2/exp/LOCK_NEWS_MANIFEST.json` was not touched). Code and raw results: `v2/bonsai/`. Compute: one owned A10G 24GB box (about 8 h, about $10), Modal queued all night and never started (nothing ran there, nothing running now). All numbers below are DEV, real greedy generation unless stated.

## 1. What was built (artifact + two commands)
Artifact dir (flat, 22 files) = `bits.bin` 1,023,529,984 + `scales.bin` 127,941,248 + `small.pt` 652,262 + tokenizer/config/vocab/merges + `chat_template.jinja` 4,063 + LICENSE/NOTICE + `bonsai_loader.py`, `common.py`, `answer.py`, `feedback.py`, `news_task.json`, `feedback_calib.jsonl` (77,803 B, 200 UNLABELED NEWS prompts, the calibration file that feedback.py needs) + `tensor_manifest.json`.
- **Size, real bytes of the written directory: 1,168,146,463 B = 7.1328% of 16,377,096,192 (= 2 x 8,188,548,096 params); cap 1,637,709,619; slack 469.6 MB.** The calibration file, scripts and manifest cost 0.12 MB in total. After feedback: +9,548 B (one fp16 row, section 3) = 1,168,156,011 B = 7.1329%.
- **Bug found and fixed in the earlier gate-1 pack:** the shipped tree had NO chat template (`chat_template.jinja` lives outside tokenizer_config.json); the tokenizer could not render a prompt. Added (+4,063 B) with LICENSE (10,174) and NOTICE (412), required by Apache-2.0. Loader now keeps the state dict on CPU (a 24GB GPU holds it; before it needed 32GB); same weights, same copy.
- **Answer command:** `python answer.py --artifact DIR --task news_task.json [--in items.jsonl|-] [text ...]` prints one generated label word per line (user mode); `--in view.json --out out.json` is the harness contract (rows `{id, query, tools}`, query = full prompt). Plain HF generate, greedy, 14 new tokens, no constrained decoding.
- **Feedback command:** `python feedback.py --artifact DIR --annotations ann.jsonl` (or `--task news_task.json --ann ann.jsonl` with `{text,label}` rows), at most 25 rows, zero rows removes `patch.pt` and leaves the tree byte-identical (tested: tree sha unchanged, harness `empty_byte_identical: True`). Method = fb_legal `mnH` (closed form, margin 4) writing the touched existing lm_head rows as fp16 into `patch.pt`; `bonsai_loader.load` overlays them. Disclosed: the weights are not frozen after feedback.
- Identity (gate 1, earlier run, original loader): 2000/2000 identical greedy sequences vs the reference HF fp16 model, logits bit-equal. Re-checked here with the final code path through harness v3: on 1200 dev exam items the shipped answer command agrees with an independent base process on 1198 of 1200 items (99.83%; the 2 flips are batch-composition numerics, bs 8 base vs bs 16 candidate), accuracies 0.7808 vs 0.7792.

## 2. Task choice (rule 8, real generation, DEV, `v2/bonsai/results/task_probe_summary.json`, 8166 items)
| task | n | Bonsai acc [Wilson 95] | chance | cells (src/class: acc) |
|---|---|---|---|---|
| **NEWS 4-label router** (World/Sports/Business/Technology) | 1600 | **0.772 [0.751, 0.792]** | 0.25 | 12 cells .58-.98: AG Tech .58, AG World .93, BBC World .71, BBC Tech .86, HuffPost Bus .67 / Sports .66 / Tech .70 / World .98 |
| TOX 5-class (safe/toxic/hate/sexual/violence) | 1600 | 0.537 [0.513, 0.562] | 0.20 | S1 sexual .06 (collapse), S3 safe .28 |
| TOX 2-class | 600 | 0.687 | 0.5 | toxic_chat Toxic .27 |
| INJR (tournament real dev data) | 1966 | 0.749 [0.729, 0.767] | 0.5 | PromptShield injection .41, benign 1.00; 104 off-menu outputs |
| INJ easy (safe-guard + jackhhao) | 600 | 0.920 | 0.5 | all .89-.96 (easy sources, do not cite as INJ) |
| FIN (Bullish/Bearish/Neutral) | 600 | 0.853 [0.823, 0.879] | 0.33 | tw Neutral .63 .. phrasebank Bearish .95 |
| SENT | 600 | 0.948 | 0.5 | .94-.97 (near ceiling, toy) |
| TOOLSEL | 600 | 0.863 | 0.5 | 75 off-menu outputs; S3 none .57 |
**Pick: NEWS.** Well above chance (3x) and well below ceiling overall, every label a real generated word (5 off-menu outputs of 1600), data exist for the exam (sealed manifest by another lane: 3200 items, 8 cells x 400, 1600 from HuffPost buckets [20,100)), an Entertainment extension is natural (feedback scenario), and the sources are broad (AG 2004-05, BBC, HuffPost 2012-22). Runner-up FIN (best scorecard shape, every cell in (chance+.15, .97) but finance-only = narrow-distribution risk and no clean unseen source). Rejected: SENT (ceiling, toy), INJ-easy (ceiling), INJR/TOX (cells at .00-.06 or 1.00: rule 8 fails per cell), TOOLSEL (S3 none .57, off-menu tool names). Caveat: HuffPost World is .98 on dev (> the scorecard's per-cell .97 ceiling) but that cell is in the S2 block that the 1200-item harness exam does not contain; the harness gate `base_above_chance_below_ceiling_every_cell` passed on the S1+S3 cells; on the sealed S2 block this cell may fail the per-cell ceiling gate (the rule text says "well above chance on the exam and below its ceiling", overall .77 satisfies that).

## 3. Feedback on a 1-bit lm_head (rules 4-6)
Earlier conclusion in FEEDBACK_LEGAL (sign-bit flips on the 1-bit row cannot teach: teach 0.00) STANDS and is bypassed, not repaired: **the touched row is simply stored as an fp16 row** (the "fp16-patch" variant that was queued and unrun). It worked first time on both scenarios. Method `mnH` margin m in {4,8,16}, 10 draws, n in {5,12,25}, controls real / shuffled (same label on unrelated inputs) / wrong / mixed(half wrong) / empty; teacher-forced on cached states, checked against REAL generation on 100 teach + 100 neighbour items: agreement 1.00 (`results/fb_*_fp16patch.json`).

NEWS + new label `Entertainment` (prompt lists 4 labels; teach set 80 unseen; neighbours 694 base-correct old items AG/BBC/HuffPost; Bonsai calls Entertainment "World" 71%, "Technology" 16% = natural mistakes). Rows touched: **1** (first token "Ent"), +8,192 B fp16.
| m | n | teach real | shuffled | real-shuffled [95% CI over draws] | wrong | mixed | OOD | neighbours kept | worst draw | min cell |
|---|---|---|---|---|---|---|---|---|---|---|
| 4 | 5 | .598 | .167 | +.43 [.35,.51] | 0 | .356 | .535 | .9990 | .996 | .991 |
| 4 | 12 | .690 | .310 | +.38 [.31,.45] | 0 | .428 | .635 | .9965 | .987 | .949 |
| 4 | 25 | .771 | .460 | +.31 [.21,.40] | 0 | .396 | .770 | .9905 | .977 | .871 |
| 8 | 5 | .701 | .230 | +.47 [.37,.56] | 0 | .466 | .660 | .9945 | .981 | .931 |
| 8 | 12 | .840 | .395 | +.45 [.35,.54] | 0 | .501 | .790 | .9866 | .958 | .840 |
| 8 | 25 | .890 | .536 | +.35 [.26,.44] | 0 | .491 | .885 | .9660 | .925 | .714 |
| 16 | 25 | .946 | .644 | +.30 [.22,.39] | 0 | .561 | .935 | .8928 | .782 | .426 |
Base teach accuracy 0.00 (never says Entertainment). Read: it teaches (CI of gain > 0 everywhere) and the wrong-label arm is exactly 0, but the shuffled control is NOT near 0 on NEWS (.17-.64): any "Entertainment" label raises that token on news-like inputs. The gap real-shuffled is +.30 to +.47 (rule 6 needs "clearly worse", CI excludes 0). Neighbours stay >=98% only for m=4 (all n) and m=8 (n<=12).

TOX 5-class + new label `selfharm` (teach 56 unseen, neighbours 520 base-correct): m=8 real .718/.786/.852 (n=5/12/25) vs shuffled 0.000/.107/.177, wrong 0, mixed .50/.38/.41; neighbours kept .9985/.9973/.9969 (worst draw .994, min cell .976-.984); 1 row. m=4: .595/.680/.734, shuffled 0/.05/.125, neighbours .9994/.9988/.9979. m=16 n=25: .920 vs .266, .9875. Cleaner than NEWS (new class far from the old ones) but the TOX 5-class base (.537, S1 sexual .06) fails rule 8, so TOX is the feedback demonstration only, not the task.

**Other placements (asked for if rows could not be taught): not needed.** The 1-bit-format edits (flips, scale-only) stay failed; the fp16 row replacement is the working placement (cost 8,192 B per touched row + about 1.3 KB container). Final-norm-scale edits were not tested (nothing to rescue).
**Rule-4 reading (honest):** no head, adapter or side memory is added and the output matrix keeps its shape; but one existing row is overwritten by a value fitted in closed form to the annotations, stored outside the 1-bit weight format (fp16 patch file), and the calibration prompts (77.8 KB, counted) are shipped. This is GRAY: "no new computation fitted to data" can be read to forbid it. Same verdict as FEEDBACK_LEGAL for Qwen3-8B; nothing here is stronger.

## 4. Harness v3 final_proof on DEV (label-mode copy `v2/bonsai/h3/`)
harness/v3 is tool-call shaped, so a copy was adapted (label scorer `labelscorer.py`, plain chat prompt, fp16 base, Bonsai pack file names on the allow-list, deterministic label derangement or an unrelated-input shuffle when all annotations carry the same new label). Exam = 1200 dev items (AG 600 as S1, BBC 600 as S3); teach 60 unseen Entertainment; neighbours 400 HuffPost dev (S2 probe, the hardest region for Entertainment); 25 annotations = the plan's `fb_annot_25` (all Entertainment); isolation = `unshare -mn` netns (no network, label dir masked), 3 base runs (bs 8/4/16). Raw: `v2/bonsai/results/harness_news_v2/`. Run v1 (annotations = 12 Entertainment + 13 old-label corrections, m=8) is kept in `harness_news/`: the old-label rows rewrote World/Sports/Technology rows and BBC World fell to 46.7% retained (exam-after 92.2%), so old-label corrections do real damage with this mechanism and must be handled before relying on it.
Run v2 (final): checks all ok (size, no-new-parts via manifest, no leak, no network, artifact unchanged by answer).
| item | value |
|---|---|
| base acc / candidate acc / agree (1200) | .7808 / .7792 / .9983; retention of base-correct certified >=98% overall and per kind/source/cell (scorecard: observed 1.0 at n=935) |
| teach (60 unseen) real / shuffled / mixed / wrong-label / empty | **.750 / .233 / .367 / .000 / .000** (base 0) |
| bytes added by feedback | real 9,548; empty 0 and tree sha identical |
| exam after feedback, retention of base-correct | **99.25%**, worst cell S3 World .972 |
| neighbours (302 base-correct HuffPost) kept | **96.4%  -> FAILS the 98% gate** (certify needs 301/302) |
Scorecard gates failed: only the dev-size ones (exam 1200 < 2000, cells < 330, OOD block), `teach_clusters_given` (one cluster) and **neighbors_kept >= 98%**. Everything else passed: retention overall/kind/source/cell, rule 8 per cell, real output, no new parts, controls clearly worse, teach gain, exam-after retention, size after feedback. (Neighbour choice is deliberately harsh: the closest 400 HuffPost items to Entertainment; the broader dev neighbour set in section 3 keeps 99.05% at m=4, n=25.)

## 5. Per-rule status (assuming the base-identity reading is granted)
| rule | status | evidence / risk |
|---|---|---|
| R1 size | PASS 7.133% real (7.133% after feedback) | all files counted incl. calibration, template, license; GRAY only on "fp16 size of the model you started from" = Bonsai's own fp16 (identical bytes to Qwen3-8B's) |
| R2 same = faithful | PASS vs Bonsai (identity; dev 99.83% pass-through, 100% retention of base-correct); **FAIL vs parent Qwen3-8B (88.9%)** | needs the maintainer's reading; not measurable by us |
| R3 exam | NOT BUILT HERE (by order). Sealed NEWS manifest exists (X15): 3200 items, 8 cells x 400, HuffPost 1600 never used in locked buckets | prompt there is the "X01 open tail", not mine (re-calibrate); same-source caveat for HuffPost buckets [0,20) |
| R4 no new trained parts | GRAY | fitted fp16 row in `patch.pt`; ship calibration prompts |
| R5 feedback teaches | PARTIAL: teach +.75, exam-after 99.25%, **neighbours 96.4% on the hardest 400 (99.05% on the broader dev set)** | at m=4; m=8 n=25 breaks neighbours (96.6%); old-label annotations break BBC World |
| R6 shuffle/empty worse | PASS on dev: real .75 vs shuffled .23, mixed .37, wrong 0, empty 0 | shuffled is not ~0 on NEWS (label-only effect); CI of real-shuffled > 0 |
| R7 real job | PASS | the label word generated; strict full-string scorer; no constrained decoding (grep clean) |
| R8 something to keep | PASS overall (.772 vs .25); per-cell: HuffPost World .98 may exceed the .97 ceiling gate on the sealed S2 block | S1/S3 cells passed in harness |
| R9 spirit | GRAY-to-risky | the entrant does no compression, 1.17 GB is the authors' release |

## 6. Remaining work and exact steps + hours to lock and score the exam (do NOT run before the freeze; one artifact per sealed exam)
Open problems first (about 6 h GPU/analysis, $8): (a) neighbour retention to >=98% on the harsh set (try m=2-3, ridge 0.05->0.2, add the answer-position states of the ~700 neighbour-like calibration prompts to M, or Entertainment-aware calibration), re-run harness v2 (1 h each); (b) decide the old-label annotation policy (either document "only new-label annotations" or fix the row damage: protect rows with a larger ridge on M); (c) re-calibrate and re-run everything with the sealed X01 open-tail prompt (`seal` prompt sha ed4c93b5...) because the sealed exam uses it; (d) decide teach/neighbour sets for the real feedback exam: >=25 unseen locked Entertainment items (HuffPost Entertainment bucket <20, `ent_sorted[225:]` is unused) and the neighbour lock.
Then, exact steps (all hours on one A10G 24GB or 1 A100; HF fp16 path; speeds measured: 15 items/s answer generation, about 3 min per 1000 items per pass, 55 min for a 1200-item run including 5 feedback arms):
1. Freeze gate (0.5 h): build the final artifact dir with `make_bonsai_artifact.py`, record tree sha256, size check (<=10%), stop editing code.
2. Materialise the 3200 sealed items from public data with `v2/exp/seal_news_lock.py`; accept only if the sealed hashes match (0.5 h CPU).
3. Lock teach (>=25 unseen), neighbours and the 25 annotations; write FEEDBACK_LOCK (0.5 h).
4. Stage 1: base pass x3 (bs 8/4/16) + candidate pass on 3200 items in netns (about 1.2 h).
5. Feedback protocol: 5 arms (real/shuffled/mixed/wrong/empty) each ~7 min + teach/neighbour eval (about 1 h), exam after feedback on 3200 (0.4 h).
6. Scorecard + PROOF_SUMMARY, per-item file, HuffPost-only retention as the rule-3/maintainer-check proxy (0.3 h).
Total about 4 h of GPU after the freeze, about 6.5 h wall including steps 1-3 and 6; about $5-6. Add 6 h/$8 for the open problems above, so about 12 h wall to a scored exam. The sealed protocol allows ONE scored artifact per sealed exam: scoring Bonsai there consumes the exam for every other candidate, so either decide the winner first or seal a second NEWS exam (the dev buckets AG [70,80) spare would give 2 x 400 only for S1; HuffPost is used up).
Maintainer check (his own data + own feedback): new labels of any word work mechanically (two tasks, two new labels); risk is neighbours on his data and the 25-annotation quality; Bonsai on foreign distributions is unmeasured (NEWS S3 BBC .71-.86 is the only OOD hint).

## 7. Limits of what was measured
Dev only; harness exam 1200 items (SE 1.2 pts); one seed of the 25 annotations in the harness run (fb_legal has 10 draws); the noise floor is 3 base runs (bs 8/4/16), but candidate vs base differ by 2 of 1200 items (batch numerics); hard-wiring to HF generate (the sealed protocol wants vLLM with identical flags for base and candidate); feedback numbers for 10 draws are teacher-forced and cross-checked on 100+100 real generations (agreement 1.00); `tensor_manifest.json` is declared (copied from the base headers), the real equality evidence is gate 1; the Modal runs never started (queue), everything ran on the owned A10G, which was shut down with `shutdown -h` and `StopInstances` (the AWS tool then timed out and the `lab` session expired, so the stopped state could not be verified: please confirm `i-0e8f20c40de73df85` is stopped).
