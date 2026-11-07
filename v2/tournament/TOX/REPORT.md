# TOX (content moderation) pipeline report, 2026-11-06 (not committed)

Recommendation: **content moderation (TOX) over ticket routing**, but TOX is NOT a clean candidate yet: three open defects below (S1/sexual cell unusable, feedback fails, base accuracy only 0.55-0.65). Routing is weaker (see section 6).

## 1. Data (data/, build_data.py, LOCK.json, SHA256SUMS; locked.jsonl chmod 444; volume `tenpct-tox-exam:/data`)
Task: input = one comment/message; output = the model GENERATES one word from {safe, toxic, hate, sexual, violence} (scorer.PROMPT, chat template, greedy). self-harm excluded: only 265 clean single-label Aegis prompts (<330).
| role | source (licence, period) | files |
|---|---|---|
| S1 build/dev/calib/pool + locked | Jigsaw civil_comments (CC0, news comments 2015-17) | s1_dev 1000, s1_calib 750, s1_pool ~25k, locked S1 2260 |
| S3 OOD dev proxy | HateXplain (CC-BY-4.0, Gab/Twitter 2019-20; only safe/toxic/hate, unanimous 3/3) | s3_dev 600, s3_pool |
| S2 locked OOD | NVIDIA Aegis 2.0 user prompts (CC-BY-4.0, LLM red-team prompts 2024) | locked S2 2260; s2_probe 437 (group-disjoint slice of the same corpus, ONLY for the rule-8 base health check) |
Locked = 4520 items, 10 cells (S1 x {safe,toxic,hate,sexual,violence}, S2 x same), every cell 450 (safe 460), min 450 >= 330, S2 2260 >= 500. Safe cells are exactly 50% "hard" (contain profanity/slur, violence, sexual, identity or insult lexicon: news about violence, quotes, in-context terms). Raw revisions + sha256 pinned in LOCK.json (civil f2970eb3, aegis d86bb8be, hatexplain parquet b0e5676a).
- Splitting by GROUP: group = near-dup component U TF-IDF/SVD/KMeans topic cluster (k=2000/250/200). civil_comments has no thread/author id in the HF release: topic clusters are a proxy, NOT a true thread split (stated limitation). 9 tests in test_data.py pass: hashes, 444, cells, source roles, text/id unique across files, groups disjoint across files, safe-hard 40-60%, and an independent exact all-pairs Jaccard>=0.6 audit across all files = 0 pairs.
- Dedup: exact (1261), near-dup J>=0.6 exact all-pairs (one representative per component; 1773 dropped), cross-source (56). Touched scan: 599 locked candidates removed because their text occurs in another file under ~/ten-percent (13+ files).
- Labels: crowd-rater thresholds (civil: toxic tox>=.7 & (insult|obscene>=.5); hate/sexual/violence subtype>=.6 & tox>=.6; safe tox==0; multi-class and ambiguous dropped). Thresholds were RAISED from .5 after a dev-only look (S1/sexual base acc 0.08 at .5). Aegis: exactly one taxonomy class; 'Needs Caution' and out-of-taxonomy-only (criminal planning, PII...) dropped. This is a design choice that removes real moderation cases; say so.
- Not trivially keyword-solvable (non-locked data only, results/baselines_keyword_lr.json): lexicon rule S1-dev 0.43, S3 0.36, S2probe 0.41; TF-IDF LR trained on S1 pool: S1-dev 0.66, S3-dev 0.44, S2probe 0.43 (chance 0.2/0.33). LR is strong on S1 safe (0.98) so within-S1 is partly lexical.
- Locked rows were never printed; only counts/aggregates computed. I viewed ~12 truncated S1 DEV (old thresholds) rows to diagnose the sexual cell. Superseded first build (.5 thresholds) was deleted.
- NOT done: no git commit (coordinator commits LOCK.json + SHA256SUMS alone). data/ files other than locked are plain-writable.

## 2. Scorer (scorer.py, test_scorer.py: 75 pass)
Strict: whole output (after one empty think block) must be exactly one label, case-insensitive, one optional trailing '.'. Empty, EOS, two labels, hedging, quotes, markdown, look-alike unicode, zero-width, prefix/sentence containing a label = wrong.

## 3. Base probes, DEV ONLY (results/*.json per item with first-token logprobs; results/summary.md)
Qwen3-8B / 14B / 32B, vLLM greedy, enable_thinking False, parse failure ~0. Qwen2.5-72B and Llama-3.3-70B NOT run (no time; accuracies are already mid-range).
| cell (acc) | 8B | 14B | 32B |
|---|---|---|---|
| S1 safe hard / plain | .82/.84 | .87/.86 | .84/.85 |
| S1 toxic | .715 | .615 | .72 |
| S1 hate | .38 | .76 | .665 |
| S1 sexual | **.02 LOW** | **.05 LOW** | **.165 LOW** |
| S1 violence | .315 | .57 | .65 |
| S3 safe hard / plain | .39/.35 | .37/.34 | .48/.32 |
| S3 toxic / hate | .81/.48 | .45/.875 | .575/.69 |
| S2probe safe h/p | .78/.94 | .66/.88 | .66/.90 |
| S2probe toxic / hate / sexual(n=37) / violence | .56/.42/.43/.67 | .31/.58/.57/.85 | .45/.57/.65/.83 |
Macro acc: S1 .45/.57/.61, S3 .55/.56/.56, S2probe .59/.62/.66 (chance .20/.33). S1->S3 gap on shared classes +0.09/+0.19/+0.19 (S1 .64-.75 vs S3 .55). Ceiling: none near 1 (max cell .94 safe-plain S2 8B). Rule 8 holds for most cells but fails at: **S1/sexual** (crowd "sexual_explicit" = sexual insults; model says toxic ~55%, safe ~25%), several S3 safe cells (model flags normal posts mentioning identities as hate), S2 toxic 8B/14B. A cell with 0.05-0.17 base accuracy leaves ~20-75 base-correct of 450 locked items: per-cell retention cannot be certified. Remedies: drop the S1/sexual cell (sexual then lives only in S2), or relabel; not done (needs rebuild + new probes). The model (32B) is far from base-accuracy-ceiling, so rule 3 "match base" is on a noisy-label task: part of "accuracy" is label-noise agreement.

## 4. Compression-agent inputs
tox_eval.py: `t_items` writes CT_ITEMS records (id, task=TOX, role, src, kind, user, answer) for research/C_tasks/dense_c.py run_ct (teacher-forced margins); `generate_eval(model,tok,rows)` works on any HF-style model object (fake-quant dense.py or dense_stream QLinear); `retention(base_rows,cand_rows)` = item retention conditioned on base-correct per source x kind with Wilson CI and the 98%/95% gates. CLI: gen/ret/titems. Untested end-to-end (compression not run by instruction); smoke-test on one model before relying on it. Prior pilot (C_tasks table): TOX 82.8% retention at 9.6% on 14B, i.e. still far from 98%.

## 5. Feedback (rules 5/6) prototype: first-token lm_head edit, fb_tox.py (fb5 lmhead_cal logic), Qwen3-1.7B / 4B / 8B, 8 draws x 25 annotations, group-cluster bootstrap CIs, results/fb/
Natural mistake = most frequent dev confusion with pool support (1.7B toxic->safe, 4B toxic->hate, 8B hate->toxic). Annotations from pool groups hash%10 in {0,1}, teach (S1 92-150 + S3 150, base wrong in the same way) groups 2-5, neighbours (base-correct, 330-370 S1 + 150 S3 = 480-520) groups 6-9; group overlap 0; spec ids sha256'd and written before any edit (SPEC_SHA in logs/results). Controls: empty (asserted byte-identical), mixed (half true/half base-wrong labels), random-wrong. Evaluation by first-token argmax (agrees with full generation 98.7-100% on base; checked against real generation after edit on 60 teach items: 8B 13% vs 15% first-token).
| model | teach S1 real (CI) | teach S3 real | neighbours S1 kept | S3 nb kept | mixed teach S1 | randwrong teach S1 |
|---|---|---|---|---|---|---|
| 1.7B | .10 [.07,.14] | .19 | .82 | .84 | .16 | .00 |
| 4B | .00 | .00 | .58 | .87 | .04 | .16 |
| 8B | .30 [.25,.33] | .24 | .87 | .90 | .45 | .06 |
Real minus mixed is NEGATIVE (-.06/-.04/-.15) at every size: the "gain" is a bias shift, not learning from labels; neighbour damage 10-42 pts (e.g. 4B toxic neighbours kept 29%); alpha hits the grid max (2.0) every time; the label first tokens are generic subwords ('to', 'h', 'viol') so the edit rewrites generic rows. **Verdict: not a clean pass; fails rule 5 (no neighbour retention, controls beat real) and, even if it worked, needs fitted statistics (LDA direction from annotation means + task-unlabeled covariance), i.e. a rule-4 risk.** Other mechanisms (down_proj MEMIT, Hessian-weighted requantisation) not tried here.

## 6. Routing vs moderation (details: routing/ROUTING_REPORT.md by the routing sub-agent)
Routing: MASSIVE/CLINC/Bitext are assistant utterances or synthetic; the only real-ticket set (one Brazilian IT helpdesk) has <=260 unique items per queue; 18/26 cells have >=330 items; Bitext base 0.92-0.96 with TF-IDF 1.00 (ceiling, trivial); label sets barely overlap across sources, so the S1->S3 gap is meaningless; menu-in-prompt is closer to multiple choice (rule 7). A 330/cell locked block of real tickets is infeasible from public data. Moderation: real public comment/prompt data, 3 sources, 10 locked cells of 450, base 0.45-0.66 (not ceiling), label IS the generated word, commercial (trust & safety). Weaknesses of moderation: labels noisy and taxonomies differ between sources, the S1 sexual cell, mid-range accuracy makes 98% retention on base-correct items hard (pilot 82.8% at 9.6%), and a reviewer can call it a narrow classifier (5 labels), though a 5-label verdict over three platforms is broader than Banking77.

## 7. Honest caveats
Not tool-calling-grade; locked set not yet committed or hashed publicly; S2probe is the same corpus as locked S2 (rule-3 purists may object; it is group-disjoint and never used for tuning); topic clusters are not true threads; touched-scan only covers local files; the first build and a mid-way builder fix (exact near-dup join, found by the independent test: 17 short-text pairs slipped past the candidate filter) were rerun, so probes are on the final dev data only. An overly broad `pkill -f build_data.py` by me during the build killed any other agent's concurrently running T1 build_data.py: tell the T1 owner to rerun. Runs launched (apps tenpct-tox-probe, tenpct-tox-fb) are logged in ~/ten-percent/RUNS.md; local logs in logs/.
