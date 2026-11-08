# L6: other-MoE base pruned + low-bit: Qwen1.5-MoE-A2.7B-Chat (DEV ONLY, locked exam NOT scored)
Task EASY-A SKILL7 (MASSIVE+CLINC150), real generated skill word, dev items (682; v2/task_search/EASY). Spend ~ $2 (two Vast boxes, destroyed, verified).
Package: answer.py (answer command), feedback.py (closed-form diffuse feedback + real/shuffled/empty controls), MANIFEST.json, results/. Compression code: v2/near_miss/a_moe/nm.py (build: `python nm.py build qwen15 L6model '{"xb":2,"xg":64,"ab":4,"ag":128,"eb":4,"eg":64,"crit":"reapmass","K":"max"}'`).

## Base choice (3 non-Qwen3 MoE bases tried, same recipe, best config each, all <= 9.9%)
| base | base acc (dev) | best retention [Wilson lo] | size |
|---|---|---|---|
| Qwen1.5-MoE-A2.7B-Chat 14.3B (60 exp, +shared) | 0.912 | **0.859 [0.829]** (RTN 2b g64, 31/60 experts, reapmass) | 9.87% |
| DeepSeek-V2-Lite-Chat 15.7B | 0.660 | 0.382 [0.339] | 9.77% |
| OLMoE-1B-7B-0125-Instruct 6.9B | 0.522 | 0.775 [0.729] (2b g128, 41/64) | 9.84% |
Best base: Qwen1.5-MoE. gpt-oss-20b / Mixtral / Qwen3.5 MoE not run.

## Compression (selected config; byte accounting in MANIFEST.json)
2-bit RTN g64 on 31 of 60 experts/layer (router rows pruned, uniform K), attn/shared/dense 4-bit g128, embed+lm_head 4-bit g64. Pruning criterion ablation (RTN 2b): mass 0.788, reap(mean gate x mean out-norm) 0.842, reapmass 0.859. GPTQ (gray, Hessian weight update) 0.741 (worse than RTN here). Ternary (K=38) 0.767. 3-bit/4-bit with fewer experts collapse (0.36 / 0.06). norm_topk_prob=True: 0.003. attn 3-bit: 0.54.
Retention overall 0.859 (n=622 base-correct), min cell (n>=20) 0.689 (massive/Music), generated, dev.

## Feedback (dev, 5 draws, 20 annotations each, 92 unseen teach items, 12 layers, lam 0.1, margin 25)
Scenario: shipped prompt lists 6 skills; NEW label 'Lists' taught by 20 annotated mistakes. Edit = sequential closed-form (MEMIT-style, no optimizer) on shared-expert down_proj of the last 12 layers, compensating the sigmoid shared-expert gate; stored as low-rank factors (feedback.npz 3.39 MB, counted); the edited model generates 'Lists' itself (real greedy generation).
| condition | teach (new label emitted) | neighbours kept (469 correct items of other 6 skills) |
|---|---|---|
| empty (no edit) | 0.000 | 0.981 |
| real labels | 0.415 (0.37-0.45) | 0.950 (min draw 0.934; worst class Music 0.927) |
| shuffled labels | 0.000 | 0.863 |
Total size with feedback 9.884%.

## 9-rule table (honest)
| # | rule | status |
|---|---|---|
| 1 size | PASS (9.884% analytic; bit-packed writer not implemented -> gap) |
| 2 faithful | **FAIL: retention 85.9% vs 98%, min cell 69% vs 95%** |
| 3 locked exam | not run (dev only; locked exam deliberately untouched) |
| 4 no new trained parts | GRAY: routing-statistics pruning; feedback stored as low-rank factors = adapter-like |
| 5 feedback teaches | PARTIAL: teach 0.415 vs 0 baseline, neighbours 0.95 (drop 3 pts) |
| 6 controls | PASS on dev (shuffled 0, empty 0) |
| 7 real job | PASS (generated skill word) |
| 8 something to keep | PASS (base 0.912) |
| 9 spirit | gray (new label placed via edit of a few late shared-expert layers) |
**Single weakest rule: Rule 2 (faithfulness 85.9% vs 98%).**
Caveats: dev data used for config choice and reporting (no separate report split); neighbours defined as items the compressed model itself gets right on the 6-label prompt.
