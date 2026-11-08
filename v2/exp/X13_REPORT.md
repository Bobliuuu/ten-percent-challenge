# X13 report: task-pruned MoE probe (Qwen3-30B-A3B-Instruct-2507), verdict KILL

Pre-registered rules (EXPERIMENT_QUEUE X13, revised: real 9.90% byte target) applied verbatim. Spend about $2.5 (one Vast A100-80, about 2 h; id logged in INSTANCES.md, destroyed and verified).

## Pipeline (v2/exp/x13/)
Pure-torch Qwen3-MoE runtime (checked against HF logits, max diff 1e-6), mass-ranked expert prune (global rank, min 16/layer, variable per-layer), calibration = 85% subsample (seed) of CAL(1000 of X01 cal.jsonl) + 150 INJ-calib + 150 C2-calib + 400 wikitext lines, attention 6-bit / embed 5-bit / head 4-bit RTN g128, kept-only router rows. REAL pack written to disk (bit-packed codes, fp16 scales, packed zeros, tokenizer, config, qmoe.py); evaluation DECODES the pack and runs REAL greedy generation (max 8 new tokens, full vocab, open-tail NEWS prompt from X01 SPLITS). Per-item outputs saved: `v2/exp/results/X13/<arm>_s<seed>_SEL.jsonl`.
Arms: M0 RTN3 K=2669; M1 GPTQ3 (batched, sequential, Hessians from routed calib tokens, damp 0.01) same K; M2 GPTQ, top-50% by mass 3-bit and rest 2-bit, K=3.2k experts; M3 no pruning (all 6144 experts) with 2-bit/1-bit mix RTN to match bytes. All packs 9.899% real bytes (size_check.py and exp/bytes.py agree: 6,044,859,434 B = 9.8992% of 61,064,245,248; 58 files; includes tokenizer, config, decoder code).

## SEL retention of base-correct (n=3073 base-correct of 3839; base acc 0.800)
| arm | s11 | s12 | s13 | mean | min |
|---|---|---|---|---|---|
| M0 RTN3 | 0.9593 | 0.9610 | 0.9636 | 0.9613 | 0.9593 |
| M1 GPTQ3 | 0.9619 | 0.9684 | 0.9632 | 0.9645 | 0.9619 |
| M2 mix 2.5b | 0.9736 | 0.9649 | 0.9704 | 0.9696 | 0.9649 |
| M3 unpruned matched | 0.007 | 0.052 | 0.076 | 0.045 | collapse |
Screen (300 base-correct SEL, 75/class, fixed hash ids): M0 0.957/0.960/0.963, M1 0.963/0.967/0.957, M2 0.967/0.960/0.967. Draw sd about 0.4 pt.
Best single run M2 s11: 0.9736, Wilson LB 0.9674, cluster LB 0.9675; classes Business 0.958, Sports 0.995, Technology 0.958, World 0.976; sources ag_news 0.976, bbc_news (SEL half A) 0.964; worst cell bbc World 0.914 (n=58).

## Decision rules fired
- SCREEN: continue only if M0 or M1 >= 0.97 on the 300-item screen: NOT met (max 0.967). (M2/M3 s11 and seeds 12-13 were already queued and ran; they changed no decision.)
- KILL: best arm SEL retention < 0.975 after M1/M2: FIRES (best run 0.9736, best arm mean 0.9696, all nine M0-M2 runs < 0.975). Tier 1.4 would also be TIER-3-adjacent (< 0.975).
- REP stays SEALED: not unsealed, no REP/HuffPost/BBC-B generation was run for candidates, so the pre-registered unseen-source block (BBC-B + HuffPost >= 0.97) was NOT measured. Only proxy: BBC half A inside SEL, 0.947-0.964 for M0-M2 (below 0.97, but SEL-grade, not the registered check). INJ-REP (report only) was not run, killed first (peer: INJ 0.941 / S2 0.851 with the same recipe, so expect fail).
- M3: unpruned at the same bytes is impossible at >= 2 bits (2-bit g128 needs about 12.9% of fp16); matched control collapses, i.e. pruning is what makes 9.9% feasible, but it does not reach 0.975.

## Honest comparison with the peer
Peer claimed NEWS 0.975 (first-token, fake-quant, closed 4-label prompt, easier source mix). With REAL decode, REAL generation, open-tail prompt and SEL class-balanced AG + BBC, the same RTN recipe gives 0.961; GPTQ + 2.5-bit mix gives 0.970. The first-token/fake-quant numbers overstated retention by about 1 pt, in line with the queue's TF-vs-GEN warning. The gap to 0.985 (queue hypothesis) is about 1.5-2 pt; GPTQ/mixed bits recover about 0.8 pt of it.
Rule 2: Business and Technology classes 0.95-0.96 are right at the 0.95 per-kind line even for the best arm; overall point 0.97 < 0.98.

## Rule 9 / different-distribution exposure (recorded, not mitigated)
Experts are kept by routing mass measured on CAL text (AG news) plus 300 other-task and 400 wikitext lines. Task-pruned experts (about 43-52% of experts per the budget; some layers keep 48, others 63+) are exactly the ones the calibration distribution uses; the maintainer's check runs the shipped system on his own held-out data from a different distribution and with his own feedback. Evidence in this run: BBC (never in CAL) is 1.0-1.5 pt below AG and its weakest cell (World, n=58) is 0.91, i.e. source shift costs real retention; the peer's broadened-calibration ablation (task-only calibration 0.81-0.88 vs 0.94-0.975) shows the pruned model's behaviour depends strongly on calibration coverage; the peer's INJ result (S2 0.851) shows far-source collapse. Reading under rules 1/9: a model whose pruned experts were chosen for a 4-topic news task is a narrow-distribution model; its feedback step must also run through pruned computation. Also rule 8: base accuracy of cell bbc World is 0.30 under the open-tail prompt (model answers 'Politics'), so the BBC World cell has little to keep; this is a prompt/task property fixed by X01, flagged for X01.
Not tested: unseen-source REP block, INJ-REP, general capability (GEN), feedback.

## Files
`v2/exp/x13/{qmoe.py,x13.py,analyze.py,screen_ids.txt}`; per-item predictions and logs `v2/exp/results/X13/`; size checks `bytes_M2_s11.json`, `size_check_M2_s11.txt`. Packs were not synced (6 GB each, box destroyed); they regenerate in about 8 min per arm (`x13.py quant`), seed-deterministic.
