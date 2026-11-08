# MoE task-specific pruning (Qwen3-30B-A3B-Instruct-2507), dev only

Setup: moe_task.py, A100-80GB, simulated (fake-quant) eval, first-token greedy vs label (format fail = wrong). Bytes exact (codes + fp16 scale + packed zero per group + bf16 norms/router). Budget 9.9% of 61,064,245,248 B. Calib = task calib (unlabeled) + 300 other-task + 400 wikitext lines; selection = per-layer mass share, global rank, min 16/layer. INJ dev subset n=1460, NEWS n=1600, C2 n=1000.

Base acc: INJ 0.8466, NEWS 0.8319, C2 0.8010. Cost ~ $3 (one ~45 min A100-80 job).

| config | K experts | bytes % | acc | faithfulness | per-label faith | per-source faith |
|---|---|---|---|---|---|---|
| inj_mass_e3 | 2677 | 9.90 | 0.8377 | 0.9409 | benign:0.892, injection:0.993 | S1:0.965, S3:0.936, S2:0.851 |
| inj_mass_e2 | 3934 | 9.90 | 0.8219 | 0.9353 | benign:0.914, injection:0.958 | S1:0.949, S3:0.952, S2:0.839 |
| inj_reap_e3 | 2677 | 9.90 | 0.7836 | 0.8697 | benign:0.775, injection:0.971 | S1:0.906, S3:0.832, S2:0.798 |
| inj_taskonly_e3 | 2677 | 9.90 | 0.7897 | 0.8778 | benign:0.791, injection:0.971 | S1:0.914, S3:0.843, S2:0.798 |
| inj_uniform_e3 | 2640 | 9.79 | 0.8329 | 0.9231 | benign:0.855, injection:0.997 | S1:0.961, S3:0.885, S2:0.845 |
| inj_quantonly_e3 | 6144 | 20.46 | 0.8486 | 0.9806 | benign:0.983, injection:0.978 | S1:0.979, S3:0.997, S2:0.952 |
| news_mass_e3 | 2677 | 9.90 | 0.8419 | 0.9752 | Business:0.945, Technology:0.984, Sports:0.979, World:0.992 | ag_news:0.980, bbc_news:0.975, huffpost:0.967 |
| news_mass_e2 | 3934 | 9.90 | 0.8275 | 0.9444 | Business:0.976, Technology:0.956, Sports:0.990, World:0.861 | ag_news:0.972, bbc_news:0.910, huffpost:0.957 |
| news_reap_e3 | 2677 | 9.90 | 0.7113 | 0.8084 | Business:0.800, Technology:0.908, Sports:0.579, World:0.989 | ag_news:0.835, bbc_news:0.776, huffpost:0.820 |
| news_taskonly_e3 | 2677 | 9.90 | 0.7212 | 0.8264 | Business:0.836, Technology:0.900, Sports:0.626, World:0.978 | ag_news:0.851, bbc_news:0.801, huffpost:0.830 |
| news_uniform_e3 | 2640 | 9.79 | 0.7831 | 0.8317 | Business:0.891, Technology:0.996, Sports:0.997, World:0.492 | ag_news:0.859, bbc_news:0.780, huffpost:0.875 |
| news_quantonly_e3 | 6144 | 20.46 | 0.8281 | 0.9760 | Business:0.967, Technology:0.940, Sports:0.990, World:0.995 | ag_news:0.978, bbc_news:0.981, huffpost:0.964 |
| c2_mass_e3 | 2677 | 9.90 | 0.7750 | 0.9363 | D:0.930, E:0.862, B:0.945, A:0.943, C:0.968 | xLAM:0.936 |

Base per-label acc INJ: {'benign': 0.8780821917808219, 'injection': 0.815068493150685}; NEWS: {'Business': 0.825, 'Technology': 0.625, 'Sports': 0.9625, 'World': 0.915}

## Findings
- NEWS (4 broad topics, task-matched calib): mass-ranked 3-bit keeps 97.5% of base-correct items (weakest label Business 94.5%); acc 0.842 vs base 0.832. Near the gate (98/95) but not on it; quant-only 3-bit (20% size) is also 97.6%, so this is close to the RTN floor.
- INJ: mass 3-bit 94.1% faithful (benign 89.2%, OOD source S2 85.1%), fails 98/95. Quant-only 3-bit is 98.1%, so pruning costs ~4 pt on INJ. 2-bit with 3934 experts: 93.5%.
- C2 (tool selection, many API domains = broad): 93.6% faithful, acc 0.775 vs 0.801: too broad for pruning.
- REAP score and task-only calibration (no OOD/generic text) are clearly worse (INJ 87-88%, NEWS 81-83%): broadened calib matters, routing mass beats REAP. Uniform per-layer counts worse than variable.
- 2-bit experts with more kept experts is not better than 3-bit here (opposite of MMLU).
- Caveat: base MoE-30B is weaker than dense Qwen2.5-32B on these tasks (dense table, different n=600 dev sets: INJ .952, NEWS .893 vs MoE .847/.832 on our bigger sets incl. OOD S2). Dense 32B at 10.03%: retention INJ .981, NEWS .965 (dense tables are retention of accuracy, MoE here is faithfulness; not identical metrics).
- Raw per-item preds not saved; keep sets are on the Modal volume moe-10pct-vol:/task (stats .pt, keep json).
