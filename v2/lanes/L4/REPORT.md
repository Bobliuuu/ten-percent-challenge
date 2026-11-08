# L4: Qwen3-30B-A3B task-pruned experts + GPTQ/Hadamard vs rival #3 (banking router + CLINC domain router)

STATUS (2026-11-08): data + harness built, first compression sweep LAUNCHED on Vast (no numbers yet). No package files (answer.py/feedback.py/MANIFEST.json) yet; packed artifact writer not implemented (sizes are EST from bit counts).

- Tasks: `v2/rival3/build_data.py` -> `v2/rival3/data/{bank,clin}` (BM25 top-8 examples in prompt; bank = 8 old families + held-out new family `security`; clin = 9 old domains + held-out new domain `kitchen`). Hash-sealed MANIFEST.json per task. Exam 2400/2700 locked (300 per kind) + `never` slice 560/630 (a hash-disjoint slice of the same dataset, NOT a different source: GRAY for rule 3).
- Compression: `v2/rival3/core.py` (routing-mass expert prune per task, RTN/GPTQ+Hadamard, mixed configs), cfgs `v2/rival3/cfgs_{bank,clin}_0.json` (base, k80 RTN b2, k80/k64 GPTQ b2, k56 b3 attn4, k48 b3, k72 global b2 attn4). EST sizes: k80 b2 9.72%, k64 b2 8.13%, k56 b3 attn4 9.51%, k48 b3 8.34%.
- Feedback: `v2/rival3/fb.py` (closed-form o_proj edit, last layers, 24 annotations, real/shuffled/wrong/empty), to be attached to the best config.
- Runs: Vast boxes tenpct-rival3_bank / _clin (launchers in v2/research/V_vast/logs/rival3_*.log, results pulled to v2/rival3/results/vast_*/).
