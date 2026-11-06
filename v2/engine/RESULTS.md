# v2/engine results (Qwen2.5-14B-Instruct, MMLU dev n=1987, base 0.8007)
Engine: sweep.py (Modal). Block-128 randomized Hadamard on input dim -> sequential GPTQ with L-level symmetric per-group (fp16 scale) grid -> entropy-coded indices (+0.5% ANS overhead) ; vocab 32k + 4-bit emb; lm_head = 4 rows; 1MB slack. Bytes are exact (ratio vs params*2).
| config | dev acc | agree | ratio_bf16 | body bpw (incl scales) |
|---|---|---|---|---|
| uniform L=8 rot g128 | 0.7635 | 0.899 | 0.1663 | 2.92 |
| uniform L=6 rot | 0.7539 | 0.871 | 0.1463 | 2.56 |
| uniform L=4 rot | 0.6623 | 0.758 | 0.1177 | 2.05 |
| uniform L=4 rot g256 | 0.6588 | 0.753 | 0.1140 | 1.98 |
| L=5 rot g128 + drop 6 layers | 0.7031 | 0.799 | 0.1172 | 2.33 |
| uniform L=3 rot | 0.5209 | 0.583 | 0.0968 | 1.68 |
| L=8 NO rotation | 0.4409 | 0.477 | 0.1629 | 2.86 |
| L=4 NO rotation | 0.2929 | 0.312 | 0.1162 | 2.02 |
Findings: (1) Hadamard incoherence is essential for a symmetric entropy-coded grid (+32 pts at L=8). (2) Frontier: ~0.166 @0.764; 0.1177 @0.70 (drop6+L5) ; 10% (L=3 uniform) falls to 0.52. 10% at base accuracy is not reached by this family; gap at 10% is ~28 pts. (3) Layer drop + higher L beats lower L without drop at equal bytes. (4) g256 saves ~0.003 ratio at ~0.4 pt.
Sensitivity tables (per-module relerr and entropy vs L in {2,3,4,6,8,16}) in tables/v2tables; alloc.py picks per-module L by Lagrangian: v_proj/k_proj get 6-16 levels, gate_proj mostly 2-3. Allocated runs (alloc1.5/1.6/1.7, cfg_C.json) were spawned then stopped on request; not completed. cfg_B (drop sweep) not completed.
