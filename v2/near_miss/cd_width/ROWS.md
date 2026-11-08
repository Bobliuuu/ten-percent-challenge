# cd_width (lane c+d) ROWS: NO RESULTS
Lane stopped by coordinator before any run. The Vast box never left "loading" (about 15 min); destroyed unused (~$0.1 spent).
| method | base | size % | retention | min-cell | feedback | rule gaps | weakest rule | notes |
|---|---|---|---|---|---|---|---|---|
| (none run) | Qwen2.5-7B | n/a | n/a | n/a | n/a | n/a | n/a | Pipeline written and syntax-checked only; never executed. |
Prior evidence (v2/prune/TSP_RESULTS.md, TF, dev): FFN/head/layer pruning + quant on 14B/32B is net-negative at <=9.9%.
Ready code: prep_data.py (disjoint calib, asserted), run_cfgs.py (Wanda-style FFN + head/GQA-group prune, Hadamard ECSQ entropy-coded bytes, vLLM gen), probe_vllm.py, analyze_cd.py, cfg_sweep1.json (11 configs). Untested.
