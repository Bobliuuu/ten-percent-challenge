# C-MOE analysis (Qwen3-30B-A3B-Instruct-2507, no training) -- plain verdict

## Verdict
No MoE operating point measured reaches 98% item retention at <=9.6% of fp16 (<=5.86e9 B). Evidence:
- k=64 experts at ~2.4 bpw (9.39% of fp16, real-rANS-verified bytes): top-1 0.692, KL 1.26, retention agnews/sst2/tool-call = 0.42/0.83/0.68 (n_base_correct 33/6/22, small). Prune-only k=64 in bf16 (52.5%) is no better (0.710, KL 1.16, 0.30/1.00/0.64): the loss is from dropping experts, not from the bits. So k<=64 is dead at any precision.
- All 128 experts at 4 bits (26.4% of fp16, 2.7x over budget): top-1 0.964, KL 0.018, retention 1.00/1.00/1.00 on the same small subsets. This is the only measured point that retains labels, and it is far outside the budget.
- Ladder (RTN probe, OLMoE-1B-7B and Qwen1.5-MoE-A2.7B, results/vol/ladder_*.json): keeping 75% of experts in bf16 already gives top-1 0.70-0.76; 50% gives 0.38-0.51. Pruning cannot be recovered without training on either model family; Qwen3-30B-A3B at k=64 matches that pattern (0.71).
- To fit 9.6% while keeping all 128 experts needs ~1.1 bpw on the experts (plus 5-bit attention, 6-bit embeddings). That point (k=128 fill) and the 3-bit/2-bit points were queued but NOT measured: the Modal workspace GPU cap starved this track for >1 h and the sweep was stopped on the coordinator's instruction to free slots for the dense track. The 4-bit result and the failure of 2.4 bpw k=64 make a pass at 1.1 bpw with scalar GPTQ+entropy coding implausible (prior v2/moe/RESULTS.md: 1-bit mixes collapse to MMLU 0.45), but it is untested here.

## Which categories could pass
- Short-label tasks (ag_news / sst2 labels): at 4 bits all-experts retention was 100% on 47/6 base-correct items; below ~4 bits unmeasured at scale; at 9.4% (k=64) 0.42/0.83. So label tasks pass only at ~26% of fp16.
- Tool-call (name+args exact): 0.64-0.68 at k=64, 1.00 at 4 bit (22 items).
- Free-form (64-token exact match): cannot pass even for bf16 base vs itself (noise floor em64 0.715; em8 0.965) -- not a usable rule-2 metric.

## Extrapolation protocol artifacts
- KL->retention fit (9 cat-config points): ret = exp(-1.12 KL); KL for 98% retention = 0.018 ~ 4-bit-all-experts quality. Budget-fitting configs sit at KL >= 1.2.
- Stage-0 ranking, additivity (stride-4) runs, bigger labelled subsets (evalL: 400 agnews, 400 sst2, 300 tool-call, built in moe_data/evalL.jsonl by prep_L.py) were implemented (cfg_final.json) but not run to completion for lack of GPU slots; app stopped.
- Predicted-vs-actual: none (no completed stride run).

## Resume (if GPUs free up)
cd ~/ten-percent/v2/compress/moe && modal run --detach moe_job.py --cfgs cfg_final.json --tag final ; python collect.py ; python analyze.py
(4 one-GPU jobs: j1 k128-fill+k64-fill with evalL, j2 quant 3-bit, j3 quant 4-bit+2-bit, j4 stride-4 x4.)
