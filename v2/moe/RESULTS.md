# v2/moe: Qwen3-30B-A3B-Instruct-2507 expert prune + quantize (no training)
Base bf16: 30,532,122,624 params, fp16 = 61,064,245,248 B, budget 6.106 GB. Dev (n=1987) acc **0.8078** (MMLU prompt per TASK.md).
Method: REAP-style expert score (mean gate*||expert_out|| over routed calib tokens) per layer, keep top fraction (router masked), quantize kept experts (asym, group 128, fp16 scale + b-bit zero, MSE clip search). Attention 4-bit g128, embedding 4/6-bit, lm_head reduced to the 4 letter rows, router rows of kept experts + norms fp16. Bytes counted exactly (raw = packed ints; ent = histogram entropy of ints + scales/zeros). Simulated (fake-quant) eval.
Hardware: Modal blocked (10-GPU workspace cap, H100 unavailable) -> AWS g7e.2xlarge (terminated).

## Sweep 1: RTN (results/sweep1_rtn.json)
| config | dev acc | agree | ratio raw / ent |
|---|---|---|---|
| attn4+emb4 only (experts bf16) | 0.7962 | 0.952 | - |
| f1.0 b3 | 0.7816 | 0.925 | 0.198 / 0.185 |
| f0.75 b3 | 0.7740 | 0.900 | 0.151 / 0.141 |
| f0.75 b2 | 0.7036 | 0.809 | 0.106 / 0.101 |
| f0.5 b4 | 0.6930 | 0.787 | 0.134 / 0.124 |
| f0.5 b3 | 0.6794 | 0.767 | 0.104 / 0.097 |
| f0.5 b2 | 0.6009 | 0.672 | 0.074 / 0.071 |
| f0.375 b4 | 0.5773 | 0.648 | 0.103 / 0.096 |
| f0.375 b3 | 0.5566 | 0.623 | 0.081 / 0.075 |
| f0.25 b4 / b3 | 0.4565 / 0.4268 | | 0.072 / 0.057 |
| f0.5 b3 freq-ranked | 0.6115 | | 0.104 / 0.097 (REAP ranking beats frequency by 7pt) |
| f0.375 b4 freq | 0.5320 | | |

## Sweep 2: sequential layer-wise GPTQ (Hessians per expert, attn GPTQ 4-bit, emb 6-bit) (results/gptq_all.json)
| config | dev acc | ratio raw / ent |
|---|---|---|
| 70% experts @2b | **0.7323** | 0.1013 / 0.0975 |
| 15%@3b + 50%@2b | 0.7227 | 0.1032 / 0.0984 |
| 45% @3b | 0.6376 | 0.0965 / 0.0901 |
| 12.5% @4b (smoke) | 0.3100 | 0.043 / 0.040 |
| 30%@2b + 70%@1b (all experts) | 0.4504 | 0.097 / 0.095 |
Remaining 1-bit mixes / g64 configs were killed on wrap-up (first attempt crashed on non-PD attention Hessian from blown-up activations; patched with NaN guard + RTN fallback).

## Conclusions
- Best at <=0.10 ratio: 0.732 dev (GPTQ, 70% experts, 2-bit), i.e. -7.5 pt vs base. Goal (within ~1pt) NOT met.
- Pruning is the dominant loss (MMLU knowledge is spread over experts); at fixed bytes, keeping more experts at fewer bits beats fewer at more bits, down to 2 bits; 1-bit collapses (0.45). GPTQ gains ~+3 pt over RTN at 2-bit.
- Quantization-only at 3 bit (f1.0) is -2.6 pt at 0.185 ratio, so even lossless pruning-free is 1.8x over budget; the MoE route needs ~1.5 bits/expert weight, which RTN/GPTQ scalar quantization cannot give without a large accuracy hit. Untested: expert merging, delta/shared-basis coding across experts, VQ/trellis codebooks for 1.5-2 bit, saliency-aware mixed bits.
- Attn4+emb4 RTN alone costs 1.2 pt; GPTQ attn fixes part of it.
Code: moe_sweep.py (RTN sweep), moe_gptq.py (sequential GPTQ).
