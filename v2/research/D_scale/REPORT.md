# D_scale: does a bigger dense base reach >=98% item retention at <=10% bytes?  (INTERIM, 2026-11-07 ~02:40Z)

Pipeline: research/D_scale/dense_stream.py (layer-streaming GPTQ + Hadamard128 + per-module level allocation + entropy-coded size accounting, 4-bit packed codes; same maths as compress/dense/dense.py). Eval: dequantised weights exported to HF bf16 (export_hf.py; fp8-dynamic for 72B) and run in vLLM greedy (vllm_base.py), prompt = base chat template with tools, enable_thinking=False, 400 new tokens, new hardened scorer.py. Items: 1300 DEV items from the rebuilt splits (800 s1_dev + 500 s3_dev, random, seed 0; never locked/S2). Calibration: s1_calib (500 multi + 500 single/nocall) + generic slice. Size = body entropy*1.005 + 11 bits/group scales + embedding + lm_head (quantised) + norms/biases + tokenizer/config files + 150 kB code.
Retention = fraction of base-correct items the candidate also gets right. STRICT gates: overall >=98% (and Wilson lo >=98), every (source,kind) cell >=95%.

## Tool-use ladder (T1, 1300 dev items)
| model | ratio | bytes | body bpw | base acc | retention (Wilson lo) | S1 | S3 | worst cell | strict |
|---|---|---|---|---|---|---|---|---|---|
| Qwen3-32B | 9.61% | 6.30 GB | 1.51 | 69.5% | 76.2% (73.3) | 84.2 | 62.3 | S3/multi 54.2 | FAIL |
| Qwen3-32B | 12.03% | 7.88 GB | 1.92 | 69.5% | 89.4% (87.2) | 90.5 | 87.3 | S1/nocall 74.7 | FAIL |
| Qwen3-32B | 15.03% | 9.85 GB | 2.40 | 69.5% | 94.8% (93.2) | 94.7 | 94.9 | S1/nocall 86.7 | FAIL |
| Qwen2.5-72B-Instruct | 9.60% | 13.96 GB | 1.52 (g=128) | 70.7% (fp8-dyn base) | 73.9% (70.9) | 80.5 | 62.1 | S1/nocall 47.5 | FAIL |
