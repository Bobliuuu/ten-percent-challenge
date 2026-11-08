# MODEL_CENSUS (first pass, 2026-11-08; probe results pending)

Source: `model_census/census.py` (HF API, exact safetensors param counts + config.json), list in `model_census/models.py`. fp16 bytes = 2 x params (all-bf16 reference). 10% budget = 0.1 x that. Average bpw at the 10% budget is 1.6 for every model by definition; the real BODY bpw is lower where embeddings and lm_head (cost 2 x V x d params untied) must be kept: e.g. Qwen2.5-0.5B/1.5B (tied 151k vocab) and Llama-3.2-1B/3B (tied 128k) lose 15-30% of the budget to the embedding, 70B+ models lose <3%. So bigger models get a BIGGER bit budget per useful weight (embedding share shrinks) and absolute capacity: 10% of 70B = 14.1 GB, of 235B = 47 GB.
Native low-bit rows: `params` is the packed count of the released repo (not the fp16 reference); their 10% budget refers to the parent fp16 model (BitNet-2B-4T: 2.4B -> 4.8 GB fp16 -> 0.48 GB budget, vs 0.85 GB packed = 17.7%; see NATIVE_SURVEY2.md for the Bonsai family, which are not on the public HF API under these ids). Gated models (Llama, Gemma) were probed through the `unsloth/` mirrors (same weights).

## Static table
| model | kind | params B | fp16 GB | 10% budget GB | bpw at 10% | arch (L/d/V/experts/topk/tie) |
|---|---|---|---|---|---|---|
| Qwen/Qwen2.5-0.5B-Instruct | dense | 0.49 | 1.0 | 0.10 | 1.60 | qwen2 24/896/151936/None/None/True |
| Qwen/Qwen2.5-1.5B-Instruct | dense | 1.54 | 3.1 | 0.31 | 1.60 | qwen2 28/1536/151936/None/None/True |
| Qwen/Qwen2.5-3B-Instruct | dense | 3.09 | 6.2 | 0.62 | 1.60 | qwen2 36/2048/151936/None/None/True |
| Qwen/Qwen2.5-7B-Instruct | dense | 7.62 | 15.2 | 1.52 | 1.60 | qwen2 28/3584/152064/None/None/False |
| Qwen/Qwen2.5-14B-Instruct | dense | 14.77 | 29.5 | 2.95 | 1.60 | qwen2 48/5120/152064/None/None/False |
| Qwen/Qwen2.5-32B-Instruct | dense | 32.76 | 65.5 | 6.55 | 1.60 | qwen2 64/5120/152064/None/None/False |
| Qwen/Qwen2.5-72B-Instruct | dense | 72.71 | 145.4 | 14.54 | 1.60 | qwen2 80/8192/152064/None/None/False |
| Qwen/Qwen3-0.6B | dense | 0.75 | 1.5 | 0.15 | 1.60 | qwen3 28/1024/151936/None/None/True |
| Qwen/Qwen3-1.7B | dense | 2.03 | 4.1 | 0.41 | 1.60 | qwen3 28/2048/151936/None/None/True |
| Qwen/Qwen3-4B | dense | 4.02 | 8.0 | 0.80 | 1.60 | qwen3 36/2560/151936/None/None/True |
| Qwen/Qwen3-8B | dense | 8.19 | 16.4 | 1.64 | 1.60 | qwen3 36/4096/151936/None/None/False |
| Qwen/Qwen3-14B | dense | 14.77 | 29.5 | 2.95 | 1.60 | qwen3 40/5120/151936/None/None/False |
| Qwen/Qwen3-32B | dense | 32.76 | 65.5 | 6.55 | 1.60 | qwen3 64/5120/151936/None/None/False |
| Qwen/Qwen3-30B-A3B | moe | 30.53 | 61.1 | 6.11 | 1.60 | qwen3_moe 48/2048/151936/128/8/False |
| Qwen/Qwen3-235B-A22B | moe | 235.09 | 470.2 | 47.02 | 1.60 | qwen3_moe 94/4096/151936/128/8/False |
| Qwen/Qwen3-Next-80B-A3B-Instruct | moe | 81.32 | 162.6 | 16.26 | 1.60 | qwen3_next 48/2048/151936/512/10/False |
| Qwen/Qwen3.5-0.8B | dense | 0.87 | 1.7 | 0.17 | 1.60 | qwen3_5 24/1024/248320/None/None/True |
| Qwen/Qwen3.5-2B | dense | 2.27 | 4.5 | 0.45 | 1.60 | qwen3_5 24/2048/248320/None/None/True |
| Qwen/Qwen3.5-4B | dense | 4.66 | 9.3 | 0.93 | 1.60 | qwen3_5 32/2560/248320/None/None/True |
| Qwen/Qwen3.5-9B | dense | 9.65 | 19.3 | 1.93 | 1.60 | qwen3_5 32/4096/248320/None/None/False |
| Qwen/Qwen3.5-27B | dense | 27.78 | 55.6 | 5.56 | 1.60 | qwen3_5 64/5120/248320/None/None/False |
| Qwen/Qwen3.5-35B-A3B | moe | 35.95 | 71.9 | 7.19 | 1.60 | qwen3_5_moe 40/2048/248320/256/8/False |
| Qwen/Qwen3.5-122B-A10B | moe | 125.09 | 250.2 | 25.02 | 1.60 | qwen3_5_moe 48/3072/248320/256/8/False |
| Qwen/Qwen3.5-397B-A17B | moe | 403.40 | 806.8 | 80.68 | 1.60 | qwen3_5_moe 60/4096/248320/512/10/False |
| meta-llama/Llama-3.2-1B-Instruct | dense | 1.24 | 2.5 | 0.25 | 1.60 | None None/None/None/None/None/None |
| meta-llama/Llama-3.2-3B-Instruct | dense | 3.21 | 6.4 | 0.64 | 1.60 | None None/None/None/None/None/None |
| meta-llama/Llama-3.1-8B-Instruct | dense | 8.03 | 16.1 | 1.61 | 1.60 | None None/None/None/None/None/None |
| meta-llama/Llama-3.3-70B-Instruct | dense | 70.55 | 141.1 | 14.11 | 1.60 | None None/None/None/None/None/None |
| unsloth/Llama-3.2-3B-Instruct | dense | 3.21 | 6.4 | 0.64 | 1.60 | llama 28/3072/128256/None/None/True |
| unsloth/Meta-Llama-3.1-8B-Instruct | dense | 8.03 | 16.1 | 1.61 | 1.60 | llama 32/4096/128256/None/None/False |
| unsloth/Llama-3.3-70B-Instruct | dense | 70.55 | 141.1 | 14.11 | 1.60 | llama 80/8192/128256/None/None/False |
| google/gemma-2-2b-it | dense | 2.61 | 5.2 | 0.52 | 1.60 | None None/None/None/None/None/None |
| google/gemma-2-9b-it | dense | 9.24 | 18.5 | 1.85 | 1.60 | None None/None/None/None/None/None |
| google/gemma-2-27b-it | dense | 27.23 | 54.5 | 5.45 | 1.60 | None None/None/None/None/None/None |
| google/gemma-3-1b-it | dense | 1.00 | 2.0 | 0.20 | 1.60 | None None/None/None/None/None/None |
| google/gemma-3-4b-it | dense | 4.30 | 8.6 | 0.86 | 1.60 | None None/None/None/None/None/None |
| google/gemma-3-12b-it | dense | 12.19 | 24.4 | 2.44 | 1.60 | None None/None/None/None/None/None |
| google/gemma-3-27b-it | dense | 27.43 | 54.9 | 5.49 | 1.60 | None None/None/None/None/None/None |
| microsoft/Phi-3-mini-4k-instruct | dense | 3.82 | 7.6 | 0.76 | 1.60 | phi3 32/3072/32064/None/None/False |
| microsoft/Phi-3-medium-4k-instruct | dense | 13.96 | 27.9 | 2.79 | 1.60 | phi3 40/5120/32064/None/None/False |
| microsoft/phi-4 | dense | 14.66 | 29.3 | 2.93 | 1.60 | phi3 40/5120/100352/None/None/False |
| microsoft/Phi-4-mini-instruct | dense | 3.84 | 7.7 | 0.77 | 1.60 | phi3 32/3072/200064/None/None/True |
| mistralai/Mistral-7B-Instruct-v0.3 | dense | 7.25 | 14.5 | 1.45 | 1.60 | mistral 32/4096/32768/None/None/False |
| mistralai/Mistral-Nemo-Instruct-2407 | dense | 12.25 | 24.5 | 2.45 | 1.60 | mistral 40/5120/131072/None/None/False |
| mistralai/Mistral-Small-24B-Instruct-2501 | dense | 23.57 | 47.1 | 4.71 | 1.60 | mistral 40/5120/131072/None/None/False |
| mistralai/Mistral-Small-3.2-24B-Instruct-2506 | dense | 24.01 | 48.0 | 4.80 | 1.60 | mistral3 40/5120/131072/None/None/None |
| mistralai/Mixtral-8x7B-Instruct-v0.1 | moe | 46.70 | 93.4 | 9.34 | 1.60 | mixtral 32/4096/32000/8/2/False |
| mistralai/Mixtral-8x22B-Instruct-v0.1 | moe | 140.63 | 281.3 | 28.13 | 1.60 | mixtral 56/6144/32768/8/2/False |
| HuggingFaceTB/SmolLM3-3B | dense | 3.08 | 6.2 | 0.62 | 1.60 | smollm3 36/2048/128256/None/None/True |
| HuggingFaceTB/SmolLM2-1.7B-Instruct | dense | 1.71 | 3.4 | 0.34 | 1.60 | llama 24/2048/49152/None/None/True |
| allenai/OLMo-2-1124-7B-Instruct | dense | 7.30 | 14.6 | 1.46 | 1.60 | olmo2 32/4096/100352/None/None/False |
| allenai/OLMo-2-1124-13B-Instruct | dense | 13.72 | 27.4 | 2.74 | 1.60 | olmo2 40/5120/100352/None/None/False |
| allenai/OLMo-2-0325-32B-Instruct | dense | 32.23 | 64.5 | 6.45 | 1.60 | olmo2 64/5120/100352/None/None/False |
| allenai/Olmo-3-7B-Instruct | dense | 7.30 | 14.6 | 1.46 | 1.60 | olmo3 32/4096/100278/None/None/False |
| allenai/Olmo-3-32B-Think | dense | 32.23 | 64.5 | 6.45 | 1.60 | olmo3 64/5120/100278/None/None/False |
| allenai/OLMoE-1B-7B-0125-Instruct | moe | 6.92 | 13.8 | 1.38 | 1.60 | olmoe 16/2048/50304/64/8/False |
| 01-ai/Yi-1.5-6B-Chat | dense | 6.06 | 12.1 | 1.21 | 1.60 | llama 32/4096/64000/None/None/False |
| 01-ai/Yi-1.5-9B-Chat | dense | 8.83 | 17.7 | 1.77 | 1.60 | llama 48/4096/64000/None/None/False |
| 01-ai/Yi-1.5-34B-Chat | dense | 34.39 | 68.8 | 6.88 | 1.60 | llama 60/7168/64000/None/None/False |
| internlm/internlm2_5-7b-chat | dense | 7.74 | 15.5 | 1.55 | 1.60 | internlm2 32/4096/92544/None/None/False |
| internlm/internlm3-8b-instruct | dense | 8.80 | 17.6 | 1.76 | 1.60 | internlm3 48/4096/128512/None/None/False |
| THUDM/glm-4-9b-chat-hf | dense | 9.40 | 18.8 | 1.88 | 1.60 | glm 40/4096/151552/None/None/False |
| zai-org/GLM-4-32B-0414 | dense | 32.57 | 65.1 | 6.51 | 1.60 | glm4 61/6144/151552/None/None/False |
| zai-org/GLM-4.5-Air | moe | 110.47 | 220.9 | 22.09 | 1.60 | glm4_moe 46/4096/151552/128/8/False |
| zai-org/GLM-4.5 | moe | 358.34 | 716.7 | 71.67 | 1.60 | glm4_moe 92/5120/151552/160/8/False |
| deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B | dense | 1.78 | 3.6 | 0.36 | 1.60 | qwen2 28/1536/151936/None/None/False |
| deepseek-ai/DeepSeek-R1-Distill-Qwen-7B | dense | 7.62 | 15.2 | 1.52 | 1.60 | qwen2 28/3584/152064/None/None/False |
| deepseek-ai/DeepSeek-R1-Distill-Qwen-14B | dense | 14.77 | 29.5 | 2.95 | 1.60 | qwen2 48/5120/152064/None/None/False |
| deepseek-ai/DeepSeek-R1-Distill-Qwen-32B | dense | 32.76 | 65.5 | 6.55 | 1.60 | qwen2 64/5120/152064/None/None/False |
| deepseek-ai/DeepSeek-R1-Distill-Llama-8B | dense | 8.03 | 16.1 | 1.61 | 1.60 | llama 32/4096/128256/None/None/False |
| deepseek-ai/DeepSeek-R1-Distill-Llama-70B | dense | 70.55 | 141.1 | 14.11 | 1.60 | llama 80/8192/128256/None/None/False |
| deepseek-ai/DeepSeek-V2-Lite-Chat | moe | 15.71 | 31.4 | 3.14 | 1.60 | deepseek_v2 27/2048/102400/64/6/False |
| deepseek-ai/DeepSeek-V3 | moe | 684.49 | 1369.0 | 136.90 | 1.60 | deepseek_v3 61/7168/129280/256/8/False |
| moonshotai/Moonlight-16B-A3B-Instruct | moe | 15.96 | 31.9 | 3.19 | 1.60 | deepseek_v3 27/2048/163840/64/6/False |
| moonshotai/Kimi-K2-Instruct | moe | 1026.41 | 2052.8 | 205.28 | 1.60 | kimi_k2 61/7168/163840/384/8/False |
| ibm-granite/granite-3.3-2b-instruct | dense | 2.53 | 5.1 | 0.51 | 1.60 | granite 40/2048/49159/None/None/True |
| ibm-granite/granite-3.3-8b-instruct | dense | 8.17 | 16.3 | 1.63 | 1.60 | granite 40/4096/49159/None/None/True |
| ibm-granite/granite-4.0-h-small | moe | 32.21 | 64.4 | 6.44 | 1.60 | granitemoehybrid 40/4096/100352/72/10/True |
| ibm-granite/granite-3.1-3b-a800m-instruct | moe | 3.30 | 6.6 | 0.66 | 1.60 | granitemoe 32/1536/49155/40/8/True |
| LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct | dense | 2.41 | 4.8 | 0.48 | 1.60 | exaone None/2560/102400/None/None/True |
| LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct | dense | 7.82 | 15.6 | 1.56 | 1.60 | exaone None/4096/102400/None/None/False |
| LGAI-EXAONE/EXAONE-3.5-32B-Instruct | dense | 32.00 | 64.0 | 6.40 | 1.60 | exaone None/5120/102400/None/None/False |
| tiiuae/Falcon3-1B-Instruct | dense | 1.67 | 3.3 | 0.33 | 1.60 | llama 18/2048/131072/None/None/False |
| tiiuae/Falcon3-3B-Instruct | dense | 3.23 | 6.5 | 0.65 | 1.60 | llama 22/3072/131072/None/None/False |
| tiiuae/Falcon3-7B-Instruct | dense | 7.46 | 14.9 | 1.49 | 1.60 | llama 28/3072/131072/None/None/False |
| tiiuae/Falcon3-10B-Instruct | dense | 10.31 | 20.6 | 2.06 | 1.60 | llama 40/3072/131072/None/None/False |
| nvidia/Llama-3.1-Nemotron-Nano-8B-v1 | dense | 8.03 | 16.1 | 1.61 | 1.60 | llama 32/4096/128256/None/None/False |
| nvidia/Llama-3_3-Nemotron-Super-49B-v1 | dense | 49.87 | 99.7 | 9.97 | 1.60 | nemotron-nas 80/8192/128256/None/None/False |
| nvidia/Llama-3.1-Nemotron-70B-Instruct-HF | dense | 70.55 | 141.1 | 14.11 | 1.60 | llama 80/8192/128256/None/None/False |
| nvidia/NVIDIA-Nemotron-Nano-9B-v2 | dense | 8.89 | 17.8 | 1.78 | 1.60 | nemotron_h 56/4480/131072/None/None/False |
| openai/gpt-oss-20b | moe | 20.91 | 41.8 | 4.18 | 1.60 | gpt_oss 24/2880/201088/32/4/False |
| openai/gpt-oss-120b | moe | 116.83 | 233.7 | 23.37 | 1.60 | gpt_oss 36/2880/201088/128/4/False |
| tencent/Hunyuan-A13B-Instruct | moe | 80.39 | 160.8 | 16.08 | 1.60 | hunyuan_v1_moe 32/4096/128167/64/None/True |
| tencent/Hunyuan-7B-Instruct | dense | 7.50 | 15.0 | 1.50 | 1.60 | hunyuan_v1_dense 32/4096/128167/None/None/True |
| baidu/ERNIE-4.5-21B-A3B-PT | moe | 21.95 | 43.9 | 4.39 | 1.60 | ernie4_5_moe 28/2560/103424/None/None/True |
| baidu/ERNIE-4.5-300B-A47B-PT | moe | 300.47 | 600.9 | 60.09 | 1.60 | ernie4_5_moe 54/8192/103424/None/None/False |
| baidu/ERNIE-4.5-0.3B-PT | dense | 0.36 | 0.7 | 0.07 | 1.60 | ernie4_5 18/1024/103424/None/None/True |
| meta-llama/Llama-4-Scout-17B-16E-Instruct | moe | 108.64 | 217.3 | 21.73 | 1.60 | None None/None/None/None/None/None |
| microsoft/bitnet-b1.58-2B-4T | native | 0.85 | 1.7 | 0.17 | 1.60 | bitnet 30/2560/128256/None/None/True |
| microsoft/bitnet-b1.58-2B-4T-bf16 | native | 2.41 | 4.8 | 0.48 | 1.60 | bitnet 30/2560/128256/None/None/True |
| tiiuae/Falcon-E-3B-Instruct | native | 0.86 | 1.7 | 0.17 | 1.60 | llama 32/2048/32768/None/None/False |
| tiiuae/Falcon-E-1B-Instruct | native | 0.53 | 1.1 | 0.11 | 1.60 | llama 24/2048/32768/None/None/False |
| tiiuae/Falcon3-10B-Instruct-1.58bit | native | 3.18 | 6.4 | 0.64 | 1.60 | llama 40/3072/131072/None/None/False |
| HF1BitLLM/Llama3-8B-1.58-100B-tokens | native | 2.80 | 5.6 | 0.56 | 1.60 | llama 32/4096/128256/None/None/False |
| 1bitLLM/bitnet_b1_58-3B | native | 3.32 | 6.6 | 0.66 | 1.60 | llama 26/3200/32002/None/None/True |
| SpectraSuite/TriLM_3.9B_Unpacked | native | 3.99 | 8.0 | 0.80 | 1.60 | llama 30/3072/50688/None/None/False |
| prism-ml/Bonsai-8B-unpacked | native | 8.19 | 16.4 | 1.64 | 1.60 | qwen3 36/4096/151669/None/None/False |
| prism-ml/Ternary-Bonsai-8B-unpacked | native | 8.19 | 16.4 | 1.64 | 1.60 | qwen3 36/4096/151669/None/None/False |
| prism-ml/Bonsai-27B | native | n/a (not found) | | | | |
| prism-ml/Ternary-Bonsai-2-27B | native | n/a (not found) | | | | |

## Quantization-friendliness evidence (literature + our own, to be extended by probe results)
- Larger dense models quantize better at equal bpw (our own 7B/14B/32B tables in task_search/results: retention at ~10% rises with size but stays well below 98% for label tasks; 32B best).
- Qwen2.5 has massive-activation outliers in a few channels, Hadamard rotation handles it (our pipeline). Llama-3 is known to be harder to quantize than Llama-2 (low-bit PTQ papers, e.g. "How good are low-bit quantized Llama3 models"); Gemma-2/3 and Mistral-Nemo tolerate int4 well. VPTQ/QuIP#/AQLM report ~1.5-2 bpw on 70B-405B with 85-95% retention of benchmark scores: the best published points are on the biggest models.
- MoE: experts are individually redundant (OLMoE, Qwen3-30B-A3B routing-mass prune works to 96/128); see moe_entry.

## Probe results: dense pipeline at ~9.9% of fp16 (SKILL7 + BANK8 dev, real greedy generation, retention = P(correct | base correct))
Pipeline: Hadamard + GPTQ + entropy-coded levels, no training (task_search/dense_core.py, target_ratio 0.099, emb/head 4 levels, g256); calibration 382 rows, items 1,382 dev rows from the sel/dev splits (no sealed file). Ranked by Wilson lower bound. Source: model_census/analyze.py, raw rows model_census/raw/. Models still running or failed are absent (failed = architecture the pipeline does not support).
| rank | model | base acc | n base-correct | ratio of fp16 | body bpw | retention | Wilson lo | worst cell (n>=8) | per task |
|---|---|---|---|---|---|---|---|---|---|
| 1 | Qwen2.5-32B-Instruct | 0.970 | 1341 | 9.91% | 1.56 | 98.9% | 98.2% | 93% BANK8/bank/card_virtual_wallet | {'SKILL7': 0.989, 'BANK8': 0.988} |
| 2 | Qwen2.5-3B-Instruct | 0.943 | 1303 | 9.95% | 1.52 | 91.9% | 90.3% | 69% SKILL7/massive/Calendar | {'SKILL7': 0.889, 'BANK8': 0.948} |
| 3 | Qwen3-32B | 0.937 | 1295 | 9.93% | 1.57 | 83.6% | 81.5% | 0% BANK8/clin/banking | {'SKILL7': 0.985, 'BANK8': 0.684} |
| 4 | Qwen2.5-1.5B-Instruct | 0.897 | 1240 | 9.95% | 1.47 | 50.4% | 47.6% | 0% BANK8/clin/travel | {'SKILL7': 0.505, 'BANK8': 0.503} |
| 5 | Qwen2.5-0.5B-Instruct | 0.458 | 633 | 9.97% | 1.28 | 0.0% | 0.0% | 0% BANK8/bank/account_identity | {'SKILL7': 0.0, 'BANK8': 0.0} |

<!-- MOE-CENSUS-B BEGIN -->
## MoE census (fork B): routing concentration and expert-prune + RTN probe (SKILL7 + BANK8 dev, real greedy generation)
Code `v2/model_census/moe/moe_probe2.py` (transformers 5.22 fused-expert layout; every router returns (logits, weights, indices); pruned experts are removed by re-selecting among kept experts with the reference's own scoring/renorm, so the kept experts get exactly the weights a physically pruned model would give). Per task: experts kept per layer by routing mass on that task's CALIBRATION text (disjoint from the dev items), expert weights RTN asymmetric g128 (fake-quant, bf16 dequantised), real greedy generation of the label (<= 8 tokens), retention = P(correct | base correct) on <= 400 dev items per task (SKILL7 = task_search/EASY dev, BANK8 = rival3 bank+clin SEL first 400 each by md5). Raw per-config rows: `v2/model_census/moe/results/*_v2.json`. Attention, embeddings and head are left bf16 in the probe (not quantised); the size column ASSUMES them at 6.25 bpw (6-bit g128 incl. scale) plus fp16 router rows, so the real size is an estimate, not a packed file. No GPTQ (plain RTN: a lower bound for retention). Dev only, n=400/task: a 99% point has Wilson lower ~98%, i.e. this is a ranking probe, not pass evidence.

### Ranked MoE bases (best point under 11.5% estimated size)
| rank | model | best point (keep f, bits) with est size <= 11.5% | est size | retention SKILL7 / BANK8 | min Wilson lower | worst cell | base acc S7 / B8 |
|---|---|---|---|---|---|---|---|
| 1 | Qwen/Qwen3.5-35B-A3B | f=0.4, 3-bit | 10.1% | 0.991 / 0.997 | 0.980 | 0.95 | 0.969 / 0.973 |
| 2 | Qwen/Qwen3-30B-A3B-Instruct-2507 | f=0.5, 3-bit | 11.3% | 0.989 / 0.995 | 0.978 | 0.95 | 0.966 / 0.975 |
| 3 | zai-org/GLM-4.5-Air | f=0.4, 3-bit | 10.0% | 0.986 / 0.997 | 0.974 | 0.95 | 0.962 / 0.985 |
| 4 | openai/gpt-oss-20b | f=0.4, 3-bit | 10.7% | 0.773 / 0.966 | 0.739 | 0.49 | 0.95 / 0.96 |
| 5 | moonshotai/Moonlight-16B-A3B-Instruct | f=0.4, 3-bit | 11.0% | 0.926 / 0.755 | 0.709 | 0.31 | 0.955 / 0.958 |
| 6 | baidu/ERNIE-4.5-21B-A3B-PT | f=0.3, 3-bit | 8.0% | 0.409 / 0.877 | 0.372 | 0.00 | 0.95 / 0.978 |
| 7 | openai/gpt-oss-120b | f=0.4, 4-bit | 10.9% | 0.333 / 0.486 | 0.296 | 0.00 | 0.846 / 0.973 |
| 8 | deepseek-ai/DeepSeek-V2-Lite-Chat | f=0.4, 3-bit | 10.6% | 0.038 / 0.0 | 0.000 | 0.00 | 0.652 / 0.95 |
| 9 | allenai/OLMoE-1B-7B-0125-Instruct | f=0.4, 3-bit | 10.1% | 0.28 / 0.0 | 0.000 | 0.00 | 0.534 / 0.497 |

### Architecture and byte budget (non-expert tensors charged at 6.25 bpw; expert bpw includes scales)
| model | MoE layers | experts | top-k | params B | expert share | fp16 GB | 10% budget GB | expert bpw if keep 100/75/50/40/30% |
|---|---|---|---|---|---|---|---|---|
| deepseek-ai/DeepSeek-V2-Lite-Chat | 26 | 64 | 6 | 15.7 | 91.6% | 31.4 | 3.14 | 1.18 / 1.57 / 2.35 / 2.94 / 3.92 |
| baidu/ERNIE-4.5-21B-A3B-PT | 27 | 64 | 6 | 21.8 | 93.4% | 43.7 | 4.37 | 1.27 / 1.70 / 2.54 / 3.18 / 4.24 |
| zai-org/GLM-4.5-Air | 45 | 128 | 8 | 106.9 | 93.3% | 213.7 | 21.37 | 1.26 / 1.69 / 2.53 / 3.16 / 4.21 |
| moonshotai/Moonlight-16B-A3B-Instruct | 26 | 64 | 6 | 16.0 | 90.2% | 31.9 | 3.19 | 1.09 / 1.46 / 2.19 / 2.74 / 3.65 |
| allenai/OLMoE-1B-7B-0125-Instruct | 16 | 64 | 8 | 6.9 | 93.1% | 13.8 | 1.38 | 1.26 / 1.67 / 2.51 / 3.14 / 4.19 |
| Qwen/Qwen3-30B-A3B-Instruct-2507 | 48 | 128 | 8 | 30.5 | 95.0% | 61.1 | 6.11 | 1.35 / 1.80 / 2.71 / 3.38 / 4.51 |
| Qwen/Qwen3.5-35B-A3B | 40 | 256 | 8 | 34.7 | 92.9% | 69.3 | 6.93 | 1.25 / 1.66 / 2.49 / 3.12 / 4.16 |
| openai/gpt-oss-120b | 36 | 128 | 4 | 116.8 | 98.1% | 233.7 | 23.37 | 1.51 / 2.02 / 3.02 / 3.78 / 5.04 |
| openai/gpt-oss-20b | 24 | 32 | 4 | 20.9 | 91.4% | 41.8 | 4.18 | 1.16 / 1.55 / 2.32 / 2.90 / 3.87 |

### Routing concentration (dev calibration text)
| model | task | experts needed 90/95/99% (fraction, layer mean) | routing entropy / ln E | mass in own top-50% | mass in own top-30% | other task's mass in top-50% |
|---|---|---|---|---|---|---|
| deepseek-ai/DeepSeek-V2-Lite-Chat | SKILL7 | 0.59 / 0.71 / 0.87 | 0.89 | 0.84 | 0.66 | BANK8 0.68, NEWS 0.65 |
| deepseek-ai/DeepSeek-V2-Lite-Chat | BANK8 | 0.70 / 0.81 / 0.94 | 0.93 | 0.78 | 0.60 | SKILL7 0.75, NEWS 0.61 |
| baidu/ERNIE-4.5-21B-A3B-PT | SKILL7 | 0.56 / 0.67 / 0.83 | 0.88 | 0.87 | 0.69 | BANK8 0.73, NEWS 0.69 |
| baidu/ERNIE-4.5-21B-A3B-PT | BANK8 | 0.63 / 0.74 / 0.90 | 0.90 | 0.83 | 0.65 | SKILL7 0.75, NEWS 0.65 |
| zai-org/GLM-4.5-Air | SKILL7 | 0.48 / 0.59 / 0.76 | 0.85 | 0.91 | 0.76 | BANK8 0.69, NEWS 0.66 |
| zai-org/GLM-4.5-Air | BANK8 | 0.58 / 0.70 / 0.88 | 0.88 | 0.86 | 0.70 | SKILL7 0.78, NEWS 0.65 |
| moonshotai/Moonlight-16B-A3B-Instruct | SKILL7 | 0.46 / 0.57 / 0.73 | 0.83 | 0.92 | 0.76 | BANK8 0.81, NEWS 0.74 |
| moonshotai/Moonlight-16B-A3B-Instruct | BANK8 | 0.48 / 0.58 / 0.75 | 0.83 | 0.92 | 0.75 | SKILL7 0.86, NEWS 0.69 |
| allenai/OLMoE-1B-7B-0125-Instruct | SKILL7 | 0.61 / 0.72 / 0.86 | 0.90 | 0.83 | 0.65 | BANK8 0.74, NEWS 0.64 |
| allenai/OLMoE-1B-7B-0125-Instruct | BANK8 | 0.63 / 0.73 / 0.88 | 0.90 | 0.83 | 0.65 | SKILL7 0.76, NEWS 0.62 |
| Qwen/Qwen3-30B-A3B-Instruct-2507 | SKILL7 | 0.33 / 0.40 / 0.54 | 0.78 | 0.98 | 0.87 | BANK8 0.88, NEWS 0.81 |
| Qwen/Qwen3-30B-A3B-Instruct-2507 | BANK8 | 0.34 / 0.43 / 0.60 | 0.79 | 0.97 | 0.86 | SKILL7 0.94, NEWS 0.83 |
| Qwen/Qwen3.5-35B-A3B | SKILL7 | 0.27 / 0.35 / 0.48 | 0.76 | 0.99 | 0.92 | BANK8 0.83, NEWS 0.81 |
| Qwen/Qwen3.5-35B-A3B | BANK8 | 0.37 / 0.48 / 0.67 | 0.80 | 0.95 | 0.85 | SKILL7 0.92, NEWS 0.76 |
| openai/gpt-oss-120b | SKILL7 | 0.32 / 0.41 / 0.54 | 0.75 | 0.98 | 0.88 | BANK8 0.90, NEWS 0.92 |
| openai/gpt-oss-120b | BANK8 | 0.33 / 0.43 / 0.61 | 0.75 | 0.97 | 0.88 | SKILL7 0.94, NEWS 0.92 |
| openai/gpt-oss-20b | SKILL7 | 0.50 / 0.60 / 0.76 | 0.80 | 0.91 | 0.76 | BANK8 0.86, NEWS 0.86 |
| openai/gpt-oss-20b | BANK8 | 0.46 / 0.56 / 0.75 | 0.79 | 0.92 | 0.78 | SKILL7 0.86, NEWS 0.85 |

### Prune + RTN probe (real greedy generation)
| model | task | kept f | bits | est size % fp16 | base acc | retention | lower | worst cell | cell | n |
|---|---|---|---|---|---|---|---|---|---|---|
| deepseek-ai/DeepSeek-V2-Lite-Chat | SKILL7 | 1 | 3 | 21.3 | 0.652 | 0.829 | 0.791 | 0.11 | Calendar | 445 |
| deepseek-ai/DeepSeek-V2-Lite-Chat | BANK8 | 1 | 3 | 21.3 | 0.950 | 0.982 | 0.962 | 0.88 | cash | 380 |
| deepseek-ai/DeepSeek-V2-Lite-Chat | SKILL7 | 0.75 | 3 | 16.8 | 0.652 | 0.604 | 0.558 | 0.1 | Lists | 445 |
| deepseek-ai/DeepSeek-V2-Lite-Chat | BANK8 | 0.75 | 3 | 16.8 | 0.950 | 0.921 | 0.890 | 0.67 | meta | 380 |
| deepseek-ai/DeepSeek-V2-Lite-Chat | SKILL7 | 0.5 | 3 | 12.3 | 0.652 | 0.445 | 0.399 | 0.18 | Music | 445 |
| deepseek-ai/DeepSeek-V2-Lite-Chat | BANK8 | 0.5 | 3 | 12.3 | 0.950 | 0.242 | 0.202 | 0.0 | incoming_topups | 380 |
| deepseek-ai/DeepSeek-V2-Lite-Chat | SKILL7 | 0.4 | 3 | 10.6 | 0.652 | 0.038 | 0.024 | 0.0 | Transport | 445 |
| deepseek-ai/DeepSeek-V2-Lite-Chat | BANK8 | 0.4 | 3 | 10.6 | 0.950 | 0.000 | 0.000 | 0.0 | account_identity | 380 |
| deepseek-ai/DeepSeek-V2-Lite-Chat | SKILL7 | 0.3 | 3 | 8.6 | 0.652 | 0.000 | 0.000 | 0.0 | Calendar | 445 |
| deepseek-ai/DeepSeek-V2-Lite-Chat | BANK8 | 0.3 | 3 | 8.6 | 0.950 | 0.000 | 0.000 | 0.0 | account_identity | 380 |
| deepseek-ai/DeepSeek-V2-Lite-Chat | SKILL7 | 0.5 | 4 | 15.2 | 0.652 | 0.465 | 0.419 | 0.03 | Calendar | 445 |
| deepseek-ai/DeepSeek-V2-Lite-Chat | BANK8 | 0.5 | 4 | 15.2 | 0.950 | 0.313 | 0.269 | 0.0 | auto_commute | 380 |
| deepseek-ai/DeepSeek-V2-Lite-Chat | SKILL7 | 0.4 | 4 | 12.9 | 0.652 | 0.000 | 0.000 | 0.0 | Calendar | 445 |
| deepseek-ai/DeepSeek-V2-Lite-Chat | BANK8 | 0.4 | 4 | 12.9 | 0.950 | 0.021 | 0.011 | 0.0 | auto_commute | 380 |
| baidu/ERNIE-4.5-21B-A3B-PT | SKILL7 | 1 | 3 | 21.0 | 0.950 | 0.736 | 0.701 | 0.11 | Transport | 648 |
| baidu/ERNIE-4.5-21B-A3B-PT | BANK8 | 1 | 3 | 21.0 | 0.978 | 0.895 | 0.861 | 0.17 | info_and_fx | 391 |
| baidu/ERNIE-4.5-21B-A3B-PT | SKILL7 | 0.75 | 3 | 16.4 | 0.950 | 0.705 | 0.669 | 0.04 | Weather | 648 |
| baidu/ERNIE-4.5-21B-A3B-PT | BANK8 | 0.75 | 3 | 16.4 | 0.978 | 0.928 | 0.898 | 0.33 | info_and_fx | 391 |
| baidu/ERNIE-4.5-21B-A3B-PT | SKILL7 | 0.5 | 3 | 11.8 | 0.950 | 0.637 | 0.600 | 0.05 | Weather | 648 |
| baidu/ERNIE-4.5-21B-A3B-PT | BANK8 | 0.5 | 3 | 11.8 | 0.978 | 0.882 | 0.847 | 0.19 | transfers | 391 |
| baidu/ERNIE-4.5-21B-A3B-PT | SKILL7 | 0.4 | 3 | 10.1 | 0.950 | 0.383 | 0.346 | 0.01 | Weather | 648 |
| baidu/ERNIE-4.5-21B-A3B-PT | BANK8 | 0.4 | 3 | 10.1 | 0.978 | 0.918 | 0.887 | 0.15 | transfers | 391 |
| baidu/ERNIE-4.5-21B-A3B-PT | SKILL7 | 0.3 | 3 | 8.0 | 0.950 | 0.409 | 0.372 | 0.07 | Weather | 648 |
| baidu/ERNIE-4.5-21B-A3B-PT | BANK8 | 0.3 | 3 | 8.0 | 0.978 | 0.877 | 0.841 | 0.0 | transfers | 391 |
| baidu/ERNIE-4.5-21B-A3B-PT | SKILL7 | 0.5 | 4 | 14.7 | 0.950 | 0.761 | 0.726 | 0.16 | Transport | 648 |
| baidu/ERNIE-4.5-21B-A3B-PT | BANK8 | 0.5 | 4 | 14.7 | 0.978 | 0.944 | 0.916 | 0.26 | transfers | 391 |
| baidu/ERNIE-4.5-21B-A3B-PT | SKILL7 | 0.4 | 4 | 12.4 | 0.950 | 0.733 | 0.698 | 0.5 | Transport | 648 |
| baidu/ERNIE-4.5-21B-A3B-PT | BANK8 | 0.4 | 4 | 12.4 | 0.978 | 0.918 | 0.887 | 0.22 | transfers | 391 |
| zai-org/GLM-4.5-Air | SKILL7 | 1 | 3 | 21.0 | 0.962 | 0.997 | 0.989 | 0.99 | Alarm | 656 |
| zai-org/GLM-4.5-Air | BANK8 | 1 | 3 | 21.0 | 0.985 | 0.997 | 0.986 | 0.95 | banking | 394 |
| zai-org/GLM-4.5-Air | SKILL7 | 0.5 | 3 | 11.8 | 0.962 | 0.992 | 0.982 | 0.98 | News | 656 |
| zai-org/GLM-4.5-Air | BANK8 | 0.5 | 3 | 11.8 | 0.985 | 0.997 | 0.986 | 0.95 | banking | 394 |
| zai-org/GLM-4.5-Air | SKILL7 | 0.4 | 3 | 10.0 | 0.962 | 0.986 | 0.974 | 0.96 | Weather | 656 |
| zai-org/GLM-4.5-Air | BANK8 | 0.4 | 3 | 10.0 | 0.985 | 0.997 | 0.986 | 0.95 | banking | 394 |
| zai-org/GLM-4.5-Air | SKILL7 | 0.3 | 3 | 8.1 | 0.962 | 0.805 | 0.773 | 0.0 | Lists | 656 |
| zai-org/GLM-4.5-Air | BANK8 | 0.3 | 3 | 8.1 | 0.985 | 0.995 | 0.982 | 0.95 | travel | 394 |
| zai-org/GLM-4.5-Air | SKILL7 | 0.4 | 4 | 12.3 | 0.962 | 0.992 | 0.982 | 0.97 | Alarm | 656 |
| zai-org/GLM-4.5-Air | BANK8 | 0.4 | 4 | 12.3 | 0.985 | 0.997 | 0.986 | 0.95 | banking | 394 |
| moonshotai/Moonlight-16B-A3B-Instruct | SKILL7 | 1 | 3 | 21.6 | 0.955 | 0.949 | 0.930 | 0.8 | Alarm | 651 |
| moonshotai/Moonlight-16B-A3B-Instruct | BANK8 | 1 | 3 | 21.6 | 0.958 | 0.851 | 0.812 | 0.44 | transfers | 383 |
| moonshotai/Moonlight-16B-A3B-Instruct | SKILL7 | 0.75 | 3 | 17.2 | 0.955 | 0.942 | 0.921 | 0.77 | Alarm | 651 |
| moonshotai/Moonlight-16B-A3B-Instruct | BANK8 | 0.75 | 3 | 17.2 | 0.958 | 0.830 | 0.789 | 0.32 | transfers | 383 |
| moonshotai/Moonlight-16B-A3B-Instruct | SKILL7 | 0.5 | 3 | 12.7 | 0.955 | 0.917 | 0.893 | 0.74 | Alarm | 651 |
| moonshotai/Moonlight-16B-A3B-Instruct | BANK8 | 0.5 | 3 | 12.7 | 0.958 | 0.736 | 0.690 | 0.17 | meta | 383 |
| moonshotai/Moonlight-16B-A3B-Instruct | SKILL7 | 0.4 | 3 | 11.0 | 0.955 | 0.926 | 0.904 | 0.84 | Calendar | 651 |
| moonshotai/Moonlight-16B-A3B-Instruct | BANK8 | 0.4 | 3 | 11.0 | 0.958 | 0.755 | 0.709 | 0.31 | small_talk | 383 |
| moonshotai/Moonlight-16B-A3B-Instruct | SKILL7 | 0.3 | 3 | 9.1 | 0.955 | 0.879 | 0.851 | 0.74 | Alarm | 651 |
| moonshotai/Moonlight-16B-A3B-Instruct | BANK8 | 0.3 | 3 | 9.1 | 0.958 | 0.567 | 0.517 | 0.04 | meta | 383 |
| moonshotai/Moonlight-16B-A3B-Instruct | SKILL7 | 0.5 | 4 | 15.6 | 0.955 | 0.954 | 0.935 | 0.84 | Alarm | 651 |
| moonshotai/Moonlight-16B-A3B-Instruct | BANK8 | 0.5 | 4 | 15.6 | 0.958 | 0.854 | 0.815 | 0.52 | transfers | 383 |
| moonshotai/Moonlight-16B-A3B-Instruct | SKILL7 | 0.4 | 4 | 13.4 | 0.955 | 0.949 | 0.930 | 0.85 | Alarm | 651 |
| moonshotai/Moonlight-16B-A3B-Instruct | BANK8 | 0.4 | 4 | 13.4 | 0.958 | 0.888 | 0.852 | 0.48 | transfers | 383 |
| allenai/OLMoE-1B-7B-0125-Instruct | SKILL7 | 1 | 3 | 21.0 | 0.534 | 0.648 | 0.598 | 0.0 | Transport | 364 |
| allenai/OLMoE-1B-7B-0125-Instruct | BANK8 | 1 | 3 | 21.0 | 0.497 | 0.955 | 0.916 | 0.88 | home | 199 |
| allenai/OLMoE-1B-7B-0125-Instruct | SKILL7 | 0.75 | 3 | 16.5 | 0.534 | 0.635 | 0.584 | 0.12 | Transport | 364 |
| allenai/OLMoE-1B-7B-0125-Instruct | BANK8 | 0.75 | 3 | 16.5 | 0.497 | 0.854 | 0.799 | 0.72 | card_virtual_wallet | 199 |
| allenai/OLMoE-1B-7B-0125-Instruct | SKILL7 | 0.5 | 3 | 11.9 | 0.534 | 0.365 | 0.318 | 0.0 | Transport | 364 |
| allenai/OLMoE-1B-7B-0125-Instruct | BANK8 | 0.5 | 3 | 11.9 | 0.497 | 0.131 | 0.091 | 0.0 | account_identity | 199 |
| allenai/OLMoE-1B-7B-0125-Instruct | SKILL7 | 0.4 | 3 | 10.1 | 0.534 | 0.280 | 0.237 | 0.0 | Lists | 364 |
| allenai/OLMoE-1B-7B-0125-Instruct | BANK8 | 0.4 | 3 | 10.1 | 0.497 | 0.000 | 0.000 | 0.0 | account_identity | 199 |
| allenai/OLMoE-1B-7B-0125-Instruct | SKILL7 | 0.3 | 3 | 8.1 | 0.534 | 0.000 | 0.000 | 0.0 | Calendar | 364 |
| allenai/OLMoE-1B-7B-0125-Instruct | BANK8 | 0.3 | 3 | 8.1 | 0.497 | 0.005 | 0.001 | 0.0 | card_delivery | 199 |
| allenai/OLMoE-1B-7B-0125-Instruct | SKILL7 | 0.5 | 4 | 14.8 | 0.534 | 0.769 | 0.723 | 0.0 | Transport | 364 |
| allenai/OLMoE-1B-7B-0125-Instruct | BANK8 | 0.5 | 4 | 14.8 | 0.497 | 0.251 | 0.196 | 0.0 | card_virtual_wallet | 199 |
| allenai/OLMoE-1B-7B-0125-Instruct | SKILL7 | 0.4 | 4 | 12.5 | 0.534 | 0.148 | 0.116 | 0.0 | Lists | 364 |
| allenai/OLMoE-1B-7B-0125-Instruct | BANK8 | 0.4 | 4 | 12.5 | 0.497 | 0.005 | 0.001 | 0.0 | account_identity | 199 |
| Qwen/Qwen3-30B-A3B-Instruct-2507 | SKILL7 | 1 | 3 | 20.7 | 0.966 | 0.992 | 0.982 | 0.97 | Calendar | 659 |
| Qwen/Qwen3-30B-A3B-Instruct-2507 | BANK8 | 1 | 3 | 20.7 | 0.975 | 0.985 | 0.967 | 0.87 | meta | 390 |
| Qwen/Qwen3-30B-A3B-Instruct-2507 | SKILL7 | 0.75 | 3 | 16.0 | 0.966 | 0.994 | 0.984 | 0.98 | News | 659 |
| Qwen/Qwen3-30B-A3B-Instruct-2507 | BANK8 | 0.75 | 3 | 16.0 | 0.975 | 0.979 | 0.960 | 0.83 | meta | 390 |
| Qwen/Qwen3-30B-A3B-Instruct-2507 | SKILL7 | 0.5 | 3 | 11.3 | 0.966 | 0.989 | 0.978 | 0.96 | News | 659 |
| Qwen/Qwen3-30B-A3B-Instruct-2507 | BANK8 | 0.5 | 3 | 11.3 | 0.975 | 0.995 | 0.981 | 0.95 | utility | 390 |
| Qwen/Qwen3-30B-A3B-Instruct-2507 | SKILL7 | 0.4 | 3 | 9.4 | 0.966 | 0.986 | 0.974 | 0.94 | News | 659 |
| Qwen/Qwen3-30B-A3B-Instruct-2507 | BANK8 | 0.4 | 3 | 9.4 | 0.975 | 0.538 | 0.489 | 0.09 | meta | 390 |
| Qwen/Qwen3-30B-A3B-Instruct-2507 | SKILL7 | 0.3 | 3 | 7.5 | 0.966 | 0.983 | 0.970 | 0.94 | News | 659 |
| Qwen/Qwen3-30B-A3B-Instruct-2507 | BANK8 | 0.3 | 3 | 7.5 | 0.975 | 0.497 | 0.448 | 0.0 | small_talk | 390 |
| Qwen/Qwen3-30B-A3B-Instruct-2507 | SKILL7 | 0.5 | 4 | 14.3 | 0.966 | 0.991 | 0.980 | 0.96 | News | 659 |
| Qwen/Qwen3-30B-A3B-Instruct-2507 | BANK8 | 0.5 | 4 | 14.3 | 0.975 | 0.997 | 0.986 | 0.96 | info_and_fx | 390 |
| Qwen/Qwen3-30B-A3B-Instruct-2507 | SKILL7 | 0.4 | 4 | 11.8 | 0.966 | 0.992 | 0.982 | 0.96 | News | 659 |
| Qwen/Qwen3-30B-A3B-Instruct-2507 | BANK8 | 0.4 | 4 | 11.8 | 0.975 | 0.810 | 0.768 | 0.3 | meta | 390 |
| Qwen/Qwen3.5-35B-A3B | SKILL7 | 1 | 3 | 21.1 | 0.969 | 0.997 | 0.989 | 0.99 | Calendar | 661 |
| Qwen/Qwen3.5-35B-A3B | BANK8 | 1 | 3 | 21.1 | 0.973 | 0.997 | 0.986 | 0.96 | info_and_fx | 389 |
| Qwen/Qwen3.5-35B-A3B | SKILL7 | 0.75 | 3 | 16.5 | 0.969 | 0.995 | 0.987 | 0.96 | News | 661 |
| Qwen/Qwen3.5-35B-A3B | BANK8 | 0.75 | 3 | 16.5 | 0.973 | 0.995 | 0.981 | 0.95 | banking | 389 |
| Qwen/Qwen3.5-35B-A3B | SKILL7 | 0.5 | 3 | 11.9 | 0.969 | 0.994 | 0.985 | 0.96 | News | 661 |
| Qwen/Qwen3.5-35B-A3B | BANK8 | 0.5 | 3 | 11.9 | 0.973 | 0.995 | 0.981 | 0.95 | banking | 389 |
| Qwen/Qwen3.5-35B-A3B | SKILL7 | 0.4 | 3 | 10.1 | 0.969 | 0.991 | 0.980 | 0.96 | News | 661 |
| Qwen/Qwen3.5-35B-A3B | BANK8 | 0.4 | 3 | 10.1 | 0.973 | 0.997 | 0.986 | 0.95 | card_virtual_wallet | 389 |
| Qwen/Qwen3.5-35B-A3B | SKILL7 | 0.3 | 3 | 8.3 | 0.969 | 0.992 | 0.982 | 0.96 | News | 661 |
| Qwen/Qwen3.5-35B-A3B | BANK8 | 0.3 | 3 | 8.3 | 0.973 | 0.961 | 0.937 | 0.5 | small_talk | 389 |
| Qwen/Qwen3.5-35B-A3B | SKILL7 | 0.5 | 4 | 14.9 | 0.969 | 0.997 | 0.989 | 0.96 | News | 661 |
| Qwen/Qwen3.5-35B-A3B | BANK8 | 0.5 | 4 | 14.9 | 0.973 | 0.997 | 0.986 | 0.95 | card_virtual_wallet | 389 |
| Qwen/Qwen3.5-35B-A3B | SKILL7 | 0.4 | 4 | 12.4 | 0.969 | 0.995 | 0.987 | 0.96 | News | 661 |
| Qwen/Qwen3.5-35B-A3B | BANK8 | 0.4 | 4 | 12.4 | 0.973 | 1.000 | 0.990 | 1.0 | account_identity | 389 |
| openai/gpt-oss-120b | SKILL7 | 1 | 3 | 20.0 | 0.846 | 0.132 | 0.107 | 0.0 | Calendar | 577 |
| openai/gpt-oss-120b | BANK8 | 1 | 3 | 20.0 | 0.973 | 0.992 | 0.978 | 0.9 | card_virtual_wallet | 389 |
| openai/gpt-oss-120b | SKILL7 | 0.5 | 3 | 10.4 | 0.846 | 0.055 | 0.040 | 0.01 | Transport | 577 |
| openai/gpt-oss-120b | BANK8 | 0.5 | 3 | 10.4 | 0.973 | 0.905 | 0.872 | 0.19 | small_talk | 389 |
| openai/gpt-oss-120b | SKILL7 | 0.4 | 3 | 8.4 | 0.846 | 0.017 | 0.009 | 0.0 | Calendar | 577 |
| openai/gpt-oss-120b | BANK8 | 0.4 | 3 | 8.4 | 0.973 | 0.781 | 0.738 | 0.35 | cash | 389 |
| openai/gpt-oss-120b | SKILL7 | 0.3 | 3 | 6.5 | 0.846 | 0.009 | 0.004 | 0.0 | Calendar | 577 |
| openai/gpt-oss-120b | BANK8 | 0.3 | 3 | 6.5 | 0.973 | 0.514 | 0.465 | 0.09 | info_and_fx | 389 |
| openai/gpt-oss-120b | SKILL7 | 0.4 | 4 | 10.9 | 0.846 | 0.333 | 0.296 | 0.11 | Transport | 577 |
| openai/gpt-oss-120b | BANK8 | 0.4 | 4 | 10.9 | 0.973 | 0.486 | 0.437 | 0.0 | small_talk | 389 |
| openai/gpt-oss-20b | SKILL7 | 1 | 3 | 21.4 | 0.950 | 0.994 | 0.984 | 0.98 | Music | 648 |
| openai/gpt-oss-20b | BANK8 | 1 | 3 | 21.4 | 0.960 | 0.995 | 0.981 | 0.96 | work | 384 |
| openai/gpt-oss-20b | SKILL7 | 0.75 | 3 | 16.9 | 0.950 | 0.985 | 0.972 | 0.94 | Music | 648 |
| openai/gpt-oss-20b | BANK8 | 0.75 | 3 | 16.9 | 0.960 | 0.979 | 0.959 | 0.91 | info_and_fx | 384 |
| openai/gpt-oss-20b | SKILL7 | 0.5 | 3 | 12.4 | 0.950 | 0.866 | 0.837 | 0.45 | Music | 648 |
| openai/gpt-oss-20b | BANK8 | 0.5 | 3 | 12.4 | 0.960 | 0.956 | 0.930 | 0.78 | work | 384 |
| openai/gpt-oss-20b | SKILL7 | 0.4 | 3 | 10.7 | 0.950 | 0.773 | 0.739 | 0.49 | Music | 648 |
| openai/gpt-oss-20b | BANK8 | 0.4 | 3 | 10.7 | 0.960 | 0.966 | 0.943 | 0.76 | banking | 384 |
| openai/gpt-oss-20b | SKILL7 | 0.3 | 3 | 9.0 | 0.950 | 0.508 | 0.469 | 0.16 | Calendar | 648 |
| openai/gpt-oss-20b | BANK8 | 0.3 | 3 | 9.0 | 0.960 | 0.945 | 0.918 | 0.8 | travel | 384 |
| openai/gpt-oss-20b | SKILL7 | 0.5 | 4 | 15.2 | 0.950 | 0.900 | 0.874 | 0.63 | Music | 648 |
| openai/gpt-oss-20b | BANK8 | 0.5 | 4 | 15.2 | 0.960 | 0.966 | 0.943 | 0.87 | work | 384 |
| openai/gpt-oss-20b | SKILL7 | 0.4 | 4 | 13.0 | 0.950 | 0.853 | 0.824 | 0.46 | Music | 648 |
| openai/gpt-oss-20b | BANK8 | 0.4 | 4 | 13.0 | 0.960 | 0.961 | 0.937 | 0.89 | card_virtual_wallet | 384 |

Not ranked / not trusted: Qwen3.5-122B-A10B (256 experts, 48 layers, 244 GB bf16; routing concentration measured: 90% of SKILL7 mass in 29% of experts, 41% for BANK8) returned 0.0 retention even at keep-100% 3-bit while base is .968/.978 (35B sibling is fine), so the quantise/prune probe failed on this model (harness issue, run OOM'd after 3 configs; raw rows in results/invalid/). Hunyuan-A13B and Mixtral-8x7B were not run (time; Hunyuan needs custom code under transformers 5). gpt-oss: the first run scored SKILL7 at 1-13% because quantised gpt-oss often fails to emit its end token after a correct label and continues "assistantanalysis..."; a re-score run (label before "assistant") is reported in the _r2 rows when present.

Reading: concentration = fraction of experts (layer mean) needed to cover 90/95/99% of the task's routing mass; the lower, the more prunable. Fine-grained models with many experts (Qwen3.5-35B-A3B 256, Qwen3-30B-A3B 128) concentrate most (90% of mass in 27-34% of experts) and survive keeping 30-50%; 64-expert models (OLMoE, DS-V2-Lite, ERNIE) need 56-63%+ and collapse below keep 50%. Cross-task overlap (other task's mass inside this task's top-50%) is 0.73-0.94, so a SKILL7-pruned model keeps most of BANK8's routing mass but BANK8 needs more experts than SKILL7 (more labels, finer boundaries).

<!-- MOE-CENSUS-B END -->
