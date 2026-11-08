# MODEL_CENSUS (first pass, 2026-11-08; probe results pending)

Source: `model_census/census.py` (HF API, exact safetensors param counts + config.json), list in `model_census/models.py`. fp16 bytes = 2 x params (all-bf16 reference). 10% budget = 0.1 x that. Average bpw at the 10% budget is 1.6 for every model by definition; the real BODY bpw is lower where embeddings and lm_head (cost 2 x V x d params untied) must be kept: e.g. Qwen2.5-0.5B/1.5B (tied 151k vocab) and Llama-3.2-1B/3B (tied 128k) lose 15-30% of the budget to the embedding, 70B+ models lose <3%. So bigger models get a BIGGER bit budget per useful weight (embedding share shrinks) and absolute capacity: 10% of 70B = 14.1 GB, of 235B = 47 GB.
Native low-bit rows: `params` is the packed count of the released repo (not the fp16 reference); their 10% budget refers to the parent fp16 model (BitNet-2B-4T: 2.4B -> 4.8 GB fp16 -> 0.48 GB budget, vs 0.85 GB packed = 17.7%; see NATIVE_SURVEY2.md for the Bonsai family, which are not on the public HF API under these ids). Gated models (Llama, Gemma) were probed through the `unsloth/` mirrors (same weights).

## Static table

## Quantization-friendliness evidence (literature + our own, to be extended by probe results)
- Larger dense models quantize better at equal bpw (our own 7B/14B/32B tables in task_search/results: retention at ~10% rises with size but stays well below 98% for label tasks; 32B best).
- Qwen2.5 has massive-activation outliers in a few channels, Hadamard rotation handles it (our pipeline). Llama-3 is known to be harder to quantize than Llama-2 (low-bit PTQ papers, e.g. "How good are low-bit quantized Llama3 models"); Gemma-2/3 and Mistral-Nemo tolerate int4 well. VPTQ/QuIP#/AQLM report ~1.5-2 bpw on 70B-405B with 85-95% retention of benchmark scores: the best published points are on the biggest models.
- MoE: experts are individually redundant (OLMoE, Qwen3-30B-A3B routing-mass prune works to 96/128); see moe_entry.

## Probe results
See below (filled by `model_census/analyze.py`).
