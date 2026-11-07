# L2 MoE base tool-calling probe (greedy, native tools template, same scorer/600 items)

| model | prec/GPU | bf16 GB | 10% GB | S1 all | S1 n/s/m | S3 all | S3 n/s/m | parse-fail | R8 | wall min | ~$ (excl. download) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| GLM-4.5-Air-FP8 (GLM-XML->json adapter) | fp8 H100:2 | 110.5 | 11.1 | 0.737 | 0.93/0.69/0.59 | 0.743 | 0.85/0.78/0.60 | 0.003 | YES | 21 | 2.8 |
| Qwen3-235B-FP8 | fp8 H100:4 | 235.1 | 23.5 | 0.853 | 1.00/0.80/0.76 | 0.730 | 0.74/0.82/0.63 | 0.003 | YES | 21 | 5.6 |
| gpt-oss-120b | mxfp4 H100:2 | 116.8 | 11.7 | 0.580 | 0.97/0.77/0.00 | 0.467 | 0.74/0.66/0.00 | 0.002 | no | 2 | 0.3 |
Notes
- Greedy, base chat template with native tools=[...], single user message, scorer.py + probe_items.jsonl copied verbatim from tournament/T1 (600 items: 300 S1-dev + 300 S3-dev). vllm 0.11.0, max_model_len 20000. Reference (SUMMARY.md): Qwen3-30B-A3B S1 .810 / S3 .650; Qwen3-14B .840/.697.
- bf16 GB = HF safetensors total params x 2 (gpt-oss-120b is natively mxfp4: 116.8 GB is actual checkpoint size; 10% budget column is 0.1 x that). 
- Precision used: Qwen3-235B = official FP8; GLM-4.5-Air = official FP8; gpt-oss-120b = native mxfp4 (reasoning_effort low, max_tokens 3000).
- GLM-4.5-Air: native output is XML (<tool_call>name<arg_key>..</arg_key><arg_value>..</arg_value></tool_call>) which scorer.py does not parse (raw: S1 .310 / S3 .283, parse-fail .695). Table row uses a mechanical XML->Qwen-JSON converter (summarize.py:glm2qwen); scorer untouched.
- gpt-oss-120b multi = 0.00: harmony stops after the first call (<|call|>), so it never emits both calls in one turn; nocall/single are fine. Per scorer, it fails R8 on that basis.
- Cost = GPUs x $3.95/h(H100) x wall (incl. weight download/load; vllm load); the first gpt-oss H100:1 attempt OOM'd (65 GB weights), re-run on 2xH100 (row says H100:2).
- Not run (queued/blocked): Qwen3-Next-80B and Mixtral-8x22B were submitted but stuck "waiting to be scheduled: H100 unavailable on starter plan, at 10 gpus" (shared workspace). Qwen3-Coder-480B-FP8, GLM-4.5/4.6-FP8 (8xH100), Llama-4 (gated, no HF token in Modal) not launched.
