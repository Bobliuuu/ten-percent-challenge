# GIANT_MOE: giant MoE at ~1.5-1.6 bits/param (2026-11-08). Verdict: NO-GO (no GPU run made; cost gate)

## 1. Byte math (v2/giant_moe/bytes.py, bytes_table.txt)
fp16 baseline = 2 B x total params; 10% = 1.6 bits/param over ALL params. Non-expert tensors (attention, shared experts, dense layers, router, embed, head, norms, tokenizer, code) are charged at 4.5/6/8 bits incl. scale; the rest of the budget is what the routed experts get, all-in (scales, codebooks, entropy coder included):

| model | fp16 GB | 10% GB | experts share | expert bpw if non-expert at 4.5 / 6 / 8 bits |
|---|---|---|---|---|
| DeepSeek-V3/R1 671B | 1342 | 134.2 | 97.5% | 1.52 / 1.48 / 1.43 |
| Kimi-K2 1.03T | 2052 | 205 | 98.9% | 1.57 / 1.55 / 1.53 |
| Qwen3-235B-A22B | 470 | 47.0 | 96.6% | 1.50 / 1.44 / 1.37 |
| Qwen3-Coder-480B | 960 | 96 | 97.5% | 1.53 / 1.49 / 1.43 |
| GLM-4.5 355B | 716 | 71.6 | 93.8% | 1.41 / 1.31 / 1.18 |
| Llama-4 Maverick | 803 | 80.3 | 96.3% | 1.49 / 1.43 / 1.35 |
| Llama-4 Scout | 217 | 21.7 | 89.0% | 1.24 / 1.05 / 0.81 (dead) |
| gpt-oss-120b | 234 (strict 2 B/param; its MXFP4 checkpoint is 65 GB = 27.8%) | 23.4 | 98.2% | 1.55 / 1.52 / 1.48 (needs re-quant from 4.25 b; also its own base fails tool-multi, R8) |

Every candidate needs experts at about 1.4-1.55 bits all-in. That is the SAME operating point as the Llama-405B dense run (body 1.531 bpw, L3_405B), which scored 99.7% on easy C_tasks INJ but 90.2% [88.5,91.6] on real INJ dev (injection class 78%). MoE only helps if expert redundancy beats dense redundancy at identical bits; there is no measurement of that yet.

## 2. Community "1.58-bit" does not fit 10% (checked against HF file listings, 2026-11-08)
Unsloth GGUF sizes vs 10% of BF16 (BF16 GGUF = 1342.3 GB for DeepSeek):
- DeepSeek-R1 UD-IQ1_S 140.2 GB = 10.44% (closest; the old 131 GB file is not in the current listing); R1-0528 TQ1_0 161.6 (12.0%); V3.1 TQ1_0 170.5 (12.7%); V3-0324 IQ1_S 186.2 (13.9%).
- Kimi-K2 TQ1_0 243.6 GB = 11.9%. GLM-4.5 TQ1_0 84.3 = 11.8%, IQ1_S 97 = 13.5%. Qwen3-Coder-480B IQ1_M 149.7 = 15.6%. Qwen3-235B-2507: smallest listed 85.7 GB (18%); no 1-bit file.
- None is <=10%, and those files are also imatrix-calibrated by a third party and loaded by llama.cpp, so size accounting would add the runtime. They are an upper bound on what 1.58-bit dynamic quant gives, not a submission.

## 3. Mixed-precision design (what we would build)
experts ~1.45-1.5 b: Hadamard-rotated, GPTQ with per-expert Hessians, 3-level (ternary-ish) ECSQ grids, entropy-coded (the L3 recipe, 11 b/group scale), with per-layer Lagrangian allocation moving bits to sensitive layers (first/last, down_proj); attention, shared experts, dense layers, router, embed/head at 4-6 b (costs 0.03-0.08 bpw of expert budget). Hard problem unique to MoE: each expert sees ~8/128 (Qwen3) or 8/256 (DeepSeek) of calibration tokens, so 128x512 tokens leaves ~4k tokens per expert for a 4096-dim gate/up Hessian (rank-deficient); you need >=1M calibration tokens (task data plus general), i.e. 10-20x the 405B calibration compute. Expert pruning (REAP) is legal but L2_moe shows it is the worst lever: 235B at 25% prune + 3 b still ~0.957 (tool calls), 37.5% prune ~0.94-0.98; bits and pruning compete for the same 10%, and 1.5 b already uses all of it.

## 4. Existing evidence (no new run)
- L2_moe, Qwen3-235B teacher-forced, easy C_tasks labels, all experts RTN3 (15% of fp16): INJ .989, SENT .993, TOOLSEL .991, FIN .989, ROUTE .978; prune 50% + RTN3 (8.2%): INJ .896. Real-INJ dev was never run on a MoE. Tool-call retention best fitting <=9.6% points 0.93-0.94.
- v2/moe Qwen3-30B-A3B: 70% experts at 2 b GPTQ at 10%: MMLU -7.5 pt; 1-bit collapses (0.45).
- L3 405B dense at 1.53 bpw: real INJ 90.2%.
- Dense 32B at 10%: best realistic 0.957-0.961 (TASK_SEARCH).
Expectation: a giant MoE at 1.5 b lands between the 405B (0.90 real INJ) and the 32B (0.957); reaching 98% overall and 95% per cell would need an unprecedented jump. Easy-data teacher-forced numbers (0.99) have already been shown not to transfer.

## 5. Probe plan and cost gate (why nothing was run)
Cheapest informative probe = DeepSeek-R1 UD-IQ1_S (the only community file within 0.5 pt of 10%) vs DeepSeek base, real greedy generation on INJ-real dev (tournament/INJ s1+s3, ~600 items), retention on base-correct with Wilson bounds. Base fp8 (671 GB) needs 8xH200 (Vast offers seen: 8xH200 ~$47/h, 4xH200 ~$29/h, 2xH100 SXM ~$4.5/h) for ~2.5 h incl. 700 GB download and load = ~$118; candidate on 2xH100 llama.cpp ~$11; total ~$130 with zero retries. Vast instances failed to load on 7 of 10 prior attempts, so a single retry exceeds the $150 cap. Also the candidate is 10.44% and R1 is a reasoning model (not the base's native short-label output), so even a pass would not be a submittable entry. Own-quantizer alternative on Qwen3-235B (4xL40S ~3-4 h GPTQ with >=1M calibration tokens, plus fp8 base on 4xH200 1.5 h at $29/h) is ~$70-110 and 1-2 days engineering with expected result at or below the 405B point. Both stopped at the gate as instructed; no instance was created, no spend.

## 6. Verdict
NO-GO for the 98%/95% gate: no candidate fits 10% with off-the-shelf quants; the budget forces experts to ~1.5 b, identical to the 405B dense point that scored 90.2% on real INJ; MoE-specific costs (under-determined per-expert Hessians, no room for pruning) cut against it. Only untested upside: expert redundancy might make real-INJ retention at 1.5 b better than dense 405B. If budget is lifted, run the Qwen3-235B own-quantizer probe with real-INJ dev first; abort if retention < 0.95 on 300 items.
