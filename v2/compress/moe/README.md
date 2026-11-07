# Track C-MOE (Qwen3-30B-A3B-Instruct-2507): first-pass status (sweep still running)

Budget: 9.6% of 61,066,575,656 B = 5.86e9 B. All numbers simulated (fake-quant), no training. Bytes = histogram entropy of GPTQ ints x1.005 + fp16 scales (g=128) + kept router rows/norms + tokenizer/config files (16.0 MB) + 150 KB code slack.
REAL rANS CHECK (constriction, layers 10 and 30, k=64 run): coded size / entropy = 1.000002 (2.4190 bpw vs 2.4190), decode verified lossless. So byte counts are trustworthy up to the 0.5% overhead already added.

## Noise floor (base bf16 vs itself, different batch composition) -- limits what ANY candidate can score
em8 0.965 / em64 0.715 / full-sequence 0.645 (set A); 0.90 / 0.725 / 0.675 (shifted set B). A greedy-transcript metric cannot reach 98% even for the base model on 64 tokens; only short-output / item-level scoring can.

## Results so far (results/table.md; raw JSON in results/vol/)
| config | %fp16 | em8 A | em16 A | em64 A | top1 A | KL A | top1 B (shifted eval) | item retention agnews / sst2 / tool-call |
|---|---|---|---|---|---|---|---|---|
| quant-only, all 128 experts, 4-bit (ceiling) | 26.4 | 0.917 | 0.830 | 0.557 | 0.964 | 0.018 | 0.986 | 1.00 / 1.00 / 1.00 |
| prune-only k=64, bf16 everything | 52.5 | 0.195 | 0.145 | 0.107 | 0.710 | 1.161 | 0.833 | 0.30 / 1.00 / 0.64 |
| k=64, ~2.4 bpw experts, attn 5b, emb 6b (fits 9.39%) | 9.39 | 0.217 | 0.168 | 0.125 | 0.692 | 1.260 | 0.806 | 0.42 / 0.83 / 0.68 |
(sst2 n=6, agnews n=47, tool-call n=90 base-correct 22: tiny samples, wide CIs.)

## Findings
- Pruning is the damage. Dropping half the experts by router frequency (no retraining) already costs top-1 0.71 and KL 1.16 with bf16 weights; quantizing the rest to 2.4 bpw adds little (KL 1.26). Same on the ladder (results/vol/ladder_*.json, RTN probe on OLMoE-1B-7B and Qwen1.5-MoE-A2.7B): prune-only to 50% gives top-1 0.51 / 0.38, 75% gives 0.76 / 0.70; 4-bit with all experts gives 0.91 / 0.90 (RTN). Same shape as Qwen3-30B-A3B, so the ladder predicts k<=64 cannot be faithful.
- Quantization alone is fine near 4 bits (top-1 0.964, KL 0.018, 100% item retention on the labelled sets) but 4 bit is 26% of fp16, 2.7x over budget. The 9.6% budget needs ~1.1-1.2 bpw if all experts are kept; k=128 fill result pending.
- Verdict so far: >=98% retention at <=9.6% is NOT reached; not reachable for k=64 (pruning loss alone breaks it). Only pending hope: k=96-128 at 1.1-1.6 bpw (running). KL for 98% retention from the first KL->retention fit (9 points, exp(-a*KL), a=1.1): KL ~ 0.018, i.e. about the 4-bit-all-experts quality, which costs 26% of fp16.
- Bug found and fixed along the way: integer clamp +-31 destroyed 5-6 bit tensors (embeddings); now +-2047. Earlier numbers with the clamp are in results/buggy_LM31 and are invalid.

## Still running / queued (Modal app tenpct-moe, id in logs/app_all.id; detached, GPU-limited)
fillA k=128/96/48/32, fillB (calibB, OOD) k=128/96/64/48/32, prune k=96/48/32, quant-only 3 bit, attn/emb ablation (8/8, 4/4), REAP ranking k=64, stride-4 additivity runs. Results land in volume tenpct-v2-hf:moe_res/ after each config. `python collect.py && python analyze.py` rebuilds results/table.md and results/analysis.md (stage-0 ranking, additivity check, KL->retention fit).
Not done: successive-halving kill driver across jobs (stage-0 numbers are logged per run; analyze.py ranks them), predicted-vs-actual for a full run (needs stride runs), per-layer bit allocation.
