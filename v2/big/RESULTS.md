# v2/big: Qwen2.5-32B / 72B (bigger bases) -- NO NUMBERS PRODUCED

Status: no accuracy or ratio measured. All runs blocked by compute, then stopped by coordinator.

## What was built
- big.py / big32.py: v2/engine/sweep.py (Hadamard-128 incoherence + sequential GPTQ, L-level grids, entropy-coded byte accounting,
  vocab trim 32768 + 4-bit emb, lm_head dropped to 4 letter rows) adapted for big models: CPU-resident model, layers streamed to ONE GPU
  (pre/post forward hooks), so 72B (145 GB bf16) needs 1 GPU + ~280 GB host RAM. cfg_1.json: base(vocab-trim only)+u3 g128 for 32B and 72B, 32B u4.
- Never executed to completion; the streaming code path is untested beyond model load (a hook-return bug, fixed, was found in the first live run).

## What failed / blockers
1. Modal A100-80GB:2 / :4 (first design): never scheduled (>12 min, 0 tasks); workspace GPU pool saturated by other agents.
2. AWS p4d.24xlarge (8xA100-40GB, quota ok at 128 vCPU): InsufficientInstanceCapacity in all us-east-1 AZs. G quota (16 vCPU) too small for g6e multi-GPU.
3. Modal single A100-80GB + 160/280 GB RAM: first run got a GPU after ~12 min, 72B download ~14 min (weights now cached in Volume hf-cache), then crashed
   on my bug (forward hook returned the module). Relaunches (apps ap-L6BW..., ap-kBzN..., ap-AFSM...) queued 30+ min without a container; stopped at wrap-up.

## Frontier (not measured, expectation only)
At ~1.6 bit/param all-in (0.10 of fp16), 72B has ~5B emb+head params that trim away (vocab 32k, 4 letter head rows) so body gets ~1.6 b/w including
group scales (g128 fp16 scale = 0.125 b). Uniform 3-level GPTQ+Hadamard is the first thing to try; prior 14B result (w3 g128, ratio 0.179, -4.5 pt) suggests
a big base would still lose several points. Unverified.

## To resume
`modal run --detach v2/big/big.py --cfgs v2/big/cfg_1.json --out out_1.json` when A100/H100 capacity is available (72B weights already in Modal Volume hf-cache).
