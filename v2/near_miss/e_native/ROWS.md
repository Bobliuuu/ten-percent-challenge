# Lane e ROWS (native low-bit starts other than Bonsai). All DEV, real greedy generation (HF transformers 4.57, bf16), retention = P(candidate correct | base correct), base = the released model run through the same pipeline.
Start model = released microsoft/bitnet-b1.58-2B-4T (packed ternary, W1.58A8 forward). fp16 reference (R1) = 2 x 2,412,820,690 = 4,825,641,380 B. Ternary body lossless, counted as IDEAL-CODER bytes (best of global/row/col table + 16-bit table entries) = 392,847,765 B (the real rANS file in v2/lowbit is 392,849,272 B); with plain 5-trits/byte it would be 416,808,960 B and every row below would be ~10.40%. Lossy levers: vocab-prefix truncation (rebuilt closed BPE, specials kept) + RTN asymmetric group-64 quantisation of the tied embedding/head (fp16 scale+min counted). No training, no calibration data (prefix truncation needs none). Dev items: T1 600, TOX 705, INJ 695, SKILL7 682 (v2/task_search/EASY dev). BANK8 not run (not in the Modal bundle).
Base accuracy (rule 8): T1 .322, TOX .338 (chance .20), INJ .725 (chance .50), SKILL7 .890. T1/TOX are weak => rule 8 marginal on those, SKILL7/INJ fine.

| method | base | size % of fp16 (bytes, ideal-coder ternary) | retention overall [Wilson lo] | min cell (n>=20) | feedback | rule gaps | weakest rule | notes |
|---|---|---|---|---|---|---|---|---|
| vocab prefix 74,167 + 3-bit emb (g64) | BitNet-2B-4T | 9.899% (477,700,385) | 95.8% [94.7] (n=1542); T1 86.5, TOX 97.5, INJ 96.2, SKILL7 97.7 [96.2] | 61.9% (T1 S1 single, n=21); SKILL7 min 91.5% (massive Music, n=47) | n/a | R2 (<98 overall), R9 (low-bit start base identity) | R2 | best config at <=9.9% |
| prefix 103,368 + 2-bit emb | BitNet-2B-4T | 9.899% | 81.6% [79.6]; SKILL7 96.2, TOX 92.9, INJ 63.3, T1 69.4 | 28.6% (T1 S1 single) | n/a | R2 | R2 | 2-bit embedding hurts INJ/T1 |
| prefix 57,830 + 4-bit emb | BitNet-2B-4T | 9.899% | 47.5% [45.0]; SKILL7 10.9, TOX 95.0 | 2.5% (SKILL7 massive Alarm) | n/a | R2 | R2 | NON-MONOTONIC: output format flips (lists "1. Alarm 2. Weather...") although token count only +1.5%; some dropped token in the prompt template triggers it. Brittle |
| prefix 30,741 + 8-bit emb | BitNet-2B-4T | 9.899% | 41.8% [39.3]; SKILL7 4.6 | 0% (SKILL7 massive Lists) | n/a | R2 | R2 | same brittleness |
| prefix 16,387 + bf16 emb | BitNet-2B-4T | 9.902% (just over 9.9) | 27.5% [25.3]; SKILL7 0.8 | 0% | n/a | R2 | R2 | token count +9..18%: tiny prefix vocab breaks the model |
Prior (v2/lowbit, not re-run): V=42k + 4-bit emb 9.453% with bytes counted via real rANS; T1 retention of 4-bit emb alone .906.
Interpretation: BitNet-2B can be put at <=9.9% (the tied 128k embedding is what costs bytes) with 95.8% item retention on dev at best; best config is brittle (neighbouring vocab sizes collapse) so it is not a robust near-miss. A frequency/task-prompt-aware vocab (bitnet2b_tv run) and Falcon-E-3B results are appended below when they land.

## Falcon-E-3B-Instruct (tiiuae, released ternary, untied 32,768 x 2,048 embedding + head), same method (dev)
fp16 reference 6,107,173,312 B (2 x 3,053,586,656). Ternary body ideal-coder 570,498,093 B (9.34% alone; 583.8 MB = 9.56% at 5/byte), so only ~0.5% of fp16 is left for embedding + head + tokenizer: vocab prefix trimmed hard. Base acc: T1 .308, TOX .308, INJ .676, SKILL7 .862.
| method | size % of fp16 | retention overall [lo] | SKILL7 | T1 | TOX | INJ | min cell | weakest rule |
|---|---|---|---|---|---|---|---|---|
| prefix 26,229 + 2-bit emb & head | ~9.9% (see cands.json) | 62.3% [59.7] | 63.3 | 87.6 | 9.2 | 75.5 | 0% (TOX S3 toxic, n=78) | R2 |
| prefix 18,771 + 3-bit | ~9.9% | 54.6% [52.0] | 58.0 | 83.2 | 22.1 | 54.0 | 0% (SKILL7 massive Music) | R2 |
| prefix 14,616 + 4-bit | ~9.9% | 50.0% [47.4] | 45.9 | 84.9 | 23.5 | 53.6 | 0% | R2 |
| prefix 7,751 + 8-bit | ~9.9% | 61.8% [59.3] | 52.9 | 74.6 | 81.6 | 58.9 | 0% | R2 |
| prefix 4,118 + bf16 | ~9.9% | 53.8% [51.3] | 51.9 | 73.0 | 67.7 | 42.3 | 0% | R2 |
Verdict: Falcon-E-3B reaches <=9.9% only with a 4k-26k token vocabulary and the retention collapses (50-62%), far worse than BitNet-2B; not a viable near-miss. (Prefix truncation of this 32k tokenizer is clearly not closed/benign; a frequency-based task vocab was not tried.)
