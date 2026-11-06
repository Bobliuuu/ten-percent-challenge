# trackD: training-free quantization (GPTQ) results
Script: gptq_run.py + lib.py (run: `python gptq_run.py '[{"name":..,"method":"gptq|rtn","bits":3,"g":64,"ebits":6}]'`, ~4 min/config on Mac).
Calibration: first 512 texts of dev2 (trackG). Eval: dev2[512:] (agree2, acc2), dev2_hard (1500, agree_hard), old SST-2 dev (2000). Never touched val/test.
GPTQ: sequential (Hessians from already-quantized prefix), act-order, damp 0.01, asym minmax groups, fp16 scale + b-bit zero, perm cost counted.
Bytes = entropy-coded codes (order-0 Shannon, ideal ANS) + scales/zeros + fp16 rest (pos/type emb, LN, biases, classifier). Word embedding = RTN group 64 at ebits. No vocab/layer pruning (trackA does that).

| config | bytes (entropy) | ratio | acc2 (base .9297) | agree dev2 | agree hard | agree old dev |
|---|---|---|---|---|---|---|
| RTN w4 g128, emb8 | 65.1MB | .149 | .9172 | .9811 | .8140 | .9970 |
| RTN w3 g64, emb8 | 56.8MB | .130 | .9003 | .9586 | .7020 | .9915 |
| GPTQ w4 g128, emb8 | 65.3MB | .149 | .9188 | .9843 | .8600 | .9995 |
| GPTQ w3 g64, emb6 | 51.1MB | .117 | .9136 | .9735 | .7653 | .9955 |
| GPTQ w2 g32, emb4 | 38.7MB | .088 | .8863 | .9381 | .6207 | .9765 |

Findings
- GPTQ beats RTN at equal bits (hard agreement +4.6 pts at w4). Entropy coding saves only ~6-7% (codes near-uniform).
- Old dev (memorized) hugely overstates quality: 99.95% there vs 86% on hard. Nothing here reaches 99.5% on dev2/hard; even w4 is 98.4% / 86%.
- 2-bit fits 43.8MB (38.7MB) but loses 7 pts accuracy; no sub-4-bit config is near-lossless without finetuning.
- Embedding (23.4M params) dominates at low bits; needs vocab pruning (trackA) combined with GPTQ to fit w4 under 43.8MB (w4 encoder ~ 85M*4.3b = 46MB alone, so also needs layer/FFN pruning or w3).
Not done (time): AWQ/SmoothQuant scaling, codebook VQ, SVD+residual, per-layer mixed precision, real ANS coder.
