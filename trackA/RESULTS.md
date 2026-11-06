# Track A: compression-only (no training). Base 109.48M params, 437.9 MB fp32.
Final: `python trackA/final.py` -> trackA/model.bin = 42,728,150 B = 42.73 MB = 9.76% of fp32 (budget 43.8 MB). Params kept 99.69% (precision reduction, not removal); 3.13 bits/param effective.
Format: bit-packed ints + fp16 scale/zero per group; biases/LN/pos/classifier fp16; zlib vocab; reload reproduces fake-quant (max logit diff 0.003, identical acc).
Config Q: GPTQ (calib dev[:1000], MSE clip search), q,k 2-bit; v,a,o 3-bit; intermediate 4-bit L0-5, 3-bit L6-11; g128 (o: g256); embeddings full vocab: train-seen tokens 3-bit, other tokens 2-bit (g128); pooler RTN 6-bit.

Final model: dev2 acc 0.9070 (base 0.9300), agree 0.9663; dev2_hard acc 0.4920 (base 0.4847), agree 0.6860; old dev acc 0.9820 (base 0.9880), agree 0.9920.

## Configs (old dev = memorized train data, saturated; dev2 numbers where measured; bytes = quantized tensors only)
| config | quant MB | old dev acc/agree | dev2 acc/agree | hard agree |
|---|---|---|---|---|
| train-vocab-pruned fp32 (11575 tok) | - | .988/.999 | .904/.961 | .776 |
| RTN enc8 | - | .9885/.9985 | .904/.961 | .771 |
| RTN enc4 g128 emb4 (pruned vocab) | ~45 | .988/.996 | .897/.951 | .683 |
| RTN enc3 g128 emb4 | ~39 | .974/.982 | .850/.895 | .591 |
| GPTQ enc4 g128 emb4 (pruned) | 50.3 | .9885/.9955 | .904/.958 | .721 |
| GPTQ enc3 g128 emb4 (pruned) | 39.7 | .982/.992 | .891/.939 | .635 |
| GPTQ enc3 g64 emb4 (pruned) | 42.4 | .985/.993 | .888/.943 | .647 |
| GPTQ enc2 g64 / g32 | 31.7/37.1 | .9415/.9605 | n/a | n/a |
| FFN neuron prune 10/20/30% fp32 | - | agree .996/.988/.980 | n/a | n/a |
| mixed (q2k2 v4 i4 o3 etc., pruned vocab, ~42MB) | ~42 | .984-.9865 / .993-.9945 | n/a | n/a |
| full-vocab embedding only, RTN 4/3/2-bit (enc fp32) | 12.5/9.5/6.6 | - | agree .995/.991/.970 | .946/.887/.758 |
| embedding SVD r128/192 (full vocab) | - | - | agree .57/.61 (unusable) | - |
| P: GPTQ enc3 uniform + full vocab emb 3/2 | 42.7 | .985/.995 | .905/.963 | .679 |
| **Q (final)** | **42.7** | .982/.992 | **.907/.966** | .686 |

Findings: old dev is useless (base memorized it). Vocab pruning to train tokens alone costs ~4pt dev2 agreement (unseen words -> UNK/split); keeping full vocab with 2-bit rows for non-train tokens fixes most of it. Encoder at ~3 bits costs ~2pt dev2 agreement; 4-bit uniform does not fit 10% (45-50 MB). q/k are least sensitive (2-bit ok), intermediate.dense most. 2-bit encoder collapses. Layer drop / FFN pruning worse than quantization per byte. Best achievable here is ~96.6% agreement on dev2 at 9.76%, not "match within 0.3pt" on dev2 (acc -2.3pt).
Files: lib.py gq.py exp.py (helpers), final.py (reproducible), sens.json, logs.
