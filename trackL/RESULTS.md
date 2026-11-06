# trackL: external correction on top of the quantized model (trackA/model.bin, 42.73MB, 9.76%)
Setup: quantized = trackA/model.bin (reloaded). Targets = base predictions (pseudo-labels). Calibration text = trackG pool minus dev2 minus dev2_hard (11,436 unlabeled sentences, imdb/amazon/tweet; base logits only). Eval = dev2 half chosen by md5(text)%2==1 (1500 ex) and dev2_hard (1500). Fits are closed-form / grid / perceptron / kNN only (no gradient-optimizer fits in the reported numbers). SST-2 val/test untouched. Script: exp.py, exp3.py, exp_e.py (exp.py also holds early LBFGS logistic fits, excluded: banned by rule clarification; they gave the same ~+0.6 to +1.0pt anyway). Eval SE ~0.4pt: treat <0.5pt differences as noise.

| method | bytes | agree dev2-ev | agree hard | acc dev2-ev (base .9347) |
|---|---|---|---|---|
| quantized, no correction | 0 | .9683 | .6860 | .9149 |
| (a2) logit-diff bias +0.5 (grid-chosen to max pool agreement) | 2 | **.9782** | .6940 | .9261 |
| (a1) least-squares scale+bias (a=.996,b=.15) | 8 | .9703 | .6887 | .9169 |
| (b) ridge pooled(768)->base diff, lam 30-300, closed form | 1.5KB | .9756-.9769 | .695-.698 | .922-.9235 |
| (b2) rank-64 PCA ridge residual | 98KB | .9776 | .690 | .9241 |
| (b2) rank-4/16 | 6/25KB | .9710/.9730 | .692/.697 | |
| (c) trackB-style perceptron lexicon (wordpiece->int8), best | 7.6KB | .9723 | .688 | .9189 |
| (d) kNN of 188 pool disagreements, sim gate .9 | 145KB | .9604 (worse) | .689 | |
| (d) + 2000 agreeing anchors | 1.7MB | .9697 | .696 | |
| (e) pooler 6bit->fp16 | +0.74MB | .9683 | .6867 | |
| (e) L11 attn fp16 / L11 ffn fp16 | +4.0/+7.7MB | .9683/.9683 | | |
| (e) L11 whole fp16 | +11MB | .9664 | .6853 | |
| (e) L10-11 fp16 / L0 fp16 | +23/+11MB | .9670/.9670 | | |
| (e) full word embedding fp16 (way over) | +38MB | .9710 | .698 | |

Bias robustness (agree vs bias added to quant logit diff): bias 0 / .25 / .5 / .75 / 1.0 -> pool .9831/.9839/.9847/.9843/.9836; dev2-cal half .9643/.967/.969/.967/.9629; dev2-ev .9683/.9716/.9782/.9756/.9716. The quantized model is biased toward class 0 (pred-positive rate .486 vs base .500); +0.5 fixes it (.499). Gain is consistent on all three sets (+0.1 pool, +0.5 cal half, +1.0 ev half, +0.8 hard).

Findings
- Best clean gain: a 2-byte bias (strictly clean, closed-form/grid): dev2 agreement .9683 -> ~.974-.978 (+0.6 to +1.0pt), hard +0.8pt, acc +1.1pt. Everything bigger (ridge on pooled, PCA low-rank, lexicon perceptron, kNN) is no better than the bias within noise; ridge pooled is equal (.977) at 1.5KB; kNN hurts; lexicon is worse (its gains are rare-word specific, and quant errors are diffuse not lexical).
- Mixed precision (e) does NOT help: restoring last layers, pooler, or even all attention/FFN of L0/L10/L11 to fp16 leaves agreement unchanged or lower. Error is spread over all layers and the embedding table (only fp16 embeddings gives +0.3pt for 38MB). Pooler fp16 (+0.74MB, fits headroom) gives 0.
- Residual disagreement (~2.2% after bias, ~31% on hard) is near-boundary flips of a diffuse noise; no cheap external component recovers them. Hard set (|margin|<1.4) is essentially coin-flip for any perturbation; ceiling there is small.
- Byte budget: headroom 1.07MB; bias costs 2-8 bytes; ridge head 1.5KB. All fit.
- Not run: GPTQ re-quantization with different bit allocations (trackA domain); stacking bias + calibration on SST-2 style text (the exam domain; pool is imdb/amazon/tweet, bias optimum might differ for SST-2 sentences. Recommend verifying the bias sign/size on unlabeled SST-2 *train* sentences vs base logits before shipping).
