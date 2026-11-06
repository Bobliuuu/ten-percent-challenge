# MDC-6 packed artifact (Qwen2.5-7B-Instruct, ~9.7% of fp16)

Built by `push.py` (quantize, `cfg.pack` captures integer indices) -> `pack_artifact.py` (entropy-code, write files) -> `artifact_src/predict.py` (loader, shipped inside the artifact dir). `verify_artifact.py` is NOT shipped; it reloads the artifact and compares logits with the run.

## Method (no weight training)
Per linear layer of the decoder (q,k,v,o,gate,up,down): block-128 Hadamard rotation with random +-1 signs on the input dimension, then sequential Hessian-aware GPTQ (calibration = MDC-6 TRAIN-split `calib.jsonl` prompts, 384x512 tokens) onto an entropy-constrained scalar quantizer: per group of 128 weights a step `s = c * rms(group)` (c=1.15), index = sign*floor(|w/s|+0.5-dz) with dead zone dz=0.15, clipped to +-7 (15 symbols). Step stored as an 8-bit code (`s=2^((code-200)/16)`). Indices and scale codes are rANS-coded with a per-module static order-0 model (constriction). Embedding: vocabulary trimmed to the 24000 tokens most frequent in MDC-6 pool+calib text (dev text NOT used) plus single chars/specials; rows 4-bit (per-64 fp16 scale); OOV tokens are greedily re-tokenized into kept tokens. lm_head: only the 14 rows for " A".." N" (label letters), bf16. Norms/biases bf16 (xz). Per-task additive logit bias (<=14 floats/task) fit on calib rows (train split); applied to both base and candidate when reported as `*_b`.

## Files (all counted by size_check)
| file | content |
|---|---|
| body.bin | uint32 words of rANS streams: for each of 196 modules, weight-index stream then scale-code stream |
| index.json.xz | per-module R,C,g,K, symbol count tables (15 + 256 ints), stream offsets/lengths, kept-token gap list, misc tensor index, bias, orig layer idx, letter ids |
| signs.bin | packed sign bits (random-sign diagonal) per rotation group |
| emb.bin | 4-bit packed embedding rows (kept tokens) + fp16 scales |
| misc.bin.xz | bf16 norms/biases + 14 lm_head rows, lzma |
| tok.tar.xz | tokenizer.json + tokenizer_config.json (Qwen2.5, full BPE) |
| config.json | Qwen2 config |
| predict.py | loader: `load(art_dir, memory=None, use_bias=True) -> f(rows)`; `f.logits(rows)`. rows = {task,text,names,inst}. Decode (constriction), dequantize (`(idx-7)*s`, inverse Hadamard * signs, bf16), build `AutoModel.from_config` Qwen2 body, fill weights, score last-token hidden @ 14 label rows |

Prompt (`fmt`): `{inst}\n\nText: {text}\n\nOptions:\nA. {name0}\n...\nAnswer:`; prediction = argmax over label-letter logits (+ per-task bias).
Requirements: torch, transformers (4.51.3 tested), constriction, numpy, a GPU (load ~3 min: ANS decode of 6.5B symbols + dequant).

## Verified numbers (dev n=2009, run on L40S) -- see results/
c=1.15, dz=0.15, vocab 24000, ncal 384: artifact size 1,482,976,475 B vs budget 1,523,123,302 (ratio 9.736%, `v2/harness/size_check.py` PASS). Reloaded artifact reproduces the run logits exactly (agree 1.0, max |dlogit| 0). Raw acc 0.8731 vs bf16 base 0.8721; with per-task bias 0.8820 vs base+bias 0.8890 (-0.7pt); agreement with the bf16 base 0.954.
Locked exam NOT touched.
