# BitNet-b1.58-2B-4T packed ternary artifact (v2/lowbit)

## Base definition (read this first)
- Model: `microsoft/bitnet-b1.58-2B-4T` (MIT). Natively trained ternary (W1.58A8) by its authors. 2,412,820,690 logical params (> 0.5B).
- **Denominator = fp16/bf16 master bytes = 2 x 2,412,820,690 = 4,825,641,380** (HF `microsoft/bitnet-b1.58-2B-4T-bf16` is 4,825,679,400 B on disk incl. header). 10% budget = 482,564,138 B.
- Two designated forms: (a) bf16 master `-bf16` (the training weights, plain bf16 forward), (b) packed release `microsoft/bitnet-b1.58-2B-4T` (ternary weights + per-tensor scale, int8 activation quant forward; this is the form the authors say to use for inference; the file itself is 1.18 GB). Dev scores: MMLU (n=1987) (a) .5229 (b) .5264; MDC-6 (n=2009) (a) .8532 (b) .8581. Both about equal, so "match the base" holds against either; the artifact's weights are exactly (b)'s ternary tensors (bit-exact round trip checked).
- Rules gray area (be honest in the issue): the "base" ternary forward IS the model, so the 10x is not a post-hoc compression of a dense fp16 model; it is a ternary-native model whose published fp16 master is 20x bigger than its released inference weights. A challenger could argue the base should be the packed 1.18 GB release (then ratio vs that would be 0.38, fail). Our reading: README measures against the model's fp16 size and says "start from one trained model"; the trained model's weights are the bf16 master. We cannot resolve this; the organizers decide. No weight training is involved; the only data-dependent steps are vocab selection (non-locked MMLU pool/calib + MDC pool/calib token counts) and nothing else.

## Artifact format (all files counted)
| file | bytes |
|---|---|
| config.xz | 480 |
| emb.bin | 53,870,080 |
| emb_min.bin | 3,366,880 |
| emb_scale.bin | 3,366,880 |
| head_letters.bin | 71,680 |
| keep_ids.bin | 168,344 |
| meta.xz | 5,020 |
| predict.py | 5,719 |
| small.bin | 881,060 |
| ternary.bin | 392,849,272 |
| tokenizer.xz | 1,592,184 |

- `ternary.bin`: for each of 210 linear matrices, quantized 3-symbol probability table (mode g=global / r=per-row / c=per-column, chosen by exact cost) + constriction rANS/ANS stream. 1.508 bits/weight incl. tables. Decoding reproduces the original packed uint8 tensors bit-exactly (verified).
- `emb.bin/emb_scale.bin/emb_min.bin`: kept-vocab rows (42,086 of 128,256 tokens: top-42,000 by count in non-locked text, all single-char tokens, specials, letter tokens) as 4-bit asymmetric RTN, group 64, fp16 scale+min. `head_letters.bin`: 14 letter-token rows in bf16 (the tied head is only used for letter logits). `keep_ids.bin`: kept ids. OOV tokens at inference are greedily re-split into kept sub-strings (0.69% of tokens).
- `small.bin`: all norms + 7x30 per-tensor weight scales, bf16. `tokenizer.xz`, `config.xz`, `meta.xz`: xz-compressed full tokenizer files, config, index tables. `predict.py`: the code.
- Verified smaller variant: V=32,768 -> 442,849,479 B (ratio 0.09177); chosen: V=42,000 -> **456,177,599 B = 0.09453 of 4,825,641,380** (size_check.py, 11 files, within budget).
- Artifact dir lives in Modal Volume `lowbit-data:/art/bitnet2b_v42k_e4/` (+ predict.py from `v2/lowbit/artifact_code/`); too large for git. Build: `build_artifact.py`; test: `dev_harness_test.py`; receipts `results/*.json*`.

## Evidence (DEV only; locked files never touched)
predict.py rebuilt the model from the artifact files only (no network: socket blocked, hf_cache_files=0, no base weights), through the harness contract:
| artifact | bytes ratio | MMLU dev acc (agree w/ packed base) | MDC dev acc (agree) |
|---|---|---|---|
| base (b) packed ternary | - | .5264 | .8581 |
| base (a) bf16 master | - | .5229 | .8532 |
| V32k e4 | 0.09177 | .5209 (.901) | .8512 (.982) |
| V42k e4 (chosen) | 0.09453 | .5259 (.919) | .8522 (.988) |
| V42k trim only, fp16 emb (ablation) | - | .5239 (.964) | .8556 (.992) |
Embedding bits 5/6/8 at V32k gave .5204-.5219 / .8532-.8552: embedding bit-width is not the lever; vocab trimming plus bf16 noise is. Differences vs base are within sampling noise (SE ~1.1 pt MMLU, ~0.8 pt MDC); paired non-inferiority at 1 pt margin should be run with the harness on the real stage (discordance: MDC ~1.2%, MMLU ~8% argmax-disagreement).

## Other native-ternary models checked (dev, same harness)
Falcon3-7B-1.58 MMLU .417, Falcon3-10B-1.58 .358, Falcon-E-3B-Instruct .561 (artifact ratio 0.0997, tight), Falcon-E-1B .515 (ratio 0.1009, over). Falcon-E dev-verification was queued behind Modal's GPU cap. TriLM etc. not evaluated. BitNet-2B is the cleanest fit.

## Not done / limits
Feedback memory (rule 5) is not part of this artifact: predict.py accepts a JSON {id: letter} override memory, but no learning mechanism is built here. Harness `candidate_view`/`worker` are MMLU-only; predict.py also handles MDC rows (inst/text/names) so a MDC harness variant can pass them through. Decode takes ~100 s on load (single-thread ANS).
