# TPQ1 - real entropy-coded artifact for the 10% challenge (model-agnostic)

One directory = the whole submission. Every byte counts (harness/v3 `checks.size_check` sums all files).

```
ART/
  model.tpq             all weights: rANS-coded integer codes + coded scales + rotation sign bits + raw (norm/bias) tensors   (the only big file)
  tensor_manifest.json  {tensor_name: [safetensors dtype, shape]} of the ORIGINAL architecture (declared; verified by decode-and-compare, see verify.py)
  config.json generation_config.json tokenizer.json tokenizer_config.json    (verbatim from the base; vocab.json/merges.txt dropped - tokenizer.json is self-sufficient)
  answer.py             `answer` CLI (loader inlined, self-contained, counted in bytes)
  feedback.py           `feedback` CLI (loader inlined)
```
No extra heads / adapters / side memory. `model.tpq` had to be added to `checks.ALLOWED_PATTERNS` (one-word harness change; nothing else in harness/v3 was modified).

## model.tpq layout
`MAGIC "TPQ1\0\0\0\0"` | data region | `zlib(JSON header)` | `u64 header_len` | `MAGIC`   (footer-indexed so the writer streams; the header is read from the end).
Header JSON `{version, hb:128, tensors:{name:...}, signs:{key:{off,len,n}}, meta}`.
* `kind:"raw"` tensor: `{dtype, shape, off, len}` little-endian bytes (norms, biases, anything not quantised).
* `kind:"alias"` tensor `{of, dtype, shape}`: a tied checkpoint that stores `lm_head.weight` as a byte-identical copy of `embed_tokens` (Qwen3-0.6B does) is declared in the manifest but stored once (the quantiser only aliases after `torch.equal`). The harness' fp16 baseline still counts both copies (it sums safetensors headers).
* `kind:"q"` tensor: `{shape [R,C], L levels, g group size, mbits, rot sign key, rpc rows/chunk, off, lens[3*nchunks], probs[L], eprobs[32]}`.
  The tensor is `ceil(R/rpc)` independent chunks (parallel decode, row-granular in-place edit). Chunk = `codes rANS words | scale-exponent rANS words | packed mantissa bits`; lens are the three byte lengths.
  * codes: uint8 level index `qi in [0,L)`, row-major `[rows, C]`, coded with `constriction` `AnsCoder` (stack/rANS, 24-bit probs) with ONE static `Categorical(probs[L], perfect=False)` per tensor (table = L float32, stored in the header and counted).
  * scales: one fp16 per (row, group of g input columns), positive. fp16 exponent field (5 bit) coded with a static 32-symbol table `eprobs`; the top `mbits` of the 10 mantissa bits stored raw-packed. **Lossless iff the quantiser rounded scales to `mbits` mantissa bits** (the quantiser does; the writer asserts it). mbits=8 -> about 8.2 bits/scale, mbits=10 -> exact fp16 at about 13 bits/scale.
  * rotation: weight = `unrot((qi - (L-1)/2) * scale)`, `unrot(M) = (M.view(R,C/128,128) @ H128) * sg` with H128 = Sylvester Hadamard / sqrt(128) (symmetric, orthonormal) and `sg` a +-1 vector of length C shared by tensors reading the same input (q/k/v; gate/up), stored bit-packed under `signs[key]` (C bits, the "seed" is the explicit sign vector: robust to RNG changes). Dequant is float32 (TF32 off) -> bf16, row chunks of 4096 are bit-identical to whole-matrix `unrot`.
* Everything is deterministic: decode needs only `constriction==0.5.0` (pin: the float->integer probability quantisation is part of the format) and torch float32 matmul.

## Commands
```
python quantize.py --model BASE_DIR --out ART --levels attn=3,o=3,mlp=3,down=3,emb=4,head=4 [--gptq --calib s1_calib.jsonl] --sim-dir SIM   # writes model.tpq + manifest + config, ART.quant_report.json, ART.sim_hashes.json, SIM/sim-*.safetensors (independent simulated bf16 weights)
python writer.py assemble --art ART                       # inline loader.py into ART/answer.py and ART/feedback.py
python verify.py --art ART --sim-dir SIM --base-arch ART.base_arch.json   # torch.equal per tensor vs simulation; exact bytes vs estimate; harness checks
python ART/answer.py   --artifact ART --in items.jsonl|view.json --out out.json [--dump-prompts p.json] [--bs 16] [--max-new-tokens 512]
python ART/feedback.py --artifact ART --annotations ann.jsonl [--method lda|mean] [--alpha 0.2] [--ridge 0.1]
python -c "import loader; loader.export_bf16('ART','/nvme/bf16_shards')"   # dequantise to a plain bf16 safetensors set (vLLM / HF); deterministic; must live outside ART
```
`answer.py` decodes all rANS chunks with a 16-process pool (constriction holds the GIL, so processes + shared memory), dequantises tensor-by-tensor on the GPU, builds the HF model with the base's own class, then greedy-decodes with the base chat template. It writes nothing inside ART (no pycache, no cache dir).

## feedback.py = pluggable closed-form lm_head row edit (fb5 `lmhead`)
1. sites = teacher-forced annotation tokens where the artifact's argmax != gold token (nothing to do -> no write at all).
2. `EDIT_FNS[method](H_by_token, W_head, alpha, {mu, sig}, ridge)` -> new bf16 rows (`lda`: `w += alpha*|w| * unit(solve(Sigma+ridge*mean(diag)I, mean(h_site) - mu))`, `mean` = fb3 stand-in). mu/Sigma come from final-normed hidden states of generic chat prompts (+ optional `--stats-items`, unlabeled items).
3. `replace_rows`: touched rows -> rotate with the tensor's own sign vector -> RTN on the tensor's own (L, g, mbits) grid -> re-encode only the touched 4M-symbol-or-smaller chunks with the SAME probability tables -> `rewrite_with` copies every untouched chunk byte-for-byte into `model.tpq.tmp`, `os.replace` over `model.tpq`. File set unchanged, size change = a few hundred bytes. Zero annotations -> `FEEDBACK_NOOP`, files never opened for writing (tree sha256 identical, tested).

## Tests
`cd tests && pytest -q` (CPU, tiny random Qwen3 + real tokenizer): bit-exactness vs simulation, codec roundtrip with many chunks, lossless-scale guard, harness size/no-new-parts/manifest checks, forbidden-word grep, `answer` == simulated-path greedy text and leaves tree sha unchanged, empty feedback byte-identical, real feedback changes only `model.tpq` (only `lm_head.weight` rows) with <0.01% size change.
