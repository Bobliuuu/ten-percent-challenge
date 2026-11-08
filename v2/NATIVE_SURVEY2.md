# NATIVE_SURVEY2: second survey of native low-bit / structurally cheap models (2026-11-08)
Lane: find any route to <=10% with >=98% retention other than the Bonsai-8B repack. Dev data only (T1 probe 600, TOX/INJ dev 1400, same files as `v2/native_lowbit/bonsai_gate1.py`). No locked file touched, nothing posted. Rules referenced: `v2/RULES.md` (R1-R9, 4d, section 6). Scripts: `v2/native_survey2/`. Status of numbers: see "RESULTS" (filled as runs finish).

## 1. Survey: what exists (verified on HuggingFace 2026-11-08)
PrismML family (all Qwen derivatives, all vendor-QAT'd, Apache-2.0). Group size 128, one fp16 scale per group; embeddings AND lm_head are low-bit too.
| model | parent (vendor card) | levels | lossless bytes vs 2x params | note |
|---|---|---|---|---|
| Bonsai-27B (1-bit) | Qwen3.6-27B | +-s | 26.87e9 x 1.125/8 = 3.78 GB + small = ~7.1% of 53.8 GB | NEW vs previous lanes; vendor avg 89.5% of parent (15 benchmarks, thinking mode) |
| Ternary-Bonsai-27B (v1) | Qwen3.6-27B | {-s,0,s} | ~10.7% (same trit entropy as v2, uniform) | vendor avg 80.98 (v2 card) |
| Ternary-Bonsai-2-27B | Qwen3.8-27B | ternary + block-1024 Hadamard folded into weights | 5.743 GB = 10.67% minimum (B/NATIVE_LOWBIT) | vendor avg 98.2% of parent (14 benchmarks) |
| Bonsai 8B/4B/1.7B (1-bit), Ternary 8B/4B/1.7B, Ternary-2 9B/4B/2B | Qwen3/3.8 | | 1-bit ~7%, ternary >10% (embedding share grows as models shrink) | |
Everything else found is either ternary (each ternary model costs >=9.9% for the trits alone, so embedding/head/scales push it over: BitNet-2B 13.6%+, Falcon-E-3B 13.8%, Falcon3-10B-1.58 ~17% (131k vocab untied bf16 emb/head 1.6 GB + trits 1.8 GB of 20.6 GB), Llama3-8B-1.58 ~21% (fp16 emb/head 2.1 GB), TriLM, Maple 20B-A1B 12.7%), or lossy PTQ checkpoints (VPTQ 1.5-1.75 bit 70B/123B/405B: lossy, same family as our own VQ, item retention for such points is 84-95% in D_scale/L3), or >=25% (QAT int4, NVFP4, gpt-oss, Kimi). No native ternary/binary checkpoint above 27B exists. No hash-routed/low-rank/Monarch LLM released natively at a useful scale was found. BITCOS (arXiv 2609.16338) shows entropy-coded ternary reaches 1.485 bpw only for sparse (zero-heavy) models; Bonsai trits are uniform (1.5848 bpw, zero slack).

## 2. Idea tested: compress Ternary-Bonsai-2-27B (base = that released model) to <=9.9%
Needs: 5.743 GB -> <=5.33 GB. Three train-free levers, all measured on the real scale statistics:
1. Group-scale log-quantisation. Scales are extremely concentrated: mean |log2(s) - rowmedian| = 0.080 octave. Entropy of quantised deltas: 0.94 bit at 4 steps/octave, 1.80 bit at 8, 2.73 bit at 16 (vs 16 bit fp16). K=8 steps (+-4.4% max rel. error) costs 1.8 bits x 210M scales = 47 MB instead of 420 MB (-373 MB = -0.69% of fp16). Source: `v2/native_survey2/scale_stats` run.
2. Vocab prefix truncation. Qwen3.8 BPE ids are in merge-rank order (verified monotonic over 247,587 merges), so keeping ids < N plus the 33 special tokens is a closed BPE (identical to training a smaller-vocab BPE merge list prefix). Embedding and lm_head are both ternary, 5120 wide: each dropped token saves 2 x 5120 trits+scales. N=140k saves 0.22 GB (-0.40%).
3. Trit coding at 5 trits/byte (1.6 bpw, +0.95% over entropy); 41 trits/65 bits would save another ~0.05 GB if needed.
Byte plan at K=8, N=140k, 5 trits/byte: see RESULTS for the REAL written size.

## 3. Pipeline (all scripts in v2/native_survey2/)
- `box/unfold_box2.py K NV OUT [layers]`: unpacks the MLX 2-bit pack, undoes the folded Hadamard (W = (W_stored @ H1024_normalised) * signs; same for embedding rows), subtracts the +1 stored in (1+w) RMSNorm weights, writes plain HF bf16 safetensors. Verified: all text tensor shapes equal Qwen3.8-27B, norm/A_log/dt_bias/conv1d values within 0.07 of the parent, so the reconstruction convention is right (the proof is the generation quality in RESULTS).
- `box/probe_box.py` (vLLM greedy, same prompts/files as gate1) and `analyze.py` (retention = P(candidate correct | base correct), per (task, source, class) cells, Wilson lower bounds).
- `box/art_build.py`: real artifact writer (trits.bin 5/byte, rowbase.bin fp16, scales.q.xz, small.pt bf16, trimmed tokenizer); prints real bytes.

## 4. Rule notes specific to this lane (v2/RULES.md section 5: low-bit starting models, vocab trimming and tokenizer changes are NOT ruled)
- R1/R9: "model you started from" = Ternary-Bonsai-2-27B (53.8 GB fp16 reference, vendor's own F16 gguf), compressed by the entrant (scale quantisation + vocab truncation + coding): a real compression of a released model, unlike a pure repack. Reading vs the grand-parent Qwen3.8-27B is the same open question as for Bonsai-8B.
- R2 vs base measured on dev only (RESULTS). Vocab truncation changes tokenisation of rare multilingual text: maintainer OOD data in other languages is the exposure.
- R4: nothing fitted, no training. Scale rounding and vocab pruning are deterministic transforms. R5/R6/MC (feedback): untouched, OPEN like every other route.

## RESULTS
(pending; filled below as the dev probes finish)
