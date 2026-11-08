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

## RESULTS (all DEV, real greedy generation, vLLM bf16, files: `v2/native_survey2/results/*.json`, analysis `analysis_*.json`)
Probe set: T1 tool calls (600; Qwen3.5+ XML calls converted to the JSON scorer format, `analyze.py`), TOX 5-class label (705), INJ Injection/Benign (695). Retention = P(candidate correct | base correct), cells = (task, source, class).
Accuracies (all / T1 / TOX / INJ): Qwen3.8-27B parent .698/.589/.836; Ternary-Bonsai-2-27B unfolded (T2B) .683/.548/.858; Qwen3.6-27B .735/.596/.866; Bonsai-27B 1-bit .623/.543/.840. Rule 8 OK for all (above chance, below ceiling).

### A. Native models vs their PARENT (the strict reading of "model you started from")
| candidate | parent | retention all [Wilson lo] | T1 | TOX | INJ | worst cell (n>=20) |
|---|---|---|---|---|---|---|
| Ternary-Bonsai-2-27B (unfolded, exact) | Qwen3.8-27B | **94.6%** [93.3] (n=1415) | 92.4 | 89.6 | 99.8 | TOX S1 violence 64%, T1 S1 nocall 80% |
| Bonsai-27B 1-bit | Qwen3.6-27B | **86.7%** [84.8] (n=1463) | 81.0 | 79.8 | 95.7 | TOX S3 toxic 38%, TOX S1 toxic 45% |
Both fail R2 against the parent. Vendor "98.2% of FP16" for Ternary-Bonsai-2 is an average-benchmark score ratio, not item retention; item retention on our tasks is 94.6%. Scale does not rescue the 1-bit model (89.5% avg per vendor, 86.7% item-level here; Bonsai-8B measured 88.9% by B_native). These are the first item-level parent-retention numbers for 27B-class native low-bit models.

### B. Compression of Ternary-Bonsai-2-27B itself (base = the released ternary model, exact unfolded reconstruction)
| variant | lossy steps | real size (written bytes / 53,808,408,928) | retention vs T2B all [lo] | T1 | TOX | INJ | worst cells (n>=20) |
|---|---|---|---|---|---|---|---|
| exact (reference) | none | (5.743 GB min lossless = 10.67%) | 100% | | | | |
| v_k8_full | scales to 8 steps/octave | n/a | 98.92% [98.2] | 98.0 | 98.7 | 99.7 | TOX S3 safe 92% (n=50) |
| v_k8_n140k | + vocab prefix 140k+33 | **9.789%** (5,267,116,016 B, art_build real files, loader verified) | 98.78% [98.1] (n=1392) | 97.8 | 98.4 | 99.7 | TOX S3 safe 92% (50), TOX S3 toxic 95% (20), T1 S3 multi 95.1% |
| v_k8_n170k | + vocab prefix 170k+33 | ~9.86% (est) | 98.85% [98.1] | 98.0 | 98.4 | 99.7 | same |
| **v_k16_n140k** | 16 steps/octave + vocab 140k | **9.835%** (5,292,128,989 B real; + ~155 KB loader/hadamard.json = 9.835%) | **99.57%** [99.1] (n=1392; shipped-path re-run 99.71% [99.3]) | 99.5 | 99.0 | 100.0 | TOX S3 toxic 90% (n=20, 2 flips), TOX S1 violence 97.4%, TOX S1 hate 98.1% |
Observations: the loss is almost entirely the scale rounding (K=8 -> 98.9%, K=16 -> 99.6%); vocab truncation to 140k tokens costs <=0.15 pt on these English tasks (170k buys nothing). K=16 is within greedy-numerics noise of the base. Bytes: trits 5 per byte (5.152 GB, 1.6 bpw), scales 2.73 bits/scale via lzma (74.0 MB), row bases fp16 8.4 MB, bf16 small tensors 52.6 MB, trimmed tokenizer 4.9 MB, rest <0.1 MB. A 41-trits-in-65-bits coder (not built) would save another ~0.05 GB.
Loader proof: `box/art_load.py` rebuilds HF bf16 tensors from the artifact; vs the variant checkpoint used for the probe: 851 tensors, 1 differing by 1.5e-8 (K8 build). K16 build: 851 tensors, 0 differing (bit-identical to the probed variant). **Shipped-path probe (weights loaded from the written artifact bytes by art_load.py): retention 99.71% [99.3] vs T2B (n=1392; T1 99.8, TOX 99.5, INJ 99.8; every cell with n>=20 >= 97.4%, lowest TOX S1 violence 97.4% n=38).** `results/loaded_k16_n140k.json`.

### C. Rules (v2/RULES.md; PASS/FAIL/GRAY, dev evidence only)
- R1 size: PASS 9.835% real (K16/N140k) of the vendor's own F16 reference for the same model; GRAY only on "model you started from" = Ternary-Bonsai-2-27B (not ruled; RULES.md section 5: low-bit starting models and vocab trimming are not ruled).
- R2: PASS-on-dev vs T2B (99.57% overall, 99.1 lower bound; every cell >=97% except a 20-item cell at 90%); FAIL vs the grand-parent Qwen3.8-27B (94.6% before our compression, so ~94% after).
- R3: UNTESTED (no locked exam built). R4: PASS (no fitting; scale rounding and prefix vocab truncation are deterministic; `signs`/Hadamard are the vendor's). R7: PASS (real label words, real tool calls). R8: PASS. R9: GRAY (compressing a released low-bit derivative; tokenizer change; same base-identity exposure as Bonsai-8B but with a real entrant-side compression step).
- R5/R6/MC (feedback): OPEN, nothing built. Per 4d, any lm_head-row edit is high risk; the ternary model has 5120-wide ternary lm_head rows so closed-form row edits would also need re-quantisation.
- MC exposure: the vocab trim drops ~108k multilingual tokens; maintainer data in another language is untested (all three probe tasks are English).

### D. Dead ends found in this lane
- Bonsai-27B 1-bit lossless: 7.1% but 86.7% vs parent (R2 FAIL on strict reading, same identity bet as Bonsai-8B; larger scale did not help).
- Other native ternary models (Falcon3-10B-1.58, Llama3-8B-1.58, BitNet-3B/TriLM, Falcon-E): size >=13.8% regardless of coding because embeddings/head are bf16 or the body alone is 9.9%.
- No native >27B low-bit exists; VPTQ/AQLM/QuIP#/QTIP 1.5-2 bit PTQ checkpoints are lossy compression of the same family as our own VQ work (84-95% item retention in D_scale/L3).
- Models whose structure is cheap (hash-routed, low-rank, Monarch, embedding-light): none released at a useful scale; Gemma-3n per-layer-embedding tables are lookup-only but 4-bit PLE still averages ~1.6 bpw over the whole checkpoint.

## E. Model lineage (verified 2026-11-08 from the model cards; precise statement of the base-identity question)
- Ternary-Bonsai-2-27B: card https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf says "Derived from Qwen3.8-27B, a 27B hybrid-attention causal language model (architecture unchanged)"; model tree base = `Qwen/Qwen3.8-27B`; NOTICE.txt (same repo, `.../raw/main/NOTICE.txt`): "This software is built from Qwen3.8-27B, Copyright 2026 Alibaba Cloud, ... Apache 2.0". Card dates it September 2026 (BibTeX month September, collection updated Sep 17). Format: ternary {-1,0,+1} with an fp16 scale per 128, Hadamard rotation (block 1024) folded into the weights; card does NOT state whether the weights were produced by QAT, PTQ or other training (the whitepaper, https://github.com/PrismML-Eng/Bonsai-demo/blob/main/bonsai-2-27b-whitepaper.pdf, could not be fetched here). The file we start from is the vendor's MLX pack `prism-ml/Ternary-Bonsai-2-27B-mlx-2bit` (text-only, "Hadamard MLX preview", chat template copied from the source GGUF); its fp16 reference size is the vendor's `Ternary-Bonsai-2-27B` F16 gguf, 53.8 GB (card file table).
- Qwen3.8-27B: https://huggingface.co/Qwen/Qwen3.8-27B, Qwen team, post-trained ("model weights ... for the post-trained model"), Apache-2.0, 27B, 64 layers, hidden 5120, "Built on the architectural foundation of Qwen3.5", compared in its own card against Qwen3.6-27B; card citation date August 2026.
- Qwen3.6-27B: https://huggingface.co/Qwen/Qwen3.6-27B, post-trained, April 2026 ("Following the February release of the Qwen3.5 series, we're pleased to share the first open-weight variant of Qwen3.6").
- Bonsai-27B (1-bit): https://huggingface.co/prism-ml/Bonsai-27B-gguf card "Derived from Qwen3.6-27B"; its NOTICE.txt: "built from Qwen3.6-27B, Copyright 2026 Alibaba Cloud". So the 1-bit and ternary-2 families have DIFFERENT parents (Qwen3.6 vs Qwen3.8); our parent-retention numbers use the right parent for each (q36 for 1-bit, q38 for ternary-2).
- Precisely stated question for the maintainer: "May the model one starts from be Ternary-Bonsai-2-27B, a third-party released ternary-weight model derived (by its authors) from Qwen3.8-27B, with the size baseline taken as its own 53.8 GB fp16 reference, and may the entrant's compression (scale quantisation, vocabulary truncation, packing) be what is judged for 'at most 10%' and '>= 98% of the items the original got right', where 'the original' is that released ternary model?" Measured: vs that model 99.7% on dev; vs Qwen3.8-27B 94.6% already before our step.
