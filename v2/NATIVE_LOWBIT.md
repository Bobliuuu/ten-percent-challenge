# NATIVE_LOWBIT: native low-bit / lossless-pack route (lane report, 2026-11-07 night)
Rules: dev only, no locked exam touched, no GitHub posting. Code in `v2/native_lowbit/`. Earlier evidence: `v2/research/B_native/REPORT.md`, `v2/lowbit/ARTIFACT.md`.
Status legend: DONE = measured here; PENDING = job written and queued, blocked by the Modal 10-GPU cap (see bottom).

## 1. Verdict (go/no-go)
- Only candidate that fits <=10% with NO lossy step: **Bonsai-8B 1-bit** (7.05%). Faithfulness vs ITSELF is 100% by construction; the open issue is the base-identity reading (parent Qwen3-8B retention 88.9%, B_native). Conditional GO, decided by the maintainer reading, not by measurement. Gate 1 identity run: PENDING (GPU cap).
- **Every ternary model is a size NO-GO without a lossy step** (numbers below): BitNet-2B 13.6% emb alone, Falcon-E-3B 13.8% lossless, Ternary-Bonsai-2-27B >=10.67% even with ideal coding. Lossy step (embedding/head 2-4 bit or vocab trim) fails rule 2 (B_native: 4-bit emb retention .906 on BitNet, ternary Bonsai emb/head to 1-bit .946-.972).
- No released 2-bit QAT / MoE-native model found that is below 10% of its own fp16 (B_native survey: Maple 12.7%, gpt-oss/Kimi/NVFP4/QAT >=25%).

## 2. Exact byte accounting (denominator = 2 bytes x logical params of the model started from)
| model | logical params | fp16 bytes | cap 10% | ternary/1-bit body (best lossless) | non-ternary (emb, head, norms) | lossless total | verdict |
|---|---|---|---|---|---|---|---|
| Bonsai-8B 1-bit (prism-ml/Bonsai-8B-unpacked) | 8,188,548,096 | 16,377,096,192 (released: 16,377,141,728 incl headers) | 1,637,709,619 | all 254 matrices incl embed+head are +-s per 128 group: 1 bit + fp16 scale = 1.125 bpw | norms only (652,655 B) | 1,154,719,808 B = **7.05%** (B_native, bit-exact roundtrip verified) | fits |
| BitNet b1.58 2B-4T | 2,412,820,690 | 4,825,641,380 | 482,564,138 | ~2.0B ternary: 392,849,272 B (1.508 bpw, rANS, 8.14%) | tied emb 128,256x2,560 = 328.3M params; bf16 = 656.6 MB (13.6%) | ~1.05 GB = 21.8% | lossless impossible; v2/lowbit fits at 9.45% only via 4-bit emb + vocab trim to 42k (lossy, dev agreement .92-.99, general ability unverified) |
| Falcon-E-3B-Instruct (tiiuae) | 3,053,586,432 (2,919,235,584 ternary + 134,350,848 bf16: embed 32,768x2,048 + untied head) | 6,107,172,864 | 610,717,286 | measured here: trit freqs .302/.396/.302, **1.5687 bpw** => 572.4 MB | 268.7 MB bf16 | 572.4+268.7+2.4 (tokenizer) = ~843.5 MB = **13.8%** | lossless NO. Only 38.3 MB left for the 134M emb+head params = 2.1-2.3 bpw all-in => lossy 2-bit class, certain rule-2 failure |
| Ternary-Bonsai-2-27B (prism-ml, derived from Qwen3.8-27B) | 26,869,760,000 ternary (+ small) | LM-only F16 gguf 53,808,408,928 | 5,380,840,892 | measured here from MLX pack codes: trit freqs .3365/.3276/.3358, global entropy **1.5848 bpw (=log2 3, zero slack)**, row-wise 1.5926 => 5.323 GB | group scales 419.8 MB fp16 (bias == -scale exactly, so scales only) | 5.323+0.420 = **5.743 GB = 10.67% minimum**, before norms/tokenizer/code; best released file (PTQ1_0, 1.75 bpw) 5,946,648,928 = 11.05% | size NO-GO lossless (needs -0.36 GB; scales cannot be dropped) |
(Falcon-E-1B: 530.7M params, packed 665 MB = 31% of its fp16 1.06 GB cap 106 MB, hopeless.) Sources: `v2/native_lowbit/t27_size.json` (Modal job, DONE), Falcon-E accounting line in Modal log (ACCOUNT, DONE; falcone.py), B_native pack logs.
Key structural fact: ternary at 1.585 bpw is 9.9% of fp16 by itself, so a ternary model has ~0.1-0.3% of fp16 left for embeddings, head, scales, tokenizer, code. Only a model whose embedding+head are ALSO ternary AND whose scale overhead is tiny can fit; Ternary-Bonsai-2-27B has both and still misses by 0.67% because 0.25 bpw of group scales cost 1.56%. Only 1-bit (Bonsai 1-bit, 1.125 bpw = 7.0%) leaves room.
Entropy-coding of ternary codes: gains only where zeros are over-represented: BitNet 1.508 bpw (40% zeros), Falcon-E 1.569; Ternary-Bonsai-2 trits are uniform (no gain).

## 3. What the "model you started from" may be (rules reading)
- Rule 1 text: start from one trained model; size vs "fp16 size of the model you started from"; maintainer: "10% of size on disk is fine compared to fp16 equivalent weights". For Bonsai-8B the fp16-equivalent is identical to the parent's, so only the identity of the base (Bonsai vs Qwen3-8B) matters, and the parent comparison fails rule 2 (88.9% overall, S3 79.6%, multi 84.7%; ternary Bonsai 91.0%). Parent retention rises slowly with scale; the 27B ternary (vendor: "98.2% of FP16 intelligence", an average-benchmark metric, not item retention) was NOT measured on item retention and is size-infeasible anyway.
- Rule 9 exposure: the entrant performs no compression; the 1.15 GB form is the authors' own release. Needs maintainer answer (B_native draft question; not posted by this lane).

## 4. Measurements
DONE: byte accounting above; Falcon-E trit entropy; Ternary-Bonsai-2-27B trit entropy and scale structure; BitNet/Falcon earlier T1 probes (B_native): BitNet bf16 master .267, packed .300 tool-call accuracy (33-38% unparseable) => fails rule 8 on T1; Bonsai-8B 1-bit T1 dev .703 (S1 .823, S3 .583), rule 8 OK.
PENDING (jobs queued on Modal, blocked by workspace 10-GPU cap for >6 h; no result yet):
- Gate 1 identity: `native_lowbit/bonsai_gate1.py` (app tenpct-native-bonsai, fn gate1). Packs Bonsai-8B-unpacked into bits/scales/small + shipped `bonsai_loader.py`, writes real byte total, then compares greedy token sequences of the shipped loader vs reference HF on 600 T1 tool-call items + 1400 TOX/INJ dev items (2000 total, dev files only) and bitwise logits on 8 prompts. Result lands in Modal volume tenpct-native:/bonsai_gate1_result.json.
- Gate 2 rule-8 on label tasks: TOX and INJ probes with `prism-ml/Bonsai-8B-unpacked` were launched through the existing `tournament/{TOX,INJ}/probe_base.py` (entries added); no result seen (same cap). Pull: `modal volume get tenpct-tox-exam results`, `tenpct-inj-exam results_v2`.
- D3 rule-8 on Falcon-E: falcone.py first run crashed (token_type_ids), relaunch stopped by me (unneeded: size already fails lossless). Rerun not required for the verdict.
- D2 parent retention of the 27B on T1: NOT run. Reason: lossless size fails (10.67% minimum) so retention cannot rescue it; additionally the student needs Apple MLX Hadamard runtime or the PrismML llama.cpp fork (custom kernels would count toward bytes).

## 5. Feedback (rules 4-6, maintainer check)
Not addressed by packing (native route leaves it fully OPEN). Per NEXT_PLAN G3, Bonsai-8B itself must carry the feedback edit; fp16 patch rows fit in the 2.95% slack. Coordination files: v2/feedback_v3 (lmhead_plain/lmhead_lda), v2/lowbit_fb (head+prototype memory = banned class under rule 4 as ruled on issue #2, do not reuse).

## 6. Recommendation
Bonsai-8B 1-bit is the only live native route (7.05%, 100% faithful to itself). It is a pure interpretation bet (rules 1/9). Drop BitNet-2B, Falcon-E and Ternary-Bonsai-2-27B as lossless routes. Spend: ~$1 so far (CPU job + 5 min GPU), well inside $75.
