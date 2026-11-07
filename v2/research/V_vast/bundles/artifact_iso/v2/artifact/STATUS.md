# STATUS - v2/artifact (real TPQ1 artifact: writer + loader + CLIs)  [updated as work proceeds]

## 2026-11-07 00:0x local (Mac, CPU)
* DONE: format + codec + writer + streaming quantiser + loader + answer.py/feedback.py + verify.py + 9 unit tests (all green, tiny random Qwen3 + real tokenizer):
  bit-exactness loader == quantiser-side simulation (torch.equal on all 16 quantised tensors, multi-chunk, 4-process shared-memory decode), answer == sim-path greedy text,
  answer leaves tree sha unchanged, empty feedback tree-sha identical, real feedback = only model.tpq changed (lm_head rows), file set unchanged, size delta < 1e-4.
* Harness change: `model.tpq` added to checks.ALLOWED_PATTERNS (only change outside v2/artifact). Everything else in harness/v3 untouched.
* Modal: deployed app `tenpct-artifact` (modal_job.py, volume tenpct-artifact), spawn with `python launch.py TAG MODEL [stages] [levels]`. (`modal run --detach` + spawn from local_entrypoint LOST the job: app ended with 0 tasks; same lesson as AGENTS.md rule 8.)

## 2026-11-07 00:5x local
* Local real-model check (Qwen3-0.6B, tied checkpoint with duplicate lm_head -> alias entry, CPU): RTN 3/4-level and GPTQ(ncal=16) both bit-exact (197 tensors, torch.equal vs quantiser-side sim);
  real artifact 9.11% of the harness fp16 baseline (GPTQ) vs 9.20% estimate -> real is 0.085% of base SMALLER than the estimate (codes -0.5% of code bytes = the 1.005 slack, scales 8.2 bits real vs 11 estimated). harness no-new-parts ok (declared_manifest).
* Modal: image-level `cpu_tests` (pytest in the Modal image, transformers 4.56.2/torch 2.5.1) PASS. GPU queue was ~35 min deep (workspace cap); 8B pipeline spawned (fc-01M6T836GA99D95ATNZ2F6F6VB), waiting for a runner.
* Lessons: import torch BEFORE putting repo dirs on sys.path in the Modal function (circular import crash in the first attempt); transformers 5.x rotary ctor has no `device=`; Qwen3-0.6B safetensors stores lm_head separately although tie_word_embeddings=true.

## 2026-11-07 01:5x  Qwen3-8B end to end on Modal A100-80GB (results/q3_8b/, volume tenpct-artifact:/q3_8b, model.tpq kept only on the volume)
* Levels attn=2 o=3 mlp=3 down=2 emb=3 head=3, g=256, scale mbits=8, GPTQ ncal=128 (picked by RTN size probes: 10.28% / 9.93% / 8.82%). Quantise+write 519 s.
* SIZE: 1,445,972,604 B = 8.827% of the harness fp16 baseline (16,381,470,720). Estimate (entropy*1.005 + 11/g, + raw + files) 8.930% -> real is 0.103% of base SMALLER (inside 0.3%). Parts: codes 1393.0 MB real vs 1400.0 est (the 1.005 slack),
  scales 40.7 MB real (exp 3.8 + mant 36.9; 8.2 bit/scale) vs 50.7 MB est (11 bit), header+signs 0.14 MB, tokenizer.json 11.4 MB, answer+feedback 48 KB, manifest 24 KB.
  If scales were kept as lossless fp16 (mbits=10) cost is ~13 bit/scale: +~0.9 bit/scale*3.2e7 scales... i.e. real would be ~+0.05% of base vs the estimate, still inside 0.3% for g=256 (for g=128 it doubles).
* BIT-EXACT: loader == quantiser-side simulated bf16 weights, torch.equal on all 254 quantised tensors (8,190,427,136 elements) on the A100 (multi-chunk, 16-process decode). Greedy outputs on 100 dev prompts (256 new tokens) real CLI vs simulated model: 100/100 identical
  after a fix (first run 92/100: my loader built the rotary inv_freq on CUDA, HF builds it on CPU; last-ulp difference flips near-ties in this 2-bit model. Fix = build on CPU then .to(dev); devgen2 result 100/100; note the dev proof run below used the pre-fix answer.py).
* LOAD: 31-34 s total (rANS decode + dequant 28-31 s with 16 workers) for 8.19 B params. GEN: 85-88 tok/s at bs=32 HF generate on A100 (100 items, 6.3k tokens, 72 s) -> ~2600 items x 100 tok x 3 passes ~ 2.5 h with HF; vLLM/larger batch needed for speed (export_bf16 written, vLLM NOT tested).
* harness/v3 final_proof on a dev mini exam (120 items, 20/cell, S3 mapped to S2; NEVER locked): ran to completion (rc 0, 2607 s total incl. quantisation). Stage-1 checks: size ok (8.83%), no-new-parts ok (declared_manifest), leak scans ok, artifact tree unchanged by answer, network blocked (guard mode only: Modal has no sudo unshare -> isolation_ok False).
  Feedback arms: empty byte-identical True (0 bytes), real changed only model.tpq (-4 B), shuffled only model.tpq (-13 B); zero added files. Accuracy base 0.79 vs cand 0.11 (2-bit attn/down on an 8B model: no quality claim at all; this is plumbing) -> scorecard FAIL on retention/exam size/teach gain, as expected.
  Scorecard gate `no_new_parts` is False for ANY artifact: final_proof.py never passes `artifact_tree` {dir, allowed_manifest, needles} to scorecard.score (harness gap, not artifact). Teach acc 0/0/0/0.
* Missing: vLLM serving path; 72B/405B-scale run (need 2 GPUs / more queue); netns isolation (needs a Linux box with sudo); real lm_head edit with an effective alpha (teach gain 0 on a destroyed 2-bit model proves nothing); per-group context models for the codes (a few % smaller possible).
