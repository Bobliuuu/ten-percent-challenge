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

## 2026-11-07 08:0x  harness wiring + Qwen2.5-72B real artifacts (built on the AWS boxes that hold the codes, pulled to the Mac)
* harness/v3: final_proof/feedback_protocol now feed scorecard v2 (base_runs K=3 bs 8/4/16, artifact_tree + needles, 5 control arms real/shuffled/mixed/wrong_tool/empty, teach clusters); harness tests 52 passed.
* 72B (Qwen2.5-72B-Instruct, base fp16 = 145,412,407,296 B): build_dequant_dir.py (lossless re-encode of D's L{i}.pt, mbits=10 = exact fp16 scales, g=128) run on tenpct-g14 (V3) and tenpct-g5 (V7), 15 min each on 6 CPU threads.
  V3: 13,882,228,218 B = 9.5468% (model.tpq 13,875,082,176; sha256 ab14637b... identical on box and Mac).  V7: 13,882,257,908 B = 9.5468% (model.tpq 13,875,111,866; sha256 f8043a87... identical).
  Estimate 13,972,214,186 (V3) / 13,972,235,416 (V7): real is 89.99 MB (0.062% of base, 0.64% of estimate) SMALLER: codes -1.005 slack (~65 MB), scales 10.65 bit real (lossless fp16: exponent 0.65 b + 10 mantissa b) vs 11 assumed (~25 MB). So the 11-bit scale assumption holds LOSSLESSLY (no scale re-quantisation needed).
  Header+signs 97.9 KB, tokenizer.json 7.0 MB, answer+feedback 48 KB, manifest 58 KB, raw norms/biases 4.3 MB (estimate said 11.5 MB).
  Bit-exact: 21 body tensors (layers 0, 40, 79 x 7 modules) torch.equal vs an independent fp32 reference from the packed codes: all True (full-tensor check of all 560 body tensors NOT run; chunk codec is lossless by construction).
  emb/head: D saved only dequantised bf16. emb re-derived by re-running RTN on the original embedding (V7: 19.3M of 1.246B bf16 elements differ from emb.pt = 1.5%, CPU-vs-GPU fp32 matmul rounding suspected; V3 used the snapped-from-bf16 variant: 279.6M differ = 22%). head recovered by snapping the saved bf16 (GPTQ not reproducible): 264.7M of 1.246B (21%) differ. => V7 body exact, emb ~exact, head approximate; D should dump head codes+scales (like L3 did) for exactness.
* Artifacts on the Mac: ~/ten-percent-artifacts/art72_V3 and art72_V7 (+ .build_report.json). Transfer lesson: ONE ssm-ssh stream = 0.77 MB/s but 12-16 parallel dd-over-ssh range streams = 11-14 MB/s (14 GB in ~20 min); scripts scratchpad fetch1.sh/loop.sh (resumable by part).
* 405B: Modal CPU job (modal_405b.py, app tenpct-artifact-405b) running with L3's emb_q/head_q dumps; result volume tenpct-artifact:/l3_405b/. 
* NOT done: 72B greedy parity (needs 2 GPUs or on-the-fly dequant, not implemented); 8B netns proof on g18 and vLLM export test (venv install started on g18, no time before the AWS cutoff; commands in research/V_vast/needs_artifact.md).

## 72B real-quality plan (V7 artifact at ~/ten-percent-artifacts/art72_V7, 13,882,257,908 B = 9.5468%)
Goal: run the real TOOLSEL/INJ dev exams on the exact bits of the real artifact. Caveats: V7 head is the snapped approximation (21% of bf16 elements differ from D's evaluated head), embedding 1.5% differ -> quality numbers are for THIS artifact, within a small perturbation of D's 98.9%.
1. Needs 2 GPUs (bf16 shards 145 GB). Modal: one container with gpu="H200:2" or "H100:2" (or A100-80GB:2 is too tight), cpu=32, memory 256 GB, ephemeral_disk 600 GB. Upload art72_V7 (13.9 GB) to volume tenpct-artifact (modal volume put, parallel) first.
2. In the container: `python -c "import loader; loader.export_bf16('ART','/scratch/bf16_72b', dev='cuda')"` (decode 72B symbols ~3-5 min with 32 workers + dequant; deterministic) -> 145 GB shards on local NVMe (allowed: the artifact itself stays 9.55%).
3. Serve: `vllm serve /scratch/bf16_72b --tensor-parallel-size 2 --dtype bfloat16 --max-model-len 8192` (vLLM parity of the export is tested on 8B in this STATUS; same code path), run D's `vllm_ct.py` teacher-forced label exams (research/D_scale/vllm_ct.py + C_tasks/items.jsonl) against it and compare with D's fp8 evaluation (note D evaluated an fp8-dynamic re-quantisation of the dequantised weights; the bf16 serve is closer to the artifact's true bits).
4. Optional: on-the-fly dequant on ONE 96 GB GPU is not implemented: keep 4-bit codes on GPU (36 GB) and dequant per layer per forward ~ +1.5 s/step for 72B (fp32 128-block Hadamard matmul), too slow for generative exams, fine for 1-3 token label exams (INJ: 2320 items x 1-3 tokens -> batch all prompts per layer pass).

## 2026-11-07 405B conversion DONE (Modal CPU, app tenpct-artifact-405b; volume tenpct-artifact:/l3_405b/art, report results/l3_405b/report_t096_full.json)
* Llama-3.1-405B L3 t096 -> TPQ1: model.tpq 78,125,928,831 B; total artifact 78,135,186,376 B = **9.527%** of 820,162,494,464 B (header-derived fp16 baseline) (9.646% vs 810e9, 9.626% vs 811.7e9). Build 3462 s. Bit-exact sample (layers 0/62/125 x 7 modules): all True. tokenizer.json 9.09 MB, raw 8.29 MB.
* H200 load test NOT run: bf16 export would be ~810 GB (needs >=6 H200 / 1 TB disk), not cheap; 72B real-exam eval prioritised.
