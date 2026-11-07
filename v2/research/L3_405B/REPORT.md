# L3: Llama-3.1-405B-Instruct at 9.6% of fp16 (GPTQ+Hadamard128+ECSQ), 2026-11-07

## Bottom line
- Teacher-forced label tasks (C_tasks, base-correct items, label-token criterion): INJ 99.7% (Wilson lo 98.7), SENT 97.7, TOOLSEL 97.5, FIN 96.6. This looks like a gate pass for INJ.
- REAL INJ exam inputs (tournament/INJ dev S1+S3, 1566 items, prompt v2, strict scorer, greedy decoding): **retention 90.2% [88.5, 91.6]**. Injection class 78% (S1 78.3, S3 78.6), benign class 99.9%. Fails the 98% / 95% gates by a wide margin. The C_tasks INJ number (safeguard_pi synthetic items, large margins) does not transfer: on the real exam the base margin for injection is small (median 4.7 logits) and the quantised model drifts toward "Ben(ign)".
- Numerics floor: base vs base at two batch settings: 0/1566 flips; candidate at the same two settings identical. The 10% loss is real, not noise.

## Artifact
- Checkpoint tensors on disk: 820.16 GB (410.08 B params; Llama-405B here has 16 KV heads; embed and lm_head 2.1 B each). t096 estimate 78,741,273,686 B = 9.601% of 820.16 GB; = 9.72% of the nominal 810 GB; = 9.70% of 811.7 GB.
- Body 77,671 MB (1.531 bpw), emb 526 MB, head 526 MB, other bf16 (norms) 8.3 MB, tokenizer+configs 9.1 MB, code 0.15 MB. Levels (fraction of linear params): L=3 77.7%, L=2 20.6%, L=4 0.84%, L=5 0.70%, rest 0.12%.
- Scale accounting caveat: the estimate charges 11 bits per fp16 group scale (g=128). With plain fp16 scales (16 bits) add 1.98 GB: 80.72 GB = 9.84% of 820.16 GB, 9.97% of 810 GB, 9.94% of 811.7 GB. The artifact writer (a2739393541b4cbb9) reports the real written size.
- Layout on Volume tenpct-l3-405b: /out/t096/q/L{i}.pt (codes packed 4-bit, fp16 scales, int8 signs, L), /out/t096/q/{emb_q,head_q}.pt (integer codes; emb exact RTN re-run, 2249/2.1e9 bf16 elements differ only through fp32 matmul shape rounding; head recovered from head.pt, 262 of 16.4M groups not bit-matched, 5274 elements), emb.pt/head.pt (dequantised bf16, used by the eval), meta.json, sens_g128.json.

## Method / pipeline
- l3.py: adapted from D_scale/dense_stream.py (layer-by-layer, ONE H200). Pass A (RTN sensitivity per module/level, 25 s/layer, 53 min), allocation (Lagrangian to 9.6%), pass B (GPTQ with Hessians from the quantised prefix, 60 s/layer, 3.05 h). Hessians come from the quantised prefix => layers are sequentially dependent, cannot be sharded.
- Calibration: 128 x 512 tokens: C_tasks calib.jsonl (598 S1-role items, 9 tasks, 80 INJ items from safeguard_pi) + 299 s1_calib tool-call items, shuffled, Llama chat template. 
- Evaluation: layer-streamed forward on one H200 (activations of all items kept on GPU; per layer: load weights from Volume (bf16 base) or dequantise the 4-bit packed codes (candidate)); no KV cache, no generation engine. Base = TRUE bf16 405B (no fp8 proxy). 25 min for 5414 items x 9 tasks on base, 13 min for 2400 items on the candidate.
- Real-exam greedy decoding is done as a speculative argmax chain: sequence = chat(prompt) + gold label tokens + "." fed in both casings (Llama answers lowercase "benign"/"injection"; the quantised model shifts toward "Ben"). The argmax at each position is exactly what greedy decoding would emit as long as the previous tokens match; outputs ending in <|eot_id|> after the label (or after label+".") are fully resolved; any divergence is a wrong output by construction. 0 items were unresolved. inj_analyze.py applies tournament/INJ/scorer.py parse_label to the reconstructed outputs.
- Llama prefers "." over <|eot_id|> after a bare label on the C_tasks prompts; for C_tasks scoring the final EOT token is excluded (label tokens only). Strict (incl. EOT) numbers are in results/ct_retention_t096_strict.txt.

## Results
### C_tasks (label-token criterion, teacher forced; dev items; base acc: INJ .965 SENT .952 TOOLSEL .862 FIN .873 ROUTE .793 TOX .785 SPAM .927 NLI .71 AIDET .69; only the four tasks below were run on the candidate)
| task | base-correct | retention | Wilson lo | worst cell |
|---|---|---|---|---|
| INJ | 579 | 0.997 | 0.987 | 0.993 |
| SENT | 571 | 0.977 | 0.961 | 0.958 (Positive) |
| TOOLSEL | 517 | 0.975 | 0.957 | 0.963 (S3) |
| FIN | 524 | 0.966 | 0.946 | 0.932 (Neutral) |
(Compare Qwen2.5-72B at 9.6%: SENT 96.8, INJ 94.5, TOOLSEL 93.1; criteria differ slightly: Qwen numbers include the EOT token.)

### Real INJ exam inputs (see results/inj_retention_t096.txt)
Base accuracy 1404/1566 = 0.8966 (S1 benign .97, S1 inj .795, S3 benign .992, S3 inj .83); 89.5% of base outputs are exactly the bare label. Candidate accuracy 0.8186; outputs exactly one label (bare or with trailing "."): 81.9% (bare 65.6%, label+"." 255 items); wrong first token 284.
Retention P(cand correct | base correct): ALL 1266/1404 = .9017 [.885,.916]; S1 636/706 = .901; S3 630/698 = .903; S1 benign 387/388 = .997; S3 benign 380/380 = 1.000; S1 injection 249/318 = .783 [.735,.825]; S3 injection 250/318 = .786 [.738,.828]; deepset .904, gandalf .898, promptshield_train .903.

## Hosting the compressed 405B for the locked INJ exam (estimate from measured runs)
- 4-bit packed codes are 203 GB (+ scales), too big for one 141 GB H200; resident serving needs 2 H200 (pipeline). Cheaper: layer streaming on 1 H200 as done here: 1566 items x 2 casings (620k tokens) = 18 min incl. loading the packed artifact from the Volume, ~6-9 s/layer, compute-bound (~330 TFLOPs). Locked set 2320 items of similar length (~0.5-0.9M tokens): ~20-30 min per pass on one H200 (~$2). One pass resolves greedy 1-3 token outputs (speculative chain); a plain autoregressive loop would need up to 3-4 passes (~$8). Feedback arms that only change lm_head can share the pass (apply different heads to the final hidden states). With N H200s sharding the items, wall time divides by N. Entropy-coded artifact: needs a GPU rANS decode of 78.7 GB per pass (not implemented here; the Volume copy used for these runs is the 4-bit-packed form).

## Rule check: calibration vs INJ dev/locked
- Calibration used: C_tasks/calib.jsonl (ids "<TASK>|<src>|cal|n", S1-role train-split items from civil_comments, safeguard_pi (80 INJ), enron, sst2, gpt_wiki_intro, xlam_dev, massive_scenario, tw_fin_news, mnli) + 299 items of D_scale/t1v2/calib_t1.jsonl (tool calling, s1_calib).
- Against the 1566 INJ dev texts: exact (normalised) duplicates 0; word-5gram Jaccard >= 0.5 near-duplicates 0 of 598 (max 0.037). No INJ dev id appears in calibration.
- Against LOCKED: not read (rule). By construction the INJ locked set is PromptShield (S1) + JailbreakHub (S2); the INJ calibration items come from safeguard_pi, a different dataset. Caveat: the INJ builder dropped safeguard (xTRam1/safe-guard) as unlicensed and seeded from jackhhao whose style leaks S2; a text-level dedup of calib INJ items against locked ids must be run by someone who may read locked (hashes only). If in doubt, requantise without the 80 safeguard_pi items.

## Spend (H200 ~$4.5/h)
Quant 3.05 h + sensitivity incl. 3.5 h; evals ~4 h (including re-runs due to stale-container/format bugs); download CPU; total ~ $50 of the ~$150 limit.

## Not done / next
- INJ-specific calibration (tournament/INJ data/calib_S1.jsonl with the v2 prompt, and the answer format the base actually emits) and/or ratio 0.0985: expected 2.2 h quant + 17 min eval each. t096_c6/t0985_c6 (C_tasks 6-task calibration) were spawned then cancelled before starting.
- TOX/ROUTE/NLI/SPAM/AIDET on the candidate not run.
- Lesson: Modal containers stay warm ~60 s after a function ends and reuse OLD code after a redeploy; rename the function (evalinj2) or wait before relaunching.
