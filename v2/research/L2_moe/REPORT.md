# L2: big MoE (REAP pruning + precision reinvestment) -- report

Bottom line: **NO-GO.** No MoE operating point that fits <=9.6% of fp16 (everything counted) reached 98% item retention on tool calling. Best fitting points sit at 0.93-0.94 (teacher-forced proxy), with multi-call S3 the binding cell. Points at ~98% exist only at 14% (prune 37.5% + RTN4: 0.978) and 22% (all experts RTN4: 0.983).

## What was run (all on dev splits; calibration = tournament/T1/data/s1_calib.jsonl in Qwen native format with gold calls + general slice; never locked/S2)
- Qwen3-30B-A3B-Instruct-2507 on g8 (1x96 GB), Qwen3-235B-A22B-Instruct-2507 on g9 (4x L40S, 470 GB bf16 streamed layer by layer; one pass over the weights evaluates many configs at once).
- Saliency per layer/expert: frequency, REAP (router-weight x expert-output-norm, mean over routed tokens), EAN. 30B: 800 tool + 200 general seqs (472k tokens). 235B: 400 tool + 100 general. Files res/sal30.pt, res/sal235.pt.
- Pruning = mask (router logit -inf before softmax; identical to removing experts, norm_topk renormalises over survivors). Uniform per layer unless "g*" (global allocation).
- Quantisation of surviving experts: RTN symmetric g128, or GPTQ (own implementation, batched over experts, Hessians from calibration items run through the already-pruned/quantised stack). Attention, embeddings, router, lm_head were bf16 in the retention runs (the byte table budgets them at 6 bit); not yet measured at 6 bit on 235B (MBITS/ABITS code exists in gen_pruned.py but the real-generation run was cancelled).
- Metric ("proxy"): teacher-forced, item ok = every token of the gold native tool call is the argmax (nocall: first token is not <tool_call>); retention = P(ok | ok in bf16 base). Validated against REAL greedy generation on 30B (below): proxy is slightly pessimistic.
- Base real-generation accuracy of 235B (FP8, probe agent, 600 probe items): S1 .853, S3 .730 (rule 8 OK). GLM-4.5-Air .737/.743 (XML tool format, needs converter), gpt-oss-120b .580/.467 with multi=0 (one call per turn): not a target. Coder-480B, GLM-4.5/4.6, Kimi-K2, DeepSeek-V3, Next-80B, Mixtral not measured (Modal H100 cap / infeasible on 192 GB boxes). See base_probe/BASE.md (its "bf16 GB" column is params in billions, not bytes).

## Byte table (Qwen3-235B-A22B, fp16 = 470.19 GB, 10% = 47.02, 9.6% = 45.14 GB). code/bytes2.py, bytes_table.txt
Experts entropy-coded (measured entropy of g128 RTN codes on real experts: 3b 2.149, 4b 3.300, 5b 4.365 bits + 0.125 fp16 scale), attn 6b, embed+lm_head 6b, kept router rows 6b, norms bf16, tokenizer/config 16 MB, code 0.15 MB.
| prune | experts 3b | experts 4b | 3b with attn/emb 4b |
|---|---|---|---|
| 0% | 15.03% | 21.98% | 14.61% |
| 25% | 11.60% | 16.81% | 11.18% |
| 37.5% | 9.88% | 14.23% | 9.46% |
| 50% | 8.17% | 11.64% | 7.74% |
| 62.5% | 6.45% | 9.06% | 6.03% |
| 75% | 4.73% | 6.47% | 4.31% |
(An earlier message of mine mislabelled prune-37.5%+RTN4 as 9.03%; it is 14.2%.)

## Qwen3-235B tool-call proxy (360 dev items, 231 base-ok; res/tool_proxy_235.txt)
| prune | experts | % fp16 | retention [Wilson 95%] | S1 | S3 | nocall/single/multi |
|---|---|---|---|---|---|---|
| 0 | RTN4 | 21.98 | 0.983 [.96,.99] | .985 | .980 | 1.00/1.00/.93 |
| 0 | RTN3 | 15.03 | 0.957 [.92,.98] | .954 | .960 | 1.00/.94/.89 |
| 25% | bf16 | -- | 0.974 [.94,.99] | .969 | .980 | .99/.99/.93 |
| 25% | RTN3 | 11.60 | 0.957 | .954 | .960 | 1.00/.94/.89 |
| 37.5% | bf16 | -- | 0.965 [.93,.98] | .969 | .960 | .99/.96/.93 |
| 37.5% | RTN4 | 14.23 | 0.978 [.95,.99] | .985 | .970 | .99/.97/.96 |
| 37.5% | GPTQ3 | 9.88 (9.46 w/ attn,emb 4b) | 0.939 [.90,.96] | .954 | .920 | .97/.96/.86 |
| 50% | bf16 | -- | 0.944 [.91,.97] | .969 | .910 | .95/.97/.89 |
| 50% | RTN4 | 11.64 | 0.952 [.92,.97] | .969 | .930 | .96/.99/.89 |
| 50% | RTN3 | 8.17 | 0.926 [.89,.95] | .931 | .920 | .96/.97/.80 |
| 50% | GPTQ3 | 8.17 | 0.931 [.89,.96] | .947 | .910 | .95/.93/.89 |
| 62.5% | bf16 | -- | 0.848 | .878 | .810 | .95/.90/.59 |
| 62.5% | RTN4 / GPTQ4 | 9.06 | 0.844 / 0.848 | | | multi .55-.59 |
| 75% | bf16 | -- | 0.805 | .870 | .720 | .93/.86/.50 |
| 50% / 62.5% | frequency score | -- | 0.459 (format collapse: single/multi 0) | | | |
Reading: retention is limited by how many experts are removed, not by bits. Going 50% -> 62.5% prune costs ~10 points even in bf16; 3->4 bit is worth ~2-3 points; GPTQ over RTN at 3 bit on 235B is +0.5 point at 50% prune (0.926 -> 0.931, within noise; no RTN3 comparator was run at 37.5%); the 8.17% and 9.88% points stay at 0.93-0.94. Binding cell: multi-call (S3 especially; overall S3 .91-.92).

## Qwen3-30B-A3B (real greedy generation, 240 dev items, 179 base-ok; res/gen30)
| config | retention | S1 | S3 |
|---|---|---|---|
| REAP 50% bf16 | 0.966 (lo .93) | .98 | .95 |
| frequency 50% bf16 | 0.888 (lo .83) | .91 | .85 |
| REAP 62.5% bf16 | 0.911 | .97 | .83 |
| REAP 50% + RTN4 (physically pruned, real int4 packed) | 0.944 | | |
Teacher-forced proxy on the same 163 base-ok items: REAP50 .939, freq50 .840, random50 .436 (single/multi 0), REAP25 .963, freq25 .982, REAP37.5 .951, REAP62.5 .859, REAP75 .748. So REAP >> frequency at >=50% (this reverses the earlier "dead at k=64" result, which used frequency from non-task data), but still short of 98%.
Layerwise-variable expert counts (global threshold on per-layer-normalised REAP mass): 50% .951 vs uniform .939; 62.5% .871 vs .859; 75% .687 vs .748 -> within noise, not a win.
GPTQ vs RTN (30B, REAP50, 163 items): 3-bit RTN .798 -> GPTQ .902; 4-bit RTN .926 -> GPTQ .914; all-experts 3-bit .920 -> .957. GPTQ helps a lot at 3 bit on the small model, almost nothing on 235B.

## Label tasks on 235B (teacher-forced argmax == label, vs bf16 base; res/lab235_summary.txt, code/lab_analyze.py), calibration was tool-call data only
| config (% fp16) | INJ | SENT | TOOLSEL | FIN | ROUTE |
|---|---|---|---|---|---|
| all experts RTN4 (21.98) | .995 | 1.000 | .989 | .994 | .982 |
| all experts RTN3 (15.0) | .989 | .993 | .991 | .989 | .978 |
| prune 37.5% RTN4 (14.2) | .965 | .993 | .991 | .966 | .961 |
| prune 50% RTN3 (8.17) | .896 | .979 | .972 | .938 | .941 |
| prune 50% bf16 | .956 | .974 | .981 | .946 | .959 |
Strict gates (>=98% overall, every source/kind cell >=95%) are met only by SENT at prune 37.5%+RTN4 and not by any <=9.6% config. Worst cells: INJ S3/Benign, FIN S1/Neutral, ROUTE small classes (n=10).

## Feedback story (30B, REAP50, closed-form covariance-whitened lm_head row edit at natural-mistake sites; 1000 dev items, 357 mistake items, half held out; res/fb30_lmhead.txt)
alpha .02: 12% of held-out mistake items fixed, neighbors kept 98.6%, shuffled-target control 0%. alpha .05-.1: more fixes (15-20%) but neighbors 92-97% and the shuffled control also moves. alpha>=.2 destroys the head. Verdict: weak, nowhere near the +15 pts / 98% neighbors G2 pattern; not evaluated on 235B.

## Artifact / inference design (not built; 235B)
- Layout: config with uniform num_experts=K per layer (REAP keep set differs per layer, stored as uint8 index map per layer, and router rows are sliced); per-matrix entropy-coded (rANS) 3/4-bit symmetric codes + fp16 scales (g128); attn/embed/lm_head 4-6 bit; tokenizer/config; ~150 KB decoder code. Raw 4-bit export is GPTQ-format compatible, so vLLM (GPTQ-Marlin MoE) could serve a 4-bit K=80 model on one 96-144 GB GPU (~73 GB raw) after a rANS decode at load.
- Measured throughput of my HF pipeline implementation (int4-packed QLinear dequant per expert call, 4x L40S, naive layer pipeline): 40 items x 256 new tokens per ~12 min = ~3 items/min, i.e. ~13 h per pass over 2.6k items: unusable. Needs fused dequant MoE kernels (vLLM Marlin): decode is weight-read bound (~45-75 GB per step on 1 H200 ~ 15-25 ms per step at large batch) -> an exam pass of 2.6k items with ~100-token outputs is minutes to ~1 h; cost ~ $5-15 per pass on an H200 class box. Not measured.

## Not done / caveats
- Real greedy generation on 235B for any config: cancelled (a 13.5-14.2% reference config; coordinator said stop). So all 235B numbers are the teacher-forced proxy (30B check: proxy underestimates REAP50 by ~2.7 pts, freq50 by ~5).
- Hadamard rotation, KL/influence-weighted per-layer bit allocation and ECSQ not run on 235B; GPTQ only plain (no rotation).
- Attention/embedding/router quantisation to 6 bit not measured on 235B; 30B previous notes say 5-6 bit is benign.
- GPTQ label-task run killed to release the box; Coder-480B, GLM-4.5/4.6, Kimi, DeepSeek, Llama-4 not evaluated (box memory / Modal cap / access).
- Single calibration set; S1->S3 gap at prune 50% is 6 pts (.969 vs .910) in proxy, the outside-source risk is real.

## Verdict / go-no-go
NO-GO for MoE at <=9.6% under the strict gates. REAP is far better than frequency pruning, but a 235B model cannot drop 50% of its experts and keep 98% of tool calls on multi-call items, and 3-bit codes at 37.5% prune are also at ~0.94. Revisit only with (i) training-free expert merging/compensation, (ii) task-matched calibration at far larger scale, or (iii) a different quantiser family (VQ/QTIP) allowing 25% prune at ~2.2 bpw: bytes 25%/2.2 bpw = ~8.4% but needs ~0.97 -> still no evidence of 98%.
