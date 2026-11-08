# WHAT_WE_TRIED: every lane, method and task, with results and dead ends

Written 2026-11-08 (about 20:40 UTC) from the committed documents: `v2/CANDIDATES.md`, `NEAR_MISSES.md`, `SOLUTIONS.md`, `lanes/*/REPORT.md`, `FEEDBACK_ABLATION.md`, `BYTE_TRICKS.md`, `QUANT_SWEEP.md`, `RED_TEAM.md`, and the archived status page. Nothing here was re-run. Deadline 2026-11-12 17:05 UTC. Rules: [`RULES.md`](RULES.md) (R1-R9 = the nine judging rules; "MC" = the maintainer's own-data check). Other entrants are "entrant #3" / "entrant #4". Nothing in this file has been posted anywhere.

Conventions. **dev** = our own development/selection splits (never pass evidence). **LOCKED** = scored once on our shared sealed exam `v2/exam` (SKILL7: 2,744 items, 7 kinds, 1,489 items from sources never used for building: HWU64 + MTOP; BANK6: 2,237 items). It is locked only in our sense (public data, one builder, hash-sealed before any run); it is not the maintainer's exam. **GEN** = real greedy generation; **TF** = teacher-forced scoring, which overstated retention by 3-10 points wherever both were measured, so TF is never pass evidence. **Retention** = P(candidate correct | base correct), Wilson 95% lower bound in brackets.

Status of the whole effort in one sentence: compression alone is solved for an easy 7-way routing task (several entries keep 99%+ of base-correct items at under 10% of fp16 on our own sealed exam), but feedback (rules 4-6) rests on a closed-form edit of existing tensors that the maintainer has not ruled on, so no entry is a clean pass.

---

## 1. Timeline in brief

| phase | dates | what happened |
|---|---|---|
| 0 | 11-05 | Small BERT-class tracks A-S (wrong base: under 0.5B and fp32 baseline). Only methods reused. |
| 1 | 11-06 early | Qwen2.5 7B-14B-30B with MMLU letters as a stand-in task. Wall: no training-free quantizer near base accuracy at 10%. |
| 2 | 11-06 | Task selection, pre-registered decision, **filed entry** (Qwen2.5-7B tool selection, letter head + memory). Issue #2 reply: feedback does not hold (fitted head). |
| 3 | 11-06 to 11-07 | Redesign for the nine rules (real generation, no head, controls). Dense 14B-405B conventional compression on real generation: stuck at 90-96% retention. Entrant #3 rejected (nearest-centroid rows). |
| 4 | 11-08 | Pivot to (a) easy 7-way SKILL7 routing task with a shared sealed exam, (b) task-pruned fine-grained MoE + 2-bit, (c) already-low-bit released models (Bonsai), (d) closed-form diffuse edits of existing tensors as feedback. Lanes L1-L13. |

---

## 2. Phase 0 and 1: legacy and stand-in work (dead ends, methods reused)

| what | result (dev) | why it stopped |
|---|---|---|
| BERT-base SST-2 tracks A, D, E, J, K (GPTQ 2-4 bit, calibration-only recovery, embedding compression) | 42.73 MB = 9.76% of fp32; dev2 acc .907 vs .930; best agree .9775-.990 | model under 0.5B, fp32 baseline (9.76% of fp32 is 19.5% of fp16). Methods only. |
| Track B feedback: sparse lexical residual memory | new-concept acc 61.9 to ~96.6 | side memory / retrieval: banned (issue #1 answer 3, rule 4) |
| Tracks G, H, I, L, M, N, P, Q, R, S (proxy data, other classifier bases, external correction, power analysis, vision classifiers, Qwen2.5-0.5B) | best agree .990 only at 12.5% ratio; 2-byte bias +0.6 to +1.0 pt; R (0.5B MMLU) ratio .42 at acc .35 vs .466 | bigger is not more robust at fixed bits; off goal |
| Rotation (Hadamard) + GPTQ + entropy coding, 14B MMLU (`v2/engine`) | L=3: ratio .0968, acc .5209 vs base .8007; rotation essential (no-rot .441 vs rot .764) | about 28 pts below base at 10% |
| 8-dim shared-codebook VQ, 14B MMLU (`v2/vq`) | 1.654 bpw, ratio .0961, acc .6195 | still 18 pts below base; about +4 pts per +0.1 bit |
| Layer drop / FFN / head prune (`v2/prune`) | tail drop 2 layers -0.4 pt for -4% bytes; FFN prune -2.5 to -4 pt at keep .9 | break-even at best |
| Entropy-coding variants (`v2/entropy`) | context modelling gains under 0.5% of index bits | dead end |
| MoE Qwen3-30B-A3B prune + quant on MMLU (`v2/moe`) | best 0.7323 vs 0.8078 (GPTQ, 70% experts, 2-bit); 1-bit collapses | -7.5 pt; method later reused with routing-mass scoring |
| 32B/72B MMLU (`v2/big`) | no numbers (GPU pool saturated, p4d no capacity) | stopped by coordinator |
| Feedback on 0.5B/1.5B stand-ins: kNN gated memory, prototype-calibrated head | teaches | retrieval/head: banned |

Lesson: broad knowledge needs far more than 1.6 bits per weight; only a narrow output space can reach 10%.

---

## 3. Task selection and the filed entry

**Task probes** (`v2/commercial`). The three maintainer-named examples on Qwen2.5 1.5B/3B/7B: PII span category (base .41/.52/.69, synthetic data), AI-vs-human (MAGE) **at chance** (base .50/.53/.50, AUROC .50-.58: fails R8), C2 tool selection among confusable tools (base .76/.75/.78 vs chance .23): chosen. MDC-6 multi-domain classification (ag_news, dbpedia, yelp, amazon, emotion, offensive) kept as fallback.

**Filed entry (C01, issue #2)**: Qwen2.5-7B-Instruct, Hadamard + GPTQ onto an entropy-constrained scalar grid + rANS, vocabulary cut to 24,000, 4-bit embedding, `lm_head` cut to 5 option-letter rows; feedback = a 476 KB PCA64+LDA head plus an 852-byte running-mean memory.
- Locked (n=2000): base .7675, candidate .799; both right on 1441 of 1535 base-correct = **93.9% [92.6]** retention. Size 9.404% real. General capability collapsed (MMLU .467 vs .693, ARC .62 vs .895).
- Fails R2 (93.9), R3 (no 500-item unseen-source block), R4 (fitted head + side memory, **ruled in #2**), R5/R6 (no pre-locked teach set, no shuffled control), R7 (letters), R9. Its locked exam is burned; not refiled.
- Related fallbacks, both letter-based with fitted heads and dead: MDC-6 7B (9.736% real, dev agree .954), BitNet-2B ternary (9.453% real, MDC-6 agree .988, but tool-call base only .267 so R8 fails).

---

## 4. Dense conventional compression on real generation (Phase 3, 11-07 to 11-08)

All DEV, GEN unless marked. Result: **every dense route lands at 90-96.8% retention at 9.6-10.0%, against 98% overall and 95% per kind.**

| candidate | result | verdict |
|---|---|---|
| Qwen2.5-72B tool selection (C03) | 95.55% GEN (TF said 98.9); worst cell S1/none 89.3; 9.61% est | R2 |
| Qwen2.5-72B injection (C04) | 95.21%; worst cells 93.0-93.6 | R2; benign near ceiling (R8 gray) |
| Qwen3-32B / Qwen2.5-72B tool calls (C05/C06) | 85.8% / 84.2%; 14B 56.1%; extrapolated 98% needs 19-25% of fp16 | dead |
| Llama-3.1-405B injection (C12) | **90.2% [88.5]** GEN (TF said 99.7) | dead |
| Qwen2.5-32B VQ-D8 K=5300, injection (C07) | 95.0% [94.2], n=3377; S3_injection 81.3, S2_benign 85.8; 9.85% est | R2 per-kind |
| Qwen2.5-32B scalar GPTQ, real injection data (C09) | 0.957 / 0.958 / 0.942 at 10.0 / 11.3 / 12.5%; draw noise about +-2 pt | R2 |
| 32B NEWS router (C08) | 96.1% [95.0] at 10.03% (over cap); Technology .90 | R1, R2 |
| 7B VQ-D8 quality push (C11) | INJ 96.2% (best of about 12 configs), TOOLSEL 94.3%; VQ buys 20-25 pts over scalar at equal bytes; MMLU .546 vs .678; one error-propagation target run destroyed the model (numerics bug) | R2 |
| 12-task search on 7B/14B/32B (C10) | 32B at 10.03%: NEWS .961, INJ-real .957, ROUTE .947, FIN .937, TOOLSEL .928. INJ-easy .981 was an easy-source probe, not evidence. EMOTION/INTENT/ROUTE cannot reach 300 per kind; LANGID at ceiling | R2, R1, R3 |
| TASK_SEARCH2 (C30-C32): tool-call arguments, PII detection/redaction, NER, log templates, SQL | 32B at 10.0/11.3/12.5%: TOOLARG 95.0/94.6/97.9; PIIDET 86.1/93.1/87.8; PIIRED 70.1/80.4/80.4; NER 30-52; LOGTPL 79-86; SQL base 4-14% | long exact-match outputs retain far worse than label words; all fail R2 |
| T3 task catalogue (14B/32B, 9.85%) | NEWS4C .916 (14B) / .949 (32B); PII2C .957; SENT2C .841; TOX2C .796; PHISH2B .807; COMM4 .750; PHISH2C .508 (class flip); 26 tasks in one calibration mix: class collapses | none reaches 98 |
| Training-free TCQ (QTIP-style trellis) vs VQ-D8 vs scalar, 32B/7B (C27 VQ_ROUTE) | TCQ 9.57%: INJ 96.8, NEWS 92.3; scalar 96.15 / 94.4 on the same items; within +-2 pt draw noise; fp-target variant collapsed | NO-GO |
| Task-calibrated structured pruning x quant (C28 prune lane) | 32B vocab-trim INJ .957, NEWS .961 (no prune baseline .930/.940); FFN .9 prune INJ .858 | structured pruning net-negative at this budget; vocab trim is a component only |
| TOX moderation (C24) | 14B 82.8% at 9.6%; 72B 95.1% TF | sexual cell unusable |
| Other bases Llama-3.3-70B, Mistral-24B, Gemma-3-27B, GLM-4-32B (C23) | 14-38% under our scorer | probe-harness artefacts; weight 0 |
| Qwen2.5-14B depth/width pruning + RTN (L8) | vocab trim alone costs retention (65% of eval prompts hit trimmed tokens); dropping 12 or more of 48 layers destroys the model at fp16; every point at 9.9% or below retains about 0 | dead |
| X03 Qwen2.5-14B NEWS router, 3 stack rungs | 9.893% real; 8 draws: 93.50, 92.92, 94.45, 96.12, 94.45, 94.72, 93.35, **65.67** (one collapsed draw, sd 10 pt); best stack +3.03 pt; one REP unseal 95.44% | FAIL 98% |
| X04a Qwen2.5-32B injection, full stack | 96.49% [96.03], worst cell 79.1 | KILL |
| Giant MoE (R1/V3/K2/Qwen3-235B/480B/GLM-4.5/gpt-oss-120b) at about 1.5-bit experts (C26) | byte math: experts must hold 1.37-1.55 bits all-in; best public 1.58-bit file is 10.44%; estimated P(98/95) about 4% | NO-GO, no run, $0 |

Cross-cutting facts: (1) the only measured way retention reaches 98% is more bits (32B tool calls 76 -> 89 -> 95% at 9.6 -> 12.0 -> 15.0%; label tasks reach 98% at 12-15%); (2) re-running the unquantised base with different batching keeps only 99.3% of its own correct items on one task, so certifying a Wilson bound of 98% needs about 99.0-99.3% observed; (3) per-kind gates at 300 items need about 97.7% observed to certify 95%; (4) TF overstates retention by 3-10 pts; (5) the gap is quantizer noise, not ambiguity (items failing at every budget are only 0.6-1.4% of base-correct).

---

## 5. The two ideas that moved the numbers

1. **Task-pruned fine-grained MoE + 2-bit experts (L3, L4, L9, L9b, L12).** On a narrow routing task most experts are never needed. Qwen3.5-35B-A3B (256 experts) keeps 99% with about 10-15% of experts before quantization (L12). More experts at 2 bits beats fewer experts at 3 bits at equal bytes (QUANT_SWEEP: BANK8 99.6-99.7 vs 94.9). Cost: R9 exposure (the model is a task specialist) and out-of-distribution risk.
2. **Easy 7-way routing task (SKILL7) on strong bases**, where the base is .96 and quantization noise rarely flips an item. Cost: R8 (headroom, base near ceiling) and R1/R9 (narrow task, "toy-shaped"). This is what every current candidate uses; a broader second task does not exist.

---

## 6. Lanes (L1-L13)

### L1: Bonsai-8B (released 1-bit) lossless repack + column-sparse feedback
Start model is a maker-built 1-bit release derived from Qwen3-8B. No compression by us (bit-exact repack, 2000/2000 identical greedy outputs). Bytes 7.44% incl. a 50 MB fp16 patch. LOCKED SKILL7: retention 1.000 (2,489/2,489, Wilson lo .9985) against the released model, all kinds and sources 1.000; against the Qwen3-8B ancestor only **88.9%** (dev). Feedback `Recipe`: teach .893 [.881,.905] (10 draws), shuffled .005, wrong .000, empty .000, neighbours .995; exam retention after feedback .9988. Mechanism: closed-form `dW = R K^T (lam (C + awt C_ans + ridge I) + K K^T)^-1` on 1,024 columns of `down_proj`, last 6 layers; 2,048 columns leaked on the shuffled arm (shuffled .84), so 1,024 shipped. NEWS secondary: teach .55-.67, neighbours .983 (two cells below .95), not shipped. Earlier NEWS router with sign-bit flips: teach 0.00 with neighbours .78-.90, or teach 1.00 with shuffled 1.00 (F07, no selective edit). Weakest: R9 (no compression by us), R4 (50 MB fitted patch), start-model reading (ruling B).
**L1+ (exploratory, dev and exam-split, not packaged):** a 0.5 MB low-rank carrier reaches teach .90-.92 with neighbours .992-.997 and shuffled 0-.012 on the shared teach split (6 draws); a 4.7 MB scale-edit carrier gives .89-.93 / .99-1.00; continual feedback over 3 new skills in sequence keeps earlier skills (Recipe .889, Email .815, Takeaway .828; neighbours .997); hostile own-data proxy with foreign labels (Translate .86-.91, Banking .82-.84, Email .93, shuffled .01-.06, old-label proxies 1.00). Raw: `v2/bonsai/l1plus/results/`.

### L2: Ternary-Bonsai-2-27B (released ternary) recompressed by us
Our compression (trits 5 per byte, log-quantised group scales, frequency-selected 165k vocabulary, no training), 5.35 GB = **9.943%** (30.6 MB slack; the denominator is back-computed, real HF fp16 of the ancestor is larger so 9.62% vs ancestor). LOCKED SKILL7: retention **100.0% [99.86]** vs the released ternary, 1 of 2,744 outputs differs; vs the fp16 ancestor 99.32% [98.93] (Calendar 96.1), i.e. recompression adds no loss. Feedback (rank <= 75 closed-form edit of two `down_proj` tensors, 4.5 MB): teach .827 (3 draws), CLI .872, neighbours 98.3-99.2 (worst cell 96.3), shuffled 0, empty 0. A mixed math/code/safety probe: 97.07% overall, math 94.2% (not passing; no general-capability claim). Weakest: R4, R9, R8 (base .967), tiny byte slack, 96 GB + cache to run, shuffled arm used random old labels (weak control per RED_TEAM), artifact only in a private bucket.
Earlier variants: a 184k-vocab variant was 10.009% (over), ternary-27B before scale coding 10.67%.

### L9b: Qwen3.5-35B-A3B packed (our compression, text-only)
Routing-mass prune (102 of 256 experts per layer on unlabeled SKILL7 prompts) + grouped RTN 3-bit g128 on kept experts, 6-bit attention/shared, 4-bit embedding/head. **9.674%** of the text-only reference (9.326% vs the full checkpoint with vision tower; 9.704% with the 21 MB carrier). LOCKED SKILL7: **99.55% [99.21]**, 3 quantizer draws 99.47-99.55, Lists 98.4 worst kind, never-used sources 99.45 [98.91]. Feedback `Recipe` (25 annotations = 12 Recipe + 13 old-skill errors; closed-form edit of the dense shared-expert `down_proj`, last 10 layers, 512 replaced columns each): teach .762/.793/.726, shuffled .110/.018/**.470**, wrong 0, empty 0, neighbours 98.56/97.99/98.28 vs 98.28 empty. Smaller carriers (5.2 MB) teach .38-.43. BANK6 near miss: LOCKED retention 97.37% [96.58] (Card 94.4) fails; feedback `Security` teach .13-.52 and neighbours damaged. Weakest: R6 variance (draw 2), R8 headroom (base .9676), R9 (60% of experts pruned for the task), highest own-data risk (BANK6 evidence), artifact must be re-packed (not in git).

### L3: Qwen3-30B-A3B-Instruct-2507, 85/128 experts + D4-K256 VQ 2 bpw (scored)
Real packed file 9.676%; with the fp16 `o_proj` edit counted in full 9.896%. LOCKED SKILL7 **99.32% [98.92]** (Calendar 96.69, never-used 99.65), teach 0 -> **79.9%** (164), shuffled 0, wrong 0, empty 0, exam 99.20 after feedback. LOCKED BANK6 99.24% [98.76], teach only **16%** (Security leaks into Card unless the edit is too weak to teach), neighbours 98.52 vs 98.79 empty. Earlier recipe RTN3 keep-55 failed on BANK8 (94.9); switched after the quant sweep. Weakest: BANK6 R5, per-cell minima 96.7-97.9, R4/R9 gray, SKILL7 neighbours not measured with the final carrier.

### L4: rival-#3 beat attempt, Qwen3-30B-A3B 80/128 + GPTQ/Hadamard 2-bit (scored)
9.895% (estimate, no packed file). LOCKED SKILL7 98.94% [98.47] (News 96.03, Wilson lo 93.65), teach .59/.63/.59 vs shuffled .18/.09/.49 vs empty 0, neighbours .989-.994. LOCKED BANK6 99.24% [98.76], teach .07-.27 with neighbours collapsing in 2 of 3 draws (.654/.608); a more protected retry fixed neighbours but teach fell to .02-.16. Dev own tasks: clin9 99.72, bank8 98.95-99.28. Uses BM25 labelled examples in the prompt (R4/R9 gray), and its `never` slice is the same dataset (R3 gray). Weakest: R5, R6, no packed file.

### L5: dense Qwen2.5-32B scalar GPTQ + Hadamard + rANS + column-sparse carrier (PENDING exam)
Measured 9.697% (9.889% with a 126 MB carrier). Dev retention .9873 [.9840] (n=5,913, worst cell .961, one quantizer draw, draw noise +-2 pt). Feedback on shared-exam splits, teacher-forced argmax: teach .671 (cross-source .633), shuffled 0, neighbours 1.000. Locked exam NOT scored; loader code not counted yet. Pending.

### L6, L7: other bases (FAIL)
L6 Qwen1.5-MoE-A2.7B (31/60 experts, RTN 2-bit): 9.87%, dev retention 85.9% [82.9]: FAIL R2. DeepSeek-V2-Lite .382, OLMoE .775; census probes of 12 more MoE bases (gpt-oss-20b 75-87, gpt-oss-120b 13-29 on SKILL7; ERNIE/OLMoE/DSV2 fail; GLM-4.5-Air 99.2/99.7 at keep 0.5 3-bit promising but not built). L7 BitNet-b1.58-2B-4T: ternary body lossless; tied 128k embedding is the cost; best at or below 9.9% retention 95.8% [94.7] and neighbouring vocab sizes collapse (47.5, 41.8): FAIL R2, R8 marginal. Falcon-E pending, no result.

### L9: Qwen3-235B-A22B-Instruct-2507 (scored on the shared exam)
88/128 experts + 2-bit RTN-MSE g128 experts + 4-bit attention/embed/head. **9.7814%** measured (46 GB); 9.896% with a 537 MB `o_proj` carrier. LOCKED SKILL7 **98.98% [98.52]** (Calendar 97.7, cell min 95.3, never-used 99.17). Feedback: teach 2.4 -> **74.4%**, shuffled 12.2%, wrong 0, neighbours 99.16% unchanged, one draw. Needs about 330 GB of GPU memory to run: impractical for the maintainer's check. Its own hash-bucket exam fails R3 per kind (RED_TEAM).

### L10: Qwen2.5-Omni-7B text path (denominator bonus)
Ship only the 7.62B thinker text path; 10% of the WHOLE 21.46 GB multimodal fp16 gives 2.25 bpw to the text body. 9.879% (9.930% with feedback), accounting only, no packed file. LOCKED SKILL7 **99.55% [99.21]** (News 98.78), feedback 0 -> 86.0% on 164 teach items, shuffled 0, neighbours 99.15 unchanged (one draw). Llama-3.2-11B-Vision text path 98.73 (base .809), Omni-3B 98.90 (Calendar 93.9 fails per-kind), BANKFAM 97.60 (Payment 88.9, not presentable). Weakest: R1 literal vs R9 spirit (denominator inflated by unused towers: ruling D), no packed file.

### L11: feedback ablation (Qwen2.5-7B, new label `Jukebox`/`Recipe`, 10 draws)
See section 7.

### L12: MoE combination ablation (research lane, not an entry)
Qwen3-30B-A3B keep 40/128 (42.9%): mass .991/.989 (SEL/REP), REAP .991/.994, frequency .962/.974; keep 25/128: mass .971/.978, REAP .962/.983, KL .964/.964, **random 0.000**, gate-free .521. With GPTQ at about 10%: .958-.973. Qwen3.5-35B-A3B keep 40/256 mass .994/.992; keep 25/256 .991/.993; merge-alignment .959-.976 (worse than prune); top-2 routing collapses (.383), top-4 .978, top-6 .992; RTN 3-bit all experts (19.5% of fp16) .991/.994; sub-rank r512 0.000. `run_final.py` (shared exam + feedback) exists but is not run: pending.

### L13: byte tricks (Qwen2.5-14B, GPTQ + Hadamard)
See section 8.

### Other lane-level rows
MDC-6/BitNet/Falcon-E/Ternary-Bonsai-8B (C15-C18): ternary 8B is 10.19% lossless, 27B 10.67% minimum, Falcon-E-3B 13.8%: size blocks, entropy coding cannot recover the gap (zero slack at log2 3). Ternary-Bonsai-8B parent retention 91.0% on tool calls.

---

## 7. Feedback mechanisms (L11 ablation and earlier)

Protocol: Qwen2.5-7B-Instruct, new label never in any prompt, n=12 annotations (5 and 25 for core methods), 10 draws, controls real / shuffled / wrong / empty, neighbours on base-correct old-skill items, teacher-forced scoring validated against real generation (agreement .98-1.00). Full tables: [`FEEDBACK_ABLATION.md`](FEEDBACK_ABLATION.md).

| mechanism | teach | shuffled | neighbours | bytes (7B, 8 layers) | verdict |
|---|---|---|---|---|---|
| `lm_head` row edits: class-mean/LDA/calibrated (F02) and closed-form min-norm `mnH` (F01) | mnH .49-.80 | .02 | .997-1.000 | about 0 | **illegal by the maintainer's words** ("same head as before"); mnH collapsed a model in 1 of 10 injection draws; never run on a compressed artifact |
| LDA/prototype head + memory (F04, filed entry) | works | | | 0.5 MB | ruled out in #2 |
| kNN / lexical retrieval memory (F05) | works on BERT | | | | ruled out in #1 |
| annotation-aware requantization (F03) | cannot install a label ranked about 30 logits down | | | | legal, cannot teach |
| `down_proj` AlphaEdit null-space k256, last 8 layers | **0.893** | 0.000 | .996 | 1.09 GB | best teach, GRAY; exam shuffled .15-.20 |
| `down_proj` protected, margin 32 | .884 | .011 | .997 | 1.09 GB | margin is the biggest lever (m 4/8/16/32 = .18/.44/.71/.88) |
| `down_proj` column-restricted, 4096 / 1024 cols | .844 / .70 | 0.000 | .998 / .999 | 178-235 / 43-59 MB | best bytes per teach |
| `o_proj` protected m32 | .773 (exam .83) | 0.000-0.013 (exam .001) | .998-1.000 | 206 MB | best controls and neighbours |
| `o_proj` column-restricted 1536 / 512 | .54 / 0.00 | 0 | 1.000 | 71 / 23 MB | too weak |
| MEMIT (covariance only) | .83 | 0.000 | .997 | 1.09 GB | ok |
| up_proj / gate_proj | .60 / .03 | 0 | 1.00 / .22 | | weak; gate edit breaks the model |
| rank-1 contrast, Steer2Edit-style neuron steering | .21-.36 | 0 | .998-1.000 | | weak |
| diagonal-only (UltraEdit-style) | .98-.99 | **.79-.81** | .96 | | teaches "say the new word" to anything: fails R6 |
| RMSNorm gains, existing v bias | .93-.98 | **.54-.97** | .97-.99 | 0.01-0.08 MB | fails R6; first label lost when a second is added |
| embedding rows (Gauss-Newton) | 0.00 | | | | dead as implemented |
| mid (10-17) / early (2-9) / every-third layers | .005 / 0 / .30 | | | | the label must be written near the output |
| 4-bit body, edited tensor re-rounded to the same grid (`requant4`) | 0.00-0.08 | | | 0 | the edit is smaller than a grid step: carrier bytes are real extra bytes |
| sign-bit flips on 1-bit weights (F07) | 0.00 or 1.00 | 1.00 in the latter | .78-.90 | | no selective edit |
| MoE `lm_head` whitened edit (L2_moe) | fixes 12% | 0 | | | weak |

Findings: protection (key covariance) is what makes an edit selective; the margin is the biggest lever; layers close to the output carry the label; continual feedback (applied twice or three times) keeps earlier labels for every protected solver; a CLI test (answer.py / feedback.py as separate processes, shards rewritten in place) emits the new label end to end. On the shared exam splits the dev separation is not guaranteed (shuffled up to .15-.37 for ridge/alpha/prot at m16 because `Recipe` is lexically close to `Lists`). BANK6 `Security` teach on the never-used source is only .18-.59. With a 9.86% compressed body no legal selective carrier fits under 9.9% (the cheapest, column-restricted `down_proj`, needs 84 MB at 14B width). Legality of every closed-form weight edit is **not ruled**; if "no retraining" (issue #1) is read to cover it, every row above is caught.

---

## 8. Quantizer and byte-trick sweeps

**QUANT_SWEEP** (MoE kept experts at 9.8-9.9%, about 12 GPU-hours, no training): rotate (random-sign Hadamard) + error-feedback rounding to a fixed-rate D4-K256 VQ (2.0 bpw, 2 KB codebook) or entropy-coded scalar g256, keep about 68% of experts: BANK8 99.61-99.67, SKILL7 98.70-99.23. Plain GPTQ int2 g128: erratic (BANK8 99.54 but SKILL7 97.05 without clip search). RTN is the floor (2-bit collapses a cell; 3-bit 98.2-98.8). Rotation barely helps RTN, helps error-feedback and VQ. L3 plan (55/128 experts, 3-bit): BANK8 94.9, NF3 93.8, AWQ hurts error feedback. TCQ accurate but 30-60x slower. Constant prompt prefix must be in the calibration set (excluding it dropped retention from 98.0 to 94.4). SKILL7 is too easy to rank methods; BANK8 separates them. Not run: AQLM proper, QuIP# E8P, PV-Tuning, learned rotations.

**BYTE_TRICKS** (L13, Qwen2.5-14B, seed noise +-1.5 pt SD): quantizing embedding+head to 2-bit (99.8 SKILL7 / 99.3 BANK6; 4-bit 100.0) frees about 9.2 pp vs fp16 tables; rANS indices 3.5 pp vs fixed packing; coded log-scales 0.57 pp; spending freed scale bytes on g64 gives +5 to +9 pts retention at less than fp16-g128 bytes; vocab trim to 65,536 ids frees 6.0 pp from fp16 but is R9 gray (40k vocab fails, 98.8/95.0 and label tokens fall outside); tokenizer.json xz 0.03 pp. Rejected: shared scales across 4 rows (collapse), g256/g512, asymmetric zero point, Hessian-trace/KL per-tensor allocation (no gain), data-free quantizers (need 2.5-4 pp more bytes), zstd/context models (nothing).

---

## 9. Review passes

**RED_TEAM** (skeptical-maintainer review of L1, L2, L9b; raw per-item JSON recomputed, numbers reproduce): as committed none of the three can run from a fresh clone (weights absent, S3-only artifact, calibration file missing, hardcoded paths); R4 GRAY-high for all (MEMIT is a key-value memory keyed on the annotations: the lm_head argument one step earlier); L1 retention 1.000 is by construction and 88.9% vs the ancestor; L2 passes even against the fp16 ancestor (99.32%) but its shuffled arm is weak; L9b has the weakest R6/R8/R9 margins and the highest own-data risk. Predicted maintainer-own-data failures: hard-coded 7-skill prompts, mutually inconsistent annotation formats across CLIs, one-shot feedback, pruned MoE out of distribution, new label close to an old one (Timer vs Alarm). Ranked fixes are listed there (publish artifacts with hashes, remove private paths, ship calibration files, rebuild covariances from unlabeled data, second broader task).

---

## 10. Dead ends (do not repeat)

Letter/MCQ heads and any head/centroid/memory feedback; lm_head row edits as feedback; RMSNorm/bias/steering-vector feedback (not selective); `requant4`; embedding-row edits; mid/early-layer edits; dense conventional compression at 9.6-10% on 14B-405B for tool calls, injection, NEWS, PII, NER, SQL, log templates (90-96.8%); 405B (TF vs GEN gap); 3-bit with fewer experts; random/gate-free/frequency expert pruning; merging experts; top-2 routing; structured FFN/layer pruning; depth/width pruning; BitNet-2B (vocab brittle, R8); Falcon-E (size); Qwen1.5-MoE/DSV2-Lite/OLMoE/gpt-oss; giant MoE at 1.5 bit; training-free TCQ (no gain over scalar/VQ); error-propagation targets (collapse); AI-vs-human detection (base at chance); the old MMLU stand-in; the filed 7B letter entry; BANK6 as a second task for pruned MoE (fails R2 or R5).

---

## 11. Where things are

| entry | package | report | raw results |
|---|---|---|---|
| L1 | `v2/lanes/L1/` | `v2/lanes/L1/REPORT.md` | `v2/lanes/L1/results/`, `v2/bonsai/fbx/results/`, `v2/bonsai/l1plus/results/` |
| L2 | `v2/lanes/L2/` | `v2/lanes/L2/REPORT.md` | `v2/lanes/L2/results/` |
| L9b | `v2/lanes/L9b/` | `v2/lanes/L9b/REPORT.md` | `v2/lanes/L9b/results/` |
| L3 | `v2/lanes/L3/` | `v2/lanes/L3/REPORT.md` | `v2/lanes/L3/results/` |
| L4 | `v2/lanes/L4/`, `v2/rival3/` | `v2/lanes/L4/REPORT.md`, NEAR_MISSES L4 | `v2/rival3/results/` |
| L5 | `v2/lanes/L5/` | SOLUTIONS C33 | `v2/lanes/L5/results/` |
| L9 | `v2/lanes/L9/` | `v2/lanes/L9/REPORT.md` | `v2/lanes/L9/results/` |
| L10 | `v2/lanes/L10/` | `v2/lanes/L10/REPORT.md` | `v2/lanes/L10/results/` |
| L11 | `v2/fb_ablation/` | `v2/FEEDBACK_ABLATION.md` | `v2/fb_ablation/results/` |
| L12 | `v2/lanes/L12/` | `agg.py` | `v2/lanes/L12/results/` |
| L13 | `v2/byte_tricks/` | `v2/BYTE_TRICKS.md` | `v2/byte_tricks/results/` |
| shared exam | `v2/exam/` | `v2/exam/EXAM_PLAN.md` | `python3 -I v2/exam/loader.py` |

Registry of every candidate with the P(all 9) estimates: [`SOLUTIONS.md`](SOLUTIONS.md). Current candidate table: [`../STATUS.md`](../STATUS.md). Entry choice: [`ENTRY_DECISION.md`](ENTRY_DECISION.md).
