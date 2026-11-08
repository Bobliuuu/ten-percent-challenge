# NEXT_PLAN: designs that could pass all 9 rules + the maintainer check
Written 2026-11-07 ~20:30 EST by the design-synthesis lane. Deadline 2026-11-12 17:05 UTC (about 4.8 days). Source numbers are cited by file. Nothing here was run by this lane (planning only, $0 spent). Other lanes' STATUS files had not appeared yet at writing; update section 7 as they land.

## 0. The one-paragraph situation
Every route that COMPRESSES a conventional model to 10% and then needs >=98% item retention has failed on real exams: Qwen3-32B tool calls 85.8%, Qwen2.5-72B 84.2% (D_scale), best real label exams 95.6% TOOLSEL / 95.2% INJ at 72B (D_scale), Llama-405B INJ 90.2% (L3), MoE 235B+REAP 93-94% (L2), 14B plain 56% (D_scale), and the extrapolated size for 98% is 19-25% of fp16. Bits per parameter, not model size, is the wall (~1.5 bpw). The only construction that gets rule 2 "for free" is a base whose forward pass is IDENTICAL to what we ship, i.e. a model that is already natively low-bit (retention 100% by identity, also on the maintainer's OOD data). That moves the risk from rule 2 to (a) the base-identity reading (rule 1/9) and (b) feedback (rules 5, 6, 4). Ranked designs below follow from that.

## 1. Rules-lawyer: exact wording and what it implies
Quotes are from GOAL_README.md.

**Size (judging rule 1)**: "On-disk bytes of everything you ship (weights, tokenizer, code, feedback storage) against the fp16 size of the model you started from." Maintainer (issue #1, q1): "10% of size on disk is fine compared to fp16 equivalent weights."
- Count: every file in the artifact dir, including tokenizer, config, chat template, our unpack/answer/feedback code, any patched tensors. Exclude torch/transformers (generic runtime), say so. Do NOT ship a custom inference engine (e.g. a vendor llama.cpp fork or kernels): its bytes would count. Plain-torch unpack-to-fp16 at load is ~6 KB.
- "fp16 equivalent" is the maintainer's own phrase: baseline = params x 2 bytes, not the on-disk bytes of whatever format the author released. For Bonsai-8B this equals Qwen3-8B's fp16 size to the byte (same architecture, 16,377,141,728 B), so the size reading cannot matter between "Bonsai fp16" and "parent fp16". Cap = 1,637,714,172 B. Bonsai 1-bit lossless pack = 1,154,719,808 B = 7.05% (B_native). Slack = 482,994,364 B (2.95%).
- "At most 10%": keep <=9.6% so rounding/scale-accounting disputes cannot bite (L3 shows 9.6% vs 9.84% depending on scale accounting). Use REAL written bytes (`size_check.py` over the dir), never entropy estimates.

**Rule 2 (same means faithful)**: >=98% of base-correct items overall, >=95% in every kind (class, source, category). "A higher score from different behaviour doesn't count." Implication: identity of the forward pass is the only construction proof; everything lossy is an empirical gate with the real exam (teacher-forced proxies did NOT transfer: D_scale, L3).

**Rule 3 (unseen ground)**: >=2000 locked, >=300 per kind, >=500 from a source/time period never built, tuned, calibrated on. If the artifact has NO calibration/tuning (lossless repack of a released model), every source is unseen by construction; still hash-lock before the first model run and keep a distinct OOD source block.

**Rule 4 (no new trained parts)**: "Don't add new computation fitted to data: no heads, classifiers, adapters (LoRA included), or side memory the system looks answers up from, and no feedback placed in the prompt. The answer has to come from the original model's own computation."
- The banned list is about ADDED computation/state. An edit of an existing tensor adds no object, but "fitted to data" modifies the weights from data; combined with "no weight training" (rule/README intro, maintainer q2/q3 "no retraining of the weights") this is the real gray zone. See feedback ladder below.
- Restricting lm_head to rows OF THE ORIGINAL head: adds no parameters and no fitted values, so it is not a head under rule 4 provided (i) the row set is chosen from generic-corpus token frequency plus all byte tokens, NEVER from task labels/outputs, and (ii) decoding is ordinary unconstrained argmax over the kept rows. If the kept rows are the task's label tokens (e.g. letters A-E as in the filed C2) it IS a classifier head in substance and a multiple-choice stand-in (rule 7). Input side: dropping embedding rows changes tokenization of rare tokens (they must be re-split into kept tokens), which changes the forward pass vs the base; on the maintainer's OOD data (other languages, code) this is a real retention risk and a rule-9 target. Vocab pruning is legal as "pruning", but treat as a last resort and measure its retention alone (B_native: lossy embed/head/vocab edits alone cost 3-5 pts on Bonsai-ternary, 9-10 pts on BitNet).
- "Side memory the system looks answers up from": prototype/kNN/LDA-mean files, logit-bias tables, steering vectors, cached examples. All banned in substance even if counted in bytes. The filed C2 head+prototype memory is exactly this.

**Rule 5 (feedback has to teach)**: <=25 annotated items "marking real mistakes the way a user would" (so mistakes must be the base's own errors, not synthetic, not oracle-selected pairs); >=25 locked unseen teach items before any feedback; teach-set must improve clearly; nearby exam items must not break (neighbor set). **Rule 6**: same feedback with shuffled labels and with empty feedback must be clearly worse. Note empty feedback must reproduce stage 1 byte-for-byte.

**Feedback legality ladder (my reading, most to least defensible)**:
1. Gradient fine-tuning, LoRA, optimizer on weights: banned.
2. Anything external looked up at inference (memory, kNN, bias tables, steering, prompt): banned by rule 4 / maintainer q3.
3. Closed-form edit of EXISTING tensor rows/weights from the annotations (feedback_v3: `lmhead_*`, `memit_*`), 0 bytes added, no gradient: not named in the banned list, "can't train the weights" literally = no training, but a maintainer can read it as "fitting weights to data = retraining". Gray; the `lmhead_plain` variant (fixed alpha, no covariance/mean) is the cleanest sub-variant but has a narrow alpha window and failed TOX new-label (feedback_v3 section 8).
4. Re-quantization with feedback items added to the calibration/Hessian of the SAME compression procedure: defensible as "a compression technique" (maintainer q2), but only available if our pipeline quantizes (not a lossless repack).
There is no zero-risk feedback. Every design below states which rung it uses. Pre-disclose in the issue.

**Rule 7 (do the real job)**: real task output. Tool call JSON generated greedily from the model's own chat template = clean. Label tasks (moderation, injection, sentiment) are clean only when the label IS the product (classification is the real job); a letter A-E over a menu is the stand-in the rule bans.

**Rule 8**: base well above chance, below ceiling. Check on a 300-item probe before locking (R8 failed for Llama/Mistral/Gemma/GLM under our scorer, L1; AIDET base is at chance).

**Rule 9 (spirit over letter)**: the main exposure of the native-low-bit designs: "entrant did no compression". Counter: README rule 1 says "any model is fine", rule 2's baseline is that model's fp16 size, and the README tagline is "Small, just as good, and still learning". Real exposure remains and cannot be argued away; only the maintainer can close it. Draft question for him is in B_native/REPORT.md (needs user OK to post; it is a new comment, not the filed proof issue).

**Maintainer check**: runs on his OWN different-distribution data with HIS feedback, no fitted computation or side memory. A lossless repack is distribution-free for retention (identical weights); only the feedback code is exposed. Any lossy design (D4) is exposed on both.

## 2. Ranked designs
Probabilities are my honest subjective estimates of "passes every rule as the maintainer will read it", dominated by the unknowns named.

### D1 (rank 1, P~25-30%): Natively 1-bit Bonsai-8B, lossless repack, label-generating task with a taxonomy-extension feedback story (D1b); tool-call variant (D1a) as the rule-7-cleanest alternative
- Base M = Bonsai-8B 1-bit (PrismML, Apache-2.0), its own fp16 checkpoint F (all 254 matrices incl. embed and untied head are exactly +-scale per 128 group, bit-exact round trip verified, B_native). Ship: lossless pack + tokenizer/config + ~6 KB unpack code; at load unpack to fp16 and run the SAME HF forward as the base reference, so outputs are identical (B_native: 39/40 greedy identical via a different kernel, 1-ulp lm_head difference only through stride-dependent cuBLAS; deploying through the same code path removes it).
- Size: 7.05% (+0 for in-place edits). Slack 2.95% can host fp16-patched rows/layers (see feedback), still <=9.6%.
- Retention vs M: 100% by construction (verify 100% output equality on the full locked exam). Rule 3 is trivially satisfied (nothing calibrated). Task choice then only has to satisfy rules 1, 7, 8 and make feedback work.
- Retention vs the parent Qwen3-8B (reading A'): 88.9% overall (S1 96.4, S3 79.6, multi 84.7) = FAIL, ternary Bonsai 91.0% (B_native). So D1 is conditional on the base-identity reading. THIS IS THE RISK. Not a measurement risk, an interpretation risk (rules 1/9).
- Task/answer format:
  - D1a tool use with generated JSON calls (T1; maintainer's own example class). Base on T1 dev: .703 (S1 .823, S3 .583), rule 8 OK. Feedback is the problem: closed-form lm_head edits do not teach natural argument-format mistakes or new multi-token tools selectively (feedback_v3 section 3-4). Probably fails rules 5/6.
  - D1b a real classification product where the label is the output and the feedback story is "user adds a class the prompt did not know" (content moderation +selfharm, sentiment +Neutral, injection +Jailbreak). Closed-form `lmhead_lda` taught +0.62-0.65 on TOX-selfharm at 4B/8B with empty/random-label controls near 0 but gray rule 4 (fitted covariance + class means) and the strict mixed-control gate fails on 8B due to saturation (feedback_v3 section 5). `lmhead_plain` is cleaner but did not move TOX. Base accuracies of Bonsai on these tasks are NOT measured (needs the probe). Rule 7 risk: low-medium (label is the real job for moderation; narrow-distribution risk for rule 1 if single-source).
- Feedback carrier on 1-bit weights: edited lm_head rows stored as fp16 patch rows in the head tensor (25 rows x 8 KB = 200 KB), or re-quantized into the 1-bit format (test which keeps the gain; this is rung 3 vs rung 4). Edited mid-layer down_proj as fp16 costs ~94 MB net each (slack allows <=3). Both stay inside "existing tensors".
- Go/no-go gates (cheapest first):
  - G0 (decision, $0): user OKs posting the B_native question to the maintainer. If he says the parent is the base, D1 is dead (88.9%). If no answer by ~11-09 12:00 UTC, decide on risk appetite; I would NOT file D1 without an answer unless the user explicitly accepts the rule-9 exposure.
  - G1 identity ($3, 1 h on one owned AWS GPU box, profile lab): run F via the reference HF path and the shipped unpack path on 2000 generation items; require 100% output equality and report bytes with the real `size_check`.
  - G2 rule 8 probe ($2, 1 h): Bonsai-8B on 300 dev items of each candidate task (TOX, SENT, INJ, ROUTE with a taxonomy hold-out class, T1): accept tasks with accuracy 0.6-0.92, >=2 classes with >=300 locked items each.
  - G3 feedback ($8-15, 1 day): on Bonsai-8B itself, 25 natural-mistake annotations, >=25 pre-locked teach items, 400 verified neighbors, 10 draws, real vs empty vs shuffled; PASS = gain >=15 pts with paired CI > 0, real - shuffled >= 10 pts, neighbors >= 98%, and the edit survives the carrier (fp16 patch AND re-quantized). Run `lmhead_plain` and `lmhead_lda` and report both. Kill if neither passes by 11-09.
  - Total new money $15-30, wall-clock 2 days to G3, 1 more day for locked exam + stage runs + proof. Fits the deadline with ~1.5 days slack.
- Why it is first: only design where rule 2 and the OOD half of the maintainer check are proven rather than hoped. Why it is not higher: depends on a reading the maintainer has not given, and feedback is unproven on a 1-bit net.

### D2 (rank 2, P~7%; but it UPGRADES D1): same lossless-repack idea on a native low-bit model whose retention vs its PARENT is >=98%
Purpose: make the design pass under both base-identity readings. Candidates: Ternary-Bonsai-2-27B (best file 11.05%, needs -1.05%), larger or newer low-bit releases from labs that release both parent and low-bit student; Bonsai 4B/1.7B (~7% / ~10.2%, but smaller models retain less).
- Evidence against: parent retention rises with scale only slowly (1-bit 88.9%, ternary 91.0% at 8B). A 98%/95%-per-kind parent retention would be a surprise.
- Size: 11.05% needs -1.05% = 0.1x of the cap; options: finer entropy coding of ternary, or trim embedding/head rows (measure alone: B_native saw -3 to -5 pts for lossy embed/head on ternary Bonsai, so trimming has to be LOSSLESS for rows used by the exam, which is not provable for the maintainer's data). Realistically only works if ternary entropy coding reaches 9.9% with no lossy step.
- Cheapest experiment ($8-15, 3 h): run Ternary-Bonsai-2-27B and its parent on the T1 dev probe (600 items), report parent retention overall and per kind, and the true entropy-coded size. Gate: parent retention >=97% overall AND size <=9.8% with no lossy step. Otherwise drop D2 and stay with D1's reading.

### D3 (rank 3, P~8%): provenance-clean natively ternary model trained from scratch (no parent) + vocabulary-only trim, label task
- Candidates: Falcon-E-3B/1B (ternary, trained as BitNet), BitNet b1.58 2B4T, TriLM. No parent model exists, so the reading A' is not available to the maintainer. Base := the ternary forward (for BitNet the bf16 master and the packed release are DIFFERENT forwards, so the base must be defined as the packed/ternary-quantized forward the authors ship; have the maintainer-readable statement ready).
- Size problem: BitNet 2B4T ternary body 8.2% leaves ~2.1 bpw for the 328M-param tied embedding (bf16 total 21.8%, 4-bit 12.0%, 2-bit 10.3%, ternary 9.65%, 1-bit 9.15%, B_native); Falcon-E ~9.97-10.09% unverified. So it only fits with a lossy embedding or a vocab trim, and B_native measured -9 pts at 4-bit embedding on BitNet. Lossless vocab trim (drop rows for tokens that are provably never produced on the exam is not provable for the maintainer's data) is the rule-9/OOD exposure.
- Capability: BitNet tool-call .267 (R8 fail on T1), so only easier label tasks are candidates; "well above chance" and "commercial, not narrow" are in tension at 2-3B.
- Cheapest experiment ($2-5, 2 h): probe Falcon-E-3B, BitNet 2B4T on TOOLSEL/INJ/SENT/TOX dev (600 items each) for rule 8 and compute exact lossless packed bytes with the true embedding. Gate: base acc >=0.65 on a non-narrow task AND lossless size <=9.6% without touching embedding rows. Likely fails the size half.

### D4 (rank 4, P~5-8%): conventional PTQ of a big dense model on a label task, all levers stacked, feedback by rung 3/4
- Base Qwen2.5-72B (best real-exam numbers: TOOLSEL 95.55%, INJ 95.21% at 9.6%, D_scale), with: BI (block-influence) layer-weighted bit allocation (gamma=1 gave +9 pts on 32B tool calls), vocab trim only on the OUTPUT side if at all, task = tool SELECTION without an irrelevance class or with it only if retention on it is fixed (binding cell S1/none 89.3%), annotation-weighted requant as feedback (rung 4).
- Needs +2.5 pts overall and +5-6 pts in the binding cell vs the best measured; no tested lever delivered that (everything else was inside noise, D_scale). Dropping the "none" class changes the product (rule 7/1 exposure: ms0-dev precedent was ruled a narrow router).
- Cost: $120-250 and 2.5 days for quantization + real-exam evaluation of 72B on Modal H100-less caps or owned AWS; exceeds my $30 lane budget, needs user OK.
- Cheapest experiment ($30-40 on one box, 1 day): re-run the best 72B config (V7) with BI gamma=1 and no-irrelevance TOOLSEL exam on 500 items; gate: >=98.5% overall AND >=96% in every kind with Wilson lower bound >=96%. Else kill. Realistic outcome: fail.

### Rejected (do not spend): filed C2 7B artifact (rule 4 head+memory, rule 7 letters, 93.5% retention); fix_real 7B real tool-call (0.934 retention, MMLU gate fails, no feedback mechanism); 405B Llama (INJ 90.2%, 100+ GB eval infrastructure); MoE+REAP (<=94% at 9.6%, only 14-22% reaches 98%); Maple/gpt-oss/Kimi/NVFP4 releases (12-31% of fp16); ternary Bonsai-8B at 10.19% (31 MB over cap; vocab trim breaks identity); Llama/Mistral/Gemma/GLM (fail rule 8 under our scorer; probe-harness artifacts, L1).

## 3. Cross-cutting decisions that need the user (nothing else blocks)
1. Post the B_native 3-question maintainer question as a NEW comment on issue #1 (not on our filed proof issue)? It is the single highest-value action: it decides D1's viability for $0. Risks: public, tips off rivals to the lossless-repack idea (ms0-dev is active). Alternative: decide by risk appetite without asking. My recommendation: ask, with the question phrased neutrally, and run G1-G3 in parallel so no time is lost.
2. Is a base-identity rule-9 exposure acceptable if he does not answer? I would not file.
3. If D1 and D2 both die by 11-09 evening, the honest outcome is "no entry that passes all nine as written" (D3 and D4 are long shots). Do not file a failing entry as a pass (DECISION.md).

## 4. Schedule (EST evening 11-07 start; deadline 11-12 12:05 EST)
- 11-07 night: G2 probe (Bonsai on candidate tasks) and G1 identity run start immediately on an owned box; D2 parent-retention probe in parallel; D3 probe optional.
- 11-08: G3 feedback experiments on Bonsai; draft + post (if approved) maintainer question; lock exam files (hash) before any exam-set model run.
- 11-09: kill/continue decision by evening (all gates reported).
- 11-10: locked exam stage 1, stage 2 (feedback), controls (shuffled, empty), validity suite.
- 11-11: proof write-up with anticipated objections (base identity, no compression done by us, feedback edit class, size counting), size_check over the real tree, repro commands.
- 11-12 by 12:00 UTC: file only if every gate passed.

## 5. What I would NOT do
- Not touch locked exams or the filed issue; no Vast; no IPs/keys in committed files.
- No teacher-forced proxies as pass evidence (they overstated retention by 3-10 pts, D_scale/L3).
- No new head, prototype file, steering vector or bias table in any feedback variant.
- No prompt-side feedback of any kind.

## 6. Open measurement facts needed (not yet in the repo)
Bonsai-8B accuracy on label tasks (TOX/SENT/INJ/ROUTE); feedback edit efficacy and carrier survival on a 1-bit net; parent retention of any larger native low-bit student; true entropy-coded size of Ternary-Bonsai-2-27B; Falcon-E exact lossless bytes.

## 7. Updates from parallel lanes
(none yet when written; append dated entries here as STATUS files appear)
