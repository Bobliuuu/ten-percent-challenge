# REDESIGN v3 — built against the live README (commit 645d1614, 2026-11-06). Goal: BEST entry, no gray.
Deadline 2026-11-12 17:05 UTC. Status: PLAN, nothing built, nothing committed.

## 0. Why the C2 artifact is dead (do not salvage)
- Rule 2 faithfulness: kept 93.5% of base-correct items (need >=98% overall, >=95% per kind). "None of these tools" kind: cand .64-.65 vs base .77.
- Rule 4: head.npz (PCA+LDA fitted to 2000 labels) + prototype memory = banned fitted part. Maintainer ruled this on issue #2.
- Rule 7: letter A-E over 5 lm_head rows = multiple-choice stand-in.
- Rules 3/5/6: no out-of-source block, no locked teach-set, no shuffled/empty controls.
Reusable: ANS entropy coder, Hadamard rotation + GPTQ pipeline, harness lock/receipt/sandbox design (`harness/exam_remote.py`), `validity/` suite.

## 0b. What "BEST" means (read from the live README; the maintainer never defines it, so this is inference)
Facts: "Among the entries that pass every rule, the best one wins." => two stages. (1) Pass/fail on all 9 rules + maintainer's check. (2) Rank the passers. Prize is paid only after the maintainer runs it on HIS OWN held-out data from a different distribution with HIS OWN feedback => OOD behaviour is the real final exam, dev numbers are not.
Ranking axes I will optimise, in priority order (a score vector; aim to dominate, not trade):
1. Passes everything with margin, not at the edge: size well under 10% (target <=9.5%), retention >=99% overall and >=97% on the worst kind, OOD-block retention as good as in-distribution.
2. Survives the maintainer's check: no task-specific code, feedback works on tool/schema styles we never saw. Gap between our S1 and a never-touched S3 is the overfit detector.
3. Feedback strength with zero side effects: biggest teach-set gain, neighbors untouched, 0 bytes added, controls (shuffled/empty) clearly worse.
4. Difficulty/utility: larger base compressed harder (e.g. 30B+ at 10%), commercial-class task from the maintainer's list (agentic/tool use, PII redaction, AI-vs-human detection), real generative output (harder than a label), more kinds covered.
5. Reproducibility: one `answer` command, one `feedback` command, hashes, receipts, a runnable proof.
Rival check (issue #3, entrant #3, Qwen3-30B-A3B banking router): 10.48% of fp16 is ABOVE the 10% cap ("at most 10%"); 10-family Banking77 router is a narrow distribution (rule 1); BM25 top-8 retrieved examples are placed in the prompt at inference (RAG, ruled out in issue #1); locked exam taken from the same dataset (no 500-item other-source block). They are likely not a passer, but they will keep fixing it: assume a clean competing entry can appear any day. Our edge must be real margins + OOD robustness, not a thinner reading of the rules.
Optional (needs user OK, it posts publicly): ask tokenbender on issue #1 or in our proof thread what "best" ranks on (size margin? retention? base size? feedback gain?).

## 1. Rule -> mechanism -> how it is proven (BY CONSTRUCTION vs EMPIRICAL GATE)
| # | Rule | Mechanism | Type |
|---|---|---|---|
| size | <=10% of fp16, all bytes on disk | ship at <=9.6% (0.4% slack for feedback re-encode); `size_check.py` sums every file, run after build AND after feedback | construction + check |
| no-train | no weight training | only PTQ/rotation/pruning/entropy coding + closed-form edits. No optimizer, no loss.backward anywhere. Repo grep `backward|optim|\.grad` in build dir must be empty; included in receipt | construction + grep |
| R4 | no new trained parts | artifact = original architecture, original tensors only (re-encoded). NO head, classifier, adapter, LoRA, memory file, prompt change. Feedback edits existing tensors in place, adds 0 new objects | construction |
| R7 | real output | model generates the tool call (JSON/XML via the base's own chat template). Full lm_head kept, no letter rows, no constrained decoding | construction |
| R3 | exam unseen | locked BEFORE any build; >=2000 items, >=300 per kind, >=500 from a source never touched (see 2) | construction (lock hashes) |
| R6 | controls | shuffled-label feedback + empty feedback run through the same pipeline | construction (scripted) |
| R2 | faithful >=98% / >=95% per kind | PTQ quality | **EMPIRICAL GATE G1** |
| R5 | feedback teaches | in-place edit generalizes to >=25 locked unseen teach-items w/o breaking neighbors | **EMPIRICAL GATE G2** |
| maint. | maintainer's own different-distribution data + own feedback | no task-specific code in either command; works for any tool-use feedback | **EMPIRICAL GATE G3** |
Honest: G1/G2/G3 can fail. Section 7 has kill criteria and a fallback; no rule is "argued" past a failure.

## 2. Task, data, exam (R3, R7, R8)
Task: agentic tool use, real output. Input = tool schemas + user request via chat template; output = the model's generated call(s) or no call. Scoring: canonicalized exact match (names + args, order-insensitive for parallel calls). "Item right" = exact match.
Sources (decide + hash-lock on day 1, before any model run):
- S1 build/dev/calib pool: xLAM function-calling (APIGen). Split by sha256 as before.
- S2 OOD locked source: a second, never-touched tool-use set (BFCL v3 / ToolACE / Hermes-FC; pick one with different schema style and time period). Fetched into the exam volume only. Never in any calib/dev/build script. >=500 items.
- Kinds (each >=300 locked): no-call/irrelevance, single call, multi/parallel call, plus OOD-source block (>=500). Locked total target 2400+.
- R8: probe base on 300 S1 dev items: must be well above chance and below ceiling (target 60-92%).
- Locked set stays on the Modal exam volume (labels never in candidate sandbox, network blocked), single-use claim per stage as in `exam_remote.py`. Extend runner to generation.
- Fix last time's weaknesses: leakage audit (`validity/leakage_audit.py`) run on S1 before lock; drop exact/near duplicates from locked.

## 2b. Third source S3 = OOD DEV PROXY (new)
Rule 3 forbids tuning on the 500-item S2 block, and the maintainer will test a different distribution anyway. So we need an OOD signal we ARE allowed to tune on: S3 (a third tool-use/task source, different style/period from S1 and S2), used only as dev. Every compression and feedback hyperparameter (how many experts to prune, bit allocation, edit layers) is chosen to maximise retention on S3-dev as well as S1-dev. S2 stays untouched until the locked run. Over-pruning to S1 routing is the classic overfit; S3-dev is what catches it.

## 2c. Model x task TOURNAMENT (do this first, cheap, 1-2 days)
Task candidates (all on the maintainer's list or clearly commercial; choose one by measured results, not taste):
- T1 agentic tool use (real output = generated call). Rich OOD sources (xLAM / BFCL / ToolACE / Hermes-FC). Feedback is hardest (multi-token names).
- T2 PII redaction (real output = redacted text or entity spans; kinds = entity types, natural OOD = other domains/languages). Per-kind retention is easy to define; feedback = a new entity type.
- T3 AI-vs-human text detection (real output = label; kinds = generator/source; natural OOD = new generators/time periods). Small output space makes faithfulness and output-row feedback easiest, but check base is above chance and below ceiling (R8).
Base candidates (fp16 size, 10% budget): Qwen3-30B-A3B-Instruct-2507 (61 GB, 6.1 GB) | Qwen2.5-32B-Instruct (65 GB, 6.5 GB) | Qwen3-14B (29.5 GB, 2.95 GB) | Qwen3-8B / Llama-3.1-8B-Instruct control (16 GB, 1.6 GB) | gpt-oss-20b (MoE; size is bf16-equivalent, check which counts) | Qwen3-Next-80B-A3B (stretch; ~160 GB, many experts, needs multi-GPU; only if budget allows).
Insight to test: on a MoE the budget can be spent on PRECISION instead of breadth. Keeping 50% of experts at ~3.2 bpw or 25% at ~6.4 bpw both cost 10%; fewer, near-lossless experts maximise retention IF the task routes to few experts, but over-pruning hurts the OOD block. S3-dev decides the point on that curve.
Tournament protocol (same harness, same scorer, dev data only, locked untouched):
1. Base score on 300 S1-dev + 300 S3-dev per (task, base): keep pairs with R8 true (well above chance, below ceiling, target 60-92%).
2. Cheap compressibility proxy per surviving pair at ~9.5%: rotation + RTN/GPTQ at 1.6-2.5 bpw (dense) or expert-prune + 4-6 bpw (MoE). Metric = retention of base-correct items on S1-dev AND S3-dev, per kind.
3. Advance the top 2 pairs to the full pipeline (section 3) and feedback test (section 4). Pick the final pair on the full axis vector of 0b, not on retention alone.
Time-box: 11-06 night -> 11-08. A pair is dropped early if proxy retention <90% at <=2.5 bpw.

## 3. Base model + compression (R1, R2, size)
The binding constraint is 1.6 bits/param all-in at 10%. Choose the base for compressibility, by probe, not by guess (2-day timebox, same 300+300 dev items, generation scoring):
- A: Qwen3-30B-A3B (MoE; ~95% of params in experts; task-specific expert pruning from routing stats = compression; competitor on issue #3 reports 96.0 vs 95.8 at 10.48%, so it is feasible but they are over 10%).
- B: Qwen2.5-32B-Instruct (dense; larger = more redundant at fixed bpw).
- C: Qwen2.5-14B-Instruct / 7B as control.
Pick by measured G1 retention at <=9.6%. Don't pick on size alone.
Pipeline (no training): Hadamard/incoherence rotation -> Hessian-aware PTQ (GPTQ/QTIP-style codebooks) -> per-tensor bit allocation by measured end-to-end sensitivity (KL of generated-token logits on S1 calib; mixed 1-4 bit) -> keep outliers/norms/embeddings at needed precision -> rANS entropy coding (existing). MoE: expert prune/merge by routing frequency on S1 calib. Closed-form bias/scale correction allowed (standard compression).
Not used (they hurt R2/R7/spirit): output-vocab trimming, lm_head row cutting, letter-only heads, constrained decoding.
Gate G1 (on S1 dev + S2 DEV-slice, NOT locked): >=98.5% of base-correct kept overall, >=96% per kind, incl. no-call kind. If G1 fails for A,B,C at 9.6%, stop and use the remaining days on the best-possible honest entry (see 7).

## 4. Feedback (R4, R5, R6) — the hard part
Constraints: no weight training, no RAG/prompt edit, no head/memory, <=25 annotations, must clearly improve unseen "what the feedback teaches" items, must not break neighbors, size still <=10% after.
Design: **in-place closed-form edit + requantize, zero added bytes.**
1. An annotation = (request+tools, model's wrong call, user's correct call). <=25.
2. Run the compressed model once; for each annotation take the hidden key at the decision token position(s) and the value delta needed to produce the correct token (closed form).
3. MEMIT-style least-squares update to the existing MLP `down_proj` of a few mid layers, regularized by the activation covariance computed from S1 calib text (same stats compression uses). No gradients, no optimizer, no new tensor.
4. Re-quantize ONLY the edited matrices with the original codebooks/scales and re-encode; recheck size. The artifact after feedback has the same file set, same architecture.
5. Same code path for any feedback. No C2/tool-name-specific logic. (G3)
Pre-registered locks before feedback: teach-set (>=25 unseen items about what the feedback teaches, e.g. same confusable tool pair / same new-schema pattern as the annotations) and neighbor-set (>=100 items touching the same tools where base is right).
Pass (G2): teach-set accuracy up by >=15 pts with paired CI excluding 0; neighbor retention >=98%; overall locked-exam retention still >=98%.
R6 controls, run through the identical pipeline: (i) empty feedback (must equal stage 1 byte-for-byte); (ii) shuffled labels (annotation corrections permuted). Both must be clearly worse than real on the teach-set (>=10 pts below real) with CIs.
Fallbacks if edit doesn't generalize (kill at day 3): (a) annotation-weighted re-quantization (feedback enters only as extra Hessian weight in the same PTQ; zero new parts); (b) in-place lm_head row edits (issue #3 style). Each must pass the same G2/R6 test. If none pass, we do not claim step 5; we submit nothing misleading.

## 5. Deliverables (README "maintainer's check")
Artifact dir; locked files; command 1 `answer` (tools+request -> call); command 2 `feedback` (annotated items -> edited artifact). Both documented, runnable offline.

## 6. Validity suite (the original ask) — a hard gate on the new artifact
Reuse `harness/validity/`: leakage audit (before lock), stratified locked re-slice (per kind, OOD block, unseen tools), general-ability probe (MMLU/ARC/HellaSwag via generation or letter logits on both models: artifact must stay within 2 pts of base), tool-order/paraphrase shift on dev. Overfit signature = gap(base,cand) larger on OOD block than on S1.

## 7. Schedule and kill criteria (today 11-06 -> 11-12 17:05 UTC)
- D0-1 (11-06/07): lock S1/S2 splits + hashes, leakage audit, generation scorer, probe bases (R8). 
- D1-3: compression sweep on A/B/C -> G1. Kill: if best <96% retention at 9.6%, widen compression search for 1 more day, else fallback.
- D3-4: feedback edit + fallbacks -> G2 + controls on dev-derived teach/neighbor sets. Kill 11-09.
- D4-5: locked exam stage1, stage2, controls (single use). Validity suite. 
- D5-6 (11-11/12): README/proof, issue filed by 11-12 12:00 UTC (5 h margin).
Fallback if G1 fails: no honest claim of R2 is possible; do not file a failing entry as passing. Report that to the user for a decision.

## 8. Cost (needs user OK before launch)
Probes + sweeps + locked runs on Modal, `--detach`, all results/logs into repo: rough $150-400 (A100/H100 hours; the 30B/32B probes dominate). Will post a concrete estimate after the day-1 probe sizing.

## 9. EXTRAPOLATE, DON'T EXHAUST (added 2026-11-06 on user request; spend cap lifted, speed is the goal)
Never run the full pipeline on a config until cheaper runs predict it will pass. Four predictors, cheapest first; each reports predicted vs actual so we know how far to trust it:
1. **Successive halving on items.** Stage 0: 64 prompts, ONE teacher-forced forward pass (top-1 agreement, KL) + 16-token greedy match. Kill anything below the stage-0 bar. Stage 1: 200 prompts. Stage 2: full eval. A config reaches stage 2 only if its stage-1 lower bound beats the current best minus a margin.
2. **Layer-subset additivity.** Quantize/prune only a spread subset of layers (e.g. every 4th), measure KL; full-model KL ~ sum of per-layer delta-KL (first-order). Calibrate the multiplier once on a model where the full run is cheap (Qwen3-8B). Gives a full-model prediction for ~1/4 of the work and ranks per-layer sensitivity for free (feeds the bit allocation).
3. **Size-scaling extrapolation.** Same recipe, same bits/param, on Qwen3-0.6B / 1.7B / 4B / 8B / 14B (dense) and a small-to-large MoE ladder (OLMoE-1B-7B, Qwen1.5-MoE-A2.7B, Qwen3-30B-A3B). Fit item-retention loss vs log(params); extrapolate to the 30B/32B target. Failing at 8B is NOT a kill by itself (larger models compress better); the fitted trend plus its CI is.
4. **Metric mapping.** Fit item-retention (the real rule-2 number, from T1-T3 dev items with gold) against teacher-forced KL / top-1 agreement across all configs seen so far. Then every cheap KL measurement converts to a predicted rule-2 retention with a CI.
Go/no-go: continue a (task, base, recipe) only if the predicted retention lower bound >= 98.5% overall and >= 96% worst kind, at <= 9.6% bytes. Kill otherwise and log why. One full run of the top candidate is mandatory to validate predicted vs actual before trusting the predictors for the rest.
Same idea for feedback: teach-gain and neighbor damage vs model size 0.6B -> 8B; extrapolate the trend before running on the target model.
