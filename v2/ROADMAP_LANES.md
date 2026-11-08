# ROADMAP: many presentable lanes (2026-11-08)

Policy (user): GUESS from the evidence; gray is FINE; we need MANY lanes that WORK end to end so we can PRESENT them. Each lane = one complete entry-shaped package with honest numbers and its weakest rule named. Not required: a clean pass. Required: it runs, it is measured, it is documented. Never claim a pass that did not pass.

## Shared defaults (guesses, so lanes are comparable)
- Primary EASY task: voice-assistant skill routing (MASSIVE + CLINC150, 7-10 skills, real generated skill name). Secondary: Banking77 grouped into 10 queues. Easy sweep (v2/EASY_TASKS.md) may replace these with better ones; lanes switch when it lands.
- Feedback default: diffuse closed-form edit of existing tensors (v2/fb_edit/), new label generated end to end; real/shuffled/empty controls.
- Package per lane in v2/lanes/<ID>/: answer.py, feedback.py, MANIFEST.json (all shipped bytes), REPORT.md (9-rule table, weakest rule, numbers, cost), results/. Push after every result.

## Lanes
| ID | Lane | Start model | Guess of weakest rule | Owner agent |
|---|---|---|---|---|
| L1 | Bonsai-8B 1-bit + column-sparse diffuse feedback, easy task | Bonsai-8B | base identity (gray) | Bonsai agent |
| L2 | Ternary-Bonsai-27B (our compression 9.8%) + feedback | Ternary-Bonsai-2-27B | base identity (gray) | rescue/27B |
| L3 | MoE task-pruned + GPTQ, beat rival #4 (easy task) | Qwen3-30B-A3B | feedback legality | MoE-#4 agent |
| L4 | MoE 96/128-style + low bit, beat rival #3 (banking10) | Qwen3-30B-A3B | feedback legality | MoE-#3 agent |
| L5 | Dense 32B/72B scalar + per-cell repair, easy task | Qwen2.5-32B/72B | retention per cell | repair agent |
| L6 | Other MoE bases pruned (OLMoE, gpt-oss, GLM Air, etc.) | best from census | size/retention | other-methods agent |
| L7 | Other native low-bit starts (BitNet-b1.58, Falcon-E) | released low-bit | base identity | other-methods agent |
| L8 | Depth/width pruning + quant | dense | retention | other-methods agent |
| L9 | Big-model (70B-235B) 1.6 bpw easy task | census pick | bytes/retention | census agent |

Support lanes: easy-task sweep, task catalogue, model census, quantization sweep, exhaustive research feed all lanes (they write v2/EASY_TASKS.md, TASK_CATALOGUE.md, MODEL_CENSUS.md, QUANT_SWEEP.md, RESEARCH_EXHAUSTIVE.md; lanes read them and adopt the best).

## Timeline (UTC)
- 11-08..11-09 20:00: every lane builds a first complete package on dev data (GUESS the config; do not wait for sweeps).
- 11-09..11-10: second pass on lanes whose dev numbers are within 3 pts of the gates; swap in the best task/model/method from the sweeps.
- 11-10 20:00: lanes freeze configs; build final artifacts; locked-exam files sealed by hash.
- 11-11 08:00: global freeze. One sealed-exam scoring per lane that is worth presenting (build a fresh sealed exam per lane if X15 is single-use).
- 11-11..11-12 12:00: write presentation (STATUS.md lane table, README), file the best entry only if honest.

## Lane scoreboard (updated as results land): see v2/NEAR_MISSES.md and STATUS.md

## PRIORITY (decided 2026-11-08 from evidence; supersedes the equal weighting above)
Evidence: dense quant (scalar/VQ/TCQ/GPTAQ) all 92-97% at <=10% -> method-insensitive, dense = fallback. MoE task-pruned experts + low-bit: rival #4 99.36% @ 9.892% (2,910 items), rival #3 96.0 vs 95.8 @ 10.48%, maintainer praised compression -> best route. Easy narrow tasks retain better. Feedback is the bottleneck: lm_head rows rejected (#3); diffuse down_proj works (teach .6-.8, shuffled ~0, neighbours .99), legality unruled. Native low-bit passes only vs its own checkpoint (identity gray).
1. L3 Qwen3-30B-A3B, ~43% experts kept by routing mass, 3-bit GPTQ+Hadamard on kept experts, easy routing task, diffuse feedback.  P(all 9) ~20%
2. L4 same model, banking10, feedback done right.  ~15%
3. L9 bigger MoE (Qwen3-235B-A22B, gpt-oss-120b, Qwen3-Next-80B) same recipe.  ~12%
4. L1 Bonsai-8B easy task + column-sparse feedback.  ~8% (identity gray)
5. L2 Ternary-Bonsai-27B + feedback.  ~6% (identity gray)
6. L6 other MoE bases.  ~8%
7. L5 dense 32B + repair (fallback only).  ~3%
Dropped: L8 depth/width pruning, VQ/trellis/GPTAQ sweeps. Quant sweep refocus: best quantizer for KEPT MoE EXPERTS at 2-3 bits. GPUs go to L3/L4/L9 first.

## Research feed adopted (v2/RESEARCH_EXHAUSTIVE.md, queue rows X16-X28; arXiv ids unverified until checked)
- MoE upgrade (X17/X18): expert scoring beyond routing mass (REAP 2510.13999, unified score 2606.15716); GPTQ/VQ on kept experts. -> L3, L4, L6, L9.
- Denominator bonus (X16): multimodal start models (Llama-3.2-11B/90B-Vision, Qwen2.5-Omni-7B) give ~2.0-2.2 bpw on the text path at 10% of whole-model fp16. Gray-low (R1 literal, R9 spirit): disclose. -> NEW lane L10.
- Diffuse feedback (X19): AlphaEdit null-space, UltraEdit, Steer2Edit, edit down_proj in 3-5 layers, no lm_head row edit, no stats side file. -> L11.
- Byte tricks (X20/X21/X24): coded scales (~0.03 bpw vs 0.125), ANS-coded indices (+1-6%), data-free quantizers (HIGGS/HyperQuant/AIMER), vocab trim with byte-fallback kept. -> L13.
- Bonsai byte audit (X22): Ternary-Bonsai-27B ~10.9%; 8-bit scales g256 -> ~10.06%, borderline. -> L2.
- Guard-model start (X23): over-confident, big margins. -> L6/support.
- Quantizer upgrades (X25): Proteus, LLVQ, residual compensation. -> quant sweep.
- Layer removal + compensation (X26-28), shared-subspace MoE (GS-MoE/D2-MoE), fine-grained Qwen3.5 MoE bases (35B-A3B 256 experts, 397B-A17B 512). -> L12, L9.
- Rival #4 note: its feedback is the row edit the maintainer rejected for us and #3, and it counts a 1 MB feedback_stats file; do not copy that.

## New lanes (added 2026-11-08)
| ID | Lane | Type | Owner |
|---|---|---|---|
| L10 | Multimodal denominator bonus (text path at ~2.1 bpw) | entry lane | L10 agent |
| L11 | Feedback ablation: AlphaEdit/UltraEdit/Steer2Edit/MEMIT/row edit on ONE fixed model; layers, edit count, key-covariance protection, paraphrase and cross-source teach; best legal-ish carrier by bytes | research/ablation | L11 agent |
| L12 | MoE combos: expert prune x merge x shared-subspace x quant (GS-MoE, D2-MoE, REAP) on Qwen3-30B-A3B, fine-grained Qwen3.5-35B-A3B | combination | L12 agent |
| L13 | Byte tricks ablation: coded scales, ANS indices, data-free quantizers, vocab trim, tied/deduped tensors, ship-code size; how many % free per lane | ablation | L13 agent |
