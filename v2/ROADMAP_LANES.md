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
