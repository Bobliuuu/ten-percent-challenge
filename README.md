# The 10% challenge: our attempt

We are attempting the [10% challenge](https://github.com/DeusExperiments/ten-percent-challenge): keep at most 10% of a trained model's fp16 size, match its behaviour on an exam it never saw, and let it learn from a few user annotations, all without training the weights. This repository is the working record: code, locked exams, results, and every dead end.

**Honest current state (2026-11-08):** no entry of ours passes all nine judging rules as the maintainer is likely to read them. We have not claimed otherwise.

- **Filed entry (issue #2): does not pass.** Qwen2.5-7B-Instruct at 9.404% of bf16 on a tool-selection task kept 93.9% of the base model's correct items on its locked exam (rule 2 needs 98%), answered with option letters (rule 7), and gave feedback through a fitted head, which the maintainer ruled out (rule 4). Its exam is used up; it will not be refiled.
- **Compression is no longer the blocker** on an easy 7-way voice-skill routing task, scored once on our own sealed exam (2,744 items). Our best entries keep 99.3% to 100% of the base model's correct items at 7.4% to 9.9% of fp16, generating a real label word.
- **Feedback is the open wall.** Our feedback is a closed-form edit of existing weights (no head, adapter, lookup or prompt; the new label comes out of the model's own forward pass). The maintainer has rejected a fitted head and a nearest-centroid row overwrite, and has not ruled on a closed-form edit of existing tensors. Until he does, rule 4 is gray for every candidate.
- **Starting from an already-compressed released model** (two of our candidates) is maybe valid and unruled.
- **If the closed-form-edit ruling goes against us, we file nothing as a pass.**

## Candidates (our own sealed exam, scored once)

Our exam is built from public data and sealed by hash before any run; it is not the maintainer's exam. "Retention" is the share of the base model's correct items that the compressed model also gets right, from real generation.

| id | start model | bytes vs fp16 | retention | feedback: new label taught, shuffled-label control | main open points |
|---|---|---|---|---|---|
| L9b (recommended primary) | Qwen3.5-35B-A3B, pruned experts + 3-bit | 9.67% (9.70% with feedback carrier) | 99.55% | 73-79%, shuffled 2-47% | one control draw is weak, base near ceiling, task-specialised pruning |
| L2 | Ternary-Bonsai-2-27B (released ternary), recompressed by us | 9.94% | 100.0% vs the release; 99.32% vs its fp16 ancestor | 83%, shuffled 0% | already-ternary start, little byte slack |
| L1 | Bonsai-8B (released 1-bit), lossless repack | 7.44% | 100% vs the release (by construction); 88.9% vs its fp16 ancestor | 89%, shuffled 0.5% | we did no compression |
| L3, L4, L9, L10 | other MoE and multimodal bases | 9.7% to 9.9% | 98.9% to 99.6% | partial, weaker, or accounting only | see [`STATUS.md`](STATUS.md) |
| L5, L12 | dense 32B; MoE ablation | pending | pending | pending | not scored yet |

Which entry we would file, and what we do if rulings go against us: [`v2/ENTRY_DECISION.md`](v2/ENTRY_DECISION.md).

## Where to read

| file | what |
|---|---|
| [`STATUS.md`](STATUS.md) | concise current status and candidate table |
| [`v2/WHAT_WE_TRIED.md`](v2/WHAT_WE_TRIED.md) | every lane, method and task we tried, with results and dead ends |
| [`v2/ENTRY_DECISION.md`](v2/ENTRY_DECISION.md) | which entry to file and the fallbacks |
| [`v2/RULES.md`](v2/RULES.md) | the rules and the maintainer's rulings, single source of truth |
| [`v2/CANDIDATES.md`](v2/CANDIDATES.md), [`v2/SOLUTIONS.md`](v2/SOLUTIONS.md) | per-entry detail and the registry of every candidate |
| [`v2/RED_TEAM.md`](v2/RED_TEAM.md) | skeptical review of our entries |
| `v2/lanes/` | one directory per lane: code, report, manifest, results |
| `v2/exam/` | the shared sealed exam and loader |
| [`GOAL_README.md`](GOAL_README.md) | our copy of the upstream rules |

Older material (the small-model tracks `track*/` and the filed 7B entry under `v2/`) is kept as a record only.

## Reproduce

Each lane directory has a `REPORT.md` and `MANIFEST.json` with the commands and byte accounting. Large artifacts are not stored in git; they are rebuilt from the public base-model snapshots with the lane's pack scripts. The shared exam loads with `python3 -I v2/exam/loader.py`. Several lanes still need their artifact published with hashes before a fresh clone can run end to end (see `v2/RED_TEAM.md`).

Most of this was built by a swarm of [Claude Code](https://claude.com/claude-code) agents working in parallel, with every result pushed as it landed.
