# ISSUE DRAFT (fallback entry L9b). NOT POSTED.

Use only if reading B (pre-compressed start) is dropped by the owner. Same placeholders and checklist as `ISSUE_DRAFT.md`. Numbers: `v2/lanes/L9b/REPORT.md`, `MANIFEST.json`, `v2/ENTRY_DECISION.md`.

---

**Title:** Entry: Qwen3.5-35B-A3B, our own compression (expert prune + 3-bit), 7-way voice-skill router, 9.70% of fp16

## What we ship
- **Start model:** Qwen/Qwen3.5-35B-A3B (35.95 B parameters, 256 experts per layer), unmodified full-precision release. **Compression is ours:** keep 102 of 256 experts per layer, chosen by routing mass on unlabeled task prompts (no labels, no gradient), 3-bit grouped RTN on kept experts, 6-bit attention and shared expert, 4-bit embedding and output layer. The vision tower is not shipped.
- **Task:** 7-way voice-skill routing; the model writes the skill word (greedy, exact match, no constrained decoding).
- **Feedback:** at most 25 annotations `{"text","label"}` (new label allowed). A closed-form edit of the shared-expert `down_proj` columns of the last 10 MoE layers, stored as replacement columns in `fb.bin` (20,981,760 B).
- **Artifact:** <<ARTIFACT LOCATION AND SHA256, owner to fill; body.bin sha256 `e0a1266db5eb7c4f877dfaeb107bf41baa5b5240ff945eb4d42a52f400efe88b`>>. Locked files: `v2/exam/` in <<REPO LINK>>.

## Size
| item | bytes |
|---|---|
| `body.bin` | 6,685,765,376 |
| tokenizer, config, template, manifest, `core.py`, `qpack.py`, `answer.py` | 20,133,949 |
| feedback carrier `fb.bin` | 20,981,760 |
| feedback code, task json, calibration file (110,789 B incl. 101,819 B calibration, 300 rows) | 110,789 |
| **total** | **6,726,991,874 = 9.704%** of the text-only fp16 size (69,321,221,376 B); 9.35% of the full checkpoint including the vision tower (71,903,645,408 B) |

The denominator is the text-only size (the vision tower is dropped for a text task). Against the full checkpoint the figure is lower, so this is the less favourable reading. Without feedback files: 9.674%.

## The 9 rules as we read them
| # | status | evidence and caveat |
|---|---|---|
| 1 | PASS | 9.704% as above; slack about 157 MB to 9.9% |
| 2 | PASS | retention 99.55% [99.21], every kind at least 98.4 (Lists), worst source-by-kind cell 97.1; never-used sources 99.45% [98.91]; 3 quantizer draws 99.47 to 99.55. One base run |
| 3 | PASS on counts, caveats | 2,744 items, 344 to 420 per kind, 1,448 base-correct from never-used sources; the exam is ours; on the exam without `seen_tox` rows, Lists and Transport are below 300 per kind |
| 4 | **GRAY** | pruning and rounding select or round existing weights using unlabeled task prompts; feedback is a closed-form edit of existing columns stored as a carrier (readings A, C, G) |
| 5 | PASS (moderate), if A holds | teach 0 to .74-.76 (8-draw means, .60 to .83) on 164 locked items; neighbours 98.0 to 98.6% vs 98.3% empty |
| 6 | **GRAY on bad draws** | shuffled mean .16 to .18 over 8 draws, range .00 to .40 (.47 on one early draw); real .74 to .76; wrong 0; empty 0. A strict-controls option (`--m 32 --lam 2000 --awt 50`) gives shuffled .02 mean (max .09) but teach only .53 |
| 7 | PASS | generated skill word |
| 8 | **WEAK** | base .9676, chance .14, only 3.2% errors left to keep |
| 9 | **GRAY** | 60% of experts removed using the task's own prompts, so the packed model is a task specialist |

## Own-data dry run (our data, not yours) and BANK6: this entry is weak here
- Banking77 (200 test queries, 77-label menu): base 70.0%, packed 61.0%; retention **79.3% [71.8]**; identical output 72.5%.
- Same queries under the router prompt: same label as base 46%.
- Generic prompts: identical continuation **11%** (for example "17 times 23" gives 17). **The packed model diverges badly from the base outside its task.**
- Unseen label `Timer`: emitted 100%, but **21.7% of Alarm requests flip to Timer**.
- Second task BANK6, same recipe with RTN: retention 97.37% [96.58], Card 94.4, Identity 91.7, which fails 98/95. A better quantizer (Hadamard plus error feedback, real packed file, 9.692%) reaches 98.51% [97.89], with Wilson lower bounds below the bar for Card and Identity, and its feedback is still partial (teach .4 to .6, shuffled up to .5). No second task ships.

## How to run
```
python answer.py --artifact <<DIR>> --task skill_task.json "wake me at six" "play some jazz"
python feedback.py --artifact <<DIR>> --task skill_task.json --ann ann.jsonl
```
Needs one GPU with at least 40 GB (tested on 80 GB and 141 GB cards). A fresh clone run of both commands worked end to end (body.bin hash verified, `Recipe` emitted after feedback). Rows: `v2/lanes/ANNOTATION_SCHEMA.md`.

## Readings we had to guess (our best-evidence reading, not settled)
A: closed-form edit of existing weights is legal feedback. C: expert pruning chosen with the task's own unlabeled prompts is data-dependent selection, not a trained part. G: a stored carrier of replacement columns is not a "new part". R9: a task-specialised model that matches its base on the task and learns still meets the spirit. Each has a stated risk above (rules 4, 6, 9).
