# ISSUE DRAFT (backup entry L9b). NOT POSTED.

Use only if reading B (pre-compressed start) is dropped by the owner. Placeholders filled; same checklist as `ISSUE_DRAFT.md`. Numbers: `v2/lanes/L9b/REPORT.md`, `MANIFEST.json`, `v2/ENTRY_DECISION.md`.

---

**Title:** Entry: Qwen3.5-35B-A3B, our own compression (expert prune + 3-bit), 7-way voice-skill router, 9.70% of fp16

## What we ship
- **Start model:** Qwen/Qwen3.5-35B-A3B (35.95 B parameters, 256 experts per layer), unmodified full-precision release. **Compression is ours:** keep 102 of 256 experts per layer, chosen by routing mass on unlabeled task prompts (no labels, no gradient), 3-bit grouped RTN on kept experts, 6-bit attention and shared expert, 4-bit embedding and output layer. The vision tower is not shipped.
- **Task:** 7-way voice-skill routing; the model writes the skill word (greedy, exact match, no constrained decoding).
- **Feedback:** at most 25 annotations `{"text","label"}` (new label allowed). A closed-form edit of the shared-expert `down_proj` columns of the last 10 MoE layers, stored as replacement columns in `fb.bin` (20,981,760 B).
- **Artifact:** not hosted: it lives on a private volume and is rebuildable from the repo scripts (`body.bin` sha256 `e0a1266db5eb7c4f877dfaeb107bf41baa5b5240ff945eb4d42a52f400efe88b`). Locked files: `v2/exam/` in https://github.com/Bobliuuu/ten-percent-challenge.

## Size
| item | bytes |
|---|---|
| `body.bin` | 6,685,765,376 |
| tokenizer, config, template, manifest, `core.py`, `qpack.py`, `answer.py` | 20,133,949 |
| feedback carrier `fb.bin` | 20,981,760 |
| feedback code, task json, calibration file (110,789 B incl. 101,819 B calibration, 300 rows) | 110,789 |
| **total** | **6,726,991,874 = 9.704%** of the text-only fp16 size (69,321,221,376 B); 9.35% of the full checkpoint including the vision tower (71,903,645,408 B) |

The denominator is the text-only size (the vision tower is dropped for a text task). Against the full checkpoint the figure is lower, so this is the less favourable reading. Without feedback files: 9.674%.

## Why this entry should pass each step and rule (rule by rule)

Labels: PASS = met as written on our exam. GRAY = depends on a reading nobody has ruled on (we argue it but do not call it a pass). WEAK = met thinly. Sources: `v2/lanes/L9b/REPORT.md` (sections numbered there), `MANIFEST.json`, `v2/ENTRY_DECISION.md`.

### The 5 challenge steps

| step | status | argument and evidence |
|---|---|---|
| 1. One trained model above 0.5B; commercial, not narrow | PASS on the model, **GRAY** on "not narrow" | Qwen3.5-35B-A3B, 35.95 B parameters, unmodified public release. Task: voice-assistant skill routing over four public datasets (reading H: seven labels is a small label set). This entry weakens the argument: the packed model is a task specialist (see rule 9 and own data) |
| 2. At most 10% of fp16 | PASS | 9.704% with carrier and code (9.674% without); against the full checkpoint 9.35% |
| 3. Match the base | PASS | retention 99.55% [99.21] of base-correct items, one base run |
| 4. Exam never seen | PASS on counts | 2,744 locked items sealed by hash, scored once; our exam, not yours |
| 5. Grow with feedback, no weight training | PASS (moderate), conditional on reading A | teach 0 to .74-.76 (8-draw means), neighbours kept |

### The 9 judging rules

**1. Size counts everything. PASS.** 6,726,991,874 B = 9.704% of the text-only fp16 size (table above), slack about 157 MB to 9.9%. Counted: `body.bin`, tokenizer, config, `core.py`, `qpack.py`, `answer.py`, `fb.bin`, feedback code, task json and the 300-row calibration file. The denominator is the text-only size, which is the less favourable of the two readings (the full checkpoint gives 9.35%).

**2. Same means faithful. PASS.** Retention 99.55% [Wilson lower 99.21] on the 2,744-item exam; every kind at least 98.4 (Lists); worst source-by-kind cell 97.1; never-used sources 99.45% [98.91]; three quantizer draws 99.47 to 99.55. The base is a single run and the exam is ours. This is our own compression of an unmodified public model, so no reading about a pre-compressed start is needed.

**3. The exam is unseen ground. PASS on counts, with caveats.** 2,744 items, 344 to 420 per kind, 1,448 base-correct items from never-used sources. Caveat: on the exam without the 532 `seen_tox` rows Lists and Transport fall below 300 per kind.

**4. No new trained parts. GRAY (readings A, C, G).** No head, classifier, adapter, lookup, side memory, prompt text or new tensor. Two things are data-dependent: (i) which 102 of 256 experts per layer to keep, chosen by routing mass on unlabeled task prompts (no labels, no gradient), and rounding to 3 bits (RTN, no learned rounding); (ii) the feedback edit, a closed-form solve of existing shared-expert `down_proj` columns stored as replacement columns in `fb.bin`. Strongest argument: pruning and rounding select or round weights the model already has, which is "compression technique" in the maintainer's wording on issue #1 ("compression techniques are fine but no retraining"); the feedback edit adds no tensor and the answer comes from the model's own forward pass. Weak side: both steps are fitted to data in some sense, and `fb.bin` looks like a stored patch.

**5. Feedback has to teach. PASS (moderate), conditional on A.** At most 25 annotated mistakes; 164 locked unseen items; teach 0 to .74-.76 (8-draw means; draws .60 to .83; cross-source MTOP and HWU64 included); neighbours 98.0 to 98.6% against 98.3% with empty feedback; worst neighbour cell 95.8 to 97.9. Weak side: teach is moderate, not high, and varies by draw.

**6. Controls. GRAY on bad draws.** Real .74 to .76; wrong labels 0; empty 0; **shuffled mean .16 to .18 over 8 draws, range .00 to .40 (.47 on one early draw)**. The gap to real is about .58 on average, so "clearly worse" holds on average and on most draws; one draw is close to half of real. A strict-controls setting (`--m 32 --lam 2000 --awt 50`) gives shuffled mean .02 (max .09) but teach only .53. We report the leak rather than hide it.

**7. Do the real job. PASS.** The model generates the skill word; exact string match; no constrained decoding.

**8. The original must have something to keep. WEAK.** Base .9676 (chance .14): only 3.2% of items are wrong, so there is little headroom. It is below the ceiling and far above chance, which is the letter of the rule.

**9. Spirit. GRAY.** 60% of experts are removed using the task's own prompts, so the packed model is a task specialist; on unrelated text it diverges from the base (own-data section: generic prompts identical 11%). Our reading: the exam is the task, the model matches the base on it (99.55%), is 9.7% of the size, and learns, which is the challenge as worded; but a judge who tries other prompts will see a degraded general model, and the second task BANK6 does not pass with the same recipe. We label this GRAY and expect it to be the entry's weakest rule.

### How it answers the earlier objections to other entries

The maintainer's written objections were: feedback that works only through a fitted classifier file; a nearest-centroid classifier computed outside the model; a grown model never run on new-label items; an answer command that cannot print a new label; output rows overwritten and called "the same head as before, just kept inside the model file".

| objection | how this entry differs | evidence |
|---|---|---|
| fitted classifier file | none. `fb.bin` holds replacement columns for existing shared-expert `down_proj` tensors; without it the model is the packed base | REPORT section 10 (fresh-clone run) |
| number computed in numpy | every teach number is real greedy generation of the grown model | REPORT feedback sections |
| grown model never run on the new label | 164 locked new-label items, 8 draws; fresh clone: `feedback.py` then `answer.py` prints `recipe` on unseen recipe requests while the other probes are unchanged | `results/fresh_clone_run.json` |
| answer command knows only old classes | `answer.py` prints generated text and takes `--task` JSON; `Recipe` and `Timer` come out although neither is in the menu | same, and own-data `Timer` emitted 100% |
| output rows overwritten | no `lm_head` or embedding row is touched; the edit is on shared-expert `down_proj` columns of the last 10 MoE layers | `fb.bin` layout, 10 layers x 512 columns |
| attributing learning to feedback | wrong 0, empty 0, shuffled mean .16 to .18 (draw-dependent, see rule 6) | REPORT section 8 |

### The maintainer's own-data check (our proxy, not his data)

Results are in the section above. What they show, stated without softening: on a different distribution the packed model **does not stay base-like**: Banking77 retention 79.3% [71.8], generic prompts identical 11%, router agreement 46%. It does keep learning: an unseen label `Timer` is emitted 100%. A caveat on our own control there: with a single new label the shuffled control is degenerate (half of the 12 labels set to `Alarm`, Timer .93 to 1.0), so it does not separate; the wrong-label control (Timer 0.0) does. If the maintainer's data looks like the router task it should behave as on the exam; if it looks like Banking77 it will not match the base. This is why L9b is the fallback entry, not the primary.

## Own-data dry run (our data, not yours) and BANK6: this entry is weak here
- Banking77 (200 test queries, 77-label menu): base 70.0%, packed 61.0%; retention **79.3% [71.8]**; identical output 72.5%.
- Same queries under the router prompt: same label as base 46%.
- Generic prompts: identical continuation **11%** (for example "17 times 23" gives 17). **The packed model diverges badly from the base outside its task.**
- Unseen label `Timer`: emitted 100%, but **21.7% of Alarm requests flip to Timer**.
- Second task BANK6, same recipe with RTN: retention 97.37% [96.58], Card 94.4, Identity 91.7, which fails 98/95. A better quantizer (Hadamard plus error feedback, real packed file, 9.692%) reaches 98.51% [97.89], with Wilson lower bounds below the bar for Card and Identity, and its feedback is still partial (teach .4 to .6, shuffled up to .5). No second task ships.
- Later try (L14, same model, broad unlabeled calibration of chat, math, code and text, Hadamard 2-bit VQ experts, 62.5% of experts kept, same bytes): own-data 90.6% / 71% / 39% (retention / router agreement / generic identical), a strict upgrade on 79 / 46 / 11 but still not base-like. The gate failed there, so it was stopped before any locked exam or packing; this entry's numbers above are unchanged.

## How to run
```
python answer.py --artifact <DIR> --task skill_task.json "wake me at six" "play some jazz"
python feedback.py --artifact <DIR> --task skill_task.json --ann ann.jsonl
```
Needs one GPU with at least 40 GB (tested on 80 GB and 141 GB cards). A fresh clone run of both commands worked end to end (body.bin hash verified, `Recipe` emitted after feedback). Rows: `v2/lanes/ANNOTATION_SCHEMA.md`.

## Readings we had to guess (our best-evidence reading, not settled)

- **A. A closed-form edit of existing weights is legal feedback (rule 4, step 5). GRAY.** For: step 5 requires learning "with no weight training", so some change to the model's own values is the only route the rules leave open (prompt, memory, head and adapter are banned); the edit has no gradient and adds no tensor. Risk: it is still a calculation fitted to annotations.
- **C. Expert pruning chosen with the task's own unlabeled prompts is not a trained part (rule 4, rule 9). GRAY.** For: it deletes whole existing experts by routing frequency, with no labels and no gradient, which is compression. Risk: it is data-dependent selection that makes the model a task specialist.
- **G. A stored carrier of replacement columns is not a new part (rule 4). GRAY.** For: the columns replace columns of an existing tensor and run through the same layer. Risk: `fb.bin` (21 MB) looks like an adapter.
- **H. 7-way routing is commercial and not narrow (step 1). GRAY.** For: it is the first decision of a voice assistant, over four datasets. Risk: seven labels, and a base at 96.8%.
- **R9. Spirit. GRAY.** Argued under rule 9; weakest here because of the generic-prompt divergence.

Each risk is stated in the rule it affects (rules 4, 6, 8, 9).
