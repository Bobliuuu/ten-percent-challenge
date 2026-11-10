# ISSUE DRAFT (entry L1). NOT POSTED.

Status: draft prepared 2026-11-09. Nothing in this file has been posted anywhere. Fill the `<<...>>` placeholders and run `CHECKLIST.md` before anyone files it. Numbers come from `v2/lanes/L1/REPORT.md`, `v2/lanes/L1/MANIFEST.json` and `v2/ENTRY_DECISION.md`; nothing was re-run for this draft.

---

**Title:** Entry: Bonsai-8B (released 1-bit) + closed-form feedback edit, 7-way voice-skill router, 7.16% of fp16

## What we ship

- **Start model:** Bonsai-8B (`Bonsai-8B-unpacked`), a natively 1-bit release built from Qwen3-8B by its makers. 8.2 B parameters. We did **not** compress it. We repack it losslessly (sign bits plus fp16 group scales; greedy outputs identical to the release on every exam item).
- **Task:** route a voice-assistant request to one of 7 skills (Alarm, Calendar, Lists, Music, News, Transport, Weather). The model writes the skill word by plain greedy decoding. No constrained decoding, no head, no lookup, no multiple-choice letters.
- **Feedback:** at most 25 annotated mistakes `{"text": ..., "label": ...}`; the label may be a word that is not in the menu (we teach `Recipe`). A one-shot closed-form calculation (no gradient, no optimiser) changes the **existing** per-128-group scales of the 1-bit MLP `down_proj` in the last 6 layers, so that the model's own forward pass writes the new word. Annotations are not stored and nothing goes in the prompt. Empty feedback leaves the tree byte-identical to the release.
- **Ship list:** weights (`bits.bin`, `scales.bin`, `small.pt`, `index.json`), tokenizer and config, licence, `answer.py`, `feedback.py`, `bonsai_loader.py`, `common.py`, `skill_task.json`, `feedback_calib.jsonl` (182 UNLABELED prompts, used only to protect old behaviour during the edit), `tensor_manifest.json`, and `patch.pt` (exists only after feedback).
- **Artifact:** https://huggingface.co/Bobliu/tenpct-l1-bonsai8b-grow (hashes in `v2/lanes/L1/build/SHA256SUMS.artifact`)
- **Locked files:** `v2/exam/` in <<REPO LINK>> (`MANIFEST.json` has a sha256 per file; `python3 -I v2/exam/loader.py` verifies them).

## Size (rule 1), real bytes, everything shipped

Denominator: Qwen3-8B fp16 = 16,381,470,720 B (Bonsai's own fp16 is 16,377,096,192 B, which gives 7.1615%).

| item | bytes |
|---|---|
| `bits.bin` (1-bit signs) | 1,023,529,984 |
| `scales.bin` (fp16 group scales) | 127,941,248 |
| `small.pt` | 652,655 |
| `index.json` | 14,593 |
| tokenizer (`tokenizer.json` 11,422,654; `vocab.json` 2,776,833; `merges.txt` 1,671,853; `tokenizer_config.json` 5,404) | 15,876,744 |
| `config.json`, `generation_config.json`, `chat_template.jinja`, `LICENSE`, `NOTICE.txt` | 18,019 |
| code: `answer.py` 2,629, `bonsai_loader.py` 2,849, `common.py` 1,924, `feedback.py` 19,159, `skill_task.json` 320 | 26,881 |
| calibration file `feedback_calib.jsonl` | 45,503 |
| `tensor_manifest.json` | 25,189 |
| **tree without feedback** | **1,168,130,816** |
| `patch.pt` after feedback (default `scale` carrier, overlay file) | 4,720,692 |
| **total after feedback** | **1,172,851,508 = 7.1596%** (slack to 10%: 465,295,564 B) |

Optional carriers for the same edit: `cols` (50.3 MB patch) total 1,218,478,299 = 7.4381%; `lr` (0.5 MB patch) about 7.134%. The `scale` patch is counted as a separate overlay file; writing the scales into `scales.bin` in place would add 0 B, which we did not do so that empty feedback restores the original tree byte for byte.

## The 9 rules, stated as we read them

Start model for the rules = Bonsai-8B (see reading B below). PASS means met as written on our exam. GRAY means it depends on a reading nobody has ruled on. WEAK means met thinly.

| # | rule | status | evidence and caveat |
|---|---|---|---|
| 1 | size counts everything | PASS | 7.16% with the patch, code, calibration, manifest, licence and tokenizer counted (table above) |
| 2 | same means faithful | **GRAY** | Retention against Bonsai-8B is 1.000 (2,489 of 2,489 base-correct items; Wilson lower .9985; every kind and every source 1.000; never-used sources 1.000, lower .9971). This is true **by construction** (lossless repack, identical outputs) and measures none of our own compression. Against the fp16 ancestor Qwen3-8B retention is **88.9%** (dev data), which would fail. It depends on reading B |
| 3 | exam is unseen ground | PASS on counts, with caveats | 2,744 items, every kind 344 or more, 1,489 items from sources never used for building (HWU64, MTOP), sealed by hash before any run, scored once. Caveats: the exam is ours, not yours. On the exam without the 532 `seen_tox` rows, three kinds have fewer than 300 base-correct items (Lists 206, Transport 233, Alarm 253), so a per-kind count on that tier is not met |
| 4 | no new trained parts | **GRAY** | No head, classifier, adapter, lookup, prompt or new tensor. But the feedback is a closed-form fit to the annotations that rewrites values of existing tensors, and 182 unlabeled calibration prompts ship. Readings A and G below |
| 5 | feedback teaches | PASS (on the letter) | New label `Recipe`, base teach 0.000. Real feedback from 25 annotations: teach .893 [.881, .905] over 10 draws on 164 locked unseen items (cols carrier), .904 over 6 draws (scale carrier, default), .909 end to end through the shipped commands. Neighbours kept .994 to .997. Exam retention after feedback .9988. Cross-source teach (MTOP .91, CLINC .89, HWU64 .81, MASSIVE .86-.88). The new word comes out of real greedy generation. The teach ceiling on a clean edit is about .945 because some "Recipe" items are ambiguous |
| 6 | controls clearly worse | PASS on SKILL7 | shuffled labels .001 to .005, wrong labels 0.000, empty 0.000, against real .90 |
| 7 | real job | PASS | the skill word is generated, exact string match, no constrained decoding |
| 8 | something to keep | PASS, with limits | base accuracy .907 (chance .143); not at the ceiling. Several single cells are near 1.0; CLINC Alarm is .52 on dev |
| 9 | spirit | **GRAY** | We did no compression of our own. If "the original" means a full-precision model we cut ourselves, this entry does not meet the spirit. Reading R9 below |

We do not claim a clean pass on rules 2, 4 or 9.

## Controls (rule 6), SKILL7, locked teach set (164 items) and locked neighbours (352 base-correct)

| arm (25 annotations) | teach | neighbours kept |
|---|---|---|
| real, `cols`, 10 draws | .893 [.881, .905] | .9952 (worst draw .9915) |
| real, `scale` default, 6 draws | .904 (.890 to .927) | .9938 (worst .9915) |
| real, `scale` default, shipped commands end to end | .909 | .9972 |
| shuffled (the new label on 25 unrelated ordinary inputs) | .001 to .005 | 1.000 |
| wrong (annotated inputs, a wrong old skill) | 0.000 | .9943 |
| empty | 0.000 (tree byte-identical) | 1.000 |

Fewer annotations teach less: 12 annotations .84 to .87 (same source), 5 annotations .64 to .78 (`scale` is weaker than `cols` at 5).

## Own-data dry run (a proxy for your check; our own data, not yours)

Run 2026-11-09 on this artifact, shipped feedback path, greedy, no constrained decoding (`v2/lanes/L1/results/owndata_summary.json`).

| set | n | result |
|---|---|---|
| Banking77 test, 77-label menu, exact label (never seen by this system) | 200 | base 45.5%; after feedback retention **100%** of the 91 base-correct items (Wilson lower 95.9); identical output 100% |
| same banking queries under the 7-skill router prompt (agreement only) | 100 | same label as base 94% |
| generic prompts (arithmetic, capitals, translation, code, haiku, email), 24 tokens | 100 | identical continuation 96%, same first line 98% |
| unseen label `Timer` (12 annotations, 28 held-out Timer requests, 60 Alarm, 60 other neighbours, 3 draws) | | `Timer` emitted .964 on all three draws; **Alarm requests flipped to Timer 21.7%** (base already 8.3%, and the base already writes `Timer` on 42.9% of the held-out Timer requests); other skills 96.7% (unchanged). Controls: wrong labels `Timer` .11 to .25, shuffled .00 to .04 |

Also on hand-written hostile data (315 old-skill utterances in 7 styles plus three new skills `Translate`, `Banking`, `Email`, calibrating on our own unlabeled prompts only): teach .87 to .93, old-skill retention 1.000 in all 7 styles, shuffled .00 to .11. That data is machine-written, so it is weak evidence.

Honest reading: the three real `Timer` draws coincide to the last digit (8 templates), so treat them as one measurement. A new label close to an old one takes some old items with it (Alarm to Timer 21.7%).

## Known weaknesses

- **Second task BANK6 is weak.** Same lossless pack on a banking task (shared exam, 2,237 items): retention 1.000 against Bonsai, but feedback for a new label `Security` teaches only **.727** (Banking77 .814, Bitext cross-source .641), neighbours .989 (one draw .977), and the shuffled control is **.101 on average with one draw at .49**. It passes by the letter and is a grayer proof than SKILL7. No second task ships in this entry.
- **Continual feedback tail.** A third skill added after two rounds can steal old items when its words overlap an old skill (Email against Lists, neighbours .863 after round 2 in one draw).
- **One narrow task.** 7-way routing is commercial but has a small label set; the base sits near its ceiling on most cells.
- **Locked-exam feedback numbers** were measured with the `cols` patch; the `scale` default was checked on the locked teach and neighbour sets (6 draws) and end to end, not on the full exam after feedback.
- **Base is a single reference run** per batch size (three batch sizes agree to .9993 to .9996).
- Hardware: one CUDA GPU with about 17 GB, `torch`, `transformers==4.56.2`.

## How to run

```
# answer (one label per line; uses patch.pt automatically if feedback was given)
python answer.py --artifact <<DIR>> --task skill_task.json "wake me at six" "play some jazz"
python answer.py --artifact <<DIR>> --task skill_task.json --in requests.jsonl

# feedback: ann.jsonl has <= 25 rows like {"text": "how long do I boil an egg", "label": "Recipe"}
python feedback.py --artifact <<DIR>> --task skill_task.json --ann ann.jsonl

# one process: feedback applied in memory, then answers
python answer.py --artifact <<DIR>> --task skill_task.json --feedback ann.jsonl "how long do I boil an egg"
```

Giving an empty `ann.jsonl` removes the patch. `--task` takes any JSON `{"prompt": "...{t}", "labels": [...]}`, so other label sets can be tried on the same weights. Rows are documented in `v2/lanes/ANNOTATION_SCHEMA.md`. A fresh-clone run (clean container, repo cloned, weights fetched, answer then feedback then answer) gave 4 of 4 unseen recipe prompts answered `Recipe` (base: Transport, off-menu, Transport, Lists), 7 of 7 neighbours unchanged, and empty feedback restored the identical tree.

## Why this entry should pass each step and rule (rule by rule)

Labels: PASS = met as written on our exam. GRAY = depends on a reading nobody has ruled on; we argue the reading but do not call it a plain pass. Numbers cite `v2/lanes/L1/REPORT.md` (L1 = sealed-exam section, L1+ = later sections), `v2/lanes/L1/MANIFEST.json`, `v2/lanes/L1/results/*.json`.

### The 5 challenge steps

| step | status | argument and evidence |
|---|---|---|
| 1. Start from one trained model above 0.5B; commercial, not narrow | PASS on the model, **GRAY** on "not narrow" | Bonsai-8B is a trained 8.2 B model (`MANIFEST.json`). The task is the intent routing a voice assistant needs, drawn from four public datasets (MASSIVE, CLINC150, HWU64, MTOP; `v2/exam/`). Strongest argument that it is not narrow: the exam spans 4 sources, 7 skills, many phrasings, and the same shipped weights score 45.5% on a 77-label Banking77 menu and answer arbitrary `--task` label sets (own-data dry run below). Weak side: the shipped menu has 7 labels and the base is near its ceiling in several cells (reading H) |
| 2. Keep at most 10% of its fp16 size | PASS | 7.16% with the patch (L1 "Bytes", table above); the 10% bar leaves 465,295,564 B |
| 3. Match the base | PASS vs the release, **GRAY** vs the ancestor | Score .9078 vs base .9074/.9074/.9078 on 2,744 items; outputs identical on 100% (`results/locked_L1.json`). By construction (lossless repack), not by our compression; see rule 2 |
| 4. Pass an exam it never saw | PASS on counts | 2,744 locked items sealed by sha256 before any run (`v2/exam/MANIFEST.json`, `python3 -I v2/exam/loader.py`), scored once, never used to build or choose. It is our exam, not yours |
| 5. Grow with feedback, no weight training, still passes the exam | PASS on the letter, depends on reading A | 25 annotated mistakes `{"text","label"}`, no gradient or optimiser; new label `Recipe` taught to .893 to .909; exam retention after feedback .9988 (`results/locked_L1.json`) |

### The 9 judging rules

**1. Size counts everything. PASS.** Every shipped byte is in the table above (weights, tokenizer, code, calibration file, tensor manifest, licence, `patch.pt`): 1,172,851,508 B = 7.1596% of Qwen3-8B fp16, 7.162% of Bonsai's own fp16. The patch is counted as a separate overlay although writing it into `scales.bin` would cost 0 B.

**2. Same means faithful. GRAY (reading B).** Retention against the released Bonsai-8B is 1.000 (2,489 of 2,489 base-correct; Wilson lower .9985). Every kind (303 to 408 items) 1.000; every source 1.000; never-used sources (HWU64, MTOP; n=1,327 base-correct) 1.000, lower .9971 (`results/locked_L1.json`). We say plainly that this is true by construction and measures none of our own compression. Against the fp16 ancestor Qwen3-8B retention is **88.9%** (dev data), which would fail. Our reading and its strongest argument are in B below.

**3. The exam is unseen ground. PASS on counts, with caveats.** 2,744 items (at least 2,000), every kind 344 or more (at least 300), 1,489 items from sources never used for building or tuning (at least 500); sealed by hash before any run; scored once. Caveats: the exam is ours; on the exam without the 532 `seen_tox` rows, three kinds have fewer than 300 base-correct items (Lists 206, Transport 233, Alarm 253), so a per-kind-count reading that excludes those rows is not met (retention there is still 1.000).

**4. No new trained parts. GRAY (readings A and G).** What is not there: no head, classifier, adapter (LoRA included), lookup table, side memory, new tensor, or prompt text; the architecture and the forward pass are the release's. What is there: a closed-form solve (no gradient) that rewrites the values of the **existing** per-128-group fp16 scales of the 1-bit MLP `down_proj` in the last 6 layers, plus 182 unlabeled prompts that only protect old behaviour. Strongest argument for passing: the rule lists what is banned (heads, classifiers, adapters, side memory, feedback in the prompt) and our edit is none of those; every channel the rules leave for step 5 (prompt, memory, head, adapter) is banned, so step 5 can only be met by changing values the model already has, and a one-shot solve with no gradient is the least "trained" way to do it. Weakest side: the edit is still a calculation fitted to the annotations, and a stored patch file can look like an adapter. We therefore ship the `scale` carrier (no new tensor; same bytes as the original if written in place) and keep `cols` and `lr` only as options. Reading A below.

**5. Feedback has to teach. PASS on the letter (conditional on A).**
- At most 25 annotated real mistakes: an annotation is accepted only if the base's own answer differs from the user's label (`feedback.py`; policy test `results/sk_policy.json`).
- At least 25 unseen items locked before any feedback: 164 locked `Recipe` items (MTOP 100, CLINC 42, HWU64 15, MASSIVE 7), base teach 0.000.
- Clear improvement: real teach .893 [.881, .905] (10 draws, `cols`), .904 (6 draws, `scale` default), .909 and neighbours .9972 through the shipped commands (`results/e2e_locked_scale_default.json`); .91 on MTOP and .81 on HWU64, sources never used for annotations or tuning.
- Without breaking nearby exam items: 352 locked base-correct neighbours kept .9952 (`cols`; worst draw .9915), .9938 (`scale`); after feedback the whole exam keeps .9988 (2,486 of 2,489; per kind at least .9967).
- Weakest side: fewer annotations teach less (12: .84 to .87; 5: .64 to .78 with `scale`); the teach ceiling on a clean edit is about .945 because 9 of the 164 items are ambiguous (for example "can i modify the apples for grapes").

**6. Prove the learning came from the feedback. PASS on SKILL7.** Same 25 annotations, shuffled labels: teach .001 to .005 (shuffled = the new label placed on 25 unrelated ordinary inputs); wrong labels 0.000; empty feedback 0.000 with the tree byte-identical; real .89 to .91 (controls table above). The gap is about .9. Weak side: on the second task BANK6 the shuffled control averages .101 and one draw reaches .49 (see weaknesses).

**7. Do the real job. PASS.** The model writes the skill word by greedy decoding and we compare the exact string. No constrained decoding, no letters, no multiple choice. A word outside the menu (`Recipe`, `Timer`) is emitted as generated text.

**8. The original must have something to keep. PASS, with limits.** Base accuracy .907 on the exam (chance .143), below its ceiling, 9.3% of items wrong. Several single cells sit near 1.0 and CLINC Alarm is .52 on dev (`REPORT.md` task health).

**9. Spirit over letter. GRAY (reading R9).** The purpose, as we read it, is a model at most 10% of a trained model that matches it and keeps learning from a few user corrections without a side store. A released 1-bit model plus an in-model, no-gradient edit does that, and all the learning happens inside the weights, which is the point of the maintainer's own wording that the objective is to improve things "without any external memory/parameters". Against: the compression was done by the Bonsai makers, not by us, so this entry demonstrates the feedback half of the challenge, not the compression half. We state that here rather than let a reader find it.

### How it answers the earlier objections to other entries

We refer to the maintainer's written objections to earlier entries (a fitted head with no effect when removed; a nearest-centroid classifier computed outside the model; a grown model never run on new-label items; an answer command that cannot output a new label; overwritten output rows described as "the same head as before, just kept inside the model file").

| objection | how this entry differs | evidence |
|---|---|---|
| feedback only works through an extra fitted classifier file | there is none. `patch.pt` holds new values for scales of existing `down_proj` tensors; delete it and the original tree is byte-identical | `results/e2e_locked_scale_default.json` (empty arm: tree sha equal) |
| the number is computed in numpy from feedback features | no number comes from outside the model. Every teach figure is real greedy generation by the grown model, exact string match | `results/locked_L1.json`, `results/e2e_locked.json` |
| the grown model is never run on the new label | it is: 164 locked new-label items, 10 draws (cols) and 6 (scale), plus the shipped CLIs end to end (teach .884 cols, .909 scale) | same files |
| the answer command only knows the old classes | `answer.py` prints whatever the model generates; the menu in the prompt does not contain `Recipe` or `Timer` and both come out | fresh-clone run: 4 of 4 unseen recipe prompts answered `Recipe` (base: Transport, off-menu, Transport, Lists), `results/dry_run.json` |
| a few overwritten output rows are the same head as before | no `lm_head`, embedding or output row is touched. The edit spreads over the scales of the last 6 layers' `down_proj` (3.6% of the weights, every output row of those matrices) and acts through the ordinary forward pass | `v2/bonsai/l1plus/results/dnorm.json` |
| learning cannot be attributed to the feedback | shuffled .001 to .005, wrong 0.000, empty 0.000 | controls table |
| the new label is the easy case | `Timer` is a label close to `Alarm`; results in the own-data section, including the 21.7% flip we report | `results/owndata_summary.json` |

### The maintainer's own-data check (our proxy, not his data)

We cannot run his data, so we ran the nearest thing on this artifact: data from a different distribution, our own annotations, the shipped path. Table above; results in `v2/lanes/L1/results/owndata_summary.json`.
- **Keeps working elsewhere:** the 77-label Banking77 task (never used by this system) gives retention 100% of 91 base-correct items after feedback (Wilson lower 95.9), identical output 100%; generic prompts (arithmetic, capitals, translation, code, haiku, email) are identical for 96% of 100 prompts (98% same first line); the same banking queries under the 7-skill prompt get the same label 94% of the time.
- **Keeps learning elsewhere:** a label the system never saw, `Timer`, is emitted .964 of the time on 28 held-out requests from 12 annotations; three hostile hand-written new skills (`Translate`, `Banking`, `Email`) teach .87 to .93 with old-skill retention 1.000 in all 7 phrasing styles, calibrating on our own unlabeled prompts only (that data is machine-written, so weak evidence).
- **Cost we report:** 21.7% of Alarm requests flip to Timer (the base already sends 8.3% there, and already writes `Timer` for 42.9% of the held-out Timer requests). The three real draws coincide to the last digit (8 annotation templates), so count them as one measurement. Controls on that arm: wrong labels Timer .11 to .25, shuffled .00 to .04.
- No side memory: annotations are not stored, the prompt is untouched, no lookup exists to consult.

### Readings we had to guess (our best-evidence reading, not settled)

Each has the strongest argument for it, then the risk. None is claimed as a pass.

- **A. A closed-form edit of existing weights is legal feedback (rule 4, step 5). GRAY.** For: step 5 says the model "learns the new and keeps the old, still with no weight training", so feedback must change behaviour by some route, and the rules ban every route except the model's own weights; our edit has no gradient, optimiser, head, adapter, lookup or prompt, adds no tensor, and the answer comes from the unchanged forward pass. Risk: it is a calculation fitted to the annotations that rewrites stored values.
- **B. Starting from an already-compressed release is allowed (rule 2, step 3). GRAY.** For: step 1 says any model above 0.5B is fine "either before or after you cut it", so a released low-bit model is a trained model to start from; the base's score is "the bar to meet", and retention and size are then measured against it, which we do (and we also report 88.9% against the ancestor). Risk: "the original" may mean the full-precision ancestor, where this entry fails rule 2.
- **G. A stored carrier (changed group scales) is not a "new part" (rule 4). GRAY.** For: it holds values for tensors the model already has, applied by overwriting, and writing it into `scales.bin` would add zero bytes and zero tensors. Risk: a separate patch file looks like an adapter.
- **H. 7-way routing is commercial and not narrow (step 1). GRAY.** For: it is the first decision a voice assistant makes, with real phrasing variety across four datasets, and the weights answer other label sets (Banking77, `Timer`). Risk: seven labels is a small label set and the base is near its ceiling in several cells.
- **R9. Spirit. GRAY.** For and against as in rule 9 above.

## Where the detail lives

`v2/RULES.md` (rules), `v2/ENTRY_DECISION.md`, `v2/lanes/L1/REPORT.md` and `MANIFEST.json`, `v2/exam/`, `v2/RED_TEAM.md`. The same repo holds a second entry, L9b, with its own compression (`ISSUE_DRAFT_L9b.md`); it is not what this issue files.
