# ISSUE DRAFT (entry L1). NOT POSTED.

Status: draft prepared 2026-11-09. Nothing in this file has been posted anywhere. Fill the `<<...>>` placeholders and run `CHECKLIST.md` before anyone files it. Numbers come from `v2/lanes/L1/REPORT.md`, `v2/lanes/L1/MANIFEST.json` and `v2/ENTRY_DECISION.md`; nothing was re-run for this draft.

---

**Title:** Entry: Bonsai-8B (released 1-bit) + closed-form feedback edit, 7-way voice-skill router, 7.16% of fp16

## What we ship

- **Start model:** Bonsai-8B (`Bonsai-8B-unpacked`), a natively 1-bit release built from Qwen3-8B by its makers. 8.2 B parameters. We did **not** compress it. We repack it losslessly (sign bits plus fp16 group scales; greedy outputs identical to the release on every exam item).
- **Task:** route a voice-assistant request to one of 7 skills (Alarm, Calendar, Lists, Music, News, Transport, Weather). The model writes the skill word by plain greedy decoding. No constrained decoding, no head, no lookup, no multiple-choice letters.
- **Feedback:** at most 25 annotated mistakes `{"text": ..., "label": ...}`; the label may be a word that is not in the menu (we teach `Recipe`). A one-shot closed-form calculation (no gradient, no optimiser) changes the **existing** per-128-group scales of the 1-bit MLP `down_proj` in the last 6 layers, so that the model's own forward pass writes the new word. Annotations are not stored and nothing goes in the prompt. Empty feedback leaves the tree byte-identical to the release.
- **Ship list:** weights (`bits.bin`, `scales.bin`, `small.pt`, `index.json`), tokenizer and config, licence, `answer.py`, `feedback.py`, `bonsai_loader.py`, `common.py`, `skill_task.json`, `feedback_calib.jsonl` (182 UNLABELED prompts, used only to protect old behaviour during the edit), `tensor_manifest.json`, and `patch.pt` (exists only after feedback).
- **Artifact:** <<ARTIFACT LOCATION AND SHA256 MANIFEST, to be filled by the owner: a public or shared link to the rebuilt folder; hashes are in `v2/lanes/L1/build/SHA256SUMS.artifact`>>
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

## Readings we had to guess (our best-evidence reading, not settled)

- **A. Closed-form edit of existing weights is legal feedback.** Our reading: it adds no head, adapter, lookup, prompt or new tensor and runs entirely through the model's own forward pass, so it is not "new computation fitted to data" in the sense of your earlier replies. Risk: it is a calculation fitted to the annotations.
- **B. Starting from an already-compressed release is allowed.** Our reading: rule 1 says a model is fine "before or after you cut it", so the released model is "the original" and retention is measured against it. Risk: "the original" may mean the full-precision ancestor (88.9%).
- **G. A stored carrier (here changed group scales) is not a "new part".** Our reading: the edited objects are values the model already has. Risk: the stored patch looks like an adapter.
- **R9. Spirit.** Our reading: the challenge asks for a model at most 10% of a trained model that matches it and still learns, and a released low-bit model qualifies. Risk: compression done by someone else may not be the intended skill.

## Where the detail lives

`v2/RULES.md` (rules and rulings), `v2/ENTRY_DECISION.md`, `v2/lanes/L1/REPORT.md` and `MANIFEST.json`, `v2/exam/`, `v2/RED_TEAM.md`. The same repo holds a second entry, L9b, with its own compression; it is not what this issue files.
