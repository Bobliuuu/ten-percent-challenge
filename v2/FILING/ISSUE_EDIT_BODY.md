Entry: Bonsai-8B (released 1-bit) + closed-form feedback edit, 7-way voice-skill router, 7.16% of fp16

<!-- Final text meant to REPLACE the title and body of our existing issue. Prepared 2026-11-10. Not posted. First line = title; everything after the blank line = body. Remove this comment before pasting. -->

## Summary

- **Start model:** Bonsai-8B, a natively 1-bit release built from Qwen3-8B by its makers (8.2 B parameters). We did not compress it. We repack it losslessly (sign bits plus fp16 group scales); greedy outputs are identical to the release on every exam item.
- **Task:** route a voice-assistant request to one of 7 skills (Alarm, Calendar, Lists, Music, News, Transport, Weather). The model writes the skill word by plain greedy decoding, compared by exact string. No constrained decoding, no head, no lookup, no multiple-choice letters.
- **Feedback:** at most 25 annotated mistakes `{"text": ..., "label": ...}`; the label may be a word that is not in the menu (we teach `Recipe`). A one-shot closed-form calculation (no gradient, no optimiser) rewrites the **existing** per-128-group scales of the 1-bit MLP `down_proj` in the last 6 layers, so the model's own forward pass writes the new word. Nothing is stored about the annotations and nothing goes in the prompt. Empty feedback leaves the tree byte-identical to the release.
- **Size:** 1,172,851,508 B = **7.16%** of Qwen3-8B fp16, every shipped byte counted (slack to 10%: 465 MB).
- **Artifact (public):** https://huggingface.co/Bobliu/tenpct-l1-bonsai8b-grow, revision `d5f0329dee86c10cd59ed5a94a04393c54fa0e18` (hashes in `v2/lanes/L1/build/SHA256SUMS.artifact`).
- **Code, results and locked exam:** https://github.com/Bobliuuu/ten-percent-challenge. The locked exam is `v2/exam/` (`MANIFEST.json` holds a sha256 per file; `python3 -I v2/exam/loader.py` verifies them). Lane folder: `v2/lanes/L1/` (`REPORT.md`, `MANIFEST.json`, `results/`).

Status words: PASS = met as written on our exam. GRAY = depends on a reading nobody has ruled on; we give our reading and its argument, and we do not call it a pass. WEAK = met thinly. **We do not claim a clean pass on rules 2, 4 or 9.**

## Commands

Needs one CUDA GPU of about 17 GB, `torch`, `transformers==4.56.2`. `<DIR>` is the folder the artifact is downloaded into.

```
# answer (one label per line; uses patch.pt automatically if feedback was given)
python answer.py --artifact <DIR> --task skill_task.json "wake me at six" "play some jazz"
python answer.py --artifact <DIR> --task skill_task.json --in requests.jsonl

# feedback: ann.jsonl has at most 25 rows like {"text": "how long do I boil an egg", "label": "Recipe"}
python feedback.py --artifact <DIR> --task skill_task.json --ann ann.jsonl

# one process: feedback applied in memory, then answers
python answer.py --artifact <DIR> --task skill_task.json --feedback ann.jsonl "how long do I boil an egg"
```

An empty `ann.jsonl` removes the patch and restores the release byte for byte. `--task` accepts any JSON `{"prompt": "...{t}", "labels": [...]}`, so other label sets run on the same weights. Row format: `v2/lanes/ANNOTATION_SCHEMA.md`. A fresh-clone run (clean container, repo cloned, weights fetched, answer then feedback then answer) gave 4 of 4 unseen recipe prompts answered `Recipe` (base: Transport, off-menu, Transport, Lists), 7 of 7 neighbours unchanged, and empty feedback restored the identical tree.

## Size (rule 1)

Denominator: Qwen3-8B fp16 = 16,381,470,720 B (Bonsai's own fp16 is 16,377,096,192 B, which gives 7.1615%).

| item | bytes |
|---|---|
| `bits.bin` (1-bit signs) | 1,023,529,984 |
| `scales.bin` (fp16 group scales) | 127,941,248 |
| `small.pt`, `index.json` | 667,248 |
| tokenizer files | 15,876,744 |
| config, generation config, chat template, licence, notice | 18,019 |
| code and task json | 26,881 |
| calibration file `feedback_calib.jsonl` (182 unlabeled prompts) | 45,503 |
| `tensor_manifest.json` | 25,189 |
| **tree without feedback** | **1,168,130,816** |
| `patch.pt` after feedback (default `scale` carrier) | 4,720,692 |
| **total after feedback** | **1,172,851,508 = 7.1596%** |

Optional carriers for the same edit: `cols` (50.3 MB patch) totals 7.438%; `lr` (0.5 MB) about 7.134%. The `scale` patch is counted as a separate overlay file; writing it into `scales.bin` would add 0 B, which we did not do so that empty feedback restores the original tree.

## The 9 rules

| # | rule | status | evidence and caveat |
|---|---|---|---|
| 1 | size counts everything | PASS | 7.16% with patch, code, calibration, manifest, licence and tokenizer counted |
| 2 | faithful | **GRAY** | Retention against Bonsai-8B is 1.000 (2,489 of 2,489 base-correct; Wilson lower .9985; every kind and every source 1.000; never-used sources 1.000, lower .9971). That is true **by construction** (lossless repack) and measures none of our own compression. Against the fp16 ancestor Qwen3-8B retention is **88.9%** (dev data), which would fail. It rests on reading B below |
| 3 | exam is unseen | PASS on counts, with caveats | 2,744 items, every kind 344 or more, 1,489 items from sources never used for building (HWU64, MTOP), sealed by hash before any run, scored once. Caveats: it is our exam; without the 532 `seen_tox` rows three kinds have fewer than 300 base-correct items (Lists 206, Transport 233, Alarm 253) |
| 4 | no new trained parts | **GRAY** | No head, classifier, adapter, lookup, prompt or new tensor. But the feedback is a closed-form fit to the annotations that rewrites values of existing tensors, and 182 unlabeled calibration prompts ship. Readings A and G |
| 5 | feedback teaches | PASS on the letter | New label `Recipe`, base teach 0.000, real teach .893 [.881, .905] over 10 draws (`cols`), .904 over 6 draws (`scale`, default), .909 end to end through the shipped commands, on 164 locked unseen items; neighbours kept .994-.997; exam retention after feedback .9988; cross-source teach MTOP .91, CLINC .89, HWU64 .81, MASSIVE .86-.88 |
| 6 | controls clearly worse | PASS on SKILL7 | shuffled .001-.005, wrong 0.000, empty 0.000, against real .90 |
| 7 | real job | PASS | the skill word is generated; a word outside the menu (`Recipe`, `Timer`) comes out as generated text |
| 8 | something to keep | PASS, with limits | base accuracy .907 (chance .143), below its ceiling; several single cells near 1.0, CLINC Alarm .52 on dev |
| 9 | spirit | **GRAY** | We did no compression of our own; this entry demonstrates the feedback half of the challenge and starts from someone else's compression. Reading R9 |

## Controls (rule 6): locked teach set (164 items), locked neighbours (352 base-correct)

| arm (25 annotations) | teach | neighbours kept |
|---|---|---|
| real, `cols`, 10 draws | .893 [.881, .905] | .9952 (worst draw .9915) |
| real, `scale` default, 6 draws | .904 (.890-.927) | .9938 (worst .9915) |
| real, `scale`, shipped commands end to end | .909 | .9972 |
| shuffled (the new label on 25 unrelated ordinary inputs) | .001-.005 | 1.000 |
| wrong (annotated inputs, a wrong old skill) | 0.000 | .9943 |
| empty | 0.000 (tree byte-identical) | 1.000 |

Fewer annotations teach less: 12 annotations .84-.87, 5 annotations .64-.78 (`scale` is weaker than `cols` at 5). The teach ceiling of a clean edit is about .945 because 9 of the 164 items are ambiguous.

## Own-data dry run (our data, shipped feedback path, greedy)

Run on this artifact; results in `v2/lanes/L1/results/owndata_summary.json`.

| set | n | result |
|---|---|---|
| Banking77, 77-label menu, exact label (never used by this system) | 200 | base 45.5%; after feedback, retention **100%** of the 91 base-correct items (Wilson lower 95.9); identical output 100% |
| same banking queries under the 7-skill router prompt (agreement only) | 100 | same label as base 94% |
| generic prompts (arithmetic, capitals, translation, code, haiku, email), 24 tokens | 100 | identical continuation 96%, same first line 98% |
| unseen label `Timer` (12 annotations, 28 held-out requests, 3 draws) | | `Timer` emitted .964 on all three draws; **Alarm requests flipped to Timer 21.7%** (base already 8.3%; the base already writes `Timer` on 42.9% of the held-out Timer requests); other skills 96.7% unchanged. Controls: wrong labels .11-.25, shuffled .00-.04 |

Honest reading: the three real `Timer` draws coincide to the last digit (8 annotation templates), so they are one measurement. A new label close to an old one takes some old items with it. On hand-written hostile data (315 old-skill utterances in 7 styles plus three new skills `Translate`, `Banking`, `Email`), teach was .87-.93 with old-skill retention 1.000 in all 7 styles; that data is machine-written, so it is weak evidence.

## Gray cells and weaknesses, with our arguments

Each reading is our best-evidence guess, not settled.

- **A. A closed-form edit of existing weights is legal feedback (rule 4, step 5). GRAY.** For: step 5 asks the model to learn "with no weight training", so feedback has to change behaviour by some route, and the rules ban every other route (prompt, memory, head, adapter). Our edit has no gradient, optimiser, head, adapter, lookup or prompt, adds no tensor, and the answer comes from the unchanged forward pass. Against: it is still a calculation fitted to the annotations that rewrites stored values.
- **B. Starting from an already-compressed release is allowed (rule 2, step 3). GRAY.** For: step 1 allows any model above 0.5B "either before or after you cut it", so a released low-bit model is a trained model to start from, and retention and size are measured against it; we also report 88.9% against the ancestor. Against: "the original" could mean the full-precision ancestor, where this entry fails rule 2.
- **G. A stored carrier (changed group scales) is not a new part (rule 4). GRAY.** For: it holds values for tensors the model already has and is applied by overwriting; written in place it adds zero bytes and zero tensors. Against: a separate patch file looks like an adapter. That is why `scale` is the default.
- **H. 7-way routing is commercial and not narrow (step 1). GRAY.** For: it is the first decision of a voice assistant, with real phrasing variety over four datasets, and the same weights answer other label sets (Banking77, `Timer`). Against: seven labels is small and the base is near its ceiling in several cells.
- **R9. Spirit. GRAY.** For: the purpose as we read it is a model at most 10% of a trained model that matches it and keeps learning from a few corrections without a side store; here all learning happens inside existing weights. Against: the compression was done by the Bonsai makers, not by us.

Weaknesses we know of:
- **Second task BANK6 is weak.** Same lossless pack on a banking task (2,237 exam items): retention 1.000, but `Security` teaches only **.727**, neighbours .989 (one draw .977), and the shuffled control averages .101 with one draw at .49. No second task ships.
- **Close new labels steal old items** (Alarm to Timer 21.7%); in continual mode a third skill whose words overlap an old one (Email against Lists) dropped neighbours to .863 in one draw.
- **One narrow task**, small label set, base near its ceiling on several cells.
- **Locked-exam feedback numbers** were measured with the `cols` patch; the `scale` default was checked on the locked teach and neighbour sets (6 draws) and end to end, not on the full exam after feedback.
- **Base is one reference run** per batch size (three batch sizes agree to .9993-.9996).
- Hardware: about 17 GB of GPU memory.

## Backup entry (L9b): our own compression, summarized

If starting from a released compressed model (reading B) is not accepted, the backup entry is **L9b**, our own compression of an unmodified full-precision release. It is weaker, and we list it so the weaknesses are visible.

- **Model and size:** Qwen3.5-35B-A3B; keep 102 of 256 experts per layer (chosen by routing mass on unlabeled task prompts), 3-bit RTN experts, 6-bit attention and shared expert, 4-bit embedding and output; vision tower not shipped. **9.704%** of the text-only fp16 (9.35% of the full checkpoint including the vision tower).
- **Feedback:** closed-form edit of the shared-expert `down_proj` columns of the last 10 MoE layers, stored as replacement columns (`fb.bin`, 21 MB).
- **Rules:** retention **99.55% [99.21]**, worst kind 98.4, never-used sources 99.45%; teach .74-.76 (draws .60-.83) with neighbours 98.0-98.6% (empty 98.3%); shuffled .16-.18 on average but up to .40, so rule 6 is GRAY on bad draws; base .9676 leaves little to keep (rule 8 WEAK); rule 4 GRAY (readings A, C, G); rule 9 GRAY (60% of experts removed with the task's own prompts).
- **Own data: weak.** Banking77 retention 79.3% [71.8], router agreement 46%, generic prompts identical 11% ("17 times 23" gives 17); `Timer` emitted 100% with 21.7% of Alarm flipped. A later try with broad unlabeled calibration and a better quantizer reached 90.6% / 71% / 39% at the same bytes (better, still not base-like) and was stopped at that gate.
- **BANK6:** the same recipe fails rule 2 (97.37% [96.58]).
- **Artifact:** not hosted; it lives on a private volume and is rebuildable from the repo scripts (`v2/lanes/L9b/`). Draft: `v2/FILING/ISSUE_DRAFT_L9b.md`.

Other lanes we measured and are not entering (details in the repo): a 64-of-256-expert 4-bit variant run end to end from packed files (7.92% text-only, retention .9898 [.9852], teach .884, controls 0, but own-data retention 71.9% and BANK6 retention 96.39%), and several dense and MoE variants that fail on retention, teach strength or controls.

## Where the detail lives

`v2/RULES.md` (rules), `v2/ENTRY_DECISION.md`, `v2/lanes/L1/REPORT.md` and `MANIFEST.json`, `v2/exam/`, `v2/RED_TEAM.md`, `README.md`.
