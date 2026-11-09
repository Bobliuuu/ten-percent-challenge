# The 10% challenge: our attempt

The [10% challenge](https://github.com/DeusExperiments/ten-percent-challenge) asks for a model that is at most 10% of the size of a trained model, behaves the same as that model on an exam it never saw, and can learn from a few user corrections, all without training the weights. This repo is our working record: code, exams, results and dead ends.

Last updated 2026-11-09. Deadline **2026-11-12 17:05 UTC**. Our own freeze (after this, nothing new is built) is **2026-11-11 08:00 UTC**.

## 1. Summary and honest status

1. We shrink big models to under 10% of their fp16 size and score them on a locked exam of 2,744 items (7-way voice-skill routing: "which skill should handle this request?").
2. Shrinking is mostly solved on this easy task: several versions keep 98% to 100% of the answers the original got right, and write a real word as the answer.
3. Learning from feedback is the hard part. We teach a new label (for example `Recipe`) from at most 25 corrections by a one-shot calculation that edits some existing weights. There is no head, adapter, lookup table or prompt. The model's own forward pass writes the new word.
4. **No version passes all 9 judging rules cleanly as the maintainer would judge them.** We have not claimed otherwise.
5. Two readings decide everything. **The maintainer has not ruled on either, so both are our best-evidence guesses, not settled.**
   - **Rule 4 is GRAY** because our feedback is a closed-form edit of existing weights. The maintainer rejected a fitted head and a nearest-centroid row overwrite. He has not said whether a calculated edit of existing tensors counts as "new computation fitted to data".
   - **Starting from an already-compressed release (L1, L2) is MAYBE-VALID and unruled.** Rule 1 says a model is fine "before or after you cut it", but the compression is then someone else's work.

If the maintainer rules against closed-form edits, **we file nothing as a pass.** Our earlier planning put the chance that any entry passes all nine rules as the maintainer reads them at about 5%. The results since then improve the compression side but not the ruling, so we have not raised that number.

The entry we filed earlier (issue #2, Qwen2.5-7B tool selection, 9.404%) does **not** pass: 93.9% retention (needs 98%), letter answers (rule 7), feedback through a fitted head (rule 4, ruled by the maintainer). It will not be refiled.

### Words used here

| word | plain meaning |
|---|---|
| fp16 size | the size of the original model with 16-bit numbers. Our size limit is 10% of that |
| retention | of the items the original model got right, the share the small model also gets right. Rule 2 needs 98% overall and 95% in every kind |
| Wilson lower bound `[x]` | a safe low estimate of retention that allows for the exam being finite |
| locked exam | `v2/exam`: 2,744 items from public data, sealed by hash before any run, scored once. It is **our** exam, not the maintainer's |
| kind | one answer class (Alarm, Calendar, Lists, Music, News, Transport, Weather) |
| expert pruning | in a mixture-of-experts model, delete the experts the task rarely uses (we pick them by how often the task's prompts route to them) |
| RTN, GPTQ, VQ, rANS | ways to store weights with fewer bits (round to nearest; error-corrected rounding; codebook lookup; entropy coding) |
| teach set | 164 locked, unseen requests for the new label. "Teach" = share where the model writes the new label |
| neighbours | old-label exam items near the new label. They must stay right after feedback |
| controls | the same feedback with shuffled labels, and with no feedback. Both must do clearly worse than real feedback |
| lane (L1, L2, ...) | one line of work, each with its own folder in `v2/lanes/` |
| ruling A to I | an unruled reading we have to guess, listed in [`v2/ENTRY_DECISION.md`](v2/ENTRY_DECISION.md) |

### The rules we are judged by

The 9 judging rules and 5 challenge rules are copied word for word in [`v2/RULES.md`](v2/RULES.md). Short version of the nine:

| rule | in a few words |
|---|---|
| 1 | size counts every shipped byte, against the fp16 size of the starting model |
| 2 | keep at least 98% of what the original got right, and at least 95% in every kind |
| 3 | exam is unseen: at least 2,000 items, 300 per kind, 500 from a source never used |
| 4 | no new trained parts (no heads, adapters, side memory, feedback in the prompt) |
| 5 | feedback must teach: at most 25 annotations, 25+ locked teach items, clear improvement, neighbours kept |
| 6 | shuffled-label and empty feedback must do clearly worse |
| 7 | the model keeps the task's real output (no multiple-choice stand-in) |
| 8 | the original must have something to keep (well above chance, below its ceiling) |
| 9 | spirit over letter: readings that defeat the purpose are rejected |

## 2. Working versions

"Working" means: scored on our locked exam, passes the retention gates there, and real feedback teaches the new label. It does **not** mean "passes the rules". Every version below has at least one GRAY or WEAK cell.

**Verdicts.** PASS = meets the rule as written on our exam. GRAY = depends on an open reading. WEAK = meets the letter but thinly, or one measurement is missing.

**Same for every version below (rule 3).** The shared exam has 2,744 items, 344 to 420 per kind, and 1,489 items from sources never used for building (HWU64 and MTOP). Caveats: it is our exam; 532 rows are flagged `seen_tox`, and on the exam without them Lists and Transport have fewer than 300 base-correct items. The base accuracy is scored on the exam as shown in each rule 8 row.

**Ranking** (from [`v2/ENTRY_DECISION.md`](v2/ENTRY_DECISION.md), refreshed 2026-11-09): L9b primary; L1, then L5, then L2 as alternates; L3, L4, L9, L10 are not filing candidates.

### L9b (primary): Qwen3.5-35B-A3B, pruned and 3-bit

- **Model:** Qwen3.5-35B-A3B, a released full-precision model (not pre-compressed).
- **How compressed (ours):** keep 102 of 256 experts per layer (chosen by task prompts, no labels), 3-bit RTN on kept experts, 6-bit attention, 4-bit embedding and output layer. Vision tower not shipped.
- **Size:** **9.674%** measured on the real packed file (9.704% with the 21 MB feedback carrier). That is against the text-only fp16 size; against the full checkpoint it is 9.326%.
- **Retention:** **99.55% [99.21]**, worst kind Lists 98.4. Three quantizer draws: 99.47 to 99.55.
- **Feedback:** closed-form edit of the shared-expert `down_proj` columns in the last 10 layers, stored as replacement columns.

| rule | verdict | how it follows the rule, and the caveat |
|---|---|---|
| 1 size | PASS | 9.674% of shipped bytes (slack 157 MB). Caveat: the reference is the text-only fp16 size. Only a private volume holds the artifact; it must be re-packed or published |
| 2 faithful | PASS | 99.55% [99.21], every kind at least 98.4, never-used sources 99.45% [98.91]. Caveat: one run of the base |
| 3 exam | PASS | see the shared note above |
| 4 no new trained parts | GRAY | pruning and RTN pick or round existing weights; no head, adapter, lookup or prompt. Feedback is a calculated edit of existing columns (rulings A and G unruled) |
| 5 feedback teaches | PASS (moderate), if A holds | teach 0 to .73-.79 on 164 locked items (8-draw mean .74-.76); neighbours 98.0 to 98.6% vs 98.3% with empty feedback |
| 6 shuffled and empty worse | GRAY on bad draws | shuffled is draw-dependent: mean .16 to .18, range .00 to .40 (.47 on the first draw 2). We tried to fix it and could not |
| feedback controls (real / shuffled / empty) | | **.74-.76 mean / .16-.18 mean / 0**; wrong labels 0. Gap about .58 |
| 7 real job | PASS | the model writes the skill word; exact match, no constrained decoding |
| 8 something to keep | WEAK | base .9676 (chance .14): only 3.2% errors to keep |
| 9 spirit | GRAY | 60% of experts removed using the task's own prompts, so it is a task specialist. The same recipe on a second task (BANK6) fails |

Also open: the maintainer's own-data check is **untested** for this version and is its highest risk (the 7-skill prompt is hardcoded; expert choice may not carry to other text).

### L1: Bonsai-8B (released 1-bit) plus a feedback patch

- **Model:** Bonsai-8B, a 1-bit release built from Qwen3-8B by its makers.
- **How compressed:** not by us. We repack it losslessly (output identical to the release).
- **Size:** **7.44%** measured (7.13% base files plus the 50 MB patch). The low-rank version (L1+) is 7.13% in total with a 0.5 MB patch.
- **Retention:** **1.000 [.9985] against the released model, by construction.** Against its fp16 ancestor Qwen3-8B it is **88.9%** (dev data).
- **Feedback:** closed-form edit of 1,024 `down_proj` columns in the last 6 layers, stored as fp16 columns.

| rule | verdict | how it follows the rule, and the caveat |
|---|---|---|
| 1 size | PASS | 7.44% of Qwen3-8B fp16 including patch, calibration prompts and code |
| 2 faithful | GRAY | 100% (2,489 of 2,489) against the release, every kind and source. That is true by construction and measures none of our compression. Against the fp16 ancestor it is 88.9%. Depends on ruling B |
| 3 exam | PASS | see the shared note. Base accuracy .907 |
| 4 no new trained parts | GRAY | no head, adapter, lookup or prompt, but 50 MB of fitted fp16 columns ship, plus 182 unlabeled calibration prompts (ruling A, G) |
| 5 feedback teaches | PASS | teach 0 to .893 [.881, .905] over 10 draws; neighbours .995; exam retention after feedback .9988; cross-source teach .81 to .91; new label from real generation |
| 6 shuffled and empty worse | PASS | shuffled .005, wrong 0, empty 0 |
| feedback controls (real / shuffled / empty) | | **.893 / .005 / 0** |
| 7 real job | PASS | the skill word is generated |
| 8 something to keep | PASS | base .907 (chance .143); some single cells are near 1.0 |
| 9 spirit | GRAY | we did no compression of our own. The reading is unruled (RULES 4f) |

Also open: second task BANK6 gives exam retention 1.000 but teach only .727 with shuffled .10 (up to .49). A new skill close to an old one (Email vs Lists) can steal old items in continual mode (neighbours down to .86).

### L5: dense Qwen2.5-32B, our compression, no pruning

- **Model:** Qwen2.5-32B-Instruct, full precision, nothing removed.
- **How compressed (ours):** GPTQ scalar quantization with Hadamard rotation and rANS coding.
- **Size:** packed body **9.797%** measured; **9.926%** with the 83.9 MB feedback carrier and code (0.03 pt over our internal 9.9% target). No single-file loader yet.
- **Retention:** **98.27% [97.70]**, worst kind News 96.4. Very thin margin over 98%.
- **Feedback:** closed-form edit of `o_proj` (8 layers, 1,024 columns).

| rule | verdict | how it follows the rule, and the caveat |
|---|---|---|
| 1 size | PASS | 9.926% total, under 10%. Caveat: over our 9.9% target; evals ran on the decoded checkpoint, not a single packed file |
| 2 faithful | WEAK | 98.27% [97.70] overall, every kind at least 96.4. Two source-by-kind cells are below 95: hwu64 x News 89.1 (n 64), hwu64 x Weather 91.9 (n 37). A strict per-cell reading fails |
| 3 exam | PASS | see the shared note. Never-used sources: 97.99% |
| 4 no new trained parts | GRAY | no head, adapter, lookup or prompt; feedback is a calculated edit of `o_proj`. GPTQ rounding is fitted to calibration prompts (flagged) |
| 5 feedback teaches | WEAK | teach .532 teacher-forced, real generation .518 to .543 (164 locked items); neighbours 1.000. Modest |
| 6 shuffled and empty worse | PASS | shuffled 0, wrong 0, empty 0 |
| feedback controls (real / shuffled / empty) | | **.53 / 0 / 0** |
| 7 real job | PASS | one-word skill generation |
| 8 something to keep | PASS | base .9690 |
| 9 spirit | GRAY | easy routing task; no packed loader and no standalone feedback command yet |

Needs only ruling A. It does not depend on rulings B (pre-compressed start) or C (task-based pruning).

### L3: Qwen3-30B-A3B-Instruct-2507, pruned and 2-bit VQ

- **Model:** Qwen3-30B-A3B-Instruct-2507, full precision.
- **How compressed (ours):** keep 85 of 128 experts, 2-bit vector-quantized experts with Hadamard rotation and error feedback, 6-bit attention.
- **Size:** packed file **9.676%** measured; **9.896%** counting the 134 MB fp16 `o_proj` edit in full (slack 0.10 pt).
- **Retention:** **99.32% [98.92]**, worst kind Calendar 96.69.
- **Feedback:** closed-form edit of `o_proj` in the last 8 layers.

| rule | verdict | how it follows the rule, and the caveat |
|---|---|---|
| 1 size | PASS | 9.896% with the edit counted in full |
| 2 faithful | PASS | 99.32% [98.92]; worst kind Calendar 96.69; worst source-by-kind cells about 96.7 to 97.2 |
| 3 exam | PASS | see the shared note. Base .9592 |
| 4 no new trained parts | GRAY | no head, adapter or side memory; the edit is a closed-form fit of `o_proj` to 24 labels, and 300 labelled old-skill prompts ship to compute it |
| 5 feedback teaches | WEAK | teach 0 to **.799** (164 items; mtop .83, hwu64 .87, massive 1.00, clinc .67); exam retention after feedback 99.20. Caveat: neighbour retention on the final carrier was not measured |
| 6 shuffled and empty worse | PASS | shuffled 0, wrong 0, empty 0 |
| feedback controls (real / shuffled / empty) | | **.799 / 0 / 0** |
| 7 real job | PASS | free-generated skill name. The menu is in the prompt (closed set) |
| 8 something to keep | PASS | base .9592 |
| 9 spirit | GRAY | task-based expert pruning (ruling C) |

Its second task BANK6 fails rule 5 (see section 3).

### L4: Qwen3-30B-A3B-Instruct-2507, pruned and GPTQ 2-bit

- **Model:** Qwen3-30B-A3B-Instruct-2507, full precision.
- **How compressed (ours):** keep 80 of 128 experts, GPTQ with Hadamard rotation at 2 bits, fp16 `o_proj` kept in the last 6 layers as the feedback carrier.
- **Size:** **9.895%, an estimate** (bit count; no packed file, code not counted).
- **Retention:** **98.94% [98.47]**, worst kind News 96.03 (its own lower bound is 93.65).
- **Feedback:** closed-form edit of `o_proj` in the last 6 layers.

| rule | verdict | how it follows the rule, and the caveat |
|---|---|---|
| 1 size | WEAK | 9.895% is an estimate. No packed file, and code is not counted |
| 2 faithful | WEAK | 98.94% [98.47]. News 96.03 passes the point gate, but its Wilson lower bound (93.65) is below 95, so the per-kind claim is not certified |
| 3 exam | PASS | see the shared note. Base .9592 |
| 4 no new trained parts | GRAY | closed-form `o_proj` edit (unruled). The task prompt also uses retrieved labelled examples (BM25), which may count as examples in the prompt; the lane report says the feedback itself is not in the prompt |
| 5 feedback teaches | WEAK | teach .585 to .634 over 3 draws; neighbours .989 to .994 |
| 6 shuffled and empty worse | WEAK | shuffled .085 to **.494**; draw 3 (.494) is close to real (.585) |
| feedback controls (real / shuffled / empty) | | **.585-.634 / .085-.494 / 0**; wrong .006 or 0 |
| 7 real job | PASS | real label word; `Recipes` (plural) counts as wrong |
| 8 something to keep | PASS | base .9592 |
| 9 spirit | GRAY | task-based pruning plus examples in the prompt |

### L9: Qwen3-235B-A22B-Instruct-2507, pruned and 2-bit

- **Model:** Qwen3-235B-A22B-Instruct-2507, full precision.
- **How compressed (ours):** keep 88 of 128 experts, 2-bit RTN with clip search, 4-bit attention, embedding and head.
- **Size:** **9.7814%** measured on the packed files (46 GB); **9.896%** with the 537 MB `o_proj` carrier.
- **Retention:** **98.98% [98.52]**, worst kind Calendar 97.7, worst source-by-kind cell 95.3.
- **Feedback:** closed-form edit of `o_proj` in layers 86 to 93, one draw only.

| rule | verdict | how it follows the rule, and the caveat |
|---|---|---|
| 1 size | PASS | 9.896% with the carrier, measured |
| 2 faithful | PASS | 98.98% [98.52]; kinds 97.7 to 99.5. Thin margin |
| 3 exam | PASS | see the shared note. Base .9657 |
| 4 no new trained parts | GRAY | no head, adapter or lookup; the carrier is a dense fitted delta of existing tensors (ruling A) |
| 5 feedback teaches | WEAK | teach 2.4% to 74.4% on 164 items, neighbours 99.16% unchanged. **One draw only**; stability is not measured |
| 6 shuffled and empty worse | WEAK | shuffled 12.2%, wrong 0.0%, empty 2.4% vs real 74.4%. One draw |
| feedback controls (real / shuffled / empty) | | **.744 / .122 / .024**; wrong 0 |
| 7 real job | PASS | one generated skill word |
| 8 something to keep | PASS | base .9657 |
| 9 spirit | GRAY | task-based pruning. It needs about 330 GB of GPU memory to run, so it is impractical for the maintainer's own check |

### L10: Qwen2.5-Omni-7B, text path only

- **Model:** Qwen2.5-Omni-7B (text, vision, audio and speech parts). We ship only the 7.62B-parameter text path and drop the other 3.12B parameters.
- **How compressed (ours):** Hadamard rotation, GPTQ and entropy-coded scalar levels.
- **Size:** **9.879%** of the **whole** multimodal fp16 size (9.930% with feedback). This is accounting only: no packed file was written.
- **Retention:** **99.55% [99.21]**, worst kind News 98.78.
- **Feedback:** closed-form edit of 256 `down_proj` columns in each of the last 6 layers (11 MB).

| rule | verdict | how it follows the rule, and the caveat |
|---|---|---|
| 1 size | GRAY | literally under 10% of the model we started from. The denominator includes parts we throw away (ruling D). Against its disk size it is 9.53%. No packed file |
| 2 faithful | PASS | 99.55% [99.21]; every kind at least 98.78 |
| 3 exam | PASS | see the shared note. Base .9676 |
| 4 no new trained parts | GRAY | closed-form edit of existing `down_proj` columns (unruled) |
| 5 feedback teaches | WEAK | teach 0 to 86.0% [79.8] on 164 items; neighbours 99.15% unchanged. One draw, no variance estimate |
| 6 shuffled and empty worse | PASS | shuffled 0%, empty 0% (one draw) |
| feedback controls (real / shuffled / empty) | | **.860 / 0 / 0** |
| 7 real job | PASS | generated skill word |
| 8 something to keep | WEAK | base .9676 |
| 9 spirit | GRAY (weakest) | the 10% budget is inflated by modules we discard (2.25 bits per weight vs 1.62 for a text-only 7B). The maintainer may read it as defeating the purpose |

### L2: Ternary-Bonsai-2-27B (released ternary), recompressed by us

- **Model:** Ternary-Bonsai-2-27B, an already roughly 1.58-bit release (ancestor: a 27B fp16 model).
- **How compressed (ours):** pack 5 ternary values per byte, log-coded scales, a frequency-selected 165k vocabulary.
- **Size:** **9.943%** measured on the exact shipped tree (slack 30.6 MB). After the later 0.68 MB feedback file it is 9.9357% (slack 34.6 MB). The reference size is the start model's own 53.81 GB. Needs a 96 GB GPU to run.
- **Retention:** **100.0% [99.86]** against the release; **99.32% [98.93]** against the fp16 ancestor (Calendar 96.1).
- **Feedback:** closed-form low-rank edit of two `down_proj` tensors (first version); default now an `o_proj` edit (4 layers, margin 64).

| rule | verdict | how it follows the rule, and the caveat |
|---|---|---|
| 1 size | PASS | 9.943% (9.9357% after the later feedback file). Tiny slack |
| 2 faithful | PASS | passes under both readings of "original": 100% vs the release and 99.32% [98.93] vs the ancestor |
| 3 exam | PASS | see the shared note |
| 4 no new trained parts | GRAY | no head, adapter, side table or prompt; the feedback is a fitted low-rank edit of existing tensors. Labelled calibration prompts ship |
| 5 feedback teaches | PASS | default m64: teach .973 (4 draws), neighbours .971 (worst cell .952). Through the shipped commands: .976 (160 of 164), neighbours .969. m16: .913 / .991 |
| 6 shuffled and empty worse | WEAK | a proper shuffled arm (labels shuffled among the annotations) teaches .756 vs real .973 at m64; .352 vs .913 at m16. Wrong labels 0, empty 0. The first "shuffled = 0%" arm was too weak a test |
| feedback controls (real / shuffled / empty) | | m64 **.973 / .756 / 0**; m16 .913 / .352 / 0 |
| 7 real job | PASS | real generated skill word |
| 8 something to keep | WEAK | base .9668 |
| 9 spirit | GRAY | already-ternary start (ruling B) and an easy task with a strong base |

## 3. Not working versions

Each box names the rule or rules it breaks. "Dev" means our own development data, never the locked exam.

> **Filed entry (issue #2): Qwen2.5-7B tool selection** — breaks: rule 2 (faithful), rule 4 (new trained parts), rule 7 (real job). Why: it kept 93.9% [92.6] on its locked exam, answered with option letters, and fed feedback through a fitted head the maintainer ruled out. Size was fine at 9.404%. Its exam is used up.

> **Dense models up to 14B (7B, 8B, 14B)** — breaks: rule 2 (faithful). Why: too few bits per weight. Best 7B result was 96.2% on injection detection; 8B was 41% to 80%; 14B tool calls kept 56.1%. Numbers: 7B/14B/8B dev at 9.6%.

> **Dense 32B to 405B on harder tasks (news desks, injection detection, tool calls)** — breaks: rule 2 (faithful), sometimes rule 1. Why: 98% needs about 12 to 15% of fp16 on these tasks. Numbers: 32B news 96.1% [95.0] at 10.03% (over the cap); 32B injection 95.0 to 95.8%; 72B tool selection 95.55%; 72B injection 95.21%; 405B injection 90.2% [88.5].

> **Spam and phishing tasks** — breaks: rule 2 (faithful), rule 8 for AI-vs-human. Why: classes flip or collapse after compression. Numbers: phishing (definition prompt, 32B) 80.7% [75.9], Legitimate .63; scam/genuine (14B) 50.8%; spam on 14B 70.8%; the email-plus-SMS spam task collapsed (email Ham cell .13 to .26); AI-vs-human detection was at chance on the base model.

> **L6: other mixture-of-experts models (Qwen1.5-MoE, DeepSeek-V2-Lite, OLMoE)** — breaks: rule 2 (faithful). Why: pruning plus 2-bit loses too much on these coarse models. Numbers: 85.9% [82.9] at 9.87% (best; min cell 69%); DeepSeek-V2-Lite 38.2%; OLMoE 77.5%. Feedback also weak (teach .415). Dev only.

> **L7: BitNet-b1.58-2B-4T and Falcon-E** — breaks: rule 2 (faithful), rules 5 and 6 (feedback), rule 1 for Falcon-E. Why: the big tied embedding forces a vocabulary trim that is brittle; neighbouring trims collapse (47.5%, 41.8%). Numbers: 97.7% on SKILL7 alone, 95.8% pooled, worst cell 91.5%; the new label scored 88.9% with no feedback at all, so real feedback (12.7%) cannot be shown to teach. Falcon-E-3B is 13.8% (too big).

> **L8: depth and width pruning of Qwen2.5-14B** — breaks: rule 2 (faithful). Why: dropping 12 or more of 48 layers destroys the model even before quantizing. Numbers: every point at 9.9% or below retains about 0%.

> **L12 feedback on Qwen3.5-35B-A3B** — breaks: rule 5 (feedback teaches) and rule 6. Why: the routed-expert `down_proj` edit never writes the new label on this model. Numbers: teach 0 / 0 / 0 for real, shuffled and empty; neighbours .963 to .977. Its compression (7.61% analytic, retention 99.17% [98.75]) is good but fake-quantized with no packer, so L12 is a research result, not an entry. The `o_proj` rerun has not landed.

> **BANK6 (second task, banking requests) on L9b** — breaks: rule 2 (faithful), rule 5. Why: the same recipe loses the Card and Identity kinds, and feedback damages neighbours. Numbers: retention 97.37% [96.58], Card 94.4, Identity 91.7; `Security` teach .13 to .52, neighbours 76.2 / 96.1 / 85.4% vs 93.9% empty. A better quantizer clears the bar on dev only (needs a real packer).

> **BANK6 on L5** — breaks: rule 2 (faithful). Why: the Bitext source (never used for building) shifts the labels. Numbers: retention .9696 [.9612], Card .871, never-used-source retention .916.

> **BANK6 on L3** — breaks: rule 5 (feedback teaches). Why: `Security` leaks into Card unless the edit is too weak to teach. Numbers: compression passes (99.24% [98.76]) but teach is .16 on 200 items.

> **BANK6 on L4** — breaks: rule 5 and rule 6. Why: neighbours collapse in 2 of 3 draws and teach is low. Numbers: teach .07 to .27, neighbours .654 / .608 / .967.

> **Last-layer row edits as feedback (`lm_head` rows)** — breaks: rule 4 (new trained parts). Why: the maintainer called this "the same head as before, just kept inside the model file" (issue #3). Numbers: teach .49 to .80, shuffled .02, but one of 10 injection draws collapsed the model.

> **Rival-style numpy centroid feedback** — breaks: rule 4 (new trained parts) and rule 5. Why: the maintainer ruled it a nearest-centroid classifier computed outside the model, with the grown model never run on the new items. That entry's reported 36.25 to 88.5% was not generated by the model. We never shipped this; our versions generate the new label inside the model.

> **X03 (Qwen2.5-14B news router) and X04a (Qwen2.5-32B injection)** — breaks: rule 2 (faithful). Why: noise across quantizer draws. Numbers: X03 at 9.893%: 8 draws from 65.67% (one collapse) to 96.12%, one unsealing 95.44%; X04a 96.49% [96.03], worst cell 79.1, killed.

> **VQ and trellis quantizers on dense models** — breaks: rule 2 (faithful). Why: no gain over plain scalar quantization at the same bytes, within about 2 points of draw noise. Numbers: trellis (TCQ) at 9.57%: injection 96.8%, news 92.3%; VQ on 32B injection 95.0% (one kind at 81.3%).

> **Giant mixture-of-experts at about 1.5 bits per weight (R1, V3, K2, 480B and others)** — breaks: rule 1 by byte arithmetic. Why: experts would need 1.37 to 1.55 bits all in; the best public 1.58-bit file is 10.44%. Never run; our estimate of a pass is about 4%.

## 4. What's left (in order)

1. **Guessed readings (listed here, not asked of anyone).** The gray rulings are our best-evidence guesses, not settled: A, a closed-form edit of existing tensors is probably not "training" or a "head" because it adds no new part and runs through the model's own forward pass; B, starting from an already-compressed release is probably allowed ("before or after you cut it"), but our own compression is the stronger story; C, pruning experts with the task's own unlabeled prompts is probably data-dependent selection, not a trained part; D, a multimodal start model's whole fp16 size is probably the literal denominator, but dropping unused towers risks the spirit rule; E, vocabulary trimming is probably fine only where label tokens survive. We plan around these guesses and keep the gray labels. Details in [`v2/ENTRY_DECISION.md`](v2/ENTRY_DECISION.md).
2. **Re-read entrant #4's issue, read-only,** for a maintainer reply. It edits existing rows of a pruned model and does run the new label end to end, so its verdict is the best available signal on A and C.
3. **Pick the entry to file** (owner's decision before the freeze). Current order: L9b, then L1, then L5, then L2. If A goes against us we file nothing as a pass.
4. **L9b open items:** own-data dry run (about 300 unrelated prompts plus a new label close to an old one, for example `Timer` next to Alarm); a fresh-clone run of the answer and feedback commands; the locked test of the protection-weight-50 variant (dev: teach .65 to .75, shuffled .00 to .01; never run on the locked sets), to try to fix the shuffled-control leak.
5. **Packers and loaders:** L5 needs a single-file loader and a standalone feedback command; L12 needs a packer and a feedback carrier that teaches (the L9b shared-expert carrier is the obvious one); L9b on BANK6 needs a real packer for the Hadamard plus error-feedback recipe.
6. **L2 default:** decide margin m64 (dev-selected, teach .973, shuffled .756) or m16 (teach .913, shuffled .352, neighbours .991, not dev-selected).
7. **Cloud tidy-up:** the AWS box sweep did not run because the `lab` login expired. The owner runs `aws login --profile lab`, then the three regions are re-swept for leftover boxes. Rotate the Vast API key if it may have leaked.
8. **Publish artifacts privately with sha256:** durable copies of the rebuilt L9b, L2 and L1 artifacts with recorded hashes, so the maintainer's check can run from a fresh clone.
9. **Freeze 2026-11-11 08:00 UTC.** **Deadline 2026-11-12 17:05 UTC.**

## 5. Reproduce and where things live

Large artifacts are not stored in git. Each is rebuilt from the public base-model snapshot with its lane's pack scripts. The shared exam loads with `python3 -I v2/exam/loader.py`.

| what | where |
|---|---|
| current status and candidate table | [`STATUS.md`](STATUS.md) |
| the rules, word for word, and the maintainer's rulings | [`v2/RULES.md`](v2/RULES.md), [`GOAL_README.md`](GOAL_README.md) |
| which entry to file, fallbacks, open rulings | [`v2/ENTRY_DECISION.md`](v2/ENTRY_DECISION.md) |
| per-entry detail | [`v2/CANDIDATES.md`](v2/CANDIDATES.md), [`v2/SOLUTIONS.md`](v2/SOLUTIONS.md) |
| every lane, method, task and dead end | [`v2/WHAT_WE_TRIED.md`](v2/WHAT_WE_TRIED.md), [`v2/NEAR_MISSES.md`](v2/NEAR_MISSES.md) |
| lane reports, manifests (byte accounting), raw results | `v2/lanes/L1`, `L2`, `L3`, `L4`, `L5`, `L9`, `L9b`, `L10` (each `REPORT.md`, `MANIFEST.json`, `results/`) |
| failed lanes | `v2/lanes/L6`, `L7`; L8 in [`v2/WHAT_WE_TRIED.md`](v2/WHAT_WE_TRIED.md) |
| L12 mixture-of-experts ablation | [`v2/MOE_COMBOS.md`](v2/MOE_COMBOS.md), `v2/lanes/L12/` |
| feedback mechanisms tried (L11) | [`v2/FEEDBACK_ABLATION.md`](v2/FEEDBACK_ABLATION.md) |
| byte-saving tricks | [`v2/BYTE_TRICKS.md`](v2/BYTE_TRICKS.md) |
| skeptical review of L1, L2, L9b | [`v2/RED_TEAM.md`](v2/RED_TEAM.md) |
| shared locked exam, plan and hashes | `v2/exam/` (`EXAM_PLAN.md`, `MANIFEST.json`, `loader.py`) |
| annotation file format (shared) | [`v2/lanes/ANNOTATION_SCHEMA.md`](v2/lanes/ANNOTATION_SCHEMA.md) |
| answer and feedback commands | `answer.py` and `feedback.py` in each lane folder |
| older small-model tracks (record only) | `track*/` |

Where two sources disagreed, the lane `REPORT.md` was used. One case: L2 size is 9.943% in the main report and 9.9357% after the later feedback file; both are shown. L12 has no `REPORT.md`, so its numbers come from `v2/MOE_COMBOS.md`.

Most of this was built by a swarm of [Claude Code](https://claude.com/claude-code) agents working in parallel, with every result pushed as it landed.
