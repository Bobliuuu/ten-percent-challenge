# 10% challenge: RULES (single source of truth)

**Every experiment, design and agent in this repo is judged against THIS file.** It combines three things, in this order of authority:

1. **The 5 challenge rules + the 9 judging rules + the maintainer's check** (upstream README, commit `645d161`, 2026-11-06T17:25Z; our verbatim copy is `GOAL_README.md`; upstream: https://github.com/DeusExperiments/ten-percent-challenge).
2. **The maintainer's ruling on our entry** (issue #2 comment, 2026-11-06T16:10Z).
3. **The maintainer's ruling on entrant #3** (issue #3 comment, 2026-11-07T14:25Z).

Plus the earlier clarifications on issue #1 (kept in section 4). Entrant #4's issue is still unanswered (check it; see section 5). Do not name other entrants by GitHub handle in any file in this repo ("entrant #3", "entrant #4").

Deadline: **2026-11-12 17:05 UTC**. Prize: the best entry among those that pass EVERY rule. User policy: pass every rule as written, no gray readings where avoidable; never file a failing entry as a pass.

---

## 1. The challenge (5 rules, verbatim from the upstream README)

1. **Start from one trained model.** Any model is fine, as long as it is above 0.5B parameters, either before or after you cut it. Pick a commercial-class task: anything commercial or of decent utility, nothing toy-shaped and nothing with a really narrow distribution. The model already does the job, and its score is the bar to meet.
2. **Keep at most 10% of its fp16 size.** You can't train the weights. Every other technique is fair game, within the rules below.
3. **Match the base.** Your model gets the same score as the one you started from.
4. **Pass an exam it never saw.** Set aside a held-out, unseen exam before you begin (see rule 3 below). Keep it locked away until the very end, and never use it to build or to choose.
5. **Let it grow with user feedback.** Feedback means annotation or labelling, where someone marks the model's work, with very few samples, the way a real user would give it. It learns the new and keeps the old, still with no weight training, and it still passes the locked exam.

## 2. How entries are judged (9 rules, verbatim)

1. **Size counts everything.** On-disk bytes of everything you ship (weights, tokenizer, code, feedback storage) against the fp16 size of the model you started from.
2. **Same means faithful.** Keep at least 98% of the exam items the original got right, and at least 95% within every kind of item (each answer class, source or category). A higher score from different behaviour doesn't count as matching.
3. **The exam is unseen ground.** At least 2,000 locked items, with at least 300 for every kind of item. At least 500 of them come from a source or time period you never built, tuned or calibrated on.
4. **No new trained parts.** Don't add new computation fitted to data: no heads, classifiers, adapters (LoRA included), or side memory the system looks answers up from, and no feedback placed in the prompt. The answer has to come from the original model's own computation.
5. **Feedback has to teach.** At most 25 annotated items, marking real mistakes the way a user would. Lock at least 25 unseen items about what the feedback teaches before giving any feedback. On those the model must clearly improve, without breaking the nearby exam items the feedback could affect.
6. **Prove the learning came from the feedback.** Also run the same feedback with shuffled labels, and an empty feedback. Both must do clearly worse than your real feedback.
7. **Do the real job.** The model keeps the task's real output. Turning it into a multiple-choice stand-in doesn't count.
8. **The original must have something to keep.** It scores well above chance on the exam and below its ceiling.
9. **Spirit over letter.** Readings that follow the wording but defeat the purpose are rejected.

## 3. The maintainer's check (verbatim)

Ship the artifact, the locked files, one command that answers questions and one that takes annotated items as feedback. Before any prize is paid, I'll run your system on my own held-out data from a different distribution and give it my own feedback. It has to keep working and keep learning there, with no new computation fitted to data and no side memory it looks answers up from.

## 4. The maintainer's rulings (verbatim, load-bearing)

### 4a. Issue #2 (our entry), 2026-11-06T16:10Z
> thank you for the submission, steps 1-4 are fine as per relaxed submission criteria.
> step 5 doesn't hold here though. the feedback only works through head.npz, which is a classifier fitted to 2000 labels, and without it your memory changes zero answers.
>
> the objective for the challenge is to work around this problem where i want to see what can be improved **without any external memory/parameters** otherwise this step gets trivialised.

### 4b. Issue #3 (entrant #3), 2026-11-07T14:25Z
> thanks for the update and for keeping everything open, it made this easy to check. i am ok with most of the steps here but this comes down to step 5 only, which has been there from the start. the system has to learn the new from feedback, and here it doesn't. the 88.5% is computed in numpy in apply_inmodel_growth.py from averages of your feedback features, which is **a nearest-centroid classifier**, and **the grown model is never run on the new-intent items**. your answer command also only lists and parses the 10 old families, so a new intent can't come out of the system you ship. the 10 overwritten rows are **the same head as before, just kept inside the model file**. the compression side is genuinely nice, and you're welcome to try again in the new repo before the deadline.

### 4c. Issue #1 (rules Q&A), 2026-11-06T05:57Z
> 1. 10% of size on disk is fine compared to fp16 equivalent weights.
> 2. can't train the weights means compression techniques are fine but no retraining.
> 3. For user feedback section, no retraining of the weights or RAG methods because just context space being edited makes it less fun.
> 4. agentic/tool use, PII redaction, AI vs Human detectors are few examples.

### 4d. What the rulings imply (hard constraints for every design)
- **Feedback must live in the model's own computation.** No external memory, head, classifier, adapter, lookup, centroid table or prompt. Overwriting a few output (`lm_head`) rows with something fitted to feedback features is "the same head as before, just kept inside the model file" (4b). Closed-form edits of existing tensors are NOT yet ruled on: treat any few-row output edit as high risk; prefer diffuse edits that act through the model's own forward pass.
- **The shipped system must be run end-to-end on the NEW class.** The answer command must be able to emit a class that was not in the original label set, and the grown model must be evaluated on the new-class items in real generation (4b). A numpy computation outside the model does not count.
- **Real label text from generation.** Letter/MCQ argmax stand-ins and trimmed 5-letter heads are exposed to rule 7 and to the maintainer's own-data check.
- **Compression side is judged "genuinely nice"/fine** (4a: "steps 1-4 are fine"; 4b: "compression side is genuinely nice"). Resubmission after a rejection is allowed (4b), so the deadline is the only wall.
- Timeline fact: the 9-rule list was published (17:25Z) AFTER the 4a reply (16:10Z) and AFTER the 4c answer; 4b (2026-11-07) was written under the full list.

## 4e. Posture (user, 2026-11-08)
The challenge is somewhat subjective: "best" is the maintainer's judgment and several rules are decided case by case. Try MANY approaches, including gray ones, and rank by fewest rules broken. Gray routes stay in the portfolio; they are labelled gray, never filed as a clean pass.

## 4f. Starting from an already-compressed released model (user ruling on how to read the rules, 2026-11-08)
Rule 1 says any model >0.5B is fine "either before or after you cut it"; nothing forbids a pre-compressed start (Bonsai, Ternary-Bonsai). "The original" / "the model you started from" = the released model we start from (e.g. Bonsai), NOT its lineage ancestor (Qwen3-8B). Retention (rule 2) and size (rule 1) are measured against the START model. Retention against an ancestor is informational only. Residual gray: rule 9 spirit (compression not ours); answer: the entry is judged on rules 4-8, where our feedback, commands and exam work is ours. Not a rule break.

## 5. Open items (check before relying on a reading)
- Entrant #4's issue (#4) has no maintainer reply yet; it edits existing output rows but does run the new label end to end. Its verdict is the best signal for what feedback edits are allowed. Check with `gh issue view 4 -R DeusExperiments/ten-percent-challenge --comments` (read only; never post).
- Not ruled: starting from an already-low-bit released model (Bonsai-type); closed-form/diffuse weight edits for feedback; vocabulary trimming; tokenizer changes; what "best" means among passing entries (size is the natural guess).

## 6. Experiment design checklist (derived; apply to every experiment)
1. Task is commercial, broad, with a REAL generated output (a label word is OK, a multiple-choice letter is not). Base well above chance and below ceiling (rule 8).
2. Locked exam: >= 2,000 items, >= 300 per kind, >= 500 from a never-used source/time; sealed (hash) before building; never used to choose; run once.
3. Calibration / config-selection / report data are DISJOINT from each other and from the exam; best-of-N is chosen on the selection set and reported on an untouched set.
4. Primary metric = retention of base-correct items (>= 98% overall, >= 95% in every kind), from real greedy generation, with confidence bounds on the lower end. Teacher-forced and easy-source numbers are labelled as such and never count as pass evidence.
5. Size = all shipped bytes (weights, scales, codebooks, tokenizer, code, calibration/feedback files) <= 10.00% of the fp16 size of the starting model; target <= 9.9% for slack.
6. No weight training (gradients, LoRA, distillation updating weights, learned rounding is gray: flag it). No new fitted computation; no heads/adapters/side memory/lookup/prompt feedback.
7. Feedback: <= 25 annotated real mistakes; >= 25 unseen locked teach items; real vs shuffled vs empty controls clearly worse; neighbours kept; the new class generated end to end by the shipped system (see 4d).
8. Maintainer's own-data check: evaluate a proxy on a source never used for building (calibrate on A, test on B).
9. Never touch locked exams or post on GitHub from an experiment. Do not name other entrants.

---

## Appendix: older scout notes (superseded by sections 1-6)
The earlier version of this file described the README at commit `7ff8d56e` (5 rules, "first complete proof", no judging list) and a conservative interpretation table. The 9 judging rules and the maintainer's check replaced those readings. The history of README commits (all by the maintainer): `b5f66935` 2026-11-05 17:02Z initial 5 rules; `90e6c43b` 17:14Z first-proof wording; `091cd670` 19:07Z submissions as GitHub issues; `7ff8d56e` 2026-11-06 05:24Z prize/deadline/>0.5B/commercial-class/fp16 size/feedback = annotation; `645d161` 2026-11-06 17:25Z judging rules, the maintainer's check, best-entry prize.
