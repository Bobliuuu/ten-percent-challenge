# Challenge intel: DeusExperiments/ten-percent-challenge (collected 2026-11-08, read-only)

Sources: `gh api` on the public repo (4 issues, 0 PRs, discussions disabled, 5 commits, events feed), web search. X/Twitter is unreachable (WebFetch 402; nitter dead), HN Algolia has 0 hits, no blog/Reddit/HF threads found. So the announcement-thread clarifications are known ONLY via the maintainer's commit message and README diffs.

## 1. Decision-relevant findings (verified unless marked INFERENCE)

1. The Bonsai / already-low-bit question is NOT answered anywhere public. Nobody has asked it; no rule or comment mentions pre-quantized, distilled or already-compressed starting models. Rule 1 literally: "Start from one trained model. Any model is fine, as long as it is above 0.5B parameters, either before or after you cut it." Rule judged-1: size is "against the fp16 size of the model you started from." All four entrants so far started from a stock bf16 model and compressed it themselves.
2. Literal reading favours Bonsai: "any model is fine", baseline = "the model you started from". But the same sentence, "The model already does the job, and its score is the bar to meet" and rule 3 (match the base, >=98% of base-correct items retained) mean the baseline score is the START model's score. If start = Bonsai-8B 1-bit, size ratio is vs Bonsai's own fp16-equivalent (its 8B params x 2 B = ~16 GB), i.e. a repack is ~7.1% of that, but the "retention vs base" is then trivially 100% (lossless). INFERENCE: this is exactly the "readings that follow the wording but defeat the purpose" the maintainer reserves rejection for (rule 9). The purpose, per the tagline and the maintainer's own words on #3 ("the compression side is genuinely nice"), is that the entrant does the compression without retraining. Starting from someone else's trained-in 1-bit model makes the compression side of the proof contribute nothing. High risk of a rule-9 rejection; no precedent either way.
3. Maintainer rules by the purpose and has actively rejected on rule 4/5 grounds (feedback must come through the original model's own computation, not an extra fitted component), after accepting the compression half. Both substantive rulings so far are about step 5 (feedback), not size. Compression/size/exam construction of entrants has NOT yet been ruled on (no rejection on those grounds, but also no acceptance).
4. Maintainer replies are slow-ish but real: #1 answered in 2 h, #2 in ~3.6 h, #3 in ~21 h (second comment). #4 (filed 2026-11-07T17:45Z) has NO response yet as of 2026-11-08. #2 (ours, Bobliuuu) is open, only the one rejection comment.
5. Rules were tightened after our submission: #2 was filed 2026-11-06T12:35Z. Maintainer reply at 16:10Z said "steps 1-4 are fine as per relaxed submission criteria". The strict "How entries are judged" 9 rules, "maintainer's check", "best" prize wording were committed 17:25:52Z, AFTER that reply. INFERENCE: the 9-rule list (esp. 4 no trained parts, 7 no MCQ stand-in) now applies to entrants by the deadline; "steps 1-4 fine" does not clear our letter-format task (rule 7) or vocab-trim-to-5-letter lm_head.
6. Prize: first version "first complete, working proof" -> changed 11-05 17:14Z -> 11-06 05:24Z -> 11-06 17:25Z to "best complete, working proof submitted by the deadline"; "Among the entries that pass every rule, the best one wins the prize." No criteria for "best" are stated. INFERENCE: size is the natural metric ("at most 10%", smaller = better), but retention/feedback strength could count. Unstated.
7. Deadline: "12 November 2026, 17:05 UTC (7 days from the announcement)". Announcement therefore 2026-11-05 17:05 UTC (first commit 17:02:50Z agrees). "Entries stay open until the deadline." Submission = a GitHub issue on the repo.
8. Maintainer final check (README): "Before any prize is paid, I'll run your system on my own held-out data from a different distribution and give it my own feedback. It has to keep working and keep learning there, with no new computation fitted to data and no side memory it looks answers up from." Needs: artifact, locked files, one command answering questions, one command taking annotated items as feedback. So a task-specific hack (e.g. 5-letter lm_head, vocab trimmed to one task's pools) risks failing on his distribution. INFERENCE for that last clause.

## 2. Entrant table

| # | Entrant | Model | Task | Claimed size / retention | Verdict (as of 2026-11-08) | Rule/reason cited |
|---|---|---|---|---|---|---|
| 1 | entrant #1 | none (rules Q&A) | n/a | n/a | answered, not a submission | see section 3 |
| 2 | Bobliuuu (us) | Qwen2.5-7B-Instruct | C2 tool selection among 4 + none (letter prompt) | 9.404%; 0.799 vs base 0.7675; 0.802 after feedback | PARTIAL REJECT: "steps 1-4 are fine", step 5 "doesn't hold" | feedback "only works through head.npz, a classifier fitted to 2000 labels"; later rule 4. Rule 7 (MCQ) not raised yet but was added after |
| 3 | entrant #3 | Qwen3-30B-A3B-Instruct-2507 | Banking77 -> 10 queues router | 10.48% disk (76.25% of params stored, 96/128 experts, ~2.2 bpw); locked 96.00% vs base 95.80% (1,000 items) | REJECTED on step 5, invited to retry in new repo (entrant #3/ten-percent-challenge-first / -best) | overwriting 10 lm_head rows with feedback centroids "is a nearest-centroid classifier"; "grown model is never run on the new-intent items"; answer cmd lists only old 10 families; "the same head as before, just kept inside the model file" |
| 4 | entrant #4 | Qwen3-30B-A3B-Instruct-2507 | voice-assistant skill routing (7+ skills, free-gen skill name) | 9.892% (6.04 GB); 99.36% of base-correct kept; 2,910 locked items, 600 CLINC150 unseen source; feedback 18.97%->64.87%, shuffled 29.23%, empty 18.97% | PENDING, no comment | n/a. Note: feedback = closed-form edit of existing output-layer rows using mean hidden state (same family of method entrant #3 was told is a centroid classifier); also lowered its own dev gate; music subset shuffled > real. Likely to meet the same objection (INFERENCE) |
| - | a fork owner, a fork owner | forks only | - | - | no submission | - |

## 3. Maintainer statements (exact quotes, short)

Issue #1 (answering entrant #1's 4 questions; the thread is the only Q&A):
- "10% of size on disk is fine compared to fp16 equivalent weights."
- "can't train the weights means compression techniques are fine but no retraining."
- "For user feedback section, no retraining of the weights or RAG methods because just context space being edited makes it less fun."
- "agentic/tool use, PII redaction, AI vs Human detectors are few examples."
- "I updated the readme now as well."
Note: direct weight editing was not explicitly answered; but see #3 where editing existing rows was rejected because it is functionally a classifier, not because editing is banned.

Issue #2 (to us): "steps 1-4 are fine as per relaxed submission criteria. step 5 doesn't hold here though. the feedback only works through head.npz, which is a classifier fitted to 2000 labels, and without it your memory changes zero answers." and "the objective for the challenge is to work around this problem where i want to see what can be improved without any external memory/parameters otherwise this step gets trivialised."

Issue #3 (to entrant #3): "this comes down to step 5 only, which has been there from the start. the system has to learn the new from feedback, and here it doesn't... a nearest-centroid classifier... the grown model is never run on the new-intent items. your answer command also only lists and parses the 10 old families, so a new intent can't come out of the system you ship. the 10 overwritten rows are the same head as before, just kept inside the model file. the compression side is genuinely nice, and you're welcome to try again in the new repo before the deadline."

## 4. Topic-by-topic

- Start from already-quantized / low-bit / distilled model: NO statement. Closest: "Any model is fine... either before or after you cut it" (rule 1; about the >0.5B floor, i.e. the model may be pruned below 0.5B after cutting). "compression techniques are fine".
- Size baseline: "fp16 size of the model you started from"; on-disk bytes of everything shipped (weights, tokenizer, code, feedback storage); "10% of size on disk is fine compared to fp16 equivalent weights." entrant #3's param-count caveat (76% of params) was not objected to. Python libs: we counted none (our own claim, unruled).
- Vocabulary pruning / tokenizer / head changes: no ruling. Our #2 vocab trimmed to 24,000 tokens + lm_head to 5 letters; maintainer said steps 1-4 fine on 11-06 16:10Z (before the strict rule list), so size/compression/exam not objected to then. Entrant #3 and #4 touched lm_head rows only for feedback. Rule 4 bans "heads, classifiers, adapters (LoRA included), or side memory"; rows "overwritten" in the existing head were ruled to be the same classifier head (#3).
- Feedback accepted/rejected: rejected = external head fitted to labels (#2); overwriting/centroid-averaging output rows from feedback hidden states, never run on the new classes and not generating the new label (#3). Nothing yet accepted. #4's variant (edit rows of the taught label tokens, new skill `recipe` generated via free generation) is the first that does run the new label end to end; its fate is the best signal of what is allowed. Context/RAG/prompt feedback banned. Weight gradient training banned.
- "Training" definition: only "no retraining"/"can't train the weights"; README rule 4: "Don't add new computation fitted to data". No statement on closed-form edits per se.
- Exam construction: README rule 3: >=2,000 locked items, >=300 per kind, >=500 from a source or time period "you never built, tuned or calibrated on". Hygiene "Keep it locked away until the very end, and never use it to build or to choose." No ruling on any entrant's exam; #3 used 1,000 items (below 2,000) and the maintainer did not cite that (his stated objection was step 5 only).
- MCQ / letter format: README rule 7: "The model keeps the task's real output. Turning it into a multiple-choice stand-in doesn't count." Not applied to anyone yet. Our #2 uses a 4-way letter prompt; high rule-7 risk. (Maintainer's 'steps 1-4 fine' predates rule 7.)
- "Best" ranking: only "Among the entries that pass every rule, the best one wins the prize." Criteria unspecified; passing all 9 first. 
- Rule 8: base must score "well above chance on the exam and below its ceiling".
- Rule 9: "Readings that follow the wording but defeat the purpose are rejected."

## 5. Bonsai: closest precedent (INFERENCE)

No direct precedent. Closest signals: (a) maintainer's repeated "the objective... is to work around this problem"/"want to see" language, i.e. he judges purpose; (b) he praised only genuine compression ("compression side is genuinely nice"); (c) rule 9; (d) the start model must "already do the job" with a score bar to meet; a lossless repack of Bonsai trivially meets the bar but the 10% comes from Bonsai authors' training, and quantization-aware-trained released models are arguably "one trained model" legitimately (README does not say the model must be full precision). Rational plan: if using Bonsai, (1) ask the maintainer on a NEW issue only with explicit approval (we were told not to post now), (2) put the fp16-equivalent of Bonsai's own parameter count as baseline AND report vs the original fp16 Qwen3-8B lineage, so the size claim holds under both baselines; (3) do real additional compression/feedback work so rule 9 cannot bite. Safest: do not rely on Bonsai-as-baseline alone.

## 6. Time / logistics

- Deadline 2026-11-12 17:05 UTC (= 12:05 EST, 09:05 PST, 22:35 IST). Today 2026-11-08; about 4 days left.
- Repo main head 645d1614 (2026-11-06T17:25:52Z); README last changed then; watch for further edits (`gh api repos/DeusExperiments/ten-percent-challenge/commits`).
- Maintainer review of entrant #3 took ~21 h on the last turnaround; entrant #3 was invited to resubmit "in the new repo before the deadline", so resubmission by edit/new issue is accepted practice.
- Maintainer's own-data check is pre-payment and uses his "own feedback"; plan answer/feedback commands to run on arbitrary data.
- Not obtainable: the X announcement thread and replies (blocked); any statement there is only reflected in the commit "Add the prize, deadline and rule clarifications from the announcement thread" (2026-11-06T05:24Z). Suggest the user read the thread manually.
