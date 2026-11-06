# 10% challenge: rules, maintainer answers, interpretation

Source: https://github.com/DeusExperiments/ten-percent-challenge (README at commit 7ff8d56e, 2026-11-06T05:24Z). Scouted 2026-11-06.

## 1. Exact rule text (current README)

Prize: 1,000 USD for the first complete, working proof.

1. **Start from one trained model.** Any model is fine, as long as it is above 0.5B parameters, either before or after you cut it. Pick a commercial-class task: anything commercial or of decent utility, nothing toy-shaped and nothing with a really narrow distribution. The model already does the job, and its score is the bar to meet.
2. **Keep about 10% of its fp16 size.** You can't train the weights. Every other technique is fair game.
3. **Match the base.** Your model gets the same score as the one you started from.
4. **Pass an exam it never saw.** Set aside a held-out, unseen subset before you begin. Keep it locked away until the very end, and never use it to build or to choose.
5. **Let it grow with user feedback.** Feedback means annotation or labelling, where someone marks the model's work, with very few samples, the way a real user would give it. It learns the new and keeps the old, still with no weight training, and it still passes the locked exam.

Submit by opening a GitHub issue on the repo by **12 Nov 2026, 17:05 UTC** (7 days from the announcement https://x.com/tokenbender/status/2118389310982643779). Only the first person to submit a complete, working proof qualifies.

README history (4 commits, all by tokenbender, no PRs):
- b5f66935 2026-11-05 17:02Z: initial 5 rules.
- 90e6c43b 17:14Z: "only first complete proof qualifies".
- 091cd670 19:07Z: submissions = GitHub issues.
- 7ff8d56e 2026-11-06 05:24Z: prize, deadline, ">0.5B", "commercial-class", "fp16 size", feedback = annotation/labelling with very few samples.

## 2. Maintainer answers (issue #1, asked by StarAtNyte 2026-11-06 03:51Z)

Link: https://github.com/DeusExperiments/ten-percent-challenge/issues/1#issuecomment-6678110736 (tokenbender, 2026-11-06 05:57Z). Only comment on the only issue. Verbatim:

> I updated the readme now as well.
> 1. 10% of size on disk is fine compared to fp16 equivalent weights.
> 2. can't train the weights means compression techniques are fine but no retraining.
> 3. For user feedback section, no retraining of the weights or RAG methods because just context space being edited makes it less fun.
> 4. agentic/tool use, PII redaction, AI vs Human detectors are few examples.

Mapping to the asker's questions: (1) disk size, not param count; baseline is fp16 equivalent. (2) compression OK, "no retraining". Direct weight editing was not addressed explicitly. (3) external memory was NOT answered directly; answer says no weight retraining and no RAG-style methods (context-only editing is disliked). (4) commercial-class examples: agentic/tool use, PII redaction, AI-vs-human detection.

## 3. Interpretation table

| Item | Status | Conservative choice |
|---|---|---|
| Size measured on disk vs fp16 equivalent of base | Allowed (maintainer) | Count every byte needed for inference as ONE deployable artifact |
| Quantization, pruning, low-rank, VQ, entropy coding, distillation-free compression | Allowed ("compression techniques fine") | Fine |
| Gradient training / fine-tuning / LoRA / distillation updating weights of the shipped model | Disallowed | Never |
| Tokenizer, config, code/decoder, codebooks, masks, index tables, labels | Ambiguous | Count ALL files needed to run (weights+tokenizer+config+loader code+memory) in the 10% bytes; stay under 10% with margin (target <=9.5%, "about 10%") |
| Generic runtime libraries (torch, transformers) | Ambiguous, near-certainly excluded | Exclude, list separately; keep custom code tiny and counted |
| Closed-form weight editing (rank-one edits, least-squares recalibration, SparseGPT/GPTQ-style reconstruction, bias correction) | Ambiguous: not gradient training but "retraining" is the banned word | GPTQ/SparseGPT are standard "compression techniques" so likely OK; avoid anything that fits to task labels via optimization on weights. Prefer calibration-only (activation stats from unlabeled/dev data), disclose clearly. Do NOT use it in the feedback stage |
| Natively ternary/low-bit model (BitNet-like) as "one trained model" | Ambiguous | Rule 1 says one trained model, any model above 0.5B "before or after you cut it". A released ternary checkpoint is a trained model, but baseline = same model's fp16 equivalent; its own bf16 shadow weights/score are the bar only if released. Safest: compress a conventional model ourselves; if using a native low-bit model, baseline must be its published score and size fp16 equivalent of its parameter count |
| >0.5B params | Required | Base (or post-cut) >0.5B params |
| Task | "Commercial-class", wide distribution, not toy | Use a known-class task (tool use/agentic, PII redaction, AI-vs-human detection) or other broad benchmarked task. Narrow single-dataset classifier (e.g. Banking77) risks "narrow distribution" rejection |
| Locked exam | Strict | Create split before any tuning, hash it, commit hash/seal publicly before building; never use for choosing |
| Feedback = annotation/labelling, very few samples | Required form | Feedback items are (input, model output, human mark/label); few (single/low double digits per concept) |
| Weight training in feedback stage | Disallowed | Never |
| RAG / context-space editing / prompt stuffing in feedback | Disallowed (maintainer) | Do not retrieve text into the prompt or edit the context |
| External kNN / key-value memory over hidden states or logit-level adjustment | Ambiguous: not retraining, not text RAG, but arguably "memory like RAG"; asker's Q3 was left unanswered | Riskiest ambiguous item. Safer variants: (a) memory inside the artifact, counted in the 10% bytes; (b) acts on activations/logits not context tokens; (c) ideally non-retrieval: parametric-free state edits like per-label bias/prototype vectors updated by counting. Disclose plainly and argue it is not RAG; keep a fallback ready |
| Closed-form/analytic update of the shipped weights from feedback | Ambiguous; "no retraining of the weights" | Treat as risky; prefer a counted side-state not in the weight tensors. Do not claim weights are frozen unless true |
| Match the base | Required | Score >= base score on the locked exam (same metric, same exam). Report CIs; aim for point estimate >= base, not just noninferiority |
| Retaining old after feedback | Required | Re-run locked exam after feedback; also report old-skill retention |

## 4. Submission-issue checklist

Deadline 2026-11-12 17:05 UTC; submit earlier (first complete proof wins, check the issue list for earlier proofs first). Open a GitHub issue on DeusExperiments/ten-percent-challenge (do not comment on #1). Include:

- [ ] Base model name, revision/hash, parameter count (>0.5B), fp16-equivalent size in bytes.
- [ ] Task description and why it is commercial-class and broad; metric.
- [ ] Final artifact: file list with byte sizes, total, % of fp16 base (<=10%, state "about" margin); sha256s; what is excluded and why.
- [ ] Method: which compression techniques; explicit statement of no gradient training and no weight retraining, incl. closed-form steps disclosed.
- [ ] Locked exam: how created, size, date, hash commitment made before building; statement it was never used for building/choosing; exam run once at the end.
- [ ] Base score vs compressed score on the locked exam, same harness, with CIs; raw outputs.
- [ ] Feedback stage: the few annotation/label samples, form, how the model learns the new; before/after on new-skill held-out cases; locked exam score after feedback; old-skill retention numbers; statement no weight training and no RAG/context editing.
- [ ] Repro: repo link, one-command script, environment, logs, timestamps.
- [ ] Anticipated objections section (tokenizer bytes, memory, closed-form edits) with our reading.
