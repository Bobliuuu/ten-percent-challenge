# trackN results: power, rules risk, issue drafts

Reproduce: `. ../.venv/bin/activate; python power.py > power.csv; python scan.py`. These are exact multinomial enumerations over (b,c) using trackC's own `ni_pvalue` (1pt margin, Berger-Boos, alpha .05) and `mcnemar_exact`. No real data was loaded.

## 1. Power / feasibility (n=872)

Key fact: the paired tests depend only on the flip rates, not on base accuracy. Results are identical for base 0.90/0.92/0.93/0.94. Base accuracy only limits what is feasible, e.g. d >= |delta|. d = disagreement rate (fraction of examples where base and candidate predictions differ in correctness). b = base right and cand wrong; c = cand right and base wrong. delta = (c-b)/n.

Probabilities for symmetric flips (delta = 0 pt):

| d | flips (~) | NI pass | exact equal acc | within 1 ex | within 0.5pt (<=4 ex) | McNemar p>.05 |
|---|---|---|---|---|---|---|
| 0.1% | 0.9 | .991 | .50 | .90 | 1.00 | 1.00 |
| 0.2% | 1.7 | .952 | .34 | .77 | 1.00 | 1.00 |
| 0.3% | 2.6 | .895 | .26 | .67 | .99 | 1.00 |
| 0.4% | 3.5 | .833 | .22 | .60 | .98 | 1.00 |
| 0.5% | 4.4 | .775 | .20 | .54 | .97 | .99 |
| 1% | 8.7 | .572 | .14 | .40 | .88 | .98 |
| 2% | 17 | .360 | .10 | .28 | .72 | .97 |
| 3% | 26 | .282 | .08 | .23 | .62 | .97 |
| 5% | 44 | .210 | .06 | .18 | .51 | .96 |

True accuracy drop (delta), at d = 0.5 / 1 / 2 / 3 / 5%:
- delta = -0.3pt: NI .37 / .26 / .17 / .14 / .12. Exact equal .09 / .09 / .08 / .07 / .06. Within 1 ex .29 / .28 / .24 / .21 / .17. Within 0.5pt .82 / .74 / .63 / .57 / .47. McNemar .93 / .93 / .94 / .95 / .95.
- delta = -0.5pt: NI .19 / .12 / .09 / .08 / .07. Exact equal .01 / .05 / .06 / .06 / .05. Within 1 ex .07 / .15 / .17 / .16 / .15. Within 0.5pt .56 / .53 / .50 / .47 / .42. McNemar .73 / .80 / .88 / .91 / .92.
- delta = -1.0pt (d = 0.5% is infeasible, since d must be >= 1%): NI .008 / .009 / .012 / .017 for d = 1 / 2 / 3 / 5%. Exact equal .00-.03. McNemar p>.05 is .13 / .52 / .68 / .78, so McNemar badly fails to flag a 1pt drop when d is large.
- delta = +0.2pt (candidate slightly better): NI .95 at d = 0.5%, .78 at d = 1%, .52 at d = 2%.

Full grid in `power.csv`.

### Disagreement needed for >=80% chance (delta ~ 0)
- trackC NI test: d <= ~0.45%, i.e. <= ~4 flipped examples of 872. 83% at 0.4%, 77.5% at 0.5%. Even a perfect-in-expectation model passes less than 80% once d >= 0.5%. The NI test is conservative because of the exact test and the Berger-Boos penalty. Passing needs a positive true delta or a very low d.
- Exact equality of accuracy: never reaches 80% unless d is essentially 0 (50% at d = 0.1%, about one flip). Realistic ceiling is 20-35%. Equality is a bad criterion to hold ourselves to, and is also unfair to a judge. Equal accuracy can also hide many offsetting flips.
- Within 1 example: d <= ~0.15%, i.e. about 1 flip.
- Within 0.5pt (<=4 examples): d <= ~1.5%, about 13 flips (88% at 1%, 72% at 2%).
- McNemar p>.05: passes at >=95% for any d up to 5% at delta = 0. This is a weak, non-evidence criterion, since failure to reject is not equivalence.

### Implication for compression error
To be convincingly "same score" under the strict test, the compressed model must be near-lossless at the decision level. That means <= ~4 flipped predictions out of 872 (<=0.5% argmax disagreement), or a systematically net-positive delta. Practically: >=99.5% top-1 agreement with the base on the calibration/dev splits (measure it on dev data, never the exam). Typical aggressive 10%-size compression (4-bit plus pruning) gives 1-3% disagreement on SST-2. That passes McNemar and the 0.5pt criterion at 60-90%, but passes trackC NI only 30-55%. To raise NI power, either hit agreement >=99.5%, or let the candidate flip to the base's errors only, which is impossible to engineer without the exam.

### Sample size a judge needs to detect a 1pt drop
Normal approximation, paired: n ~ (1.96+0.84)^2 * d / 0.01^2 (two-sided, 80% power).
- d = 1%: ~785. d = 2%: ~1,570. d = 5%: ~3,925.
- For NI to pass with 80% power when the true delta = 0: d = 1% needs ~620, d = 2% ~1,240, d = 5% ~3,090.
- Unpaired (different test sets, accuracy ~92%): ~11,500 per arm.
- So n = 872 can detect a 1pt drop only if d <~ 1.1%. A judge using SST-2 dev alone cannot reliably distinguish "-1pt" from "same" unless the models are very close. That cuts both ways: a failed pass may be noise, and a passed exam is weak evidence. Expect to need to argue paired analysis.

## 2. Rules-risk matrix

Rules, from the image: (2) keep ~10% of the base, "you can't train the weights", "every other technique is fair game". (4) the exam is "never used to build or to choose". (5) user feedback must produce growth with "no weight training". Risk is the likelihood that a judge calls it "training the weights".

| Technique | Risk | Why | Disclosure language |
|---|---|---|---|
| GPTQ / Hessian-based weight updates | HIGH (60-75%) | Compensation updates modify weights using calibration data and second-order gradient information. Calls itself "post-training" but is data-driven weight optimization. | "Uses N unlabeled/dev calibration sentences (no labels, none from the exam) to compute a layer-wise Hessian; weights are rounded with error feedback. No gradient descent on the task loss, no labels, no fine-tuning." Pre-clear in issue #1 first. |
| SparseGPT | HIGH (60-75%) | Same as GPTQ, plus the remaining weights are updated. | Same, plus "pruning mask chosen and surviving weights re-solved in closed form per layer." |
| Closed-form compensation (least-squares bias/scale correction) | MED (35-50%) | No iterative training, but it changes values from data. Defensible as calibration. | "One-shot closed-form least-squares correction from N calibration activations; no SGD." |
| AdaRound | HIGH (65-80%) | Optimizes rounding via gradient on reconstruction loss. Looks like training of a rounding variable. | Avoid, or disclose as "learned rounding" and accept risk. |
| Pseudo-label-fit residual correctors | HIGH (70-85%) | A small learned model (trained on the base's own outputs) added on top. Fitted parameters are trained weights, even if the base is frozen. Distillation-adjacent. | Disclose as an added learned module with its parameter count included in the 10% budget. Likely violates rule 2 spirit, and also rule 5's "no weight training". |
| kNN memory (embedding store) | LOW-MED (15-30%) on training, but HIGH on size: the store counts toward the 10% under a strict reading | No weights changed. Issue is what counts as "size". Fits rule 5 well (grows with feedback). | "Memory is an append-only store of (embedding, label) pairs from user feedback; no weight updates. Included in size (bytes) at X." |
| Lexicon memory | LOW (10-20%) | A rule or token table, no weights touched. Also counts toward size. Not a "trained model" argument is possible. | Same as kNN. Disclose exact entries and provenance (not derived from the exam). |
| Model editing (ROME/MEMIT, rank-one) | HIGH (70-85%) | Directly edits weights. Not gradient descent, but a literal weight change. | Avoid for compression. For rule 5 only, ask the maintainers first. |
| Vocabulary trimming | LOW (5-10%) | Removing unused embedding rows is structure removal and does not alter kept weights. Risk is on exam-use if the vocab is chosen using the exam text (must not be). | "Vocab pruned to tokens occurring in non-exam data (list provenance). Kept rows byte-identical." |
| Entropy coding (Huffman/ANS) of weights | NONE-LOW (<5%) | Lossless. The open issue is size meaning (stored vs. in-memory). | "Lossless coding; decoded tensors are bit-identical to quantized tensors." |

General rule: any step that fits parameters to data (labels or pseudo-labels) invites the "training" label. Pure deterministic transforms (rounding, pruning by magnitude, vocab removal, entropy coding) are safe. State clearly that no exam data entered any step.

## 3. Drafts (NOT posted)

### Comment for issue #1
> Thanks for the challenge, and thanks to the author of this issue for raising these points. A few clarifications would help anyone planning a submission:
> 1. Size: is "~10%" measured by parameter count, bytes of the stored artifact, or the memory needed at inference? Do tokenizer/vocabulary files, and any external memory, count toward it?
> 2. Does calibration-based quantization or pruning (one-shot rounding or pruning that uses a small set of unlabeled sentences, with no gradient descent on the task loss and no labels) count as "training the weights", or as an allowed technique?
> 3. Does external memory (a store that grows from user feedback, with no weight updates) count toward the 10%, and is it allowed under rule 5?
> 4. "Same score": is it exact equality, within a stated margin, or a statistical test, and on what exam size?
> Thanks!

(This reveals no method detail.)

### Submission issue template
```
Title: Submission: <base model> at <X>% size, exam <dataset/split>

## Claim
Base model: <name, HF revision SHA>   Submission size: <X>% of base (<bytes/params, how measured>)
Score on locked exam: base <acc> / mine <acc>, n=<N>

## Evidence checklist
- [ ] Base model identity: weights SHA-256, config, revision
- [ ] Size: script and output (params + stored bytes + memory, tokenizer/memory included or excluded and why)
- [ ] Method summary, plus a list of every step that touches weights
- [ ] Data audit: each dataset used for build/calibration/selection, with split names and example hashes
- [ ] Exam lock: exam rows were never loaded before the final run; hash of exam file; timestamp of lock; harness commit
- [ ] Paired result: b, c, n, McNemar p, NI test (margin <m>), one-sided 95% lower bound on delta
- [ ] Both predictions files (base and candidate, per-example) so judges can recompute
- [ ] Reproduction command and environment (lockfile, hardware, seed)
- [ ] Rule 5 (if claimed): feedback format, update procedure, before/after exam score, proof that no weights changed (artifact hash before/after)
- [ ] Disclosure: calibration/pseudo-labeled data, learned modules, external memory; exact language from section 2
- [ ] Declaration: no exam data used to build or choose
```

## Repo / competition status (checked 2026-11-06)
- DeusExperiments/ten-percent-challenge: 3 commits, last 2026-11-05 19:08Z ("Take submissions as GitHub issues"). Earlier commit: "Only the first complete, working proof qualifies". The first-valid-proof rule makes the race matter.
- 1 open issue (#1, by entrant #1, updated 2026-11-06 03:52Z), 0 comments. No PRs.
- Stars 6, forks 1 (a fork owner, no new commits since upstream). Bobliuuu/ten-percent-challenge (pushed 2026-11-06) is not a fork and appears to be your own repo. No competing submission issues or attempts seen yet.
- Note: GitHub MCP server failed to connect (auth header); `gh` CLI worked fine.
