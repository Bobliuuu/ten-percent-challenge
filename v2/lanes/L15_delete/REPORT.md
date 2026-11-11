# L15 REPORT: feedback that only deletes weights (Deletion Tuning on Bonsai-8B)

Last updated 2026-11-11. Branch `feat/deletion-feedback`. Task, exam, splits, draws and controls are the L1 ones (SKILL7, new label `Recipe`, 164 locked teach items, 352 base-correct neighbours, 2,744-item locked exam). Code: [`README.md`](README.md). Runs: `RUNS.md`, section "L15". Modal spend about **$6.4** (app `tenpct-l15`, `modal billing report --for today`).

## 1. Bottom line

1. **Deletion teaches.** With 25 annotations, deleting about 1.0 million existing `gate_proj` weights in the last 6 layers (0.33-0.35% of those matrices, 0.012% of all parameters; `deletions.bin` 1.6-1.7 MB) makes the model write `Recipe` on **.971** of the locked teach items (8 draws, range .957-.982). The shipped commands, run end to end, give .970. Neighbours are kept at .990.
2. **It fails rule 6.** The shuffled-label control (25 ordinary requests labelled `Recipe`) teaches `Recipe` on the same locked items at **.927** (range .884-.976). That is not "clearly worse". The leak appeared in every one of six dev configurations, and none closes the gap. L1's closed-form carrier gets .001-.005 on the same control.
3. **Exam retention after feedback is .9851 [.9796] overall but Alarm drops to .908** (both measured draws). That fails our per-kind 95% gate, where L1's scale carrier keeps .9988.
4. **On rule 4 it is the cleanest carrier we have.** No value is written: the artifact is a list of positions to zero, and an empty feedback restores the tree byte for byte. But the positions are **found by gradient descent** (AdamW on per-weight mask logits). That is a different gray: rule 2 says "can't train the weights", and RULES.md section 6 item 6 lists learned rounding as gray. The weights never change value. The search over which ones to remove is trained.
5. **Verdict: not an entry.** The carrier does not pass rule 6 and fails a per-kind exam gate after feedback. It remains a useful negative result. A deletion-only edit is expressive enough to teach a new label word, but this mask learning is not selective, so the learning is not traceable to the labels.

## 2. What was built

- `deletion.py` is a re-implementation of Deletion Tuning (Lu et al., NeurIPS 2026, arXiv 2611.10866). Details were checked against the reference code (github.com/agentic-learning-ai-lab/deleting-weights: `src/learnable_sparsity/*`, `src/wanda_calibration.py`, `scripts/train_sft.sh`):
  - **Logits:** one fp32 logit per `gate_proj` weight, drawn from N(0, 0.01^2).
  - **Wanda prior:** the 2^-11 lowest-scoring weights (|W_ij| * ||x_j||_2, activations of the annotated sequences) are shifted down by 3 sigma and all others up by 3 sigma.
  - **Training mask:** `W * sigmoid(s * logit)`, with s rising linearly from 100 to 1000.
  - **Optimiser:** AdamW with lr 5e-4, wd 0 and gradient clip 1.0.
  - **Regulariser:** `- 1e-5 (1 - t/T) sum ||W*m||^2`. Its logit gradient is added analytically, and a custom autograd function saves only the layer input, which keeps memory low.
  - **Deploy:** a weight is zeroed where its logit is <= 0.
  - **L15 additions:**
    - a protection term `beta * CE` on the model's own greedy answers to the 182 shipped unlabeled calibration prompts (the same file L1 ships);
    - an optional restriction to the last k layers.
- **What "delete" means on a 1-bit model.** Bonsai weight = sign * group scale, so a single weight has no zero state. Two options were possible:
  - zero a group scale, which deletes 128 weights at once and is coarse;
  - zero individual entries, which is what we do. Those positions become 0 and the matrix is {-s, 0, +s} there. `bits.bin` and `scales.bin` are untouched. The loader zeroes the listed positions after unpacking.

  No new value is introduced in either option. Every run used the released unpacked checkpoint `prism-ml/Bonsai-8B-unpacked@376f381`, whose greedy outputs equal the L1 pack on the full locked exam (L1 `cand_vs_base_bs16_identical_outputs = 1.0`).
- `deletions.bin` holds per layer the sorted flat indices, delta-coded as uint32 and zlib-compressed. `feedback.py` and `answer.py` have the same contract as L1. Zero annotations remove the file.

## 3. Selection (dev only: ann_pool and sel; the locked teach set and the exam were not used)

Dev protocol: 25 `ann_pool` items labelled `Recipe` (real) are compared with 25 base-correct `sel` items labelled `Recipe` (shuffled). The selection rule was fixed in RUNS.md before the results came in: the highest real-minus-shuffled dev-teach, with real dev-neighbours >= .98. Ties would go to fewer deletions. Measures: dev-teach = the other 275 `ann_pool` items; dev-neighbours = 279 base-correct `sel[:300]` items. Two draws per config.

| cfg | setting | dev-teach real | dev-teach shuffled | real - shuffled | dev-neigh real / shuffled | deleted (real) | bytes |
|---|---|---|---|---|---|---|---|
| c0 | all 36 layers, 200 steps, beta 1 | .967 | .987 | -.020 | .987 / .961 | 0.21% of gate_proj | 6.6 MB |
| c1 | all layers, beta 4, cal batch 16 | .969 | .985 | -.016 | .991 / .957 | 0.20% | 6.2 MB |
| c2 | all layers, beta 0 (paper, no protection) | .982 | 1.000 | -.018 | **.862 / .000** | 0.075% | 2.2 MB |
| c3 | all layers, 400 steps, batch 4, beta 4 | .960 | .995 | -.035 | .991 / .952 | 0.15% | 4.7 MB |
| **c4** | **last 6 layers**, 200 steps, beta 4 | .953 | .938 | **+.015** | .991 / .946 | 0.34% of those 6 | 1.65 MB |
| c5 | all layers, lr 1e-4, beta 4 | .933 | .978 | -.046 | .996 / .953 | 0.17% | 5.4 MB |

All six configurations teach, and in all six the shuffled control teaches about as much. The paper setting without protection (c2) also destroys old behaviour, with the shuffled mask taking neighbours to 0. c4 was chosen by the rule and is the shipped default. Smokes: Qwen3-0.6B (100 steps): real .94, shuffled .31. Bonsai-8B (20 steps, 40-item limit): real .93, shuffled 1.00. A local CPU smoke ran out of laptop RAM at step 0. The unit tests (tiny Qwen3, CPU) pass.

## 4. Locked result (config c4, 8 draws = the L1 draws, seeds 2500+d; `results/locked_c4.json`)

| arm | teach on 164 locked items | neighbours kept (352) | exam retention after feedback (2,490 base-correct) | deleted weights | `deletions.bin` |
|---|---|---|---|---|---|
| empty | **0.000** (no file) | 1.000 | 1.000 (byte-identical tree, checked through the CLI) | 0 | none |
| **real** | **.971** mean (.957-.982) | **.990** (.983-.994) | **.9851 [.9796]**, d0 and d1. Per kind: Alarm **.908 / .911**, others .991-1.0 | 0.99-1.07 M | 1.61-1.73 MB |
| shuffled | **.927** mean (.884-.976) | .946 (.909-.983) | .9522 [.9431] (d0); Alarm .905, Transport .921, Lists .944, Weather .943 | 2.10-2.37 M | 3.1-3.5 MB |
| wrong (old wrong labels) | 0.000 | .982 (.974-.994) | .9827 [.9768] (d0); Alarm .918 | 1.33-1.49 M | 2.1-2.3 MB |

- **Shipped commands end to end** (`results/e2e_c4.json`, draw 0): `feedback.py` deleted 1,050,801 weights and wrote 1,703,353 B. `answer.py` then gave teach .970 (159/164). Empty feedback left the tree byte-identical (True).
- **Wrong-label arm:** the 25 annotated items return their wrong label in 8/8 draws (`wrong_label_hit` 1.0). The deletion memorises whatever it is given.
- **Off-menu outputs** on the exam after real feedback: 77-82, against 4-6 `Recipe` outputs. Neighbours by kind after real feedback (d0): Alarm .80 (small cell), Calendar 1.0, Lists .991, Transport 1.0. Where the lost Alarm items go was not analysed per item, because outputs per item are not stored.
- **Base reference:** the unpatched model's own greedy outputs (base accuracy .9074, same as L1).

## 5. Size

Shipped bytes = the L1 pack without its patch (1,168,094,685 B, which includes L1's code and calibration file) + L15 code (18,417 B) + `deletions.bin`. The mean real feedback gives **1,169,779,102 B = 7.141%** of Qwen3-8B fp16 (7.143% of Bonsai fp16). The largest (shuffled) file would be 7.152%. This is smaller than L1's 4.72 MB `scale` overlay (7.16%).

## 6. Rule-by-rule verdict (as the deletion carrier on the L1 base; our best-evidence readings, nothing here is a ruling)

| rule | verdict | how it follows the rule, and the caveat |
|---|---|---|
| 1 size | ✅ PASS | 7.141% including every shipped byte (deletion list 1.6-1.7 MB) |
| 2 faithful | ⚠️ GRAY | before feedback: 1.000 against the release by construction, 88.9% against the Qwen3-8B ancestor (L1 reading B). The deletion mask is found by gradient descent on mask logits. The weights never change value, but the search is trained (learned-rounding-type gray, RULES.md 6.6) |
| 3 exam | ✅ PASS | the shared SKILL7 exam (2,744 items, 1,489 from never-used sources), same caveats as L1 |
| 4 no new trained parts | ⚠️ GRAY (cleanest of our carriers) | no head, adapter, lookup, prompt or new value: the artifact is a list of existing weights to zero. Gray only because the list is fitted to data, and 182 unlabeled calibration prompts ship |
| 5 feedback teaches | ⚠️ WEAK | teach 0 -> .971 on 164 locked items, from real greedy generation; neighbours .990. But exam retention after feedback has Alarm at .908-.911, below the 95% per-kind line (overall .9851) |
| 6 shuffled and empty worse | ❌ FAIL | empty 0 and wrong 0, but **shuffled .927** (draws .884-.976) against real .971. Not clearly worse |
| 7 real job | ✅ PASS | the skill word is generated, no constrained decoding |
| 8 something to keep | ✅ PASS | base .907 |
| 9 spirit | ⚠️ GRAY | L1's pre-compressed start (no compression of ours); a mask trained per feedback round |

## 7. Why the shuffled control leaks, and what was not tried

- **Why it leaks.** Before feedback, the base model already treats recipe requests as off-menu ("Lists", "Transport" and other words with low confidence). Any feedback that makes the word `Recipe` reachable flips them.
  - The gradient-learned mask is a global edit: it makes `Recipe` reachable for whatever the model is unsure about.
  - L1's closed-form edit acts only along the annotated activation directions, which is why its shuffled arm stays near 0.
  - The shuffled masks are about twice as large (2.1-2.4 M against 1.0 M deletions) and damage neighbours (.946) and the exam (.952). This is a signature, but rule 6 is about teach, and it does not rescue the arm.
- **Not tried:**
  - selection restricted to the Wanda candidate set (a capacity cap);
  - a deletion budget or L0 penalty;
  - fewer steps;
  - deleting only along the annotated directions (a closed-form deletion score, e.g. first-order Taylor);
  - group-scale deletion;
  - BANK6;
  - the own-data check;
  - more than one exam-after draw per control arm.

  A closed-form, locality-preserving deletion score is the most promising follow-up if a deletion carrier is wanted.
