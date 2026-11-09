# L5 REPORT: dense Qwen2.5-32B-Instruct, scalar body + closed-form feedback, SKILL7 (2026-11-08/09)

Start model Qwen2.5-32B-Instruct (65,527,752,704 fp16 bytes). Task SKILL7 (7 voice skills, one-word real generation), shared sealed exam `v2/exam` (2744 items, verified, scored ONCE, real greedy generation). Machine-readable: `MANIFEST.json`. Raw results: `results/` (draws, final, final_b6, cfgs in `cfgs/`).

## Protocol actually followed
1. Calibration 128 task prompts (EASY calib), selection = lane dev set 5913 items (MASSIVE + CLINC150, disjoint from calibration and from the exam). 5 quantizer draws (seeds 0-4) at the nominal 9.85% config: dev retention .9873 / .9859 / .9839 / .9852 / .9875 (worst-kind .961 / .949 / .947 / .935 / .940). Selected draw 0 by (worst-kind retention, then overall) BEFORE touching the exam. A second config (nominal 9.75%) was also run 5 draws (.9838 / .9829 / .9827 / .9848 / .9820); not used (lower retention, packed body would leave carrier room but retention margin was thinner).
2. Selected draw re-run once with the same seed on `exam + 500 dev check items`, packed with the rANS packer (real bytes) and saved; the 500 dev rows reproduce the earlier draw (.998 per-item agreement). Base model scored on the same items with real generation.
3. Feedback config fixed before any teach/neighbour item was scored, from `v2/FEEDBACK_ABLATION.md` (o_proj protected closed form, margin 32, last 8 layers, column-restricted); primary = 1024 columns (fits the 10% line), 1536 and 2048 reported as secondary rows.
4. Draw count: 5 per config. Exam never used to choose draw or feedback config.

## Nine-rule table (SKILL7 entry)
| # | Rule | Result | Verdict |
|---|---|---|---|
| 1 | Size counts everything | packed body 6,420,039,377 B (9.797%; includes tokenizer, norms, embeddings, lm_head, rANS tables, packer code) + feedback carrier 83.9 MB (8 o_proj x 1024 columns, fp16) + code ~36 KB = **9.926%** | pass (target 9.9% missed by 0.03 pt, line 10% met) |
| 2 | Same means faithful | exam retention of base-correct **.9827** (2613/2659, Wilson LB .9770); per kind min **News .964**, Lists .978, Weather .978, Alarm .990, Music .986, Calendar .988, Transport .997; per source min hwu64 .965, massive .977. clean-only tier .9808 (min kind News .965) | pass; thin margin. Source x kind cells hwu64/News .891 (n 64) and hwu64/Weather .919 (n 37) are below 95%: a strict reading of "each kind x source" would fail |
| 3 | Unseen exam | 2744 items, every kind 344-420, 1489 from never-used HWU64 + MTOP (retention on them .9799). 532 items are `seen_tox` tier (clean-only row above) | pass (clean-only Lists 223 / Transport 236 < 300 as documented in EXAM_PLAN) |
| 4 | No new trained parts | no head/adapter/side memory/prompt; weights are scalar-quantized originals; feedback is a closed-form edit of existing o_proj tensors | gray: closed-form weight fitting unruled (RULES 4d); GPTQ-style Hessian rounding is calibration-fitted (flag) |
| 5 | Feedback teaches | 25 annotated real mistakes, 164 locked unseen teach items (121 cross-source MTOP/HWU64). Teach teacher-forced .532 (same-source .441, cross .570); REAL generation emits `Recipe` on 51.8-54.3% (3 draws); neighbours (379 exam ids, quantized-model-correct) kept 1.000, worst draw 1.000 | pass (teach modest) |
| 6 | Controls | shuffled labels 0.000, wrong labels 0.000, empty feedback 0.000 on teach (5 draws TF, 3 draws real generation); neighbours 1.0 | pass |
| 7 | Real job | one-word skill generation | pass |
| 8 | Original has something to keep | base exam accuracy .9690 (below ceiling .985 rule-of-thumb, far above chance 1/7) | pass |
| 9 | Spirit | easy 7-way routing (canonical EASY-A); packed file decoded for verification but the evals ran on the dequantized HF checkpoint; feedback uses a column-restricted o_proj rewrite | gray: task easiness, no single-file packed loader yet |

Secondary feedback rows (same run): 1536 columns teach .673 (total 9.99%), 2048 columns .718 (total 10.05%, over the line); n=12: .202 / .357 / .507.

## BANK6 (shared exam 2237 items): FAILS
Selected draw 1 of 5 (dev+sel retention .9882, worst kind .981), real bytes 9.685%. Exam retention **.9696** (LB .9612), Card **.871** (Bitext Card, never-used source), never-used-source retention .916. Overall <98% and Card <95%: not a candidate; the exam shift from Banking77 to Bitext is the failure. Base exam accuracy .9110.

## Honest gaps
- Margin on rule 2 is small (.9827 vs .98); another draw could fall below .98 (dev draw spread .9839-.9875).
- Answer/feedback commands: `answer_s7.py` works on the decoded checkpoint; the feedback is run through the research harness (fb_ablation/fblib via `val_gen.py`), no standalone CLI that edits the packed file; no single-file packed loader.
- Total bytes 9.926% (above the internal 9.9% slack target).
- Per-kind sources: News and Weather rest mostly on MTOP/HWU64, whose labels are noisy.

## Compute
Vast boxes used (our `tenpct-` boxes): repair, l5a x2; all destroyed at the end. Packed body sha256 in `results/final/pack_sha256_S7F_d0.txt`; the 6.4 GB body is not in the repo (reproducible from `cfgs/cfgs_s7final.json`, seed 0).

## Update 2026-11-09: shipped packed loader verified (folder `ship/`)
- **Packed loader**: `ship/answer.py` / `feedback.py` run straight from the rANS body (`packed_loader.py`, no dequantised file written). Rebuilt artifact (same config S7F_d0, seed 0): body.bin 6,410,187,128 B, sha256 list `results/ship/pk_sha256.txt` (private Modal volume only; not in repo). Decode vs the dequantised checkpoint: 771 tensors, 735 bit-equal, max abs diff 5.7e-8 (fp16 norm/bias/embedding rounding), 200 dev items identical answers (`results/ship/verify.json`). Load 107 s, peak 69.7 GB on an A100-80GB.
- **Exam on the rebuilt body (scored once, packed loader, real generation)**: retention of base-correct **.9801** (2606/2659), min kind News .961, min source hwu64 .967, clean-only .9776 (`results/ship/exam_new_body.json`). The earlier (first-build) number was .9827; the rebuild differs in bytes and this .9801 is now the headline. Margin over the 98% line is 0.01 pt: thin, one more draw could fall under.
- **Locked feedback through the shipped commands** (`results/ship/eval_c1024.json`, `eval_c1536.json`, 3 draws x real/shuffled/wrong + empty, one draw also as true CLI subprocesses with identical outputs): 1024 columns (83.9 MB, total ~9.93%) teach .45-.46, 1536 columns (125.9 MB, ~9.99% before code) teach .59-.62, shuffled 0, wrong 0, empty 0, neighbours .997-1.000. `feedback.py` default is 1536 columns. At 1536 columns the size line is met only barely (body 9.797% + carrier 0.192%); 1024 columns is the safer size.
- **Fresh-clone dry run** (`results/ship/dry_run.json`, `modal/modal_dryrun.py`): shallow clone, weights from the private volume (0 sha mismatches), answer -> feedback (8 Recipe annotations, 103 s) -> answer: 2/4 unseen recipe prompts answer Recipe, 7/7 neighbours unchanged; empty feedback removes patch.pt, tree identical, answers equal base. Answer takes about 93 s per call (model decode dominates).
- Gaps left: no public/durable copy of body.bin, no single-file bundle (it is one folder), one-GPU 80 GB minimum.
