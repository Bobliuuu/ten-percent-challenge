# L9 report: Qwen3-235B-A22B-Instruct-2507 at 9.78% of fp16 (SKILL7 voice-skill routing)

Start model: Qwen/Qwen3-235B-A22B-Instruct-2507 (235,093,634,560 params, fp16 = 470.19 GB, 10% = 47.02 GB). Task: SKILL7 (7 skills, one-word answer, MASSIVE + CLINC150 + HWU64 + MTOP). Everything below was run on the REAL packed artifact (written by `pack235.py`, reloaded by `build_model`, byte-audited with `v2/tools/bytes_audit.py`), not on a simulation, except where marked.
Recipe: keep 88/128 routed experts per layer chosen by routing mass on 1,231 unlabeled calibration rows (`results/kept_k88.json`), grouped asymmetric RTN with MSE clip (2-bit, g128, fp16 scale, 2-bit zero point) on the kept experts, 4-bit g128 attention / embedding / lm_head, router rows of kept experts in bf16, norms and tokenizer unchanged. No gradient, no calibration-fitted rounding.

## Numbers
| item | result |
|---|---|
| Shipped bytes (audit of all files incl. tokenizer, config, code) | **45,990,926,086 B = 9.7814%** of 470.19 GB (slack 1.03 GB); `results/audit_k88.json` |
| Shared locked exam `v2/exam` SKILL7 (2,744 items), scored once, real greedy generation | base acc .9657; **retention 98.98% [Wilson lo 98.52]** (2623/2650); per kind: Alarm 99.0, Calendar **97.7**, Lists 99.1, Music 98.6, News 99.5, Transport 99.2, Weather 99.5; per source: clinc150 99.4, hwu64 98.7, massive 98.2, mtop 99.5; never-used sources (HWU64+MTOP, 1,489 items) **99.17 [98.56]**; `clean_only` tier (2,212 items) **98.83 [98.27]**, Calendar 97.6; min (source x kind) cell 95.3 |
| Dev SEL (963) / REP (932, untouched report split), packed artifact | 99.22 [98.39] / 99.43 [98.67] |
| 3 extra routing-mass calibration draws (800 random cal rows each), REP | 99.54 / 99.43 / 99.43 (lo 98.83 / 98.67 / 98.67); min cell 98.1 / 97.1 / 97.1 |
| Config scan (dev SEL/REP, RTN, same recipe) | keep 72 at 8.16%: 98.9/99.3; keep 64 at 7.35%: 98.9/99.3; keep 48 at 5.74%: 98.9/98.8; keep 32 at 4.12%: 97.2/97.5; keep 16: collapse (77/79); 3-bit keep 60 at 9.80%: 98.9/99.3; attn/emb/head 3-bit: 98.3 |
| BANK8 dev (single source, 1600 items), keep 88 RTN2, simulated | 98.96 [98.32], base .964 (not repacked, not exam) |

## 9 judging rules (honest)
| # | rule | status |
|---|---|---|
| 1 | size counts everything | PASS 9.7814% measured on the packed files; with the feedback carrier (dense fp16 o_proj delta of 8 layers, 537 MB) 9.8956% (<9.9) |
| 2 | >=98% overall, >=95% per kind | PASS on the shared exam: 98.98 [98.52]; kinds 97.7-99.5; cell min 95.3 |
| 3 | exam >=2000, >=300/kind, >=500 unseen source | PASS by the shared exam (2,744; 344-420/kind; 1,489 never-used-source items). GRAY: 532 items are tier `seen_tox` (clean_only sensitivity row above); our calibration is on MASSIVE/CLINC train rows, disjoint from the exam by construction |
| 4 | no new trained parts | GRAY: no head/adapter/lookup; expert pruning and rounding are deterministic transforms; the feedback is a closed-form edit of existing o_proj tensors (a dense delta carrier is shipped) |
| 5 | feedback teaches (<=25 annotations, 25+ unseen locked teach items, neighbours kept) | 2.4% -> 74.4% on 164 locked teach items, neighbours 99.16% unchanged (357); GRAY legality (see rule 4) |
| 6 | shuffled / empty controls clearly worse | PASS on 1 draw: 12.2% shuffled, 2.4% empty, 0.0% wrong vs 74.4% real |
| 7 | real task output | PASS: one generated skill word, exact match |
| 8 | base has something to keep | PASS: base .9657 on exam (chance .14); near the ceiling but below it; dev base .926/.942 |
| 9 | spirit | GRAY: the model is task-pruned (experts chosen by routing mass on the task prompts), the same ruling as the 30B lane; 2-bit experts; base identity = the released 235B instruct model |

## Feedback on the packed artifact (new skill `Recipe`, never in the prompt menu)
Shared exam loader: teach = 164 locked Recipe items (100 MTOP, 42 CLINC, 15 HWU64, 7 MASSIVE), neighbours = 357 base-correct items of the 379 `skill7_neighbours` exam ids, annotation = 25 items (12 Recipe from `ann_pool` + 13 real old-skill mistakes of the compressed model on calibration rows). Edit: closed-form ridge solve on o_proj of layers 86-93 (lam 5000, margin 16; chosen on dev), applied on the packed artifact's dequantised weights; carrier = 8 x 4096 x 8192 fp16 = 536,871,560 B. Real greedy generation, exact label `Recipe`.
| condition | teach `Recipe` emitted | neighbours correct (357) | min (src x kind) cell | leak of `Recipe` onto neighbours |
|---|---|---|---|---|
| empty feedback | 2.4% (4/164, natural errors) | 99.16% | 98.0 | 0 |
| **real feedback** | **74.4% (122/164)** | 99.16% | 98.0 | 0 |
| shuffled labels (same 25 inputs) | 12.2% | 99.16% | 98.0 | 0 |
| wrong labels (random other skill) | 0.0% | 99.16% | 98.0 | 0 |
Only one annotation draw (seed 0) was completed on the exam; a second real/shuffled draw (seed 1) was killed to free the box (not done). Dev history: with lam 1000 on the packed artifact two of three annotation draws were unstable (one broke the model: neighbours 5% and 92%), lam 5000 gave teach .61 and neighbours .9915 on the dev teach set; 30B-style fake-quant path, 3 draws: teach .48-.57 real vs .005-.085 shuffled vs 0 wrong. So the effect is clear (74% vs 2%/12%/0%) but its stability over annotation draws on the exam is not measured, and legality of editing o_proj is the open rule-4/5 question (the carrier is a dense fitted delta of existing tensors, not a new module).
Rule 5 gate "clearly improves without breaking neighbours": teach 2.4% -> 74.4%, neighbours unchanged. Rule 6: both controls clearly worse (12.2%, 0.0% vs 74.4%).

## Hardware needed to run answer.py
`answer.py ART "text"` rebuilds the dequantised bf16 model in GPU memory (no fused low-bit kernel): about 330 GB for the 88 kept experts (156 B params x 2 B) + 13 GB attention + 2.5 GB embedding/head, i.e. 4 x 275 GB B300 (tested), or 8 x 80 GB, or 5 x 141 GB H200 class; 1.5 TB host disk is not needed (artifact is 46 GB). Load takes about 5 min from local NVMe. A true low-bit kernel would need ~46 GB + KV cache (a single 80 GB GPU) but is NOT built.

## Gaps (not done)
BANK6 exam run, Security feedback, D4/ECSQ quantiser, GPTQ on the experts (RTN already passes), Qwen3-Next-80B (different architecture, patch not ported), a fused low-bit inference path, end-to-end CLI test of `answer.py` / `feedback.py` on the packed artifact (the box was destroyed while the test was loading; an earlier `answer.py` run without the routing prompt template printed a refusal fragment, which was fixed by wrapping the request in the SKILL7 prompt but NOT re-run; the artifact itself was exercised through `eval_art.py` / `eval_fb_exam.py`, which call the same `build_model` / `make_carrier`). A second annotation draw, BANK8/BANK6 on the packed artifact, and a locked-exam run of the feedback-edited model are not done.
Files: `core235.py` (multi-GPU port of moe_entry core: routing mass, prune, RTN), `pack235.py` (writer/loader/generation), `eval_art.py`, `eval_fb_exam.py`, `answer.py`, `feedback.py`, `build_exam.py` (own sealed exam, superseded by the shared exam), `results/`.
Compute: Vast 4 x B300 (about 5 h at ~$45/h, destroyed at the end); an earlier 8 x RTX PRO 6000 box was destroyed after 10 min (HF CDN stalled).


## Update 2026-11-09 (restart): artifact rebuilt on Modal, loader verification and locked feedback draws queued
- The original 46 GB artifact lived on a destroyed box, so it was re-packed from the public snapshot on Modal (`modal/pack_job.sh k88 cfg_art.json results/kept_k88.json`, 1x H200, about 10 min, same kept-expert file and RTN config). Audit of all shipped files (`results/audit_k88_rebuild.json`): **45,991,253,393 B = 9.7815%** of 470,187,269,120 B (the earlier box measurement was 45,990,926,086 B; the difference is the shipped code files). Artifact sits in a private Modal volume (no public copy); sha256 list produced by the pack job.
- Two earlier pack attempts of the restart failed with CUDA OOM because stale concurrent pack processes shared the GPU; the clean retry succeeded (see RUNS.md).
- Queued (multi-GPU capacity on Modal was the bottleneck, 4x H200 never scheduled, so H100:8, H200:3 and A100-80GB:6 variants were added to `modal/modal_l9.py`): `verify_loader.py` (packed loader vs the original fake-quant path on 200 dev SEL items) and `fb_draws.py locked` (seeds 0-5, lam 5000, margin 16; real / shuffled / wrong per seed, empty once; teach = 164 locked items, neighbours = base-correct locked neighbour ids). Results are appended below when they land.
