# FEEDBACK_ABLATION: which closed-form edit of EXISTING tensors teaches a new label best (lane L11, 2026-11-08)

Code, configs, raw results: `v2/fb_ablation/` (`fblib.py` = all mechanisms, `run_abl.py` = ablation runner, `run_exam.py` = runner on the shared locked-exam splits, `cli/answer.py` + `cli/feedback.py` = end-to-end commands, `cfgs/`, `results/modal/*.json`, `analyze.py`, `gen_tables.py`; Modal apps `tenpct-fbabl`, `tenpct-fbcli`; run ids in `RUNS.md`). Dev data only (SKILL7-derived, hash buckets [0,70)) for the ablation; the final ranking (T7-T9) uses the shared exam splits `v2/exam` (ann_pool, teach, neighbour ids) with configs fixed BEFORE that run. No sealed/locked file of any lane was read, nothing posted. Companion docs: `FEEDBACK_EXPERIMENTS.md`, `FEEDBACK_LEGAL.md` (earlier single-mechanism work), `RULES.md` 4a-4d.

## 0. Protocol (identical for every mechanism)
- Model: Qwen2.5-7B-Instruct, full model, real forward pass (no cached tails), edited tensors stored back as bf16 (what a shipped bf16 body holds). Second arm: 4-bit RTN g128 body (all q/k/v/o/gate/up/down of the 28 layers). Third arm: Qwen3-14B (4-bit RTN body and bf16), the model family of the coordinator's 9.86% SKILL7 body.
- Task: SKILL7-derived routing, 5-skill menu (Alarm, Weather, Calendar, Lists, Transport). NEW label `Jukebox` = the held-out Music class (never in a prompt menu; the base cannot emit it, base teach 0.000; the label is 4 tokens `J|uke|box|<|im_end|>`). Second label `Briefing` (held-out News) for the continual test. Sources MASSIVE + CLINC150; buckets: calibration [0,40), annotation pool [40,55), teach [55,70) (96 unseen Music items, 46 massive + 50 clinc) plus 150 more unseen items from calibration-bucket Music (`teach2`), neighbours = 714 old-skill dev items of which 690 base-correct (cells source x skill, min 20 items).
- Feedback: n annotated REAL mistakes (n = 12 headline; 5 and 25 for the core methods) drawn from the annotation pool (every Music item is a mistake of the base because it cannot say Jukebox; base answers Lists/Alarm/None...), 10 draws (8 for Qwen3-14B and the exam runs, 6 or 4 for the slow Gauss-Newton family). Controls per draw: real, shuffled (same n labels `Jukebox` moved onto unrelated ordinary old-skill requests), wrong (real Music requests with a wrong old label), empty (= base, byte-identical, teach 0.000). Cross-source: annotations from MASSIVE only evaluated on the CLINC half of the teach set, and the reverse. Paraphrase: base-model paraphrases of the annotated items. Neighbour retention is on base-correct items (aggregate, worst draw, worst cell), plus agreement of the model's own greedy answers on 135 held-out calibration-format prompts (`cal agree`).
- Evaluation = teacher-forced greedy match of the whole label string (4 tokens incl. end-of-turn) in one forward pass; validated against REAL greedy generation (T6: agreement 0.98-1.00).
- All edits are closed form from the annotated sites (teach-forced corrected answer; sites = positions where the base argmax differs). Target: the final-residual change `dx = rms * (gap + m) * u/(u.u)`, `u = gamma*(w_new - w_competitor)` (margin m logits), split over the carrier layers sequentially (layer l delivers 1/(layers left) of the CURRENT remaining dx, recomputed through the real forward). The only gradient use is the Jacobian rows of the Gauss-Newton family (one linearised min-norm step, <= 3 iterations), no optimiser anywhere.
- Calibration (key covariance, never labels): 300 task-format prompts of the old skills + the model's own greedy answers (ships as text, counted in bytes).

## 1. Mechanisms tested (tag prefix -> what it is)
| tag | mechanism |
|---|---|
| `down_ridge` | MEMIT closed form on `down_proj`: dW = R K^T (lam C + K K^T)^-1, C = key covariance of calibration tokens (no answer-position protection) |
| `down_prot` | our whitened/protected version: C + awt * C_ans (key covariance at the old-label/EOS positions), Woodbury solve; `lam`, `awt`, margin `m` swept |
| `down_alpha` | AlphaEdit-style null-space projection: keys projected off the top-k eigenspace (k = 256/1024/4096) of the calibration key covariance before the ridge solve |
| `down_ultra` | UltraEdit-style: no covariance matrix, diagonal (running-rms) key normalisation only, ridge solve ("memory-free, lifelong") |
| `down_contrast` | rank-1 whitened diff-of-means (annotated keys vs calibration answer keys), LDA-like |
| `down_steer2` | Steer2Edit-style: steering direction from the targets distributed over the top-r contrast neurons (rank-1, r = 64/512) |
| `o_prot` / `o_alpha` | same solvers on attention `o_proj` (3584-d keys, writes into the residual) |
| `up_prot` / `gate_prot` | MLP up / gate edits: target through a damped min-norm pseudo-inverse of down_proj and the silu*up linearisation |
| `norm_post`, `norm_in`, `norm_final` | RMSNorm gain edits (post-attention norm of last k layers, input norm, final norm): Gauss-Newton one-step min-norm, metric = calibration Fisher diagonal |
| `vbias` | existing v_proj bias of the last 8 layers (a Steer2Edit-style global steering vector through the existing bias), Gauss-Newton |
| `embed_gn` | input-side embedding rows of the annotated request words, Gauss-Newton, metric = token frequency in calibration prompts |
| `lm_mnH` | lm_head row edit (the earlier mnH): UNSHIPPABLE reference (4a/4b caught it) |
| `L3/L5/L8/L12/mid8/early8/spread8/all28` | layer set of the carrier (last 3/5/8/12 layers, layers 10-17, 2-9, every third, all 28) |
| `m4/m8/m16/m32`, `autocal` | edit margin; `autocal` = the feedback command picks the smallest m (4,8,16,32,64) with annotated accuracy >= 0.95 and calibration agreement >= 0.985, using ONLY annotations + calibration prompts |
| `cols*` | column-restricted carrier: only the top-r input columns of each edited tensor (by key energy at the annotated sites) are rewritten (r columns x rows x 2 bytes) |
| `requant4` | q4 body: edited tensor re-rounded onto the SAME 4-bit grid with the original scales (0 extra bytes) vs `fp16ovr` (edited tensors kept bf16/fp16 = counted bytes) |
