# L3 REPORT: Qwen3-30B-A3B-Instruct-2507, task-pruned experts + 2-bit VQ + closed-form o_proj feedback (2026-11-08)

Base `Qwen/Qwen3-30B-A3B-Instruct-2507`, fp16 = 61,064,245,248 B. Tasks: SKILL7 (EASY-A) and BANK6 (EASY-B), shared locked exam `v2/exam` (scored ONCE per task, loader+report, `results/*/REPORT_scored.json`). Code: `v2/moe_entry/` (core.py harness, fb.py, qlib*.py quantisers from the quant-sweep lane), user commands `answer.py`, `feedback.py`, `runtime.py`, `eval_art.py`, `final_report.py`. Raw: `results/`, logs `v2/moe_entry/logs/`, near misses `v2/NEAR_MISSES.md`.

## Recipe (frozen from dev, not from the exam)
- Expert pruning: 85 of 128 routed experts kept per layer by routing mass on task-prompt calibration text (SKILL7 `cal_big` 2,031 prompts incl. the shared menu prefix; BANK6 `cal` 698), router rows sliced, experts exact slices.
- Quantisation (no weight training): kept experts D=4, K=256 Gaussian-codebook VQ (2.000 bpw, one fp16 sigma per expert matrix, shared fp16 codebook), two-sided random-sign Hadamard rotation, GPTQ/LDLQ-style error feedback with per-expert Hessians from the same calibration prompts (4-layer groups, earlier layers already quantised). Attention 6-bit g128 RTN, embeddings 5-bit, lm_head 4-bit (fp16 scales, int zero points), router rows + norms fp16.
- Real packed file (`weights.bin` + `index.json` + config + tokenizer) = **5,908,472,562 B = 9.676%**. Feedback replaces the 8 last-layer `o_proj` tensors by fp16 tensors (`edit_o_proj.pt`, 134,217,728 B counted in full) => **6,042,759,448 B = 9.896%** in `MANIFEST.json` (a repack would give about 9.81%).
- Feedback (`feedback.py`): closed-form diffuse edit of existing `o_proj` tensors, last 8 layers, no optimiser, no head, no new tensor, nothing in the prompt, empty feedback = no edit file. 24 annotated items = 12 items of the new skill + 12 real mistakes of the compressed model on old skills. New label is generated end to end by the unmodified answer path. Hyper-parameters chosen on dev grids (`NEAR_MISSES.md`): SKILL7 m=32; BANK6 lam3000 awt50 m8.

## Numbers on the shared locked exam (greedy generation, exact label, retention of base-correct)
| | SKILL7 (2,744 items) | BANK6 (2,237 items) |
|---|---|---|
| base accuracy | 95.92% | 88.69% |
| retention overall [Wilson lo] | **99.32% [98.92]** (2,614/2,632) | **99.24% [98.76]** (1,969/1,984) |
| worst kind | Calendar 96.69% (n 302) | Card 97.89% (n 379) |
| per source | massive 98.34, clinc 99.62, hwu64 99.66, mtop 99.64 | banking77 99.60, bitext 98.20 |
| never-used source (rule 3 >= 500) | 1,489 items, retention 99.65 [99.19] | 600 items, retention 98.20 [96.62] |
| clean-only tier retention | 99.34 [98.89] (Lists 207 / Transport 233 base-correct, below 300) | 99.15 [98.58] (Exchange 141, Identity 162 below 300) |
| after feedback, retention overall | 99.20 (min src x kind cell 97.2) | 99.19 [98.69] |
| teach set (locked, new label) | **79.9%** (164 items; mtop 83, hwu64 87, massive 100, clinc 67) in the first harness run; **87.1% mean** over 6 draws of the packed artifact with the final carrier (`prot` solver, last 8 layers, m32; 84.8-90.2) | 16.0% (200 items) with the old carrier; **46.4% mean** (28.5-55.5) over 8 draws with the final carrier (20 layers, 1638 columns, prot, lam3000, awt50, m28) |
| shuffled-label control (strict derangement) | 0.0% | 0.0% |
| wrong-label control / empty feedback | 0.0% / 0.0% | 0.0% / 0.0% |
| neighbours (exam ids near the new label) after feedback | 99.01% mean over 6 draws (min 98.87; 354 base-correct items; empty 100%); min cell .972 | 92.1% mean over 8 draws (min 83.4, max 98.5; 745 items; empty 99.2%); min cell .595 |
| size, all shipped bytes | 9.896% | 9.896% |

## 9 rules
| rule | SKILL7 | BANK6 | note |
|---|---|---|---|
| 1 size counts everything | PASS 9.896% (edit counted in full) | PASS 9.896% | slack 0.10 pt; analytic == real file within 0.003 pt |
| 2 faithful (>=98% overall, >=95% per kind) | PASS 99.32 / worst kind 96.69 | PASS 99.24 / 97.89 | per-cell (src x kind) minima: massive|Calendar 96.9 type cells, below the rival's 98.5 |
| 3 exam unseen (>=2000, >=300/kind, >=500 never-used) | PASS counts (2,744; 344-420; 1,489) | PASS counts (2,237; 330-400; 600) | clean-only tier misses 300 for Lists/Transport (SKILL7) and Exchange/Identity (BANK6): GRAY |
| 4 no new trained parts | PASS (no head/adapter/side memory) but GRAY: closed-form fit of o_proj to 24 labels; `calib_prompts.jsonl` (300 labelled old-skill prompts, 20 KB, counted) is used to compute the edit | same | maintainer has not ruled on closed-form weight edits |
| 5 feedback teaches (<=25 labels, >=25 locked teach items, clearly improves, neighbours kept) | PASS 0 -> 79.9%, neighbours 99.2 | FAIL/weak 0 -> 16% (Security leaks into Card unless the edit is too weak to teach) | BANK6 dev grid: stronger edits teach 36-44% but drop neighbours to 88-94% |
| 6 real vs shuffled vs empty | PASS 79.9 vs 0 vs 0 | weak 16 vs 0 vs 0 | |
| 7 real output | PASS free-generated skill name | PASS | the menu is in the prompt (closed-set choice) as in the rival and EASY tasks: GRAY |
| 8 base has something to keep | PASS .959 | PASS .887 | |
| 9 spirit | OK | OK | shared exam tier `seen_tox` (532 SKILL7 items) disclosed in EXAM_PLAN |

## Versus rival entry #4 (9.892%, retention 99.36%, per-cell >= 98.56, teach .649/.547 cross-source, shuffled .292)
Size 9.896% (not below 9.892); retention 99.32 (about equal, lower bound 98.92); per-cell minimum 96.7-97.9 (below the 98.5 target); teach 79.9% overall and 83-87% on never-used sources with shuffled 0% and empty 0% (beats the rival's 64.9 / 29); feedback does not touch lm_head and ships no statistics file; BANK6 feedback does not meet the bar.

## Shipped-command checks (packed artifact, `runtime.load`, not the in-memory simulation)
`eval_art.py` on the first 400 SKILL7 DEV items: base .9475, packed artifact .950, retention 99.74% (378/379). `feedback.py` with 12 Recipe annotations (m=32) wrote `edit_o_proj.pt` (134,217,728 B) and `answer.py` then emitted `Recipe` for 57/60 teach texts (before: 0/60; before-feedback outputs were Lists 35, None 14, Cooking 6, Weather 2) and 60/60 correct on 60 DEV items; `feedback.py` on an empty file removes the edit. (Dev-grade smoke tests; the exam numbers above come from the harness run.)

## Weakest rule
SKILL7: rule 4 (closed-form o_proj fit is an unruled weight edit; calibration prompts shipped) and the per-cell 98.5 target. BANK6: rule 5 (teach 16%).

## Evidence trail (what did not work)
Pure 3-bit RTN/GPTQ at 55 experts: 98.0-98.7 on dev (VOICE10 / SKILL7) and 94.9 on BANK8; excluding the shared prompt prefix from calibration collapsed retention (94.4); short names-only prompt hurt (95.9); mislabelled early GPTQ rows were RTN (deployment staleness, see NEAR_MISSES). The switch to 2-bit VQ on 85-87 experts (recipe from the quant-sweep lane) raised dev retention to 99.2-99.9.

## Cost / ops
One Vast H100 NVL box (about 9 h, about 28 USD), Modal queue starved (no Modal spend). Packed artifact (5.9 GB) lived on the box only; it is reproducible from `cfgs_final_*.json` in about 8 min (deterministic seeds; codebook shipped). Box destroyed at the end (see STATUS).


## Update 2026-11-09 (restart): final carriers measured on the PACKED artifact (`runtime.load`, real greedy generation, locked teach/neighbour sets)
Code `jobs/fb_eval.py` + `fbl3.py`; raw `results/fb/fb_locked_skill7x_s7_prot_L8_m32.json`, `results/fb/fb_locked_bank6x_b6_final.json` (dev sweeps 1-5 `results/fb/fb_dev_*`).
- SKILL7 (carrier 134,217,728 B, last 8 layers, prot, m32), 6 annotation draws, shuffled / wrong controls on every draw: real teach .848 / .860 / .866 / .902 / .890 / .860 (mean .871); shuffled .000-.024; wrong 0; empty 0. Neighbours: real .9915 / .9915 / .9915 / .9887 / .9887 / .9887 (354 base-correct locked neighbour ids; empty 1.000), min cell .972. This closes the open "SKILL7 neighbours after the final carrier" item: neighbours stay at 98.9-99.2%.
- BANK6 dev sweep 1-5 (`NEAR_MISSES.md`, `results/fb/fb_dev_bank6x_dev*`): the strongest stable edit found was 20 layers (28-47), 1638 columns, `prot` solver, lam 3000, awt 50, margin 28 (carrier 134,250,480 B, 0.10 pt slack kept). Locked run, 8 draws: real teach .285-.555 (mean .464; banking77 .56-.58, bitext_bank .20-.55), shuffled 0.000 on all draws, wrong 0.000, empty 0.000. Neighbours (745 base-correct ids) are UNSTABLE: .985 / .983 / .980 on three draws but .833-.899 on four (min cell .595), also hurt under the shuffled (.807-1.000) and wrong controls (.683-1.000). Verdict: BANK6 feedback teaches (0 -> .46, up from .16) but breaks neighbours on about half the annotation draws, so rule 5 remains FAIL for BANK6 (draw-dependent), SKILL7 is unchanged and remains the L3 entry task. Stronger margins (m36-48) teach .58-.81 but drop neighbours to .85-.96 (dev sweep 5).
- A duplicate relaunch (`results/fb/fb_locked_bank6x_b6_final_m28.json`, 6 draws, same seeds) reproduced draws 0-5 exactly (deterministic), so the run is reproducible.
