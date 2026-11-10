# L14 plan (2026-11-09): own compression that stays base-like outside the task

Idea: our L9b failed own-data (Banking77 retention 79.3, router agreement 46, generic identical 11) because experts were pruned on SKILL7 prompts only. L14 scores experts (routing mass) AND collects the GPTQ Hessians on BROAD unlabeled text = SKILL7 task prompts (2031) + general chat/QA (dolly 696), math (gsm8k 400), code (CodeAlpaca 400), wikitext (400). Budget is recovered with Hadamard + error-feedback (GPTQ/LDLQ) 2-bit-class VQ experts (bank6q recipe), real bytes counted.
Byte reality: at 10% the expert budget is about 2.06 bpw for 160/256 experts (62.5%), 1.84 bpw for 70%, 1.6 bpw for 80%. So 70-90% keep needs sub-2-bit VQ (tested as D4 K128 1.75 bpw k184 = 72%, D4 K64 1.5 bpw k205 = 80%).
Gates, pre-registered before any run (Qwen3.5-35B-A3B, fake-quant, stock expert forward):
 1. dev retention on SKILL7 SEL >= 98% overall (L9b k102: 99.48).
 2. own-data dry run (same prompts as L1/L9b/L12): Banking77-200 retention >= 95 (L1 100, L9b 79.3), router agreement on banking >= 85 (L1 94, L9b 46), generic prompts identical 24 tokens >= 70 (L1 96, L9b 11). Timer arm only if the three pass.
 3. only if 1+2 pass: locked exam ONCE + feedback carrier.
Control: t0_task = same recipe with task-only calibration (isolates the effect of the broad calibration).
Code: build_cal.py (data), l14_run.py (fork of L9b/bank6q/qrun.py), modal_l14.py/launch_l14.py, jobs/*.json. Results: results/.
