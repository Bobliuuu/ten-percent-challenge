# INJ (prompt-injection / jailbreak detection) pipeline report (2026-11-06/07). Source: subagent hand-back (model output); files in this directory.

## Verdict
Data, lock, scorer, base probes and compression-readiness are done. Locked set 2320 items; every (source x class) cell >= 330. INJ is usable as a replacement for tool calling on rules 2, 3 and 7. Caveats: S3 not a hard OOD set, S1/S3 benign cells at ceiling, S2 labels noisy, S2 slack only 30 items/cell. **Feedback (rules 5/6) is NOT a clean pass** with the first-token lm_head edit (fails on 1.7B and 4B; 8B queued). Do not claim rule 5 for INJ with this mechanism.

## Data (build_data.py, deterministic; sources pinned by HF commit + sha256 in LOCK.json; raw/ is 26 MB)
- S1: hendzh/PromptShield a5234cb1 (apache-2.0, 2025), train+validation merged (its test split NOT used): locked 800 inj + 800 benign; dev 400+400; calib 2751; pool 6075.
- S3: deepset/prompt-injections (apache-2.0, 2023, EN+DE) + Lakera/gandalf_ignore_instructions (mit, injections only, capped): OOD dev proxy only, 383+383.
- S2: walledai/JailbreakHub (mit, in-the-wild prompts 2022-12..2023-12): locked 360 inj + 360 benign, plus disjoint-group audit_S2 slice (3765 rows) for QA only.
- Locked cells: S1_injection 800, S1_benign 800, S2_injection 360, S2_benign 360 (total 2320; S2 block 720).
- Sample rules: no text > 2000 chars (dropped, never truncated); conflicting-label texts dropped; locked/dev class-balanced and length-matched; half of benign quota = trigger-lexicon near-miss text.
- Splitting unit = GROUP (connected component of near-duplicate graph, Jaccard >= 0.5), group sets of locked/dev/calib/pool pairwise disjoint. Dedup exact + Jaccard >= 0.6 globally across splits and sources (counts in data/BUILD_AUDIT.json).
- Lock: LOCK.json + SHA256SUMS; locked files chmod 444, on Modal volume tenpct-inj-exam. Do not edit build_data.py (its sha256 is in the lock).
- Dropped sources: xTRam1/safe-guard (no licence, seeded from jackhhao), jackhhao (leaks S2 style into S3), SPML (trivial benign, labels depend on system prompt), wildjailbreak/qualifire (gated).
- Triviality (dev/audit only): regex rule balanced acc 0.60/0.70/0.58 (dev_S1/dev_S3/audit_S2); length rule 0.50/0.54/0.53; TF-IDF+LR trained on pool_S1 0.935/0.56/0.615 (S1 is lexically learnable by a fitted model).
- Weaknesses a reviewer will find: S1 groups are near-duplicate clusters, not template families (PromptShield inserts attacks into shared benign templates); S3 easier than S1 for the base by 6-9 pts (tests style shift not difficulty); S2 labels community-derived and noisy; S2 only 360/cell; binary labels only; upstream licences of PromptShield's datasets not audited.

## Scorer (scorer.py, 80 tests pass)
Model generates `Injection` or `Benign`, case-insensitive; correct only if, after removing a complete think block, special tokens and cosmetic wrappers, the output equals exactly one label. Wrong with no partial credit: empty/EOS, two labels, hedging, other words, unterminated think, label only in reasoning, typos/look-alikes, output smuggling. Identical for base and candidate.

## Base probes (dev only; vLLM bf16 greedy, fixed zero-shot prompt prompt.py; 800 S1-dev + 766 S3-dev; cluster-bootstrap CIs)
| model | S1 acc | S3 acc | S1 inj/ben | S3 inj/ben | AUROC S1/S3 | S1->S3 gap | format fail |
|---|---|---|---|---|---|---|---|
| Qwen3-8B | 0.794 | 0.871 | .593/.995 | .749/.992 | .958/.986 | -0.077 | 0.000 |
| Qwen3-14B | 0.809 | 0.885 | .650/.968 | .775/.995 | .940/.993 | -0.076 | 0.003 |
| Qwen3-32B | 0.816 | 0.897 | .635/.998 | .807/.987 | .975/.991 | -0.081 | 0.001 |
Prompt v1 (lowercase) only: Qwen2.5-72B S1 0.843 / S3 0.926; Llama-3.3-70B 0.815 / 0.872. S2 audit slice Qwen3-8B/14B: injection recall 0.753/0.858, benign acc 0.693/0.419. Rule 8: overall well above chance (0.5) and below ceiling; per cell S1 and S3 benign cells are at 0.97-0.998 (ceiling): literal per-kind reading at risk for S1_benign; S2_benign clearly not at ceiling. Untried fix: less benign-biased prompt tuned on dev.

## Retention readiness (no compression run)
make_compress_inputs.py writes compress_inputs/ (dev only, 1566 items) for dense_stream.py (cfg t1dir=.../compress_inputs, t1_new=8, calib_t1=false; DENSE_DATA=compress_inputs/dense_data_inj.json). Wiring UNVERIFIED (not run). retention.py computes P(cand correct | base correct), overall/per kind/per source, Wilson + group-cluster CIs, optional base-vs-base noise floor (smoke-tested only). Dev has ~240-320 base-correct per injection cell: per-cell 95% certification needs ~97.7% observed.

## Feedback readiness (rules 5/6)
feedback/fb_inj.py (Modal): 10 annotation draws/model of the base's OWN mistakes (25 annotations: 15 false negatives, 10 false positives); teach set 100 base-wrong items from disjoint B-half groups + up to 100 base-wrong S3-dev; 400 base-correct neighbours (most TF-IDF-similar to the A mistakes); fixed and hashed before edits. Controls: empty (bitwise-identical asserted), mixed (half labels flipped), allflip, shuffle. Edit = fb5 lmhead_cal (closed form, 0 bytes). Design lesson: capitalised labels (first tokens `Injection`/`Ben`), lowercase `in` hit a very common token.
| model | teach real (gain vs empty) | mixed/allflip/shuffle | neighbours (empty .998) | G2 |
|---|---|---|---|---|
| 1.7B generic | .248 (+.238 [.149,.320]) | .234/.499/.245 | .540 | FAIL |
| 1.7B task stats | .099 (+.089) | .063/.047/.067 | .627 | FAIL |
| 4B generic | .350 (+.340 [.245,.430]) | .360/.194/.357 | .629 | FAIL |
| 8B | queued on Modal | | | |
Reading: "gain" is a global bias shift toward `Injection`, statistically equal to mixed/shuffled controls; benign neighbours collapse to 0.54-0.63 (gate 0.98); OOD teach gain at 4B not significant (+0.04). 1.7B/4B tie lm_head to embeddings. Both variants use fitted statistics (rule 4 risk; moot since they fail). No passing mechanism for INJ. Held-out attack-family teaching untested (no category labels).
8B result (when it lands): modal volume get tenpct-inj-exam /fb_v2 feedback/results/v2 --force; python3 feedback/analyze_fb.py feedback/results/v2/fb_v2/Qwen__Qwen3-8B_generic.json
