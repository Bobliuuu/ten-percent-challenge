# Track C: label-output tasks with large decision margins (option 2)

Status 2026-11-07. Dev-only probes: 300 S1-role + 300 S3-role items per task from public third-party train splits, hash-ordered. No locked or S2 rows were read or scored. No commits.
Files: `build_tasks.py`, `items.jsonl`, `calib.jsonl`, `dense_c.py` (copy of `compress/dense/dense.py` plus a `ctasks` teacher-forced evaluator), `analyze.py`, `margin_analysis.py`, `perkind.py`, `results/` (raw per-item json, tables, plots, logs).

## 1. Bottom line

**No task retains >=98% of base-correct items at <=10% of fp16, for Qwen3-8B or Qwen3-14B.** This is the existing GPTQ + Hadamard + ECSQ pipeline with the real `target_ratio=0.096` allocator.
- Best at 9.6% on 14B: prompt-injection detection 0.958 [0.938, 0.972] and tool selection 0.883. The rest of the 8 usable tasks are 0.71 to 0.83.
- On 8B at 9.6%, retention is 0.41 to 0.80.
- The margins really are large. Median base first-token margin is about 14 to 22 logits (tool-call generation is much lower), and only 1 to 5% of base-correct items have margin <2. But at 1.45 bpw the quantization noise on the margin is of the same size: std 13.1 logits (8B) and 7.0 logits (14B), with a mean shift of -11 and -6.8. Large margins help (see section 4) but cannot rescue a 1.45 bpw model on 8B or 14B.
- **Retention is steep in both bpw and model size.** 14B at 15% of fp16 reaches >=98% overall on tool selection (0.992), financial sentiment (0.987), sentiment (0.986) and prompt injection (0.980, Wilson lower bound 0.964). 8B at 15% gets only tool selection (0.990) and financial sentiment/injection at 0.98. That is the 15% setting, not the 10% cap.
- Going 8B to 14B at 9.6% lifted injection from 0.605 to 0.958 and tool selection from 0.798 to 0.883. A 30B+ base at 9.6% is therefore the one plausible route to >=98% on a label task. **I did not measure it.** The 48 GB L40S cannot hold a 32B bf16 model for the eval step, and the g2 GPU was shared.
- Practical recommendation (section 5): label-output tasks improve retention versus tool-call generation (14B at 9.6%: 0.72 to 0.96 versus 0.56), but the 10% cap with 8B/14B still fails. Pick the label task with the best margins, go to the largest affordable base, and treat >=98% at 9.6% as an open experiment, not a given.

## 2. Ranked candidates (17 considered; 9 probed)

Scoring: C = commercial legitimacy, D = distinct public sources/time periods (S1 build, S3 OOD dev, S2 locked OOD), K = classes, R = rule risk (rule 7 stand-in, rule 1 narrow). Base acc and retention are measured on my dev probes (Qwen3-14B base acc; ret = P(correct | base correct), 14B at 9.6% / at 15%). "n/p" = not probed.

| rank | task | C | sources (S1 / S3 / S2 candidates) | K | rule risk | 14B base | 14B ret 9.6% | 14B ret 15% | notes |
|---|---|---|---|---|---|---|---|---|---|
| 1 | Prompt-injection / jailbreak detection | high (LLM-app security) | safe-guard-pi (synthetic, 10k) / jackhhao jailbreaks / deepset-prompt-injections (662), in-the-wild jailbreaks (1.4k, positive only), toxic-chat jailbreaking flag | 2 | medium: sources are small and partly synthetic, so rule 1 "narrow" and per-kind>=300 on S2 are tight | 0.913 | **0.958** | 0.980 | best margins/retention; base near ceiling in S1 (0.95 on 8B) |
| 2 | Tool SELECTION as a label (name or `none`), router style | high (agentic, maintainer's list) | xLAM-style dev / S3 dev / one more tool-use set (BFCL, ToolACE, Hermes) | 2 kinds (none/tool), many names | medium: output is a name only (no arguments), a reviewer may call it a reduced tool-use task; names are multi-token but first token is distinct | 0.827 | 0.883 | **0.992** | highest margins (median 19.5 to 22); S3 half retains worst at 9.6% (0.75) |
| 3 | Content moderation / toxicity | high | civil_comments (2017, news) / toxic-chat (2023, LLM prompts) / measuring-hate-speech 135k, Jigsaw 160k, ToxiGen 9k, tweet_eval hate | 2 | low, label is the real output; label noise is high | 0.805 | 0.828 | 0.961 | "Toxic" kind is the weak one (0.74 at 9.6%, 0.92 at 15%) |
| 4 | Financial headline sentiment (3-class) | medium | twitter-financial-news (2022) / phrasebank / FiQA 1.2k, topic set | 3 | low to medium (finance domain) | 0.877 | 0.778 | 0.987 | |
| 5 | Request routing with menu (voice-assistant scenario, support dept) | high | MASSIVE scenario (18) / Bitext support (11) / HWU64, CLINC, Banking77 intents | 11 to 18 | medium (menu in prompt is a long-ish stand-in; Bitext is synthetic) | 0.792 | 0.811 | 0.975 | many small kinds, so per-kind>=300 needs a big locked set |
| 6 | Spam / phishing | high | Enron (2006) / UCI SMS (2012) / Deysi spam 8k | 2 | low | 0.805 | 0.708 | 0.959 | old, easy data; weak retention |
| 7 | NLI / grounding check | medium | MNLI / FEVER-NLI / ANLI, SNLI | 3 | low | 0.725 | 0.722 | 0.945 | lowest margins of the useful tasks |
| 8 | Review sentiment (control) | medium | SST-2 / Yelp / IMDB | 2 | high: base near ceiling (0.94), "toy" | 0.938 | 0.801 | 0.986 | control only |
| 9 | AI-vs-human text detection (maintainer's list) | high | GPT-wiki-intro / dmitva essays / Ateeqq 23k | 2 | base is chance or worse | 0.588 (8B: 0.32) | 0.26 | 0.92 | fails "base well above chance"; 8B is below chance (biased to answer "AI") |
| 10 | Intent classification (Banking77 / CLINC) | high | Banking77, CLINC150, HWU64, MASSIVE | 64 to 150 | high: Banking77 = narrow domain (entrant #3 precedent), 77 classes | n/p | n/p | n/p | many labels, per-kind>=300 impossible |
| 11 | Support-ticket triage (queue/priority) | high | Tobi-Bueck 62k (synthetic) only | ~10 | high: single synthetic source | n/p | n/p | n/p | no real OOD source found |
| 12 | PII presence / type | high | ai4privacy 300k (all positive, synthetic) | 2+ | high: no natural negatives, synthetic only | n/p | n/p | n/p | trivial or ill-posed as a label |
| 13 | Legal clause tagging | medium | LEDGAR (100 classes, one family), UNFAIR-ToS | 100 | high: classes, one source | n/p | n/p | n/p | |
| 14 | Medical triage code assignment | high | none open and licence-clean | many | high | n/p | n/p | n/p | |
| 15 | Log anomaly classification | medium | loghub HDFS, BGL | 2 | medium | n/p | n/p | n/p | zero-shot base near chance on raw logs |
| 16 | Code vulnerability yes/no | high | Devign, BigVul, PrimeVul | 2 | base near chance | n/p | n/p | n/p | |
| 17 | Language / dialect ID, news topic (AG News) | low | many | 4 to 100 | base at ceiling, toy | n/p | n/p | n/p | |
| (18) | Resume screening | high | no clean public data | n/a | legal/bias, no data | n/p | n/p | n/p | |

Not probed means ranked on data availability and rule risk only.

## 3. Retention vs bpw (dev probes, 600 items per task, Wilson 95% CIs)

Config = existing allocator (`target_ratio`), emb/head quantized at L=4/6/8, g=256/128. "ratio" is the true byte ratio including embedding, head and fixed files. Calibration text = 40 S1-role calibration items per task (disjoint from the dev probes) mixed across 9 tasks. A single-task calibration should be at least as good.

### Qwen3-14B (bf16 item acc 0.808 overall)
| task | base acc | ret 9.63% (1.48 bpw) | ret 15.03% (2.39 bpw) |
|---|---|---|---|
| INJ | 0.913 | 0.958 [0.938, 0.972] | 0.980 [0.964, 0.989] |
| TOOLSEL | 0.827 | 0.883 [0.852, 0.908] | 0.992 [0.979, 0.997] |
| FIN | 0.877 | 0.778 [0.740, 0.811] | 0.987 [0.973, 0.994] |
| SENT | 0.938 | 0.801 [0.766, 0.832] | 0.986 [0.972, 0.993] |
| TOX | 0.805 | 0.828 [0.792, 0.859] | 0.961 [0.939, 0.975] |
| ROUTE | 0.792 | 0.811 [0.773, 0.843] | 0.975 [0.957, 0.986] |
| SPAM | 0.805 | 0.708 [0.666, 0.747] | 0.959 [0.937, 0.973] |
| NLI | 0.725 | 0.722 [0.678, 0.762] | 0.945 [0.919, 0.963] |
| AIDET | 0.588 | 0.261 [0.218, 0.309] | 0.924 [0.891, 0.947] |

The 14B t20/t30 runs were not completed (stopped after t15 to free the box).

### Qwen3-8B (bf16 item acc 0.761)
| task | base acc | ret 9.64% (1.45 bpw) | ret 15.04% (2.38 bpw) | ret 20.0% (3.25 bpw) | ret 23.5% (3.75 bpw, t30 caps here) |
|---|---|---|---|---|---|
| TOOLSEL | 0.858 | 0.798 [0.761, 0.830] | 0.990 [0.977, 0.996] | 0.992 [0.980, 0.997] | 0.998 [0.989, 1.000] |
| INJ | 0.950 | 0.605 [0.565, 0.645] | 0.982 [0.968, 0.990] | 0.986 [0.973, 0.993] | 0.991 [0.980, 0.996] |
| FIN | 0.862 | 0.406 [0.365, 0.449] | 0.981 [0.965, 0.989] | 0.990 [0.978, 0.996] | 0.992 [0.980, 0.997] |
| SENT | 0.940 | 0.615 [0.574, 0.654] | 0.966 [0.948, 0.978] | 0.980 [0.965, 0.989] | 0.995 [0.984, 0.998] |
| TOX | 0.795 | 0.759 [0.719, 0.795] | 0.931 [0.904, 0.950] | 0.962 [0.941, 0.976] | 0.983 [0.967, 0.991] |
| ROUTE | 0.751 | 0.670 [0.626, 0.712] | 0.941 [0.916, 0.959] | 0.954 [0.931, 0.970] | 0.978 [0.961, 0.988] |
| SPAM | 0.702 | 0.703 [0.658, 0.745] | 0.855 [0.818, 0.886] | 0.962 [0.939, 0.976] | 0.957 [0.933, 0.973] |
| NLI | 0.673 | 0.411 [0.364, 0.459] | 0.837 [0.797, 0.869] | 0.975 [0.955, 0.987] | 0.988 [0.971, 0.995] |
| AIDET | 0.322 | 0.171 [0.124, 0.230] | 0.829 [0.770, 0.876] | 0.922 [0.876, 0.952] | 0.927 [0.882, 0.956] |

Plots: `results/retention_Qwen3-8B.png`, `results/retention_Qwen3-14B.png`. Full S1/S3/ALL tables: `results/table_Qwen3-*.md`.

Comparison point from the existing tool-call probe at the same pipeline and size (`compress/dense/t1_retention.md`): 14B at 9.6% keeps 0.561, 8B 0.219. Label tasks are clearly better (14B 0.71 to 0.96, 8B 0.41 to 0.80).

### Margins explain only part of it
Retention by base minimum-token margin (pooled, AIDET excluded):

| model / cfg | [0,2) | [2,4) | [4,8) | [8,12) | [12,16) | >=16 |
|---|---|---|---|---|---|---|
| 8B t096 | 0.39 | 0.34 | 0.45 | 0.60 | 0.65 | 0.88 |
| 8B t15 | 0.47 | 0.66 | 0.85 | 0.97 | 1.00 | 1.00 |
| 8B t20 | 0.64 | 0.78 | 0.98 | 1.00 | 1.00 | 1.00 |
| 14B t096 | 0.56 | 0.59 | 0.66 | 0.74 | 0.87 | 0.95 |
| 14B t15 | 0.68 | 0.74 | 0.93 | 0.98 | 1.00 | 1.00 |

At 9.6% even items with margin 12 to 16 logits are flipped 13% (14B) to 35% (8B) of the time: quantization noise std (7 to 13 logits) is the same order as typical margins. At 15% the noise (std 3 to 4.5) is well below the typical margin and only the low-margin tail fails. So the question is "can the base be compressed to a 3-logit-noise model at 10%", not "does the task have big margins".

## 4. Per-kind at the closest points (rule 2 needs >=95% per kind)
- 14B @ 9.6%: INJ Attack 0.985, Benign 0.931; TOOLSEL none 0.961, tool 0.827 (S3 half 0.751); TOX Toxic 0.738.
- 14B @ 15%: INJ Attack 1.000, Benign 0.960; TOOLSEL none 1.000, tool 0.986, S3 0.990; TOX Toxic 0.923 (fails per-kind); FIN kinds 0.968 to 1.000.
- 8B @ 15%: INJ 0.978 / 0.986; TOOLSEL 0.990 / 0.991 (S1 1.000, S3 0.977).
Small kinds (ROUTE classes) have n<30 per class in my probe, so only the pooled number means anything there.

## 5. Recommendation

1. Do not expect option 2 to remove the compression problem. At <=10% there is no passer on 8B or 14B. Do not pick an 8B/14B label task and plan on ">=98% at 9.6%".
2. If the 10% cap stays, the only credible lever is base size: the 8B to 14B step took injection 0.61 to 0.96 and tool selection 0.80 to 0.88 at 9.6%. Test Qwen3-32B (or the 30B-A3B MoE) on the top two tasks below before committing. That needs a box with >=80 GB for the eval step; `dense_c.py` already supports it unchanged apart from the memory.
3. Top two tasks to carry forward:
   - **A. Prompt-injection / jailbreak detection (label Attack/Benign).** Best retention (0.958 at 9.6% on 14B, 0.980 at 15%). Risks: all sources are small and partly synthetic; need a real never-touched S2 source of >=500 items (deepset prompt-injections 662 is the obvious candidate and already includes both kinds; I did not read it) and >=300 per kind locked from the larger pool (safe-guard 10k plus jackhhao 1.3k). Rule 1 "narrow distribution" is the main review risk.
   - **B. Tool SELECTION (tool name or `none`).** Highest margins, near-ceiling retention at 15% (0.992) and the closest to the maintainer's agentic example. It reuses the xLAM-style S1/S3 pools and the existing T1 build. Risk: name-only output is a smaller task than a full call; S1 base is 0.99 on 8B (ceiling), so use S3-like and harder pools to keep base below ceiling. Content moderation is a lower-margin but legitimately broad alternative if the reviewers dislike A and B.
4. Be explicit with the maintainer-side question: ask whether a label (name/class) counts as real output for tool use. Tool selection stays inside rule 7 only if the label is the production output of a router, which is the usual commercial deployment.

## 6. Data/OOD feasibility and feedback story (top 2)

**A. Injection.** S1: `xTRam1/safe-guard-prompt-injection` (train 8k+, hash split). S3 dev proxy: `jackhhao/jailbreak-classification`. S2 candidates (not touched): `deepset/prompt-injections` (546 train + 116 test), `TrustAIRLab/in-the-wild-jailbreak-prompts` (1405, positives only so it needs benign mixed in), `lmsys/toxic-chat` jailbreaking flag. Kinds = Attack / Benign: >=300 each in the locked set is feasible from safe-guard + jackhhao; the S2 block of >=500 has to be deepset (about 662 rows) so it is thin.
**B. Tool selection.** S1/S3: the T1 xLAM-style pools (`~/work/app/s1_dev.jsonl`, `s3_dev.jsonl` have 1.5k nocall and 3.2k single in S1, only 163 nocall in S3, so S3 `none` is the scarce kind) plus one never-touched set (BFCL / ToolACE / Hermes-FC) for S2. Kinds: none (>=300) and tool (>=300). Scarce `none` in S3 is the feasibility risk.
**Feedback.** A user's annotated mistake is (input text, correct label). For both tasks the label's first token is a single lm_head row (Attack/Benign; `none` vs the first token of the tool name), so the existing lm_head-row edit from `feedback_v3/` applies: add a rank-1 update along the final-hidden direction of the mistaken input to the correct-label row (and subtract from the wrongly predicted row), in closed form, zero new bytes. This was not run here. The large margins (median 14 to 22 logits) mean a feedback edit has to move a lot of logit mass for a few items, so the neighbour-damage check in G2 matters more than for a low-margin task; expect the edit size to scale with the margin of the item being fixed.

## 7. Caveats
- Probe prompts are one fixed zero-shot instruction per task, no examples, no fitted parts. Scoring is teacher-forced exact match on label tokens plus `<|im_end|>`, which equals greedy generation producing exactly the gold string.
- Sources differ between S1 and S3 only in role; ROUTE uses a different label menu per source, and NLI S3 is FEVER-NLI.
- Class-balanced samples (150/150 binary, 100 per class for 3-class), so base accuracy is not the natural-prior accuracy.
- Retention was computed on the same 600 items per task; CIs treat items as independent.
- 14B GPU run OOMed once at the lm_head quantization step; it was rerun with layers on CPU (identical maths). Logs in `results/log_run*.txt`.
- 8B t30 hit the allocator's level cap (23.5% instead of 30%).
- Boxes: `tenpct-g6` (L40S) used for all reported runs; the g2 attempt was abandoned (slow shared disk). The g6 job has been stopped.
