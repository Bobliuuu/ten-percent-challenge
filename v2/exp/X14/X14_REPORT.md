# X14 report: guard-model probe (step 1 only; KILLED)

Data: 300 dev items (`bundle/data/items300.json`), BTSAFE (beavertails 330k + 30k) and TOX (civil_comments + toxic_chat), 8 cells x 37 (+4 in one), dev pools only, no locked file read. Real vLLM 0.11 greedy generation, bf16, one RTX A6000 (about $0.15 total; instance destroyed and verified). Each model uses its own native prompt format (Qwen3Guard / Llama Guard 3 chat templates; generic Qwen3-8B uses the dev item prompt, no thinking). Verdict-word scorer. Margin = top1 - top2 logprob (= logit gap) at the verdict token; "min-line" = min over tokens up to the verdict token.

| model | acc | BT Safe | BT Unsafe | TOX Safe | TOX Unsafe | median verdict margin | ratio vs generic |
|---|---|---|---|---|---|---|---|
| Qwen3Guard-Gen-8B (Controversial=wrong) | 0.737 | 0.81 | 0.80 | 0.87 | 0.49 | 5.88 | 0.36 |
| same, Controversial=Unsafe | 0.870 | 0.81 | 0.89 | 0.87 | 0.91 | | |
| Llama-Guard-3-8B | 0.693 | 0.97 | 0.55 | 0.97 | 0.30 | 5.50 | 0.34 |
| Qwen3-8B generic (control) | 0.827 | 0.78 | 0.88 | 0.93 | 0.72 | 16.25 | 1.00 |

Min-line median ratio: 0.32 / 0.35. Margin histograms are in `results/step1_summary.json`.

## Decision (pre-registered rule, unchanged)
KILL after step 1: median margin must exceed 1.5x generic; it is 0.36x (Qwen3Guard) and 0.34x (Llama Guard). The guard models are LESS confident than the generic model on the same items (p10 margin 1.1 / 1.75 vs 4.75). Hypothesis (larger margins) is rejected. No step-2 VQ run, no further spend. Rule-8 band: all three are inside 0.60-0.92 overall, so that was not the blocker. Caveat: margins are in the model's own logit scale and the generic model answers a single-word-forced prompt while guards have a calibrated two-sided template; the rule was fixed in advance, so we do not reinterpret it.

## Licences / rules
- Qwen3Guard-Gen-8B: Apache-2.0 (ungated). Qwen3-8B: Apache-2.0.
- Llama-Guard-3-8B: Llama 3.1 Community License (gated by Meta; we used an unverified third-party bf16 mirror because no HF token; provenance not hash-checked against Meta). Terms for redistributing a derived artifact: must ship the licence and "Built with Llama" notice, derived-model naming prefix "Llama", acceptable-use policy applies, 700M MAU clause. Manageable but extra shipped text counts toward rule-1 bytes.
- Rule 1 "any model is fine": a released fine-tune (Qwen3Guard) is plausibly "the model you started from"; rule-9 risk LOW-MEDIUM (task is broad commercial safety, but the model is the task's own purpose-built tuning, which the maintainer might read as pre-fitted). Moot given the kill.
- Real output: Qwen3Guard emits "Safety: X / Categories / Refusal" lines; the verdict word is the real output; 19% of Qwen3Guard answers are "Controversial", a third class not in our gold labels (a 3-way plugin would have needed a mapping decision, a rule-9 exposure).
