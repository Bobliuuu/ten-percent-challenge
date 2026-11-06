# TASK C / C2: tool use (function calling): tool selection + irrelevance

Maps to maintainer example "agentic/tool use". Data: argilla/apigen-function-calling (109k; xLAM-60k-style + APIGen synthetic, verified executable function calls, thousands of distinct real-API-style tools across many domains). Items kept: unique queries whose gold answer calls exactly one tool name (possibly several times with different arguments). Files `C/*.jsonl` (random distractors) and `C2/*.jsonl` (hard distractors), `LOCK.json`.
**Item**: user query + 5 options: 4 candidate tools ("name: description (params: ...)") + "None of these tools can do this". 20% of items are irrelevance items (gold tool removed; answer = E). Output = one letter. Metric: accuracy of the argmax over 5 letter logits (also: accuracy on relevant vs irrelevant items).
 - **C**: distractors = other tools of the same item, topped up by random tools of the same split.
 - **C2**: same items/splits, distractors = 3 TF-IDF nearest-neighbour tools (same-API-family confusers); irrelevance items show the 4 nearest neighbours of the (removed) gold tool. Built after seeing C is near ceiling (see below); its locked file was not scored before locking.
**Splits**: h=int(sha256("C|"+query)[:8],16); %7==0 LOCKED 2000, ==1 DEV 2000, ==2 CALIB 2000, rest POOL 4000 (caps; tool pools for distractors are drawn within the same split).
**Prompt**: as MDC-6 letter format (`inst`, `Text: query`, options, `Answer:`); few-shot = 5 calib examples (one per letter).
**Secondary (generative, not compression-friendly)**: chat-template prompt with the item's tool JSON, greedy decode a JSON list of calls; exact call match (name + stringified args, order-insensitive) on first 300 dev items.

## Base dev scores (n=2000 letter-logit; gen n=300)
| base | C zs | C 5-shot | C zs relevant / irrelevant | gen exact-call | gen name-match |
|---|---|---|---|---|---|
| Qwen2.5-1.5B-Instruct | 0.958 | 0.890 | 0.973 / 0.891 | 0.607 | 0.797 |
| Qwen2.5-3B-Instruct | 0.703 | 0.954 | 0.639 / 0.989 | 0.657 | 0.877 |
| Qwen2.5-7B-Instruct | 0.800 | 0.975 | 0.760 / 0.981 | 0.783 | 0.980 |
Chance (majority) 0.225. C is easy (random distractors) and zero-shot scores swing 0.70-0.96 purely on how often the model picks the "None" option (calibration, not capability), i.e. a poor, unstable bar. C2 (hard) dev scores below.
Generative exact tool-call match (0.61/0.66/0.78) has real headroom and is the closest to production "agentic" use, but the output space is open JSON (arguments), harder to compress-and-match and noisier to score.

## C2 (hard distractors) dev scores (n=2000, letter-logit)
| base | zs | 5-shot | zs relevant / irrelevant | 5-shot relevant / irrelevant |
|---|---|---|---|---|
| Qwen2.5-1.5B-Instruct | 0.761 | 0.705 | 0.841 / 0.406 | 0.833 / 0.134 |
| Qwen2.5-3B-Instruct | 0.692 | 0.746 | 0.685 / 0.722 | 0.846 / 0.302 |
| Qwen2.5-7B-Instruct | 0.768 | 0.782 | 0.780 / 0.714 | 0.865 / 0.414 |
Majority chance 0.225. Headroom is real (relevant-tool accuracy 0.84-0.87 few-shot; irrelevance detection is the weak spot and swings with prompt/shots, so decide ONE frozen prompt (zero-shot or fixed shots) before building and keep it). Raw: `base_results.log`.
