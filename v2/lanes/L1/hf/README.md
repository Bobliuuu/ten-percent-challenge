---
license: apache-2.0
base_model:
- prism-ml/Bonsai-8B-unpacked
- Qwen/Qwen3-8B
tags:
- 1-bit
- bonsai
- qwen3
- ten-percent-challenge
- feedback
---
# tenpct-l1-bonsai8b-grow

Entry L1 for the 10% challenge: a natively 1-bit Bonsai-8B (lossless repack: sign bits + fp16 group scales) that routes a voice-assistant request to one of 7 skills (Alarm, Calendar, Lists, Music, News, Transport, Weather) by plain greedy generation, and that learns a NEW skill word (we test `Recipe`) from at most 25 annotated mistakes through a closed-form edit of existing tensors (no gradient, no optimiser, annotations are not stored).

Code, results, locked exam files and the full report: https://github.com/Bobliuuu/ten-percent-challenge (see `v2/lanes/L1/REPORT.md`, `v2/lanes/L1/MANIFEST.json`, and the locked exam in `v2/exam/` with a sha256 per file in `v2/exam/MANIFEST.json`; the exam is not duplicated here).

## Size (counted bytes, denominator Qwen3-8B fp16 = 16,381,470,720 B)
| item | bytes |
|---|---|
| this tree without feedback (weights, tokenizer, config, licence, code, calibration file, tensor manifest) | 1,168,130,816 (7.131%) |
| `patch.pt` after feedback, default `scale` carrier (created by `feedback.py`, absent in this repo) | 4,720,692 |
| **total after feedback** | **1,172,851,508 = 7.1596%** |
| optional `cols` carrier (`--carrier cols`), total | 1,218,478,299 = 7.4381% |

`MANIFEST.json` has the per-file sizes and the bytes audit; `SHA256SUMS.artifact` has the sha256 of every weight/tokenizer file. This model card and `MANIFEST.json` are small extras (a few KB).

## Run (CUDA GPU with ~17 GB, `torch==2.5.1`, `transformers==4.56.2`, `numpy`, `accelerate`)
```
hf download Bobliu/tenpct-l1-bonsai8b-grow --local-dir art && cd art
python answer.py --artifact . --task skill_task.json "wake me at six" "play some jazz"
# feedback: <= 25 rows {"text": ..., "label": ...}; the label may be a new word
printf '{"text":"how long do I boil an egg","label":"Recipe"}\n' > ann.jsonl
python feedback.py --artifact . --task skill_task.json --ann ann.jsonl     # writes patch.pt
python answer.py --artifact . --task skill_task.json "how do I roast a turkey"
```
An empty annotation file removes `patch.pt` (tree back to the original).

## Results (details and caveats in the GitHub report)
Locked exam (2744 items, 7 skills): retention of the base model's correct answers 1.000 (outputs identical to the release); feedback on the locked teach set (164 unseen `Recipe` items): about 0.89-0.91 taught, neighbours kept about 0.995, shuffled-label control about 0.005. Several rules are GRAY readings; see the report. The start model is not compressed by us.

## Licences and attribution
Weights are a bit-exact repack of `prism-ml/Bonsai-8B-unpacked` (Apache-2.0, Prism ML), which is built from `Qwen/Qwen3-8B` (Apache-2.0, Alibaba Qwen team). `LICENSE` and `NOTICE.txt` are included unchanged. Created using Bonsai by Prism ML.
