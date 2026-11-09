# L5: Qwen2.5-32B-Instruct, packed rANS body, SKILL7 router with closed-form feedback (shipped folder)

Task: route a voice request to one of 7 skills (Alarm, Weather, Music, Calendar, Lists, News, Transport) by plain greedy generation of the skill word (`skill_task.json`).
Weights: `body.bin` + `index.json` + `other.npz` (scalar GPTQ-style Hadamard-rotated 2/3/4-bit indices and fp16 group scales, per-tensor rANS) + tokenizer/config. Needs ONE CUDA GPU with >= 80 GB, `torch`, `transformers==4.57.1`, `constriction`, `numpy`.
No dequantised checkpoint is ever written: `packed_loader.py` decodes the rANS body tensor by tensor (CPU worker pool) and dequantises + un-rotates on the GPU (about 100 s on an A100-80GB).

| file | what |
|---|---|
| `answer.py`, `feedback.py`, `common.py` | the two commands |
| `packed_loader.py`, `pack.py` | rANS decode to GPU; `pack.py` is the packer/reference decoder |
| `fb_core.py` | closed-form feedback edit (L11 recipe) |
| `skill_task.json`, `feedback_calib.jsonl` | task prompt; 300 UNLABELED prompts (key covariance only) |
| `eval_locked.py` | reporting harness (not part of the entry) |
| `patch.pt` | exists only after feedback was given; absent = untouched model |

## Answer
```
python answer.py --artifact DIR "wake me at six" "play some jazz"
python answer.py --artifact DIR --in requests.jsonl       # plain text or {"text": ...} per line
```
## Feedback (annotated mistakes, at most 25)
`ann.jsonl` rows `{"text": "how long do I boil an egg", "label": "Recipe"}` (the label may be a word that is NOT in the menu).
```
python feedback.py --artifact DIR --ann ann.jsonl [--cols 1536]
```
Closed-form edit (no training, no gradient) of the existing `o_proj` tensors of layers 56-63: margin 32, protected by the calibration key covariance (lam 1000, awt 10), only the top `--cols` input columns of each tensor are rewritten. The replaced columns (bf16) and their indices are stored in `patch.pt` (1024 columns: 83.9 MB, 1536 columns: 125.8 MB). Annotations are not stored. An empty annotation file removes `patch.pt`. A new feedback run always starts from the unpatched model (one round).
