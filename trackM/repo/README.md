# ten-percent-challenge (DRAFT, not pushed)

Task: "small, just as good, and still learning". Base = `textattack/bert-base-uncased-SST-2` (fp32 safetensors 437,958,644 B, 109.48M params). Keep ~10%, no weight training, match the base on a locked unseen exam, grow with feedback through a memory file only.

**Status: pipeline complete; numbers below are PLACEHOLDER-CONFIG numbers on a dev stand-in, NOT the locked exam.** The real exam has not been run.

## Artifact (everything counted)
| file | bytes |
|---|---|
| `artifact/model.qpk` (bit-packed 2/3/4-bit GPTQ weights + fp16 scales/zeros + zlib full vocab + fp16 biases/LN/classifier) | 42,728,150 |
| `artifact/predict.py` (tokenizer + BERT forward + memory reader; code) | ~9,300 |
| memory file `mem1.bin` (token->int8 weights + 1000 replay anchors) | 25,659 |
| **total** | **~42,763,000 = 9.76% of base (budget 43,795,864; headroom ~1.03 MB)** |

Breakdown of `model.qpk` (`logs/build_log.json`): encoder linears 31.85 MB packed + 2.21 MB scales/zeros; embeddings 6.97 MB + 0.73 MB (train-seen tokens 3-bit, others 2-bit, full 30,522 vocab kept); pooler 6-bit 0.44 MB; small fp16 tensors 0.35 MB; vocab 0.11 MB. Effective 3.13 bits/param; 99.7% of params are kept (quantized, not pruned).
Whole-file lzma would save ~4.2 MB (not applied; would need an lzma decoder in predict.py, stdlib so cheap): headroom for a richer config.

## The 5 rules, evidence
| # | rule | evidence | caveat |
|---|---|---|---|
| 1 | one trained model | `harness/lock.py` pins base weights sha256 + id in `LOCK.json`; base evaluated by the same harness on the same exam | |
| 2 | keep ~10%, no weight training | `harness/size_check.py` (in every receipt): all bytes of code+vocab+weights+memory = 9.76%. Weights are post-training quantized (GPTQ uses a calibration pass on SST-2 *train* text to choose rounding; no gradient step on any weight, base weights are only rounded) | "GPTQ is not training" is an interpretation; stated openly. Param-count ratio from size_check reads 0% because packed bits are not a recognised tensor file; real figure is 109.14M/109.48M params at 3.13 bits |
| 3 | match the base | paired report: McNemar, 1-pt non-inferiority exact test, in the receipt | dev stand-in: base 98.80 vs artifact 98.20 (stage1); NI NOT shown (b=14,c=2) -- but dev is SST-2 train which the base memorised, so it is not a fair proxy. Real number pending |
| 4 | unseen exam locked to the end | lock-first ordering, content hash, O_EXCL single-use receipts, exam text only ever read by `final_exam.py`; builders use `splits.json` dev/feedback (train rows) only | nobody-peeked is not machine-enforceable |
| 5 | grows with feedback, no weight training, still passes | stage 2 requires identical artifact tree hash and a changed memory; memory counted in size; fresh NI test | stand-in stage 2 (mem1, 14 pending errors + 1000 anchors): dev 97.60 (-1.2 vs base): the memory currently COSTS accuracy on dev; "learns the new" is only shown by trackB's synthetic lexical scenario (new-concept 62%->97%), not on the exam |

## No-base / no-network proof
`verify.py` runs the artifact copy in a macOS `sandbox-exec` (network denied, HF cache and repo unreadable), with transformers/datasets/safetensors imports poisoned and an audit hook on every `open()`. Result (`logs/verify_output.txt`): predictions identical to in-process, no forbidden module, no forbidden path opened, socket connect fails. Only `predict.py` and `model.qpk` (+ memory) are opened. Runtime deps: python, numpy, torch.
Tokenizer is a re-implementation of BERT WordPiece: identical ids to HF on 2000 dev sentences + unicode edge cases; one known difference: a literal "[SEP]" in raw text is not treated as a special token.

## Layout
`artifact/` the submission dir (predict.py; model.qpk is ~42 MB, publish via release/LFS, not committed) | `compress/` trackA GPTQ+pack code | `memory/lex.py` lexical residual memory (fit side) | `build.py` `feedback.py` `verify.py` `reproduce.sh` | `harness/` lock/exam/size gate | `receipts_standin/` dev stand-in receipts | `logs/`.

## Honest caveats
- All accuracy numbers here are dev stand-in (train rows, base 98.8% because it memorised them); no locked-exam result exists yet.
- Strict NI at 1 pt needs b+c <= ~10 discordant on 872 items; this config agrees with base on 99.2% (dev), so passing is not assured.
- GPU/MPS GPTQ is not bit-reproducible; packaging after it is (sha256s in the log).
- Memory hyperparameters were tuned on trackB's synthetic scenario.
- The exam harness sandbox did not set TMPDIR, so any torch candidate crashed on import; predict.py works around it (sets tempfile dir). Proper fix: pass `TMPDIR=<scratch>` in `final_exam.py`'s child env (changes harness hash, so do before lock).
