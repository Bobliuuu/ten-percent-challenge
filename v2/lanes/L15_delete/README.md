# L15: deletion-mask feedback carrier (Bonsai-8B, SKILL7 + `Recipe`)

Feedback that **only deletes existing weights**. From at most 25 annotated mistakes, `feedback.py` learns which entries of the existing MLP `gate_proj` matrices to set to zero, and writes `deletions.bin`: a delta-coded, zlib-compressed list of their positions. It stores no values and adds no tensor. `answer.py` loads the model, zeroes those positions and answers by plain greedy generation. An empty annotation file removes `deletions.bin`, so the artifact is byte-identical to the start.

The method is Deletion Tuning / Bit-Mask Tuning (Lu et al., NeurIPS 2026, arXiv 2611.10866; reference code github.com/agentic-learning-ai-lab/deleting-weights), re-implemented in `deletion.py`. There is one logit per weight, initialised from Wanda scores. The mask is `sigmoid(sharpness * logit)`, with sharpness annealed from 100 to 1000. AdamW updates the logits, and a weight is deleted where its logit is <= 0. L15 adds an optional protection term: cross-entropy on the model's own answers to the 182 shipped unlabeled calibration prompts (the same file L1 ships).

**Result in one line:** deletion teaches (locked teach .971, 8 draws), but the shuffled-label control teaches almost as much (.927), so rule 6 fails. Exam retention after feedback is .9851, with Alarm at .908. See [`REPORT.md`](REPORT.md).

| file | what |
|---|---|
| `deletion.py` | mask learning (Wanda init, annealed sigmoid, memory-light custom autograd), apply/undo, `deletions.bin` format |
| `feedback.py`, `answer.py` | the two shipped commands (same user/harness modes as L1; reuse L1's `bonsai_loader.py`, `common.py`, `feedback_calib.jsonl`, `skill_task.json`) |
| `run_lane.py` | dev selection (ann_pool/sel only) and the locked protocol (L1 draws, controls, exam after feedback) |
| `modal_l15.py` | Modal deploy + spawn (`lane`, `e2e`); results to volume `tenpct-l15` and RESULT_JSON |
| `test_deletion.py` | CPU unit tests (gradient check, save/load/apply/undo exactness, learn leaves weights untouched) |
| `summ.py` | prints a results JSON as a table |
| `results/` | smoke, dev selection (c0-c5), locked (`locked_c4`), end to end (`e2e_c4`) JSON and logs |

```
python feedback.py --artifact <DIR> --task skill_task.json --ann ann.jsonl      # writes deletions.bin (defaults = dev-selected c4: last 6 layers, 200 steps, beta 4)
python answer.py   --artifact <DIR> --task skill_task.json "how long do I boil an egg"
python -m pytest v2/lanes/L15_delete/test_deletion.py -q
modal deploy v2/lanes/L15_delete/modal_l15.py
python v2/lanes/L15_delete/modal_l15.py lane locked_c4 locked '{"steps":200,"beta":4.0,"cal_bs":16,"layers":6}' 8
python v2/lanes/L15_delete/modal_l15.py e2e e2e_c4 '{"steps":200,"beta":4.0,"cal_bs":16,"layers":6}'
```
`<DIR>` = a Bonsai-8B folder (the L1 pack, or the released `prism-ml/Bonsai-8B-unpacked` checkpoint) with the L1 helper files copied in. Needs one CUDA GPU of about 40 GB (the mask logits and Adam state for 6 layers add about 4.5 GB, or about 29 GB for all 36 layers).
