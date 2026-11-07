# STATUS - v2/artifact (real TPQ1 artifact: writer + loader + CLIs)  [updated as work proceeds]

## 2026-11-07 00:0x local (Mac, CPU)
* DONE: format + codec + writer + streaming quantiser + loader + answer.py/feedback.py + verify.py + 9 unit tests (all green, tiny random Qwen3 + real tokenizer):
  bit-exactness loader == quantiser-side simulation (torch.equal on all 16 quantised tensors, multi-chunk, 4-process shared-memory decode), answer == sim-path greedy text,
  answer leaves tree sha unchanged, empty feedback tree-sha identical, real feedback = only model.tpq changed (lm_head rows), file set unchanged, size delta < 1e-4.
* Harness change: `model.tpq` added to checks.ALLOWED_PATTERNS (only change outside v2/artifact). Everything else in harness/v3 untouched.
* Modal: deployed app `tenpct-artifact` (modal_job.py, volume tenpct-artifact), spawn with `python launch.py TAG MODEL [stages] [levels]`. (`modal run --detach` + spawn from local_entrypoint LOST the job: app ended with 0 tasks; same lesson as AGENTS.md rule 8.)
