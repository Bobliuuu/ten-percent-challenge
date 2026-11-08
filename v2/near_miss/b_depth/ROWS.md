# b_depth near-miss rows
No experiment completed. Harness (core.py/gen.py/run_modal.py, jobs_7b.json) written and launched on Qwen2.5-7B-Instruct, but the Modal app produced no result within ~25 min (still in the sensitivity stage, no volume output) and was stopped on coordinator instruction. Spend < $1.5. Untested harness: may contain bugs. No retention numbers exist.
| method | base | size % | retention | min-cell | feedback | rule gaps | weakest | notes |
|---|---|---|---|---|---|---|---|---|
| (none) | Qwen2.5-7B-Instruct | n/a | n/a | n/a | n/a | all untested | n/a | 7B budget 9.9% = ~1.5 GB; est. needs ~2 bit + ~20% layers dropped, or ~3 bit + ~45% dropped |
