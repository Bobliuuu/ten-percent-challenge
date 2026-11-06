# Agent rules for this repo (read first)

1. NEVER plain `modal run`. Always `modal run --detach` (or `modal deploy`). Laptop crashes kill non-detached apps ("local client disconnected"). This cost us ~40 killed apps on 2026-11-06.
2. Results must not depend on the local client: write to a Modal Volume / print JSON to app logs. Do not rely on `local_entrypoint` writing files.
3. Keep EVERYTHING in this repo: code, specs, results JSON, run logs, Modal logs (`modal app logs <id>`), notes.
4. Commit and push after EVERY result. Log each launched run (app id, command, time) in `RUNS.md`.
5. Pull remote logs/results into the repo before stopping or terminating any box.
6. Pass these rules to every subagent.
7. SOURCE OF TRUTH for the goal: GOAL_README.md (github.com/DeusExperiments/ten-percent-challenge). Base must be >0.5B params; size is 10% of fp16; commercial non-narrow task; locked exam; feedback with few annotations. Re-read it before planning.
8. (Learned 2026-11-06) Plain `modal run --detach` entrypoints with .map/.remote can still LOSE jobs (v2/engine found this). For anything long: `modal deploy` the app and spawn jobs with `.spawn()` from a launcher (see v2/engine/launch.py), write results to a Modal Volume, and print RESULT_JSON.
9. GPU capacity: Modal workspace cap ~10 GPUs, no H100; AWS p4d/multi-GPU g6e had no capacity; a g7e.2xlarge in us-east-2 worked (profile `lab`). Terminate AWS boxes when done and pull results into the repo first.
10. Task results so far: MMLU (14B, MoE, VQ, entropy, prune) cannot reach 10% at base accuracy; AI-vs-human (task B) base is at chance. Live candidates: native ternary BitNet (v2/lowbit), Qwen2.5-3B on MDC-6 (v2/altpush).
