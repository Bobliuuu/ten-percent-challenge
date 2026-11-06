# Agent rules for this repo (read first)

1. NEVER plain `modal run`. Always `modal run --detach` (or `modal deploy`). Laptop crashes kill non-detached apps ("local client disconnected"). This cost us ~40 killed apps on 2026-11-06.
2. Results must not depend on the local client: write to a Modal Volume / print JSON to app logs. Do not rely on `local_entrypoint` writing files.
3. Keep EVERYTHING in this repo: code, specs, results JSON, run logs, Modal logs (`modal app logs <id>`), notes.
4. Commit and push after EVERY result. Log each launched run (app id, command, time) in `RUNS.md`.
5. Pull remote logs/results into the repo before stopping or terminating any box.
6. Pass these rules to every subagent.
7. SOURCE OF TRUTH for the goal: GOAL_README.md (github.com/DeusExperiments/ten-percent-challenge). Base must be >0.5B params; size is 10% of fp16; commercial non-narrow task; locked exam; feedback with few annotations. Re-read it before planning.
