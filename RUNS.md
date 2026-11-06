# RUNS (detached relaunch 2026-11-06, after laptop crash)
trackQ missing models relaunched with modal run --detach: vitl_c100 in_vitl in_deitb in_cnxb in_r50 in_r101 (specs trackQ/out/*_2.spec.json, logs trackQ/logs/*_3.log)
trackS: sweep_bases.py --detach (trackS/run2.log)

## 01:30-01:40 EST relaunches (all `modal run --detach`)
- trackD: c3,c4,c5 -> o3,o4,o5 (trackD/m{3,4,5}_d.log); msweep.py go() prints RESULT_JSON
- trackE: sweep.py (fixed missing `mk`, the real cause of the empty sweep1.json) -> trackE/sweep3.log, prints RESULT_JSON
- trackR: run.py c3 -> r3 (trackR/r3_d.log); prints RESULT_JSON
- trackS: sweep_bases.py (trackS/run2.log)
- trackQ: 6 missing models (trackQ/logs/*_3.log); trackQ/salvage.py merges partial cfgs from all logs
- CAVEAT: Modal says detached local entrypoint keeps only the last triggered function alive after the client dies; .map() fan-outs (D,E,R,S) may lose results if the Mac dies. Recover from `modal app logs` RESULT_JSON lines.
- Deadline: 7am 2026-11-07. Fallback if Modal limits hit: AWS.
- 02:25 stopped off-goal trackS-bases, trackE (BERT/small). Q,D finished. Kept trackR (Qwen 14B). Logs in modal-logs/final.

- v2/base: frozen Qwen2.5-14B MMLU splits + LOCK.json + v2/TASK.md (no remote run; base_acc 0.8007 reused from trackR r3_d.log)
- v2/feedback: fb.py Qwen2.5-0.5B dev run, app ap-GCpvnS4S856sWJvhetuJbl (modal run --detach, 01:58)
- v2/feedback MMLU stand-in (Qwen2.5-0.5B) fb.py: ap-RfMCr7dRtZEyaMdesBgDJJ (earlier SST-2 app ap-GCpv... stopped, superseded)
- 02:05 harvested trackR r3.json (14B). v2 agents running: engine ap-ur2s..., exam-stand-in, feedback.

## v2/engine (workstream 2, 14B base) 2026-11-06
- smoke 0.5B (v2/engine/smoke_out.json): engine works.
- launched `modal run --detach v2/engine/sweep.py --cfgs cfg_tables.json` (4 table shards, per-module proxy err/entropy vs L) -> v2/engine/log_tables.txt
- launched cfg_A.json (14B uniform L in {3,4,5,6,8}, rot vs norot, g256, drop6) -> v2/engine/log_A.txt
