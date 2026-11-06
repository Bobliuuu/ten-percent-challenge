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
- relaunch (run.remote): ap-MFVbuvP2PA7PZBG1ZF36dh
- v2/alt: MDC-6 multi-domain classification hedge splits built+locked (modal --detach ap-KxQu03girs6DrjhkPdspoF, 02:07)
v2/entropy run.py L{4,6,8,16}x rot{0,1} launched Fri  6 Nov 2026 02:10:04 EST log v2/entropy/launch.log

## v2/big (workstream: Qwen2.5-32B/72B) Fri  6 Nov 2026 02:10:21 EST
- big.py = engine sweep with device_map multi-GPU. cfg_1.json: 72B/32B base(vocab trim only)+u3 g128, 32B u4. app ap-w8qFX9hULeT0K9n9ntxLLD, log v2/big/log_1.txt

## v2/moe (Qwen3-30B-A3B-Instruct-2507 prune+quant sweep) 2026-11-06
- 02:09 modal run --detach v2/moe/moe_sweep.py -> ap-Nd5vsC3yUJbqKwqz0sBpza (H100:2/A100-80GB:2; results /vol/results.json in volume moe-10pct-vol). Earlier apps ap-1h8zo.., ap-r9nr.. stopped (spawn never scheduled).
- v2/alt base_eval.py (4 models MDC-6 dev) modal run --detach ap-maRPOeFFWWZpFZ3jKp8KQo 02:24
- v2/prune smoke: ap-Ti6TO54xMEQVtjDHUHiBUe (02:10, modal run --detach prune.py --cfgs smoke.json), results volume tenpct-prune
- v2/alt probe.py cfg_p1.json (8 cfgs 7B/3B/1.5B) --detach ap-85gE4bWbBMMQHJsRo2gsTP 02:29
- RESTART: Modal multi-GPU never scheduled; AWS p4d no capacity us-east-1. big.py now streams layers CPU->1xA100-80GB. app ap-SjLIvn2lK3iXZYsXaxkizC (cfg_1.json, same configs)
- cfg_B.json (14B layer-drop x uniform L, g256) -> v2/engine/log_B.txt
- v2/moe sweep1 (RTN prune+quant) run on AWS g7e.2xlarge i-0198fb51c29cfc844 us-east-2 (Modal blocked: 10-GPU cap); results v2/moe/results/sweep1_rtn.*
- relaunch (hook return bug fixed): ap-L6BWuA91mY58R5HPIZxQyo
- v2/lowbit eval_base.py (6 native-ternary models, MMLU+MDC dev) --detach ap-ClEyfeb37OiaBTZN0VR5EU 02:44; results Volume lowbit-data:/base
- v2/prune p1+p2 (54 cfgs, removal-only + dense GPTQ) ap-pJN1EDGqCWbeSc6GNLkwPe 02:50 detached; smoke ok (drop4+fullpatch acc .806/.8125 on n=160); results in vol tenpct-prune + v2/prune/results/
- v2/commercial build_data.py ap-SPmnKvzT9zzcRJjlIrrKSF (02:45, done, locked); base_eval.py 3 Qwen2.5-Instruct bases ap-9kKyWhppjlUwKM3R8CjGn5 (02:49) --detach
- relaunch w/ GPU fallback list: ap-kBzNrHTOH6QarbefrjofPT
- v2/alt probe relaunch ap-mPCfr6Mzw2aaNluQlRxUZw (earlier ap-85gE... queued 20min w/o GPU, left running)
- v2/feedback2 feat.py (1.5B feature extraction for no-RAG feedback study) modal run --detach ap-SJ6DvDkqMKmI47PMMZ3bPL 02:43; vol tenpct-fb2:/f2
- cfg_C.json (14B sensitivity-allocated per-module L at body bpw targets 1.5..2.4, g128) -> v2/engine/log_C.txt; tables in v2/engine/tables

## v2/vq (workstream: sub-2-bit VQ on Qwen2.5-14B) 2026-11-06
- Modal app ap-1oMnzKOD2q3XLQnR7fkqot (`modal run --detach v2/vq/vq.py --cfgs cfg_modal_r1.json`): m_k14, m_mlp12_attn16, m_mlp13_attn14; results -> Modal Volume `vq-results` /<name>.json + RESULT_JSON in logs.
- Lambda A100 (ssh lambda-final, ~/vq): `vq_local.py cfg_14b_r1.json` (b14_k13, b14_k16, b14_k12) -> ~/vq/res/*.json, log ~/vq/r1.log. (Modal was GPU-queued for >2h; AWS p4d/p4de had no capacity.)

## v2/altpush (MDC-6 max compression) 2026-11-06
- AWS g6e.2xlarge (L40S) i-076f90d86ce833547 us-east-2, ip REDACTED-IP, key swarm-gpu, name altpush-l40s; runs via v2/altpush/run_local.py (modal stub). TERMINATE at end. (Modal probe ap-85gE4... had 0 tasks scheduled for 15+ min, so used AWS.)
- a3b: Qwen2.5-3B uniform L=2/3/4 g128 -> results/out_a3b_*.jsonl. L3: ratio .0966 acc .7685 (base .834) agree .845; L4: ratio .1176 acc .8188 agree .917
- v2/compress_B sweep_B.py cfg1.json (47 cfgs 1.5B/3B/7B drop x L, task B dev) --detach ap-UsSTn86RRIJ8e3GahrSVbH 02:53; results vol hf-cache:/compressB/*.json

## v2/compress_C (task C tool use) Fri  6 Nov 2026 03:03:13 EST
- smoke cs.py ap-ATMlSLGcerWPVo3x5HKZmM; sweep1.json (24 cfgs: 3B/7B/14B base+uniform L, 14B alloc) ap-2rnXuDRmAVFJPg3VT2bxRn; results vol tenpct-C-res, logs v2/compress_C/
- v2/commercial base_eval relaunch w/ GPU fallback list: ap-B0ywSZ8eQoTmust48jvK15 (03:03); ap-9kKy.. stopped (never scheduled)
- v2/prune p3 (ffn recon variants skip0/lam, late-block drops) ap-L0HhmWzmbdSMtbOOjfoKiP 03:20 --detach
- sweep1 relaunched after remap fix: ap-aQKtFFclKXKorbsukxGXxY
- v2/compress_A sweep_A.py (task A PII, engine adapted): cfg_1.json 15 cfgs (fp baselines + uniform GPTQ L=3/4/6 on 1.5B/3B/7B) ap-ADnjPMffxvs1SzL3H1jINo 03:06 --detach; GPU list fallback (L40S/A100/A10G); Modal GPU-starved by other agents, AWS p4d no capacity.
- v2/feedback2 ana.py local numpy on features (results_q15_L8.json); q05 feat app ap-02rVhZxnSMRFvbZgat0HnN stopped (no A10G capacity)
- b3b: 3B ECSQ (c*group-rms step, deadzone dz, 8-bit scales) c1.2/dz.15: ratio .1005 acc .8039 agree .8875, acc_b(bias) .8547 vs base_b .868; c1.4/dz.1: ratio .0926 acc .781; results/out_b3b_*. c: 7B c1.2dz.15, 7B c1.0dz.25, 3B c1.0dz.25 running (out_c_*)
- v2/lowbit build_artifact.py: bitnet2b ap-l8gkEU3Yzje0hEn7Skk78l (03:20), falcon-e 3B+1B ap-j6uIBTKSolIvazueZFm1TG (03:35); artifacts in Volume lowbit-data:/art/<tag>
- v2/alt: Modal GPU queue stuck ~40min (backlog, 0 runners) for probe fn; deployed app tenpct-alt-probe3 (spawn_probe.py cfg_p1.json), results via 'modal app logs tenpct-alt-probe3'
- big32.py (32B only, mem 96GB, broader GPU list) ap-AFSM7dGkglhpKVb9v1tT7M; 72B app ap-kBzN... still queued
- compress_B cfg1 stopped early (base zero-shot at chance on B dev); results v2/compress_B/res1
- 03:35 stopped MMLU-family apps (v2big, moe, prune, engine, vq) to free capped Modal GPUs for lowbit + altpush; logs in modal-logs/final
- v2/commercial base_eval DEPLOYED app tenpct-commercial-base (modal deploy + spawn_eval.py, 03:25) after run --detach spawn jobs sat at 0 tasks; fc ids 01M6QZDXCK.., 01M6QZEC3S.., 01M6QZETVG..
- v2/moe finished: GPTQ sweep on g7e box, box terminated; see v2/moe/RESULTS.md
