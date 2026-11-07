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
- v2/engine: cfg_B/cfg_C aborted (apps stopped on request); see v2/engine/RESULTS.md
- c: 7B c1.2dz.15 vocab16k->12000 ratio .0948 acc .8596 vs base .8721 agree .9457 (acc_b .8766 vs .889). vocab study (cfg_v, L=0): 3B vocab12k agree .977, 33k agree .9925 (ebits4), ebits3/2 hurt. Next: 7B vocab33k c1.15 (out_e_*), 3B ncal512 (out_d_0)
- sweep1 trimmed (GPU cap, 3 containers): sweep2.json relaunched (see modal app list tenpct-C newest)
- v2/lowbit build sweep ebits 5/6/8 bitnet2b: ap-EulMhMKtTwhhOxte5j0VDy
- v2/alt probe p1 (spawned on deployed tenpct-alt-probe3, ids in v2/alt/ids.txt, collect.py): 7B L3 g128 ratio 0.0919 acc .8547 vs base .8726; p2 mixed-bit queue ids_p2.txt
- v2/alt p3 (14B/32B on tenpct-alt-probe80, A100-80GB; ids_p3.txt)
- e_0: 7B vocab33k(dev-incl, optimistic) c1.15 dz.15 ncal128: ratio .1007 acc .8671 vs .8721 agree .9577. NOTE vocab was built incl DEV text; fixed (nodev) in push.py; honest vocab study out_v2: 3B 33k agree .9886, 24k .9881, 16k .9821. Added pack capture + pack_artifact.py + artifact_src/predict.py + verify_artifact.py
- v2/commercial C2 (hard distractors) build_c2.py + dev eval of 3 bases via spawn_eval.py C2 (redeployed tenpct-commercial-base) 03:52; first results log v2/commercial/base_results.log (app ap-bY3PXKNev6XDXLOu2UH4Ev)
- v2/commercial bino.py (Binoculars zero-shot probe, B dev) deployed tenpct-commercial-bino 03:53
- build bitnet2b V42000 e4
- sweep3 (32B base/L3/g256/drop4, 14B cal512) launched 03:59
- v2/compress_A AWS box g5.12xlarge eu-north-1 i-09f24d8981aa83348 (Modal starved): runner_aws.py cfg_2.json (7B/3B/1.5B GPTQ sweep) 09:3x UTC; TERMINATE at end
- lowbit dev_harness_test v32k_e4 & v42k_e4 (above)
- sweep4 (module-type mixed L patterns 14B/7B) launched
- v2/lowbit_fb feat extraction (frozen features of decoded BitNet-2B artifact; MDC-6 calib/dev/pool + hold prompts): deployed app ap-J6HRpwu4XyNcBx8c7vSIWC (tenpct-lowbit-fb-feat), spawned run(calib|dev|pool|hold) 04:2x EST 2026-11-06; out Volume lowbit-data:/fb/*.npz
- v2/alt: results in v2/alt/results_probe.json (7B best: attn L4 + rest L3: ratio .0944 acc .8651 vs base .8726 agree .94). 14B/32B queued on tenpct-alt-probe80 (A100-80GB scarce), ids_p3.txt
- g: 7B vocab24k(nodev) c1.15 dz.15 ncal384, PACKED ARTIFACT on box ~/art7b (local copy ~/ten-percent-artifacts/art7b_c115): size_check PASS 1,482,976,475 B = 9.736%; reloaded artifact logits identical to run; dev acc .8731 vs base .8721 (raw), .8820 vs .8890 (bias). h: c1.09 dz.15 and dz.2 (out_h_0) running. See v2/altpush/ARTIFACT.md
- v2/commercial C2 + bino results pulled: v2/commercial/{base_results,bino_results}.log, TASK_A/B/C.md, RECOMMENDATION.md
- v2/lowbit_fb 7B: Modal app tenpct-lowbit-fb-feat7b (deployed; GPU starved) AND AWS g6e.2xlarge i-0aad61d1dd9c5c95e us-east-2 name lowbit-fb-7b (key swarm-gpu) launched 2026-11-06 ~05:00 EST for 7B feature extraction + ingest/stage-2 verification. TERMINATE at end. Artifact copy: ~/ten-percent-artifacts/art7b_c115_fb, Modal volume lowbit-data:/art7b
- h: 7B c1.09 dz.15 (not packed): acc .8611 acc_b .8785 (noise vs c1.15 .8731/.882). Artifact uploaded to Modal volume altpush-artifacts:/art7b_c115; MANIFEST.json written; fast-decode test: no load gain.
- v2/alt: 14B (A100-40GB, ncal64) L3 ratio .0953 acc .8457 vs base .8721; attn4 ratio .100 acc .8522; no better than 7B. 80GB queue (32B) cancelled (no capacity). results_14B.json
- v2/compress_C push_c2.py (altpush recipe on C2, 7B) c2_sweep1.json 6 cfgs (ECSQ c in 1.1-1.3), pack to vol tenpct-C-res; 04:52
- v2/compress_A wave1 results (7B u3 g128 ratio .0919 acc .621 vs base .657; calibrated bias .667) in v2/compress_A/results; wave3 mixed-bit cfg_3.json running on AWS eu-north-1 i-09f24d8981aa83348
- v2/compress_A 14B sweep sweep_A14.py cfg_14.json (6 cfgs) ap-MFklLcNtzsaBSsWCjaHyZr 10:3x --detach (local process must stay alive for spawn stagger)
- C2 7B sweep1 results v2/compress_C/c2res; finish (pack+verify) launched for c114_dz15 (ap-moylEuKrVoyDi2KkXPxQZm) and c120_dz15
- v2/compress_A wave4 (7B first/last-layer high-bit variants, cfg_4.json 8 cfgs) Modal ap-Au2nHMXAkz5bzLJB5scUIG 11:0x --detach
- 2026-11-06 v2/c2_fb feat extraction C2 dev/calib/pool: modal app tenpct-c2fb-feat (deploy+spawn: dev fc-01M6R5VXTVNXP34G972FMVF2CJ, calib ...YH9Q123HAM79QJS1, pool ...0XDZK2G4VCE7E7QCEC) -> Volume tenpct-c2fb:/f
- v2/compress_A 14B wave5 cfg_14b.json (8 cfgs) ap-YFsncRqBuJd1IVvyhbtjJy ~11:20; FINDING: 14B u3 g128 ratio .0953 acc .752 vs base .748 (dev)

- v2/lowbit_fb DONE: 7B + BitNet feedback verified on box i-0aad61d1dd9c5c95e (terminated). See v2/lowbit_fb/RESULTS.md
- v2/compress_A DONE: AWS box terminated; results v2/compress_A/results (modal_results.jsonl, wave14_a.json, 7B jsons). Best: 14B u3 g256 ends4 ratio .0935 acc .747 vs base .747; 14B u3 g128 .0953 acc .752
- 2026-11-06 v2/c2_fb verify_final: modal app tenpct-c2fb-final (deployed, spawn fc-01M6RAZX9J4Q4NYMT1V1PWYTSX) -> Volume tenpct-c2fb:/final/result.json; final artifact tenpct-c2fb:/art_c2fb
- 2026-11-06 v2/c2_fb: canonical artifact = ~/ten-percent-artifacts/artC2_fb_fixed + tenpct-c2fb:/art_c2fb (good body.bin 0304a803); re-verified, harness tree 6290414d
- 2026-11-06 07:28 EST REAL C2 stage 1 PASS: base .7675 cand .799 (+3.15pt, CI [+1.60,+4.70]), NI p=.001, 9.404%, tree 6290414d; harness lock 0d963d1
- 2026-11-06 07:40 EST REAL C2 stage 2 PASS_BOTH_STAGES: cand .802 vs base .7675 (+3.45pt, CI [+1.85,+5.05]), NI p=.001, same tree 6290414d, memory 852B sha 3cc96ec2
- 2026-11-06 07:50 EST SUBMITTED proof issue https://github.com/DeusExperiments/ten-percent-challenge/issues/2 (C2, 9.404%, stage1 .799 vs .7675, stage2 .802; artifact download link still to add)
- 2026-11-06T13:28:22Z v2/demo/demo_run.py: app ap-8VGp2T5nOkAe8cC2AjoEsw (modal run --detach demo_run.py), result Volume tenpct-demo:/demo_outputs.json
- 2026-11-07T02:36:20Z v2/research/L3_405B: apps l3-405b-dl (download, done), l3-405b (quant/evalrun), Volume tenpct-l3-405b
2026-11-07T02:45:56Z modal probe_base 8B,14B launched (tenpct-tox-probe)
2026-11-07T03:09:52Z modal fb_tox 1.7B,4B launched
- 2026-11-07T04:15:09Z L3_405B: spawned quant t096_c256 (fc-01M6T3F4V824676N65YSV963MP) in app l3-405b

## Vast (04:50Z 2026-11-07): no tenpct-* instances rented; create blocked by account instance cap (see v2/research/V_vast/INSTANCES.md)
- 2026-11-07 ~05:25Z D_scale: Modal app tenpct-dscale (deployed, fn run) call fc-01M6T6ZGZ3ASW8HK1HAX2A5MA5 = m72_gam1_lcm_nc (Qwen2.5-72B gamma1 + C_tasks label calib + nocall-heavy T1 calib, H200); launcher v2/research/D_scale/modal_launch.py mvars3.json; results -> Volume tenpct-dscale:/72B/m72_gam1_lcm_nc.json
- 2026-11-07 07:05Z vast smoke rental id 59535711 label tenpct-smoke (V100, $0.124/h) created+destroyed to confirm the account cap cleared (account had 0 instances)
- 2026-11-07T07:05Z vast id 59536016 label tenpct-smoke GPU RTX 2080 Ti 22GB $0.079/h (+5% guard) started
- 2026-11-07T07:19Z vast id 59537416 label tenpct-smoke GPU Tesla P40 24GB $0.107/h (+5% guard) started
- 2026-11-07T07:20Z vast id 59537416 tenpct-smoke destroyed after 0.01 h (~$0.00)
- 2026-11-07T07:20Z vast id 59537533 label tenpct-smoke GPU RTX 3080 20GB $0.135/h (+5% guard) started
- 2026-11-07T07:25Z vast id 59537930 label tenpct-smoke GPU Q RTX 6000 22GB $0.135/h (+5% guard) started
- 2026-11-07T07:26Z vast id 59537943 label tenpct-smoke GPU Tesla V100 32GB $0.201/h (+5% guard) started
- 2026-11-07T07:27Z vast id 59537930 tenpct-smoke destroyed after 0.02 h (~$0.00)
- 2026-11-07T08:17Z vast id 59542098 label tenpct-artifact_iso GPU RTX 3090 24GB $0.177/h (+5% guard) started
- 2026-11-07T08:17Z vast id 59542099 label tenpct-INJ_q4b GPU RTX 3090 24GB $0.166/h (+5% guard) started
- 2026-11-07T08:25Z vast id 59542882 label tenpct-artifact_iso GPU RTX 3090 24GB $0.179/h (+5% guard) started
- 2026-11-07T08:25Z vast id 59542886 label tenpct-INJ_q4b GPU RTX 3090 24GB $0.177/h (+5% guard) started
- 2026-11-07T08:34Z vast id 59543597 label tenpct-artifact_iso GPU RTX 3090 24GB $0.201/h (+5% guard) started
- 2026-11-07T08:36Z vast id 59542886 tenpct-INJ_q4b destroyed after 0.10 h (~$0.02)
- 2026-11-07T08:42Z vast id 59544206 label tenpct-artifact_iso GPU RTX 3090 24GB $0.201/h (+5% guard) started
- 2026-11-07T08:52Z vast id 59544206 tenpct-artifact_iso destroyed after 0.12 h (~$0.02)
- 2026-11-07 v2/harness/validity/general_remote.py: Modal app tenpct-general ap-FV8ChwJroAUj8HROcrJBlX call fc-01M6V228PZK0MSXZT6NNCSXEX8 (--detach), Qwen3-8B artifact vs base, result Volume tenpct-artifact:/q3_8b/general/general_result.json
- 2026-11-07 general check DONE (ap-va9imwk0raYVaRctLfJmu4): q3_8b artifact at chance on MMLU/ARC/HellaSwag/TQA, ppl 11.09->642.7; results v2/harness/validity/q3_8b_general_result.json
- 2026-11-07 v2/harness/validity/general_c2_remote.py: Modal app tenpct-general-c2 ap-uRyjoRqY1RCZIT13zjEH0y fc-01M6V3WK01BKQTS6PAZ8HCCR2R (--detach), filed C2 artifact general check, result Volume tenpct-c2fb:/general/general_c2_result.json
- 2026-11-07 general check filed C2 DONE (ap-YGbm6ItpQZ3PBpggJcY8f4): cand MMLU .467 vs base .693 etc; v2/harness/validity/c2_general_result.json
