# L2_moe STATUS
- 2026-11-07 ~01Z: using tenpct-g8 (was idle, 0 MiB). g3/g4/g5 busy with others (not touched). Work dir on g8: /opt/dlami/nvme/work/l2.
- Base probes of big models on Modal: sub-agent, results in base_probe/.
- g8: Qwen3-30B-A3B saliency + REAP-vs-freq sanity (work in /opt/dlami/nvme/work/l2).
- g9 (4xL40S, ap-south-1a): Qwen3-235B-A22B-Instruct-2507 bf16 download to /opt/dlami/nvme/work/l2/hf (started ~01:40Z). g10, g11 unused so far.
- 01:50Z: 235B saliency rerun (g9 GPU0, ~19 min) + quant-only/base teacher-forced run (g9 GPU1). Then pruned configs on GPUs 0-3. g8: real generation check of 30B (res30). g11 idle (tell coordinator).
- 02:55Z: g9 GPUs 0,1,3 running label-task teacher-forced (lab235_g*.json, ends ~03:40Z), GPU2 extra tool-call proxy configs (tf235_g2b). g8: GPTQ vs RTN on 30B (tf30g). After labels: 235B real generation (gen_pruned.py) uses all 4 GPUs of g9. g11 idle (coordinator may terminate).
- Code in code/, raw results in res/. rsync of box files into res/ done by hand every ~15 min.
- 04:50Z FINAL: all jobs finished/killed. g8, g9 idle (no GPU procs); g11 never used. Results synced in res/, logs/, REPORT.md written. Safe to terminate g8/g9/g11.
