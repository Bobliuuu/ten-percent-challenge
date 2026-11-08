# MoE combinations (lane L12)

Simulated (fake-quant) engine, real greedy generation on the SEL set (n=963; retention = share of base-correct items kept; 95% lower bound in brackets). Size = analytic bytes / fp16. Full table: `python3 v2/lanes/L12/agg.py` (raw JSON in `v2/lanes/L12/results/vast{A,B,C}`). Code: `v2/lanes/L12/l12_core.py`.

Models: Qwen3-30B-A3B-Instruct-2507 (128 experts, base .925) and Qwen3.5-35B-A3B (256 experts + shared expert, hybrid linear attention, base .944).

## Headline
- **Best at <=10%: Qwen3.5-35B-A3B, REAP/mass-style pruning to 25% of experts + RTN 4-bit everything (experts, attention, embeddings, head, shared expert): 7.61% of fp16, SEL retention .982 [.972], min cell .957, REP .980.** No calibration-heavy quantiser needed.
- Qwen3-30B-A3B cannot hold more than ~.96 at 10% (GPTQ2 on 70% of experts: 10.28%, .958; GPTQ3 on 45%: 9.80%, .962, min cell .884).
- A single task-specific pruning score does most of the work; merging, shared-subspace SVD, adaptive per-layer counts add nothing over it.

## Ablations (SEL retention)
| Axis | Qwen3-30B | Qwen3.5-35B |
|---|---|---|
| Prune score @k40 (bf16 experts) | mass .991, reap .991, reapsum .990, kl-LOO .990, freq .962, gatefree .939, random .024 | freq .992, mass .994, reap .981, gatefree .974, random .572 |
| Prune score @k25 | mass .971 (min cell .61), reap .962 (min .83), reapsum .975, kl .964, freq .891, gatefree .52, random 0 | mass .991, reap .976 |
| Layer-adaptive counts @k30 | uniform .978, kl0.5 .984, cover .976, global .964, kl1.0 .557 | uniform .988, global .979 |
| Merge vs prune | align-merge k40 .966 (no-LSE .985), avg .957, k25 collapse; prune .991 | align-merge .959 (k25) / .976 (k40) vs prune .991 / .994 |
| Top-k at inference | top6 .988, top4 .966, top3 .937, top2 .05 | top6 .992, top4 .978, top2 .383 |
| Non-expert quant only | q4 .973, q3 .951 | q4 .985 |
| RTN all experts | 4b .961, 3b .974, 2b .939 | 4b .987, 3b .991 |
| Shared-subspace SVD coding | 0.000 at every rank (256-1024) | 0.000 |
| GPTQ+Hadamard on REAP-pruned | 2b/70% .958 @10.3%; 3b/45% .962 @9.8%; 4b/35% .962 @10.0% | **all 0.000: engine bug/incompat on fused Qwen3.5 experts, not a quality result (RTN on the same cfg works)** |
| RTN on pruned | 3b/k40 .961 @8.8%; 2b/k50 .921 @7.7% | 4b/mass k25 **.982 @7.61%** |

Takeaways: (1) with a task-specific score, pruning is nearly free; random is catastrophic; frequency is a bad proxy on the coarse model. (2) Fine-grained Qwen3.5 tolerates both pruning and low-bit RTN far better than Qwen3-30B. (3) Top-k 8 to 6 is free on Qwen3.5. (4) Data-free shared-subspace SVD of experts does not work (adapter-like, flagged). (5) Neuron-aligned merging is worse than pruning.

## Caveats
- Numbers are simulated quantisation (fake-quant), analytic bytes; no packed exporter measured. REP/SEL are L12's own sets (0 overlap with the locked exam); locked-exam score below.
- Qwen3.5 GPTQ configs returned 0.000 for every cell; treat as a defect in l12_core's GPTQ path for Qwen3.5 fused-expert layout, unresolved.
- Min-cell metrics are small-n tails; the pass rules (95% Wilson per kind) are stricter than these retentions imply.

## Locked shared exam (SKILL7, run once, best config Q35_B3_rtn4_mass_k25, 7.61% of fp16)
Real greedy generation, `v2/exam/loader.py` (manifest verified), n=2655: base acc .9676, retention **.9917** (Wilson lo .9875); gate overall >= .98 PASS, gate per-kind >= .95 PASS (min kind .972). Outputs: `v2/lanes/L12/results/final/`.  Bytes are analytic and the model is fake-quant, not a packed export.

Feedback on this model (25 real annotated mistakes, closed-form down_proj edit on the last 4 layers, 53 experts edited): **teach accuracy 0.0 for real, shuffled and empty controls** (the new label is never produced), exam retention .9902 real / .9906 shuffled / .9917 empty, neighbours .963 / .969 / .977. So the closed-form feedback edit (built for Qwen3-30B) does not inject the label on Qwen3.5-35B-A3B (R5/teach-rate rule NOT met here); it is harmless but ineffective. Compression-only result stands. All boxes destroyed.
