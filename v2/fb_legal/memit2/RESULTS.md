# memit2: non-head closed-form feedback edit (MEMIT-style, MLP down_proj, 5 consecutive layers)

Qwen3-8B, bf16. Edit = dW_l = R K^T (lam C + K K^T)^-1 on `mlp.down_proj` (4096x12288) of layers 26-30, 0 added bytes, no weight gradients,
lm_head / embeddings untouched. Code: core.py (+ memit2.py launcher). Data, draws, hash locks, controls = fbcmp harness (same conds()).
Keys = down_proj input at the last prompt token and each teacher-forced label position; C = E[k k^T] (fp32, per-token mean, 114k tokens of the task's
unlabeled calib prompts, ridge 1e-3*mean diag); residual spread 1/(L_last-l+1) with keys recomputed after each layer; lam=15000.
Extras that made it work: (a) protection keys: 300 unlabeled calib prompts with the model's OWN greedy continuation, residual 0, weight 10;
(b) global edit gain 0.5 (chosen on dev draws 0-2, then all final numbers on fresh draws 100-109). Target v* (tgt=adam): 30-step Adam on a free
activation vector at the last edited layer (no weight training). Closed-form logit-lens target (tgt=closed) FAILED (huge delta norm, neigh 0.0 at L8-12): not defensible/usable.

## Final (fresh draws, n=12, TF = teacher-forced exact match of all label tokens incl. EOS)
| task | teach real | shuffle | wrong | ood | neigh kept | mnH (fbcmp) |
|---|---|---|---|---|---|---|
| TOX selfharm, 10 draws | 0.789 | 0.045 | 0.000 (mixed_wrong 0.43) | 0.741 | 0.993 | .557 / .018 / kept .9992 |
| routing recipe+music, 5 draws | 0.819 | 0.127 | - | 0.846 | 0.999 | .704 / .116 / kept .9992 |
| TOX, new name "Zyntrel" (not in prompt), 2 draws | 0.768 | - | - | 0.706 | 0.986 | - |
Base teach = 0.0 (TOX), 0.037 (route). Empty edit = base.
Decisive checks (TOX, draws 100-102): edited weights saved to safetensors, model restored, weights RELOADED from disk, REAL greedy generate:
teach gen-acc 0.821/0.750/0.839 (TF 0.804/0.732/0.839, agree 98-100%); neighbour gen-acc 0.993/1.000/0.987.
4-bit RTN (g128) of the 5 edited matrices: teach 0.810, neigh 0.980 (4-bit of the unedited matrices alone: 0.996).
Unit (n=1, L26-30, gain 1): site flips 100%, neigh 0.994 (results/b1.json). 

## What went wrong in the first attempt / what mattered (sweeps in results/b1..b5b.json)
- layers 4-12: neighbours 0.83-0.96 even at n=1; 14-20: 0.95-0.97; 26-30 best; 31-35 breaks (delta norm 600).
- C regularisation lam=15000 (paper value) is right order; lam 150000 does not flip, lam<=3000 no better.
- continuation positions (self,h,arm,EOS) are harmless (neigh 1.000); all neighbour damage comes from the decision-position edit.
- protection keys: weight 1 -> neigh .967, 10 -> .976, 30/100 -> .973 (saturates; ~2% of neighbours are near-boundary and flip under any perturbation; 4-bit rounding alone flips 0.4%).
- the last 1% was bought with gain: gain 1.0 teach .89/neigh .975; 0.75 .875/.986; 0.5 .80/.990. bf16 rounding is NOT the cause (fp32 path identical).
## Caveats (honest)
- gain, protection weight, layer set are tuned hyperparameters (on draws 0-2 incl. neighbour retention); final draws are fresh (100+).
- shuffle control still teaches 4.5% (TOX) / 12.7% (route), comparable to mnH route; shuffled/wrong edits wreck neighbours (shuffle neigh ~0.99 at gain .5, was 0.66-0.78 at gain 1).
- Covariance + protection keys use unlabeled task-format calibration prompts (same source as the fbcmp calibration set), not generic wikitext (cov_src=wiki implemented, not run).
- 4-bit neighbour retention .980 vs .996 floor: just under the .99 bar.
- ~4 GPU-hours total on Modal A100-80GB (hot-reload server job to avoid re-queueing).
