# feedback_v3: closed-form in-place edits for the feedback rule (partial, honest)

Code: `fb3.py` (Modal job: data, edits, sweep, controls, W4 sim), `analyze.py` (uses `harness/scorecard.py` paired_boot; writes `table.md`, `table.json`), `sync.sh`. Raw per-item results: `results/*.json` (arrays of per-item 0/1). `results_stale/` = B runs with an invalid shuffled control (see below). Launch logs `launch*.log`.

## Setup actually run
- Qwen3 ladder submitted: 0.6B, 1.7B, 4B-Instruct-2507, 8B. Finished with final phase: 4B (A, A-late), 8B (A, C), 1.7B (A-late, C-late). 0.6B and 1.7B base-sweep runs were killed unfinished (workspace GPU cap + time). **Task B has no valid result** (first run's shuffled control was a no-op because all B annotations share one label; fixed in code, rerun never got a GPU). **Qwen3-8B on B and 0.6B anything: none.** The ladder has 2-3 points per cell and is only indicative.
- Proxy data is written by the base model itself (LLM-generated tickets / customer messages / PII templates), hash-split into annotation / dev / teach; neighbors = items of old classes that the base answers correctly (A: 5 old queues; C: std-PII texts). Task A has 10 new queues, 25 annotations (about 2 per label; 5 each would be 50 > cap). Task C neighbor sets are small for some sizes (1.7B C: none survived, neighbor = nan).
- Edit = no backward, no optimizer. Sites = teacher-forced positions where argmax != correct token; target delta from the unembedding direction with a forward-only scale search; MEMIT-style ridge least squares on `down_proj` (keys covariance from wikitext + chat text), or `o_proj`; `lm_head` row edits (mean = centred mean final hidden; lda = covariance-whitened). Bytes added = 0 in all cases (in-place).
- Config per method chosen on a separate dev set + 100 neighbor-dev items (sweep of 27 configs), then 3 annotation draws on the teach set (100 items, A; 51 for 8B C) and >=100 neighbors. Controls: shuffled labels (empty == base). Drift = token agreement with base on 50 generic prompts.
- G2 here = teach gain >= +15 pts with paired-bootstrap lo > 0, neighbors >= 98%, real-minus-shuffled >= +10 pts with lo > 0, 0 bytes. (Locked-exam retention not measured: no exam in this proxy.)

## Results (method x proxy task)  — see `table.md` for all rows
| model | task | method | teach gain (95% lo) | neighbors | real - shuffled | G2 |
|---|---|---|---|---|---|---|
| 4B | A new label | memit_down | +0.44 (+0.38) | 0.868 | +0.42 | FAIL (neighbors) |
| 4B | A | memit_o | +0.36 | 0.882 | +0.36 | FAIL (neighbors) |
| 4B | A | lmhead_mean | +0.17 | 0.990 | +0.01 (lo -0.03) | FAIL (control) |
| 4B | A | **lmhead_lda** | +0.40 (+0.34) | 0.992 | +0.24 | **PASS** |
| 4B (late-layer sweep) | A | memit_down | +0.41 | 0.953 | +0.35 | FAIL (neighbors) |
| 4B (late) | A | lmhead_lda | +0.25 (+0.20) | 0.995 | +0.19 | PASS |
| 8B | A | memit_down / memit_o | +0.03 / +0.15 | 0.912 / 0.912 | +0.03 / +0.14 | FAIL |
| 8B | A | lmhead_mean | +0.18 | 0.985 | +0.04 | FAIL (control) |
| 8B | A | **lmhead_lda** | +0.47 (+0.41) | 0.983 | +0.20 | **PASS** |
| 1.7B (late) | A | best (lmhead_lda) | +0.20 (+0.16) | 0.960 | +0.06 | FAIL |
| 8B | C new PII type | memit_down / memit_o | +0.06 / +0.25 | 0.957 / 0.928 | -0.01 / +0.25 | FAIL (neighbors) |
| 8B | C | lmhead_mean | +0.17 | 1.000 | +0.06 | FAIL (gain, control) |
| 8B | C | lmhead_lda | +0.33 (+0.26) | 0.957 | +0.12 | FAIL (neighbors) |
| 1.7B (late) | C | memit_down / memit_o | +0.56 / +0.55 | n/a (no neighbors) | +0.42 / +0.55 | undetermined |
| any | B tool pair | all | 1.7B first run: memit +0.83..0.87 teach, neighbors 0.99/0.82/0.99 (seed-dependent), but control invalid | | | not evaluated |

W4 (4-bit RTN of all layers, then memit_down edit, snapped to original grid / re-quantised): 4B A: gain fp-edit +0.37, snap-orig +0.30 (or +0.01 in the late-layer run), requant +0.35 (or +0.02); 8B C: +0.30 / +0.24 / +0.31. Snapping to the original grid can erase the edit (small deltas below one grid step); not reliable.

## Size ladder (teach gain, extrapolated, log10 params; 2-3 points, bootstrap over items only) — `table.md`
- A lmhead_lda: +0.20, +0.25, +0.47 for 1.7B/4B/8B (neighbors .96/.995/.983); predicted @14B +0.53 (+0.46,+0.61), @32B +0.67 (+0.56,+0.78). Gain increases with size; CI is item-bootstrap only (3 points), so treat as indicative.
- A memit_down/memit_o: no clean trend (+0.07/+0.41/+0.03), neighbors 0.91-0.95 at every size: fails neighbors at all sizes tested.
- C lmhead_lda: +0.18 (1.7B) -> +0.33 (8B); memit_*: gain high but neighbors <=0.96.

## Verdict
- Only `lmhead_lda` (covariance-whitened, class-conditional edit of the correct output token's lm_head row; closed form, 0 bytes, no backward) passes G2, on task A at 4B and 8B (3 annotation draws each). It fails at 1.7B (A gain +0.20 but control separation only +0.06) and fails neighbors on 8B C (0.957).
- MLP down_proj / o_proj MEMIT-style edits give large teach gains but break neighbors (0.85-0.95) at every size; the shuffled control separation is real, so the edit does learn, but it is not selective. Later layers (65-90% depth) help neighbors (4B A: 0.87 -> 0.95) but not enough. lm_head "mean" variant (issue #3 style) is too weak / not label-specific (shuffled control within noise).
- Not a pass for C, none for B; the single-token-first-position mechanism is natural for classification-like feedback (A) and brittle for long copy outputs (C).
- Caveats: LLM-generated proxy data (possible near-duplicates across splits), dev-selected config, only 3 annotation draws, no 0.6B, no valid B, ladder from 2-3 points, 4B uses Instruct-2507 while the others use Qwen3 base-chat, W4 simulated by RTN not the real codebook quantiser. Next: rerun B with fixed control, finish 0.6B/1.7B/8B for A,B,C, widen the lmhead_lda alpha/regularisation sweep (alpha grid edge at 0.4 on 4B), and try lmhead_lda + a late-layer down_proj blend.
