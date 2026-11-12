# DECISION_FINAL (running log, decisions delegated to the assistant by the user, 2026-11-12)

Rule for updating the filed entry (issue #2): a new carrier must strictly beat the L1 scale carrier on rules 5 and 6 with no new failure (shuffled and wrong near 0, empty unchanged, every kind >= 95% after feedback, exam retention >= 98%). Choosing a config on the locked data is not allowed, so a failed locked run cannot be retuned and rerun.

## L1 deletion B (gradient-free, zeroes existing down_proj group scales), checked 14:17 UTC
- Real teach .942 (scale carrier .899-.909), shuffled 0.000, wrong 0.000, empty byte-identical: better on rule 6.
- Neighbours kept .9595 (worst .940) against .9936 for the scale carrier; Lists .944-.956 after feedback in 2 of 3 exam draws: fails the 95% per-kind bar as written.
- Verdict: does NOT strictly beat the filed carrier, introduces a per-kind failure. Not filed. Kept as a documented near miss (`v2/lanes/L1/deletion_b/REPORT.md`). A retune with higher nu would need a fresh locked exam, which does not exist before the deadline.
- Filed entry stays L1 scale carrier unchanged unless L9b deletion (C) clears the bar.

## L9b deletion C (gradient-learned mask on shared-expert down_proj), checked after its final report
- One-shot locked run 1: real teach .954, shuffled .796 (range .61-.93), wrong 0, exam retention 98.04-98.46%, worst cell massive/Music 88.5%. Rule 6 FAILS (shuffled leaks, same as PR #1 L15: shuffled .927).
- Runs 2 and 3 (higher KL weight) were chosen after seeing run 1 on the locked exam: not eligible, shuffled still .77-.81.
- Verdict: no deletion carrier (L15, B, C) clears the bar. FILED ENTRY STAYS UNCHANGED (L1 scale carrier, issue #2 as posted 2026-11-11). Documented as negative results in the repo only.

## Final strict check, 2026-11-12 15:47 UTC (readings counted as pass only if the rule's words are met)
| candidate | R1 | R2 | R3 | R4 | R5 | R6 | R7 | R8 | R9 | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| L1 scale carrier (FILED) | PASS | gray (100% vs release, 88.9% vs ancestor) | PASS | gray (closed-form edit, maintainer rejected a similar edit on #4) | PASS | PASS on Recipe (shuffled .001-.005); shuffled arm exam retention .9783 | PASS | PASS | gray | no clean pass |
| L1 deletion B | PASS | PASS overall, FAIL Lists .944-.956 | PASS | gray | WEAK, neighbours .9595 | PASS (0.000) | PASS | PASS | gray | does not beat L1 |
| L9b deletion C (run 1) | PASS | WEAK (98.04-98.46%, Music cell 88.5%) | PASS | gray | PASS | FAIL (.796) | PASS | WEAK | gray | fail |
| L15 deletion (PR #1) | PASS | gray, Alarm .908 | PASS | gray | WEAK | FAIL (.927) | PASS | PASS | gray | fail |

Decision: no new variant strictly beats the filed L1 scale carrier without a new failure. The filing on issue #2 is NOT edited. No candidate is a clean pass; the filing already says so (it claims no clean pass on rules 2, 4, 9). Deletion results stay in the repo as documented negative results only; nothing about them goes into the public filing text.
