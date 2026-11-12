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
