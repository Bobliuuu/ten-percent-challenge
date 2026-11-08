# Scout log (rules/competition)

## 2026-11-06 (scouted via gh api only, read-only)
- Repo DeusExperiments/ten-percent-challenge created 2026-11-05 17:02Z; 8 stars, 1 fork, 1 open issue, 0 PRs, 4 commits (all by tokenbender, README only). Latest push 2026-11-06 05:24Z.
- Issue #1 "few questions about the rules" by entrant #1 (2026-11-06 03:51Z); one reply by tokenbender 05:57Z (see RULES.md for verbatim). Issue was renamed once. No submission issues exist as of scout time.
- Reply implications: disk size vs fp16 is the metric; compression OK; NO RAG / context-space editing for feedback; kNN-memory question (asker Q3) not directly answered. Weight-editing question answered only as "compression ok, no retraining".
- README was materially tightened at 05:24Z: >0.5B params, commercial-class (no toy/narrow), fp16 baseline size, feedback = annotation/labelling with very few samples, prize 1000 USD, deadline 2026-11-12 17:05Z.
- Forks: a fork (no commits beyond upstream). Bobliuuu/ten-percent-challenge is ours.
- Other attempts found: isHeSatoshi/tokenbender (public, pushed 2026-11-06 04:19Z): "frozen-weight Banking77 subnetwork experiment". Base johnsonfzzz/Qwen2.5-1.5B-Banking77-Intent (1.54B params), selects (never changes) weights via mask search, vocab pruned to ~10k rows, kNN-style int8 feedback memory counted in the entry budget, bubblewrap sandbox. States "No parity result is claimed", locked exam unopened, no real mask search done yet, no qualifying A/B proof. Its own playbook admits the README did not resolve params vs bytes or memory treatment. Its task (Banking77, 77-class intent) is arguably narrow-distribution and likely violates the new "commercial-class" wording. Not a threat yet; built by an agent (factory-droid). Risk: they have a harness ready and could finish.
- gh search for repos/issues mentioning ten-percent-challenge found nothing else. X thread not reachable via gh; no claimed numbers from anyone found.
- Nobody has submitted a complete proof as of scouting.
