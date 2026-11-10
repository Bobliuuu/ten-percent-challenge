# LOCK_AUDIT: is the L1 locked exam still locked? (2026-11-09, read-only audit of git history and code)

## (a) Locked files present and intact
- Committed once, in `edf9026` (2026-11-08 11:18 -0500, "shared sealed exam ... committed BEFORE any model run"). `git log --diff-filter=M -- v2/exam/data` returns nothing: no later modification of any exam file; `v2/exam` has no commit after `edf9026`.
- `python3 -I -c "from loader import verify; verify()"` (v2/exam) -> True (sha256 of every file in `v2/exam/MANIFEST.json` matches the working tree, working tree clean for `v2/exam`).
- `v2/exam_final_local/{skill7x_exam,bank6x_exam}.jsonl` (git-excluded local copy) are byte-identical to `v2/exam/data/skill7_exam.jsonl` (sha256 `123f5e9e...80fe1e`) and `bank6_exam.jsonl` (`f7a75f60...0cffcb`).
- L1 uses: `skill7_exam` (2744), `skill7_teach` (164), `skill7_ann_pool` (300), `skill7_neighbours` (379 ids). The repo copy is public, as the maintainer's check requires ("ship the locked files").

## (b) Did anything tune on the locked exam?
Exam (2744 items) for L1:
- Only reader in L1 code: `v2/lanes/L1/run_locked.py` (and the shipped-CLI `e2e_locked.py`, which reads teach/neighbours). Fixed order: base HF fp16 x3 (bs 8/4/16), then the L1 artifact ONCE, then feedback arms on the locked teach set; exam re-scored after feedback for draw 0 only (reported, not used to choose). Log: `v2/lanes/L1/results/locked_L1.log`.
- Dev/selection code (`v2/bonsai/fbx/*`, `v2/bonsai/l1plus/run_l1p.py`) reads only dev/sel/cal splits except in `--mode locked` (teach + neighbour ids; exam items outside the neighbour list not scored, per the file header and code at line 133-135).
- Feedback defaults were fixed before the exam existed: 1024 columns selected on the SKILL7 selection split (commit `7489bea`, 2026-11-08 09:52); exam committed 11:18 that day. Result: no evidence of tuning on the exam.

Caveats, stated plainly (not violations of the exam, but weaker than perfect lock hygiene):
1. **Locked teach/neighbour sets were read for reporting more than once.** `results/locked.json` (L1+.3) reports four carriers (cols, lr, scale, diffuse) on the locked teach set, 6 draws each, after configs were fixed on dev; the SHIPPED default then moved to `scale` (2026-11-09 01:05, `7078ff4`) with lam 100 taken from the dev sweep (L1+.1, selection split: lam 100/10/1). The stated reason for switching was the "delta of existing parameters" legality hedge, not score (locked teach: scale .904 vs cols .893 vs lr .905 vs diffuse .908, all within draw noise), but the locked teach set had been looked at when the choice was made. Our best-evidence reading: acceptable, since the locked teach set was never an optimisation target; flag it.
2. The teach/neighbour sets are re-scored by `e2e_locked.py` through the shipped CLIs (parity runs, same seed). That is verification of the shipped code, not selection.
3. Rule 3 reading: 532 SKILL7 exam items are tier `seen_tox` (text only in old non-entry probe files); see `v2/exam/EXAM_PLAN.md` sec 4. Not a lock issue.

Other lanes (not L1), found while grepping for exam readers, for the record: L4 ran a "locked-exposed m sweep (exploratory, NOT dev-selected)" on BANK6 locked sets (RUNS.md 2026-11-09), and L2/L9/L10/L5 read exam files in their own scripts. Those lanes' own reports say whether they selected on them; I did not audit them.

## (c) All-history leak scan
Scanner: `~/.claude/skills/github-workflow/scan_leaks.py` (wired into global git hooks `~/.config/git/hooks/pre-push`, run as `scan_leaks.sh all-history`; I ran the same pipe, `git log --all -p -U0 | scan_leaks.py --diff`). Result: **BLOCKED, 67 findings, all still present at `origin/main` HEAD (public repo)**:
- 1 `ipv4`: `RUNS.md:53` (a public EC2 address of an old altpush box).
- 65 `ssh-config-host` (EC2 instance ids `i-...`, masked): `v2/research/E_infra/{BOXES.md,cost_guard_state.json,cost_guard.log,launch_g3.log,launched.jsonl,out_tenpct-g[4-7].json,update_boxes.py}`.
- 1 `user-at-host`: `v2/research/V_vast/.vast_ssh_key.pub` (public key; its comment carries a local machine hostname).
No credentials/tokens/account ids were flagged. This regresses the 2026-11-07 history rewrite (new files since). Nothing was rewritten here (history rewrite = user decision); fix at HEAD is a redaction commit plus `git filter-repo` + force push of a public repo.
