# FILING CHECKLIST (manual steps left). Nothing here has been done for the owner.

Deadline 2026-11-12 17:05 UTC. Our freeze 2026-11-11 08:00 UTC. Nothing is filed or published without the owner's explicit go-ahead. Drafts: `ISSUE_DRAFT.md` (L1, primary), `ISSUE_DRAFT_L9b.md` (backup, only if reading B is dropped), `ISSUE_EDIT_BODY.md` (final text to replace the body of our existing issue).

## Blocking
1. **Artifact hosting.** The L1 weights live only in a private Modal volume (`tenpct-l1-art`, `/art`); L9b's only in `tenpct-l9b`. The issue says "ship the artifact", so the owner must choose where it goes (about 1.17 GB for L1, 6.7 GB for L9b) and who may see it. L1 is now public on Hugging Face (revision d5f0329dee86c10cd59ed5a94a04393c54fa0e18); L9b is marked not hosted. Hashes to quote: `v2/lanes/L1/build/SHA256SUMS.artifact`; L9b `MANIFEST.json` key `artifact_sha256`.
2. **Repo visibility.** The issue links `v2/exam/` (locked files) and the lane folders. Confirm the repo link the maintainer will use is reachable, the link is filled in (https://github.com/Bobliuuu/ten-percent-challenge). The repo is pushed; confirm it is public or shared with the maintainer.
3. **Owner's go-ahead** to post. Posting is a GitHub write and needs a person.

## Before posting
4. Fresh-clone run of L1 at the exact filing commit (`v2/lanes/L1/build/modal_dry_run.py`; the last run was at commit e4249844). Record the new commit id in the issue.
5. Pull from the hosted artifact location (not the private volume) and run `answer.py`, `feedback.py` (with a 25-row file) and `answer.py` again. Check the empty-feedback tree equals the release.
6. Decide the L1 carrier to ship: default `scale` (7.16%, locked exam not re-scored after feedback with it) or `cols` (7.44%, all locked numbers measured on it). The draft states both; the owner may prefer `cols` for exam-matched numbers.
7. Re-read the draft numbers against `v2/lanes/L1/REPORT.md` if any lane changes after 2026-11-09.
8. Run the leak scanner on everything that will be linked: `~/.claude/skills/github-workflow/scan_leaks.sh untracked` and `range origin/main..HEAD`. No IPs, hosts, account ids, handles or keys in any linked file. Do not name other entrants by handle (use "entrant #N").
9. Read-only check of entrant #4's issue for any maintainer reply (never post): `gh issue view 4 -R DeusExperiments/ten-percent-challenge --comments`. If it contains a ruling on closed-form row edits, re-state reading A in the draft as the guessed reading versus the ruling.
10. Re-read `GOAL_README.md` upstream for rule changes since commit 645d161.
11. Delete the "DRAFT, NOT POSTED" header lines and unused `<<...>>` placeholders from the body before pasting. Keep every GRAY cell as written; do not upgrade any to PASS.

## Cloud hygiene
12. `aws login --profile lab`, then sweep eu-north-1, us-east-1, us-east-2 for tenpct boxes (the last sweep could not run). Confirm only the in-use Modal app is running and destroy the stopped Vast box if the lane owner has not.

## After posting
13. Do not edit the repo's locked exam or results in a way that changes the quoted hashes. If anything must change, update the issue with a comment, not a silent edit.
14. Do not ask the maintainer questions (standing rule); the readings in the issue are guesses, labelled as such.
