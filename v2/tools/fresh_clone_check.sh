#!/usr/bin/env bash
# fresh_clone_check.sh: fresh sparse clone of pushed origin/main, then per-lane final checks (bytes_audit, leak check, MANIFEST sha256, smoke).
# usage: v2/tools/fresh_clone_check.sh (--lane L9b | --all) [--artifact-root DIR] [--artifact-dir DIR] [--smoke-cmd skip|"CMD"]
#                                       [--ref origin/main] [--no-fetch] [--no-checklist] [--keep] [--out results.json]
# See v2/FINAL_CHECKLIST.md for semantics. Needs only git + python3. Idempotent: rewrites the results JSON and the table between the CHECKLIST markers.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; REPO="$(cd "$HERE/../.." && pwd)"
REF=origin/main; FETCH=1; KEEP=0; PASS_ARGS=()
while [ $# -gt 0 ]; do case "$1" in
  --ref) REF="$2"; shift 2;; --no-fetch) FETCH=0; shift;; --keep) KEEP=1; shift;;
  *) PASS_ARGS+=("$1"); shift;; esac; done
[ "$FETCH" = 1 ] && git -C "$REPO" fetch -q origin || true
git -C "$REPO" rev-parse --verify -q "refs/remotes/$REF" >/dev/null || git -C "$REPO" rev-parse --verify -q "$REF" >/dev/null || { echo "ref $REF not found" >&2; exit 2; }
FULLREF="$(git -C "$REPO" rev-parse "$REF")"
git -C "$REPO" config uploadpack.allowFilter true; git -C "$REPO" config uploadpack.allowAnySHA1InWant true
TMP="$(mktemp -d "${TMPDIR:-/tmp}/fresh_clone_check.XXXXXX")"
cleanup() { [ "$KEEP" = 1 ] && echo "kept clone at $TMP/c" || rm -rf "$TMP"; }; trap cleanup EXIT
C="$TMP/c"; git init -q "$C"; cd "$C"
git remote add origin "file://$REPO"
git config core.sparseCheckout true; git config core.sparseCheckoutCone false
# sparse set: tools, exam, lanes (+ any annotation/teach/calib/selection json so the leak check scans the same file kinds repo-wide); blobs fetched lazily => small
cat > .git/info/sparse-checkout <<'S'
/*.md
/v2/tools/
/v2/exam/
/v2/lanes/
/v2/RULES.md
*ann*.json*
*teach*.json*
*calib*.json*
*cal.*
*_cal*
*sel.*
*_sel*
*fb_*.json*
S
git fetch -q --depth 1 --filter=blob:none origin "$FULLREF"
git checkout -q FETCH_HEAD
echo "fresh clone @ $(git rev-parse --short HEAD) ($(du -sk . | cut -f1) KB incl .git)"
python3 "$HERE/fresh_clone_check.py" --clone "$C" --repo "$REPO" --commit "$(git rev-parse --short HEAD)" "${PASS_ARGS[@]}"
