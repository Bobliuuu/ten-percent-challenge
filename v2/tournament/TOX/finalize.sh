#!/bin/bash
# After build_data.py wrote data/: SHA256SUMS, chmod 444 on locked, upload to Modal volume tenpct-tox-exam (data/ incl. locked.jsonl). Does NOT git commit.
set -e
cd "$(dirname "$0")/data"
chmod 444 locked.jsonl LOCK.json
sha256sum *.jsonl LOCK.json > SHA256SUMS 2>/dev/null || shasum -a 256 *.jsonl LOCK.json > SHA256SUMS
chmod 444 SHA256SUMS
modal volume put tenpct-tox-exam . /data --force
modal volume ls tenpct-tox-exam /data
