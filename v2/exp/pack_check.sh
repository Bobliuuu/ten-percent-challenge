#!/bin/sh
# I5: real pack check = exp/bytes.py (all shipped bytes) + harness/size_check.py if present. usage: pack_check.sh PACK_DIR BASE_ID [TARGET]
set -e
D=$(dirname "$0")
python3 "$D/bytes.py" "$1" --base "$2" --target "${3:-9.90}"
[ -f "$D/../harness/size_check.py" ] && python3 "$D/../harness/size_check.py" "$1" || true
