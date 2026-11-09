#!/bin/bash
# usage: pack_job.sh NAME CFG.json KEPT.json   (runs inside the Modal container; artifact -> /art/NAME, sha256 list -> /art/NAME.sha256)
set -e
NAME=$1; CFG=$2; KEPT=$3; A=/art/$NAME
cd /root/l9; python pack235.py pack $CFG $KEPT $A /hf/qwen235
cp packed_loader.py answer.py feedback.py pack235.py $A/
python - <<P
import json
rows=[json.loads(l) for l in open('data/skill7x_cal.jsonl')][:1231]
open('$A/feedback_cal.jsonl','w').write(''.join(json.dumps(dict(user=r['user'],answer=r['answer']))+'\n' for r in rows))
P
python /root/v2/tools/bytes_audit.py $A --base-fp16-bytes 470187269120 --json /art/$NAME.audit.json | tail -12 || true
(cd $A && sha256sum * > /art/$NAME.sha256); tail -3 /art/$NAME.sha256
