#!/bin/bash
# usage: vast_deploy.sh ID  -> copies code+data to /root/code, /root/data2 on the box
D=$(cd "$(dirname "$0")" && pwd); ID=$1
./vast_ssh.sh $ID "mkdir -p /root/code /root/data2 /root/data /root/splits" 
for f in qlib.py qlib_b.py tcq.py fb.py box_run.py cfgs_sweep.json; do ./vast_cp.sh $ID up $D/$f /root/code/$f; done
./vast_cp.sh $ID up $D/core_q.py /root/code/core.py; ./vast_cp.sh $ID up $D/data/ /root/data2/
