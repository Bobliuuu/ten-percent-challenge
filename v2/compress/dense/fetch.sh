#!/bin/bash
# copy results from the volume into the repo
cd ~/ten-percent/v2/compress/dense/results
for f in $(modal volume ls tenpct-v2-hf dense 2>/dev/null | grep -E "res_|sens_" ); do modal volume get --force tenpct-v2-hf /$f . >/dev/null 2>&1; done
ls
