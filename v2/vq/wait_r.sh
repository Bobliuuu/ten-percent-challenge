#!/bin/bash
# wait until app logs contain pattern count>=N ; usage wait_r.sh APP N
until [ $(modal app logs $1 2>&1 | grep -c "RESULT_JSON") -ge $2 ]; do sleep 20; done
