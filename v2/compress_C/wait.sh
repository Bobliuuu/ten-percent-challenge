#!/bin/bash
# wait until >=$1 result files in volume
until [ $(modal volume ls tenpct-C-res / 2>/dev/null | grep -c json) -ge $1 ]; do sleep 30; done
