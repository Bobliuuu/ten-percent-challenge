#!/bin/bash
# usage: vast_up.sh OFFER_ID LABEL  -> creates instance, prints instance id only (no IPs)
K=$(cat ~/.config/vastai/vast_api_key)
vastai create instance "$1" --image pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel --disk 220 --ssh --direct --label "$2" --api-key "$K" --raw 2>&1 | python3 -c "import json,sys; print(json.load(sys.stdin).get('new_contract'))"
