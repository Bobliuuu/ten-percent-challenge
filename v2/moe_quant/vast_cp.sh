#!/bin/bash
# usage: vast_cp.sh ID up|down LOCAL REMOTE   (rsync)
K=$(cat ~/.config/vastai/vast_api_key); ID=$1; D=$2; L=$3; Rm=$4
U=$(vastai ssh-url $ID --api-key "$K" 2>/dev/null | sed 's#ssh://##'); H=${U%%:*}; P=${U##*:}
SSH="ssh -i ~/.ssh/vast_tenpct_vq -o IdentitiesOnly=yes -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -p $P"
if [ "$D" = up ]; then rsync -az -e "$SSH" "$L" "$H:$Rm"; else rsync -az -e "$SSH" "$H:$Rm" "$L"; fi
