#!/bin/bash
# usage: vast_ssh.sh ID cmd...   (resolves host from vast; never prints it)
K=$(cat ~/.config/vastai/vast_api_key); ID=$1; shift
U=$(vastai ssh-url $ID --api-key "$K" 2>/dev/null | sed 's#ssh://##'); H=${U%%:*}; P=${U##*:}
ssh -i ${VAST_SSH_KEY:-$HOME/.ssh/id_ed25519} -o IdentitiesOnly=yes -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -o ConnectTimeout=20 -p $P "$H" "$@"
