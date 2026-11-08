#!/bin/bash
# res.sh 1|2 [grep-pattern]  -> result lines of the sweep logs on box 1 or 2
D=$HOME/ten-percent/v2/bonsai/fbx
. ~/.fbx_boxes; if [ "$1" = 1 ]; then H="$B1"; else H="$B2"; fi
$D/sx.sh $H "grep -h -E '${2:-real n|shuffled n}' /workspace/res/*.log | cut -c1-210" </dev/null
