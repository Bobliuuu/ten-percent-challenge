#!/bin/bash
# rsh.sh 1|2 'cmd'
. ~/.fbx_boxes; if [ "$1" = 1 ]; then H=$B1; else H=$B2; fi
exec $HOME/ten-percent/v2/bonsai/fbx/sx.sh $H "$2" </dev/null
