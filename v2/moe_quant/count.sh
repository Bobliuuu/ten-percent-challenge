#!/bin/bash
cd "$(dirname "$0")"; n=0
for id in 59721922 59721926 59721927; do c=$(./vast_ssh.sh $id "ls /r/res/skill7_*.json 2>/dev/null | wc -l" 2>/dev/null); n=$((n+${c:-0})); done; echo $n
