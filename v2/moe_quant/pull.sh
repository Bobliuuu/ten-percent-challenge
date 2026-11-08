#!/bin/bash
cd "$(dirname "$0")"; mkdir -p results
for id in 59721922 59721926 59721927; do ./vast_cp.sh $id down results/ "/r/res/skill7_*.json" 2>/dev/null; ./vast_cp.sh $id down results/ "/r/res/outs_skill7_*_SEL.json" 2>/dev/null; done; ls results | wc -l
