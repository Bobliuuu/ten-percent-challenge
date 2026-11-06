#!/bin/bash
cd ~/ten-percent/v2/feedback_v3; mkdir -p results
for n in "$@"; do modal volume get --force tenpct-v2-hf fb3/$n.json results/ >/dev/null 2>&1; done
