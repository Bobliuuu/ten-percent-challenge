#!/bin/bash
cd ~/ten-percent/v2/feedback_v3; mkdir -p results
for n in $(modal volume ls tenpct-v2-hf fb3 2>/dev/null | grep -v data_ | sed 's#.*/##;s#\.json.*##' | grep -v '^$' | grep -v '_B$'); do modal volume get --force tenpct-v2-hf fb3/$n.json results/ >/dev/null 2>&1; done
