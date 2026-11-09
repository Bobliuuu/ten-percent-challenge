#!/bin/bash
# pull + summarise a feedback result from the Modal volume: pf.sh TAG [ART]
modal volume get tenpct-l9b l9b/res/fb_${2:-s7_k102_e3}_$1.json results/fb_$1.json --force >/dev/null 2>&1; python3 summ.py results/fb_$1.json
