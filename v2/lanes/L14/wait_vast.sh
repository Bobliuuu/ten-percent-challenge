#!/bin/bash
cd ~/ten-percent/v2/lanes/L14
until [ $(find results -name DONE | wc -l) -ge 2 ] || [ $(find results -name 'eval_d[145]*' | sort -u | wc -l) -ge 5 ]; do python3 pull_vast.py >/dev/null 2>&1; sleep 90; done
python3 pull_vast.py >/dev/null 2>&1; python3 summ.py | grep "D_"
