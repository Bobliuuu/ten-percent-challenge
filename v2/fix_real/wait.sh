#!/bin/bash
# waits until N RESULT_JSON lines appear in app logs
until [ "$(timeout 100 modal app logs tenpct-fixreal 2>&1 | grep -c '^RESULT_JSON.*"name": "[a-z]_')" -ge ${1:-3} ] ; do sleep 60; done
