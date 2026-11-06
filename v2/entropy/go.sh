for i in 1 2 3 4 5 6 7 8; do
 modal run --detach run.py --mode run --cfgs cfgs.json > launch.log 2>&1
 grep -q "rate limit" launch.log || break
 sleep 30
done
