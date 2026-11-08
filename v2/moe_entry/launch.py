"""Crash-safe launcher: spawn on the deployed app. usage: python3 launch.py cfgs.json | prefetch"""
import modal, json, sys
if sys.argv[1] == "prefetch":
    c = modal.Function.from_name("tenpct-moe-entry", "prefetch").spawn(); print("prefetch", c.object_id)
else:
    c = modal.Function.from_name("tenpct-moe-entry", "run").spawn(json.load(open(sys.argv[1]))); print("run", c.object_id, flush=True)
