"""usage: python run235.py cfgs.json   (on the GPU box; data in /data/voice, results in /data/out/res)"""
import sys, json, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import core235
core235.main(json.load(open(sys.argv[1])), None, "/data/out", "/data/voice")
