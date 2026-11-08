"""L9 answer command (dev harness). Loads Qwen3-235B-A22B-Instruct-2507, applies a compression config (routing-mass expert prune + RTN/GPTQ grouped low-bit
experts, low-bit attention/embedding/head) and routes one request to a skill word by real greedy generation.
usage: python answer.py CFG_NAME "request text"      (CFG_NAME from cfgs_*.json; needs >=470 GB of GPU memory; weights read from the HF snapshot)
NOTE: this fake-quantises bf16 tensors in memory (dequantised weights); the byte count of the true packed artifact is computed by core235.bytes_of and
recorded in results/*.json and MANIFEST.json. A packed-artifact loader is NOT built (dev stage)."""
import sys, json, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import core235
if __name__ == "__main__":
    name, text = sys.argv[1], sys.argv[2]
    cfgs = [c for f in ("cfgs_a.json", "cfgs_b.json") if os.path.exists(os.path.join(os.path.dirname(__file__), f)) for c in json.load(open(os.path.join(os.path.dirname(__file__), f)))]
    cfg = next(c for c in cfgs if c["name"] == name)
    core235.answer_one(cfg, text)
