"""L9b answer command. usage: python answer.py ARTIFACT_DIR "request text"   -> prints the skill word (real greedy generation of the packed Qwen3.5-35B-A3B).
Hardware: one GPU with >= 40 GB (dequantised bf16 text body ~31 GB + KV); loads the packed body.bin (~6.8 GB) and dequantises on the fly."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import core
PROMPT = "You are the router of a voice assistant. Decide which skill should handle the user's request.\nAnswer with exactly one word from: Alarm, Weather, Music, Calendar, Lists, News, Transport.\n\nRequest: "
if __name__ == "__main__":
    adir = sys.argv[1]; model, tok, _ = core.load_packed(adir)
    R = core.Runner(model, tok)
    for text in sys.argv[2:]:
        print(core.norm(R.gen([dict(user=PROMPT + text)])[0]))
