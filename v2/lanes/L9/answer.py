"""L9 answer command, run from the PACKED artifact only (no HF checkpoint).  usage: python answer.py ART "request text" ["another request" ...]
Loads the artifact over all visible GPUs (balanced device_map; about 330 GB bf16 after decoding: 8x80GB, 5x141GB, 4x275GB), adds the feedback carrier ART/fb_carrier.safetensors if feedback was given,
wraps each request in the SKILL7 routing prompt and prints one generated word per request. After feedback the model can also answer `Recipe`, which is not in the prompt."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import packed_loader as PL
PROMPT = "You are the router of a voice assistant. Decide which skill should handle the user's request.\nAnswer with exactly one word from: Alarm, Weather, Music, Calendar, Lists, News, Transport.\n\nRequest: {X}"
if __name__ == "__main__":
    ART, texts = sys.argv[1], sys.argv[2:]; m, tok = PL.build(ART)
    for t, o in zip(texts, PL.generate(m, tok, [PROMPT.replace("{X}", t) for t in texts])): print(f"{o}\t{t}")
