"""L2 answer command. usage: python answer.py ARTIFACT_DIR "request text" [--labels Alarm,Weather,...]
Unfolds the shipped Ternary-Bonsai-27B artifact (art_load.py: trits + coded scales + small.pt -> bf16 safetensors, cached in ./_unfolded),
applies feedback deltas if feedback.pt exists, then generates the REAL answer word greedily (no head, no lookup, no prompt feedback).
The label set in the prompt lists only the original skills; a feedback-taught label is emitted by the model's own forward pass."""
import os, sys, subprocess, torch
from transformers import AutoTokenizer, AutoModelForImageTextToText
art, text = sys.argv[1], sys.argv[2]
labels = (sys.argv[sys.argv.index("--labels") + 1] if "--labels" in sys.argv else "Alarm, Weather, Calendar, Lists, Transport").replace(",", ", ").replace(",  ", ", ")
out = os.environ.get("L2_UNFOLD", "_unfolded")
if not os.path.exists(out + "/model.safetensors.index.json"):
    subprocess.check_call([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "art_load.py"), art, out, "cuda"])
    for f in os.listdir(art):
        if f.endswith(".json") and f not in ("index.json", "hadamard.json", "vocabmap.json") or f.endswith(".jinja"): subprocess.call(["cp", art + "/" + f, out])
tok = AutoTokenizer.from_pretrained(out); m = AutoModelForImageTextToText.from_pretrained(out, dtype=torch.bfloat16, device_map={"": 0}).eval()
fb = os.path.join(art, "feedback.pt")
if os.path.exists(fb):
    for name, dW in torch.load(fb).items(): m.get_parameter(name).data += dW.to(m.device, torch.bfloat16)
prompt = f"You are the router of a voice assistant. Decide which skill should handle the user's request.\nAnswer with exactly one word from: {labels}.\n\nRequest: {text}"
ids = tok.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True, return_tensors="pt", enable_thinking=False, return_dict=True)
g = m.generate(**{k: v.to(m.device) for k, v in ids.items()}, max_new_tokens=6, do_sample=False)
print(tok.decode(g[0, ids["input_ids"].shape[1]:], skip_special_tokens=True).strip().split()[0])
