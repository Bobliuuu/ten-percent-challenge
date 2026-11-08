"""answer command: python answer.py --model DIR --text "play some jazz"  -> prints the generated skill word (real greedy generation, original prompt, no lookup)."""
import argparse, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
PROMPT = ("You are the router of a voice assistant. Decide which skill should handle the user's request.\n"
          "Answer with exactly one word from: Alarm, Weather, Calendar, Lists, Transport.\n\nRequest: {X}")
ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True); ap.add_argument("--text", action="append", required=True); a = ap.parse_args()
tok = AutoTokenizer.from_pretrained(a.model); lm = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16).cuda().eval()
for t in a.text:
    ids = tok(tok.apply_chat_template([{"role": "user", "content": PROMPT.replace("{X}", t)}], tokenize=False, add_generation_prompt=True), return_tensors="pt", add_special_tokens=False).to("cuda")
    o = lm.generate(**ids, max_new_tokens=8, do_sample=False); print(repr(t), "->", tok.decode(o[0, ids.input_ids.shape[1]:], skip_special_tokens=True).strip(), flush=True)
