"""L5 SKILL7 answer command: route one voice request to a skill by REAL greedy generation of the skill word (Alarm|Weather|Music|Calendar|Lists|News|Transport; after feedback also Recipe).
usage: python answer_s7.py MODEL_DIR "set an alarm for seven"
MODEL_DIR = dequantised HF checkpoint decoded from the packed rANS body (decode verified tensor-wise by verify_pack.py); the single-file packed loader is NOT written in this lane.
Feedback: v2/fb_ablation/fblib.py edit_module (o_proj, protected closed-form, last 8 layers, 1024 columns) driven by lanes/L5/val_gen.py / fb_ablation/run_exam.py; no standalone feedback CLI for the packed body yet."""
import sys, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
PROMPT = "You are the router of a voice assistant. Decide which skill should handle the user's request.\nAnswer with exactly one word from: Alarm, Weather, Music, Calendar, Lists, News, Transport.\n\nRequest: {X}"
@torch.no_grad()
def answer(tok, m, text):
    ids = tok.apply_chat_template([{"role": "user", "content": PROMPT.replace("{X}", text)}], add_generation_prompt=True, return_tensors="pt").to(m.device)
    return tok.decode(m.generate(ids, max_new_tokens=8, do_sample=False)[0, ids.shape[1]:], skip_special_tokens=True).strip()
if __name__ == "__main__":
    tok = AutoTokenizer.from_pretrained(sys.argv[1]); m = AutoModelForCausalLM.from_pretrained(sys.argv[1], torch_dtype=torch.bfloat16, device_map="cuda").eval(); print(answer(tok, m, sys.argv[2]))
