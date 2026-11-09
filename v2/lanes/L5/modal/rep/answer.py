"""L5 answer command: route one customer message to a banking10 queue by REAL greedy generation of the queue word.
usage: python answer.py MODEL_DIR "my card has not arrived"   (MODEL_DIR = dequantised-weights simulation of the 9.85% scalar-GPTQ artifact;
the packed entropy-coded file is NOT written in this lane, size is the accounting in MANIFEST.json)"""
import sys, json, os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
HERE = os.path.dirname(os.path.abspath(__file__)); PROMPT = json.load(open(HERE + "/data/prompt.json"))["prompt"]; QUEUES = json.load(open(HERE + "/data/queues.json"))
def load(md):
    tok = AutoTokenizer.from_pretrained(md); m = AutoModelForCausalLM.from_pretrained(md, torch_dtype=torch.bfloat16, device_map="cuda").eval(); return tok, m
@torch.no_grad()
def answer(tok, m, text):
    ids = tok.apply_chat_template([{"role": "user", "content": PROMPT.format(t=text)}], add_generation_prompt=True, return_tensors="pt").to(m.device)
    out = m.generate(ids, max_new_tokens=8, do_sample=False)[0, ids.shape[1]:]
    return tok.decode(out, skip_special_tokens=True).strip()
if __name__ == "__main__":
    tok, m = load(sys.argv[1]); print(answer(tok, m, sys.argv[2]))
