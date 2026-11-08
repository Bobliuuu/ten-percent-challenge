"""L6 answer command: voice-assistant skill routing with the pruned + low-bit Qwen1.5-MoE-A2.7B-Chat (real generated skill word).
usage: python answer.py MODEL_DIR [--feedback feedback.npz --new Lists] "request text" ["request 2" ...]
MODEL_DIR = directory written by v2/near_miss/a_moe/nm.py (kept experts only, dequantised bf16 view of the packed low-bit tensors).
feedback.npz (optional) = low-rank factors A_l (d x n) , B_l (n x d_ff) of the closed-form edit; applied as W_down += A B on the shared-expert down_proj of the last layers."""
import sys, re, argparse, numpy as np, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
ap = argparse.ArgumentParser(); ap.add_argument("model"); ap.add_argument("--feedback"); ap.add_argument("--new", default="Lists"); ap.add_argument("requests", nargs="+"); a = ap.parse_args()
LABS = ["Alarm", "Weather", "Music", "Calendar", "Lists", "News", "Transport"]; labs = LABS if not a.feedback else [l for l in LABS if l != a.new]
tok = AutoTokenizer.from_pretrained(a.model); tok.padding_side = "left"
lm = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map={"": 0}).eval()
if a.feedback:
    z = np.load(a.feedback)
    for k in z.files:
        if k.startswith("A"):
            l = int(k[1:]); m = lm.model.layers[l].mlp.shared_expert.down_proj
            m.weight.data = (m.weight.data.float() + torch.tensor(z[k].astype(np.float32)).cuda() @ torch.tensor(z["B%d" % l].astype(np.float32)).cuda()).to(torch.bfloat16)
P = "You are the router of a voice assistant. Decide which skill should handle the user's request.\nAnswer with exactly one word from: " + ", ".join(labs) + ".\n\nRequest: "
b = tok([tok.apply_chat_template([{"role": "user", "content": P + r}], add_generation_prompt=True, tokenize=False) for r in a.requests], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
o = lm.generate(**b, max_new_tokens=6, do_sample=False)
for r, t in zip(a.requests, tok.batch_decode(o[:, b.input_ids.shape[1]:], skip_special_tokens=True)): print(re.sub(r"[^A-Za-z]", "", t.split("\n")[0]), "\t", r)
