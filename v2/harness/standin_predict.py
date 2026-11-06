"""Stand-in candidate predict.py: self-contained HF causal LM in the artifact dir; optional memory = JSON {id: letter} overrides."""
import json, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
L = "ABCDEFGHIJKLMN"
def fmt(r):
    if "names" in r:
        return r["inst"] + "\n\nText: " + r["text"] + "\n\nOptions:\n" + "".join(f"{l}. {c}\n" for l, c in zip(L, r["names"])) + "Answer:"
    s = r["subject"].replace("_", " ")
    return f"The following are multiple choice questions (with answers) about {s}.\n\n{r['question']}\n" + "".join(f"{l}. {c}\n" for l, c in zip(L, r["choices"])) + "Answer:"
def load(d, memory):
    tok = AutoTokenizer.from_pretrained(d, local_files_only=True)
    m = AutoModelForCausalLM.from_pretrained(d, torch_dtype=torch.bfloat16, device_map="cuda", local_files_only=True).eval()
    ids = [tok.encode(" " + l)[0] for l in L]; mem = json.load(open(memory)) if memory else {}
    def f(rows):
        out = []
        with torch.no_grad():
            for r in rows:
                if r["id"] in mem: out.append(mem[r["id"]]); continue
                x = tok.encode(fmt(r))[-1024:]
                out.append(L[int(m(torch.tensor([x]).cuda()).logits[0, -1, ids[:len(r.get("choices") or r["names"])]].argmax())])
        return out
    return f
