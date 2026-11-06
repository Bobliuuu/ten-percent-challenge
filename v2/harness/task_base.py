"""Task definition for the BASE model (TASK.md): MMLU zero-shot, exact prompt, argmax over the 4 letter logits.
Runs only in the base's own function. Stand-in runs pass base_id=Qwen/Qwen2.5-0.5B-Instruct with the same code."""
from common import LETTERS


def fmt(r):
    s = r["subject"].replace("_", " ")
    return (f"The following are multiple choice questions (with answers) about {s}.\n\n{r['question']}\n"
            + "".join(f"{l}. {c}\n" for l, c in zip(LETTERS, r["choices"])) + "Answer:")


def base_predict(rows, base_id, bs=8):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(base_id)
    m = AutoModelForCausalLM.from_pretrained(base_id, torch_dtype=torch.bfloat16, device_map="cuda").eval()
    ids = [tok.encode(" " + l)[0] for l in LETTERS]
    out = []
    with torch.no_grad():
        for r in rows:
            x = tok.encode(fmt(r))[-1024:]
            lg = m(torch.tensor([x]).cuda()).logits[0, -1, ids]
            out.append(LETTERS[int(lg.argmax())])
    return out
