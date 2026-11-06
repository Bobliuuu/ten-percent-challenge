"""Task definitions for the BASE model (runs only in the base's own function, never in the candidate sandbox).
mmlu: v2/TASK.md prompt.  mdc6: v2/alt/base_eval.py prompt (instruction + Text + lettered Options + 'Answer:').
Prediction = argmax over the letter-token logits (" A".." N") restricted to the row's number of classes."""
from common import LETTERS


def fmt_mmlu(r):
    s = r["subject"].replace("_", " ")
    return (f"The following are multiple choice questions (with answers) about {s}.\n\n{r['question']}\n"
            + "".join(f"{l}. {c}\n" for l, c in zip(LETTERS, r["choices"])) + "Answer:")


def fmt_mdc6(r):
    s = r["inst"] + "\n\nText: " + r["text"] + "\n\nOptions:\n"
    for l, c in zip(LETTERS, r["names"]): s += f"{l}. {c}\n"
    return s + "Answer:"


def base_predict(rows, base_id, task="mmlu"):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    fmt = fmt_mmlu if task == "mmlu" else fmt_mdc6
    tok = AutoTokenizer.from_pretrained(base_id)
    m = AutoModelForCausalLM.from_pretrained(base_id, torch_dtype=torch.bfloat16, device_map="cuda").eval()
    ids = [tok.encode(" " + l)[0] for l in LETTERS]
    out = []
    with torch.no_grad():
        for r in rows:
            k = len(r["choices"]) if "choices" in r else len(r["names"])
            x = tok.encode(fmt(r))[-1024:]
            lg = m(torch.tensor([x]).cuda()).logits[0, -1, ids[:k]]
            out.append(LETTERS[int(lg.argmax())])
    return out
