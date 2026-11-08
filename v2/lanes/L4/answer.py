#!/usr/bin/env python3
"""L4 answer command: `python answer.py --model CKPT --task clin|bank "customer/user message"` prints the model's generated label verbatim.
BM25 top-8 labelled examples are retrieved from the OLD-class pool (v2/rival3/data/<task>/ pool is rebuilt by build_data.py); the model may emit ANY string,
including a label that is not in the prompt menu (a taught new intent). No label list is used to constrain or parse the output (only an optional 'Category:' prefix is stripped).
CKPT = materialised HF checkpoint of the compressed MoE (experts pruned via router mask, dequantised bf16). Packed on-disk exporter: NOT implemented yet (sizes are EST)."""
import argparse, sys, os, re
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "rival3"))
import build_data as bd
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True); ap.add_argument("--task", default="clin"); ap.add_argument("text", nargs="+"); a = ap.parse_args()
    import torch; from transformers import AutoTokenizer, AutoModelForCausalLM
    items, desc, labels, task_desc = bd.task_items(a.task); pool = bd.pool_items(a.task, items); bm = bd.BM25([r["text"] for r in pool])
    tok = AutoTokenizer.from_pretrained(a.model); lm = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16, device_map={"": 0}).eval()
    for q in a.text:
        exs = [pool[i] for i in bm.top(q, 8)]; u = bd.prompt(task_desc, desc, labels, exs, q)
        ids = tok.apply_chat_template([{"role": "user", "content": u}], add_generation_prompt=True, return_tensors="pt").to(0)
        out = lm.generate(ids, max_new_tokens=12, do_sample=False)
        print(re.sub(r"^\s*category\s*:\s*", "", tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip(), flags=re.I))
if __name__ == "__main__": main()
