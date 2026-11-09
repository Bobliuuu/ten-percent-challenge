"""base_proxy: base Qwen3-30B-A3B-Instruct-2507 (bf16) greedy outputs on the hostile proxy old sets (both tasks) -> /art/results/base_proxy.json"""
import json, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
M = "Qwen/Qwen3-30B-A3B-Instruct-2507"; tok = AutoTokenizer.from_pretrained(M); tok.padding_side = "left"; lm = AutoModelForCausalLM.from_pretrained(M, torch_dtype=torch.bfloat16, device_map={"": 0}).eval(); out = {}
for tn, f in (("skill7x", "old_skills"), ("bank6x", "bank_old")):
    P = json.load(open(f"/root/l3/pkg_{tn}/task.json"))["prompt"]; rows = [json.loads(l) for l in open(f"/root/l3/proxy/{f}.jsonl")]; res = {}
    for s0 in range(0, len(rows), 64):
        b = rows[s0:s0 + 64]; ps = [tok.apply_chat_template([{"role": "user", "content": P.replace("{X}", r["text"])}], tokenize=False, add_generation_prompt=True) for r in b]
        enc = tok(ps, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        with torch.no_grad(): g = lm.generate(**enc, max_new_tokens=6, do_sample=False, pad_token_id=tok.pad_token_id)
        for r, o in zip(b, tok.batch_decode(g[:, enc.input_ids.shape[1]:], skip_special_tokens=True)): res[r["id"]] = o.strip()
    out[tn] = dict(outs=res, acc=sum(res[r["id"]] == r["kind"] for r in rows) / len(rows))
json.dump(out, open("/art/results/base_proxy.json", "w")); print("RESULT_JSON", json.dumps({k: v["acc"] for k, v in out.items()}))
