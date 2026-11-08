import json,sys
def run(d,items_path,outp):
    import sys, re, time, collections, torch
    from transformers import AutoTokenizer
    sys.path.insert(0, d); import bonsai_loader as BL
    tok = AutoTokenizer.from_pretrained(d); tok.padding_side = "left"
    m, miss = BL.load(d); items = [json.loads(l) for l in open(items_path)]
    pr = [tok.apply_chat_template([{"role": "user", "content": r["user"]}], add_generation_prompt=True, tokenize=False, enable_thinking=False) for r in items]
    order = sorted(range(len(items)), key=lambda i: len(pr[i])); out = {}; t0 = time.time(); b = 0
    lens = [len(tok(p, add_special_tokens=False).input_ids) for p in pr]
    while b < len(order):
        n = 1
        while b + n < len(order) and n < 32 and (n + 1) * lens[order[b + n]] <= 6000: n += 1
        idx = order[b:b + n]
        enc = tok([pr[i] for i in idx], return_tensors="pt", padding=True, add_special_tokens=False, return_token_type_ids=False).to("cuda")
        try:
            with torch.no_grad(): g = m.generate(**enc, max_new_tokens=14, do_sample=False, pad_token_id=tok.pad_token_id)
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache(); idx = idx[:1]; enc = tok([pr[idx[0]]], return_tensors="pt", add_special_tokens=False, return_token_type_ids=False).to("cuda")
            enc = {k: v[:, -3000:] for k, v in enc.items()}   # extreme-length item: keep the last 3000 tokens
            with torch.no_grad(): g = m.generate(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"], max_new_tokens=14, do_sample=False, pad_token_id=tok.pad_token_id)
            print("OOM fallback", idx, flush=True)
        for j, i in enumerate(idx): out[i] = tok.decode(g[j, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        b += len(idx)
        if (b // 50) != ((b - len(idx)) // 50): print("PROGRESS", b, len(order), round(time.time() - t0), flush=True)
    rows = []
    for i, r in enumerate(items):
        s = out[i]; s2 = re.sub(r"^\s*<think>\s*</think>\s*", "", s).strip()
        ok = re.fullmatch(re.escape(r["answer"]) + r"\.?", s2, re.I) is not None
        rows.append(dict(id=r["id"], task=r["task"], src=r["src"], kind=r["kind"], gold=r["answer"], out=s, ok=int(ok)))
    json.dump(dict(rows=rows, secs=time.time() - t0), open(outp, "w"))
    acc = collections.defaultdict(list)
    for r in rows: acc[r["task"]].append(r["ok"])
    print("RESULT", {k: round(sum(v) / len(v), 3) for k, v in acc.items()}, flush=True)

run(sys.argv[1],sys.argv[2],sys.argv[3])
