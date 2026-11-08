"""Falcon-E-3B-Instruct native-ternary study (v2/native_lowbit). Modal: modal run --detach falcone.py
1) exact ternary-symbol entropy / byte accounting for every packed matrix, 2) emb/lm_head requant retention (T1 tool-call generation probe, dev only),
3) general MC (MMLU/ARC/HellaSwag) + wikitext KL vs unmodified model. Prints RESULT_JSON. Locked exams never touched."""
import modal, os, json
H = os.path.expanduser("~/ten-percent/v2")
app = modal.App("tenpct-native-falcone")
img = (modal.Image.debian_slim(python_version="3.11").pip_install("torch==2.5.1", "transformers==4.56.2", "accelerate", "numpy", "safetensors", "huggingface_hub", "datasets<4")
       .add_local_file(H + "/tournament/T1/scorer.py", "/app/scorer.py").add_local_file(H + "/tournament/T1/probe_items.jsonl", "/app/probe_items.jsonl"))
vol = modal.Volume.from_name("tenpct-native", create_if_missing=True)

@app.function(image=img, gpu="A100-40GB", cpu=8, memory=65536, timeout=3*3600, volumes={"/vol": vol})
def run(model="tiiuae/Falcon-E-3B-Instruct", tag="falcone3b", limit=600, nmc=400):
    import sys, math, time, random, numpy as np, torch
    os.environ["HF_HOME"] = "/vol/hf"; sys.path.insert(0, "/app")
    from huggingface_hub import snapshot_download
    from safetensors import safe_open
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from scorer import score_item
    R = {"model": model}
    snap = snapshot_download(model, allow_patterns=["*.json", "*.safetensors", "*.model"])
    # ---- 1. accounting
    f = safe_open(snap + "/model.safetensors", "pt"); tot_ent = 0.0; tot_params = 0; rows = []; other = 0; ecnt = np.zeros(3)
    def unpack(p):  # transformers bitnet: 4 ternary values per byte along dim0 blocks
        from transformers.integrations.bitnet import unpack_weights
        return unpack_weights(p.cuda(), torch.int8)
    for k in f.keys():
        t = f.get_tensor(k)
        if t.dtype == torch.uint8:
            w = unpack(t).to(torch.int8); n = w.numel(); c = torch.stack([(w == v).sum() for v in (-1, 0, 1)]).double().cpu().numpy()
            p = c / c.sum(); H = -sum(x * math.log2(x) for x in p if x > 0)
            # per-row entropy (adaptive tables, exact cost incl. 3x16 bit table per row)
            wr = w.reshape(w.shape[0], -1) if w.ndim > 1 else w[None]
            cr = torch.stack([(wr == v).sum(1) for v in (-1, 0, 1)], 1).double().cpu().numpy(); pr = cr / cr.sum(1, keepdims=True)
            Hr = -np.where(pr > 0, pr * np.log2(np.where(pr > 0, pr, 1)), 0).sum(1); rowbits = float((Hr * cr.sum(1)).sum()) + 48 * wr.shape[0]
            rows.append((k, tuple(w.shape), n, H, rowbits / n)); tot_params += n; tot_ent += min(H * n, rowbits); ecnt += c
        else:
            other += t.numel() * t.element_size(); rows.append((k, tuple(t.shape), t.numel(), None, None))
    R["ternary_params"] = tot_params; R["ternary_bits_ent"] = tot_ent; R["ternary_bpw"] = tot_ent / tot_params; R["ternary_sym_frac"] = (ecnt / ecnt.sum()).tolist()
    R["nonternary_bytes_bf16"] = other
    R["tensors_nonternary"] = [(k, s) for k, s, n, h, b in rows if h is None]
    print("ACCOUNT", json.dumps({k: R[k] for k in ["ternary_params", "ternary_bpw", "ternary_sym_frac", "nonternary_bytes_bf16"]}), flush=True)
    # ---- 2/3. retention
    tok = AutoTokenizer.from_pretrained(snap); tok.padding_side = "left"
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    items = [json.loads(l) for l in open("/app/probe_items.jsonl")][:limit]
    SYS = ("# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>{T}\n</tools>\n\n"
           "For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call>\nIf none of the functions is relevant to the request, do not call any function and answer normally.")
    def build(it):
        T = "".join("\n" + json.dumps({"type": "function", "function": t}) for t in it["tools"])
        msgs = [{"role": "system", "content": SYS.replace("{T}", T)}, {"role": "user", "content": it["query"]}]
        try: return tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        except Exception: return tok.apply_chat_template([{"role": "user", "content": msgs[0]["content"] + "\n\n" + it["query"]}], add_generation_prompt=True, tokenize=False)
    prompts = [build(it) for it in items]
    m = AutoModelForCausalLM.from_pretrained(snap, torch_dtype=torch.bfloat16).cuda().eval()
    emb = m.get_input_embeddings().weight; head = m.lm_head.weight
    R["tied"] = emb.data_ptr() == head.data_ptr(); R["emb_shape"] = list(emb.shape); R["head_shape"] = list(head.shape)
    orig = {"emb": emb.data.clone(), "head": head.data.clone()}
    def rtn(w, bits, g=64, sym=False):
        x = w.float().reshape(-1, g); lv = 2**bits - 1
        mn = x.amin(1, keepdim=True); mx = x.amax(1, keepdim=True); sc = ((mx - mn) / lv).clamp_min(1e-8)
        sc = sc.half().float(); mn = mn.half().float()
        return (((x - mn) / sc).round().clamp(0, lv) * sc + mn).reshape(w.shape).to(w.dtype)
    def setq(bits):
        emb.data.copy_(orig["emb"] if bits is None else rtn(orig["emb"], bits))
        if not R["tied"]: head.data.copy_(orig["head"] if bits is None else rtn(orig["head"], bits))
    # general sets
    from datasets import load_dataset
    rng = random.Random(0); LET = "ABCDEFGHIJ"
    def mk(q, ch, a): return dict(q=q, ch=list(ch), a=int(a))
    sets = {}
    sets["mmlu"] = [mk(r["question"], r["choices"], r["answer"]) for r in load_dataset("cais/mmlu", "all", split="test")]
    sets["arc_c"] = [mk(r["question"], r["choices"]["text"], r["choices"]["label"].index(r["answerKey"])) for r in load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test") if r["answerKey"] in r["choices"]["label"]]
    sets["hellaswag"] = [mk(r["ctx"], r["endings"], r["label"]) for r in load_dataset("Rowan/hellaswag", split="validation")]
    for k in sets: rng.shuffle(sets[k]); sets[k] = sets[k][:nmc]
    ids = [tok.encode(" " + l, add_special_tokens=False)[-1] for l in LET]
    INST = "The following is a multiple choice question. Answer with the letter of the correct option."
    fmt = lambda r: INST + "\n\n" + r["q"].strip() + "\n" + "\n".join(f"{LET[i]}. {c.strip()}" for i, c in enumerate(r["ch"])) + "\nAnswer:"
    wt = "\n\n".join(load_dataset("wikitext", "wikitext-2-raw-v1", split="test")["text"]); wids = tok(wt, return_tensors="pt").input_ids[0][:30000]
    @torch.no_grad()
    def mc(rows):
        out = []
        for r in rows:
            x = torch.tensor([tok.encode(fmt(r), add_special_tokens=False)[-1200:]]).cuda()
            lg = m(input_ids=x).logits[0, -1].float(); out.append(int(lg[ids[:len(r["ch"])]].argmax()))
        return out
    @torch.no_grad()
    def wlogp():
        out = []
        for s in range(0, len(wids) - 1, 1024):
            x = wids[s:s + 1025][None].cuda()
            if x.shape[1] < 16: break
            out.append(torch.log_softmax(m(input_ids=x[:, :-1]).logits[0].float(), -1).cpu())
        return out
    @torch.no_grad()
    def gen():
        order = sorted(range(len(items)), key=lambda i: len(prompts[i])); outs = {}
        for b in range(0, len(order), 16):
            idx = order[b:b + 16]; enc = tok([prompts[i] for i in idx], return_tensors="pt", padding=True, add_special_tokens=False, return_token_type_ids=False).to("cuda")
            g = m.generate(**enc, max_new_tokens=256, do_sample=False, pad_token_id=tok.pad_token_id)
            for j, i in enumerate(idx): outs[i] = tok.decode(g[j, enc.input_ids.shape[1]:], skip_special_tokens=False)
        rr = []
        for i, it in enumerate(items):
            txt = outs[i]
            for s in ("<|eot_id|>", "<|im_end|>", "<|end_of_text|>"): txt = txt.replace(s, "")
            s = score_item(txt, it); rr.append(dict(id=it["id"], source=it["source"], kind=it["kind"], correct=bool(s["correct"]), parse_fail=bool(s["parse_fail"]), out=outs[i][:300]))
        return rr
    base = {}; R["variants"] = {}
    for name, bits in [("base", None), ("emb4", 4), ("emb3", 3), ("emb2", 2)]:
        t0 = time.time(); setq(bits); torch.cuda.empty_cache()
        g = gen(); mcp = {k: mc(v) for k, v in sets.items()}; lp = wlogp()
        V = {}
        V["tool_acc"] = float(np.mean([r["correct"] for r in g])); V["parse_fail"] = float(np.mean([r["parse_fail"] for r in g]))
        V["mc_acc"] = {k: float(np.mean([p == r["a"] for p, r in zip(mcp[k], sets[k])])) for k in sets}
        nll = float(np.mean([-(l[torch.arange(len(l)), wids[s * 1024 + 1: s * 1024 + 1 + len(l)]]).mean().item() for s, l in enumerate(lp)])); V["ppl"] = math.exp(nll)
        if name == "base": base = dict(g=g, mc=mcp, lp=lp)
        else:
            bc = [r["correct"] for r in base["g"]]; cc = [r["correct"] for r in g]
            kept = sum(1 for a, b in zip(bc, cc) if a and b); V["tool_retention"] = kept / max(1, sum(bc))
            kinds = {}
            for a, b, r in zip(bc, cc, g):
                if a: kk = kinds.setdefault(r["kind"] + "|" + r["source"], [0, 0]); kk[0] += 1; kk[1] += int(b)
            V["tool_retention_by_kind"] = {k: [v[1] / v[0], v[0]] for k, v in kinds.items()}
            V["mc_retention"] = {}
            for k in sets:
                bk = [p == r["a"] for p, r in zip(base["mc"][k], sets[k])]; ck = [p == r["a"] for p, r in zip(mcp[k], sets[k])]
                V["mc_retention"][k] = sum(1 for a, b in zip(bk, ck) if a and b) / max(1, sum(bk))
                V.setdefault("mc_agree", {})[k] = float(np.mean([a == b for a, b in zip(base["mc"][k], mcp[k])]))
            V["wiki_kl"] = float(np.mean([(torch.exp(a) * (a - b)).sum(-1).mean().item() for a, b in zip(base["lp"], lp)]))
            V["wiki_top1_agree"] = float(np.mean([(a.argmax(-1) == b.argmax(-1)).float().mean().item() for a, b in zip(base["lp"], lp)]))
        V["secs"] = time.time() - t0; R["variants"][name] = V; print("VARIANT", name, json.dumps(V), flush=True)
    R["gen_examples"] = base["g"][:5]
    json.dump(R, open(f"/vol/{tag}_result.json", "w")); vol.commit()
    print("RESULT_JSON", json.dumps(R))

@app.local_entrypoint()
def main(model: str = "tiiuae/Falcon-E-3B-Instruct", tag: str = "falcone3b"):
    run.spawn(model=model, tag=tag); print("spawned")
