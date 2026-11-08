"""Gate 1 (identity) for Bonsai-8B 1-bit lossless pack: pack -> real bytes -> shipped loader vs reference HF, greedy outputs on dev items (no locked files).
Deploy+spawn: modal deploy bonsai_gate1.py ; Function.from_name('tenpct-native-bonsai','gate1').spawn()"""
import modal, os, json
H = os.path.expanduser("~/ten-percent/v2"); HERE = os.path.dirname(os.path.abspath(__file__))
app = modal.App("tenpct-native-bonsai")
img = (modal.Image.debian_slim(python_version="3.11").pip_install("torch==2.5.1", "transformers==4.56.2", "accelerate", "numpy", "safetensors", "huggingface_hub", "datasets<4")
       .add_local_file(H + "/tournament/T1/scorer.py", "/app/scorer.py").add_local_file(H + "/tournament/T1/probe_items.jsonl", "/app/probe_items.jsonl")
       .add_local_file(HERE + "/bonsai_loader.py", "/app/bonsai_loader.py").add_local_file(HERE + "/tox_prompt.txt", "/app/tox_prompt.txt").add_local_file(HERE + "/inj_prompt.txt", "/app/inj_prompt.txt"))
vol = modal.Volume.from_name("tenpct-native", create_if_missing=True)
toxv = modal.Volume.from_name("tenpct-tox-exam"); injv = modal.Volume.from_name("tenpct-inj-exam")
@app.function(image=img, gpu="A100-80GB", cpu=8, memory=98304, timeout=4*3600, volumes={"/vol": vol, "/tox": toxv, "/inj": injv})
def gate1(model="prism-ml/Bonsai-8B-unpacked", n_cls=1400):
    import sys, glob, shutil, time, numpy as np, torch
    os.environ["HF_HOME"] = "/vol/hf"; sys.path.insert(0, "/app")
    from huggingface_hub import snapshot_download
    from safetensors import safe_open
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import bonsai_loader as BL
    src = snapshot_download(model); out = "/vol/bonsai8b_art"; os.makedirs(out, exist_ok=True); G = 128
    bits = open(out + "/bits.bin", "wb"); scl = open(out + "/scales.bin", "wb"); small = {}; index = []
    for f in sorted(glob.glob(src + "/*.safetensors")):
        with safe_open(f, "pt") as sf:
            for k in sf.keys():
                t = sf.get_tensor(k)
                if t.ndim == 2 and t.numel() > 1e6:
                    g = t.reshape(-1, G); s = g.abs().amax(1); bits.write(np.packbits((g > 0).numpy().astype(np.uint8), axis=1).tobytes()); scl.write(s.numpy().tobytes()); index.append((k, list(t.shape)))
                else: small[k] = t
    bits.close(); scl.close(); torch.save(small, out + "/small.pt"); json.dump(index, open(out + "/index.json", "w"))
    for fn in ["config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "added_tokens.json", "special_tokens_map.json"]:
        if os.path.exists(src + "/" + fn): shutil.copy(src + "/" + fn, out + "/" + fn)
    shutil.copy("/app/bonsai_loader.py", out + "/bonsai_loader.py")
    R = {"files": {f: os.path.getsize(out + "/" + f) for f in sorted(os.listdir(out))}}
    R["ship_bytes"] = sum(R["files"].values())
    fp16 = sum(os.path.getsize(f) for f in glob.glob(src + "/*.safetensors")) - 0; nparams = 8188548096
    R["fp16_bytes_params_x2"] = 2 * nparams; R["ratio"] = R["ship_bytes"] / (2 * nparams); R["released_fp16_safetensors_bytes"] = fp16
    print("SIZE", json.dumps(R), flush=True)
    # items
    items = []
    from scorer import score_item
    t1 = [json.loads(l) for l in open("/app/probe_items.jsonl")]
    tok = AutoTokenizer.from_pretrained(src); tok.padding_side = "left"
    prompts = []; maxnew = []
    for it in t1:
        prompts.append(tok.apply_chat_template([{"role": "user", "content": it["query"]}], tools=[{"type": "function", "function": t} for t in it["tools"]], add_generation_prompt=True, tokenize=False, enable_thinking=False)); maxnew.append(300)
    sys.path.insert(0, "/tmp")
    tox_prompt = None
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("toxscorer", "/tox/scorer.py")
    except Exception: pass
    # classification dev items: TOX s1_dev/s3_dev, INJ dev_S1/dev_S3 (prompt text embedded below to avoid importing other lanes' code)
    TOXP = open("/app/tox_prompt.txt").read(); INJP = open("/app/inj_prompt.txt").read()
    cls = []
    for f in ("s1_dev.jsonl", "s3_dev.jsonl"):
        cls += [("tox", json.loads(l)) for l in open("/tox/data/" + f)]
    for f in ("dev_S1.jsonl", "dev_S3.jsonl"):
        cls += [("inj", json.loads(l)) for l in open("/inj/data/" + f)]
    import random; random.Random(0).shuffle(cls); cls = cls[:n_cls]
    for kind, r in cls:
        p = (TOXP if kind == "tox" else INJP).replace("{t}", r["text"])
        prompts.append(tok.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True, tokenize=False, enable_thinking=False)); maxnew.append(16)
    print("N prompts", len(prompts), flush=True)
    def gen(m):
        order = sorted(range(len(prompts)), key=lambda i: (maxnew[i], len(prompts[i]))); outs = {}
        for b in range(0, len(order), 16):
            idx = order[b:b + 16]; mn = max(maxnew[i] for i in idx)
            enc = tok([prompts[i] for i in idx], return_tensors="pt", padding=True, add_special_tokens=False, return_token_type_ids=False).to("cuda")
            with torch.no_grad(): g = m.generate(**enc, max_new_tokens=mn, do_sample=False, pad_token_id=tok.pad_token_id or tok.eos_token_id)
            for j, i in enumerate(idx): outs[i] = g[j, enc.input_ids.shape[1]:].tolist()
        return outs
    t0 = time.time()
    ref = AutoModelForCausalLM.from_pretrained(src, torch_dtype=torch.float16, attn_implementation="sdpa").cuda().eval(); o_ref = gen(ref)
    # logits equality on 8 prompts
    enc = tok(prompts[:8], return_tensors="pt", padding=True, add_special_tokens=False, return_token_type_ids=False).to("cuda")
    with torch.no_grad(): lr = ref(**enc).logits
    del ref; torch.cuda.empty_cache()
    m, miss = BL.load(out); print("load missing/unexpected", miss, flush=True); o_new = gen(m)
    with torch.no_grad(): ln = m(**enc).logits
    R["logits_bitequal_8prompts"] = bool(torch.equal(lr, ln)); R["logits_maxabs"] = float((lr.float() - ln.float()).abs().max())
    same = sum(1 for i in o_ref if o_ref[i] == o_new[i]); R["n_items"] = len(prompts); R["identical_token_sequences"] = same; R["identical_frac"] = same / len(prompts)
    # score T1 part with reference outputs
    c = 0
    for i, it in enumerate(t1):
        txt = tok.decode(o_ref[i], skip_special_tokens=False).split("<|im_end|>")[0].replace("<|endoftext|>", ""); c += int(score_item(txt, it)["correct"])
    R["t1_acc_ref"] = c / len(t1); R["secs"] = time.time() - t0
    diffs = [i for i in o_ref if o_ref[i] != o_new[i]][:10]; R["diff_examples"] = [(i, tok.decode(o_ref[i])[:120], tok.decode(o_new[i])[:120]) for i in diffs]
    json.dump(R, open("/vol/bonsai_gate1_result.json", "w")); vol.commit(); print("RESULT_JSON", json.dumps(R))
