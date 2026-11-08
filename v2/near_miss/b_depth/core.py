"""Depth prune (BI / task-loss sensitivity) + RTN-group quant + task vocab trim, fake-quant (dequantized bf16) for vLLM eval, exact byte accounting."""
import os, json, re, glob, math, random, subprocess, shutil, time, sys
import torch
D = "/code/data"; WORK = "/work"
def snap(model):
    from huggingface_hub import snapshot_download
    return snapshot_download(model, allow_patterns=["*.json", "*.safetensors", "*.txt", "merges.txt", "vocab.json", "*.model"])
def load_texts(tok):
    """calibration material, DISJOINT from eval ids. returns list of (prompt_text, answer_text or None, raw_text_for_vocab)"""
    ev = {json.loads(l)["id"] for l in open(D + "/probe_items.jsonl")}
    TOXP = open(D + "/tox_prompt.txt").read(); INJP = open(D + "/inj_prompt.txt").read()
    cal = []
    for l in open(D + "/c_t1.jsonl"):
        r = json.loads(l)
        if r["id"] in ev: continue
        ans = "\n".join("<tool_call>\n" + json.dumps(dict(name=g["name"], arguments=g["arguments"])) + "\n</tool_call>" for g in r["gold"]) or None
        cal.append(("t1", r, ans))
    for l in open(D + "/c_tox.jsonl"):
        r = json.loads(l); cal.append(("tox", r, r["gold"]))
    for l in open(D + "/c_inj.jsonl"):
        r = json.loads(l); cal.append(("inj", r, "Injection" if r["gold"] != "benign" else "Benign"))
    return cal, TOXP, INJP
def render(tok, k, r, TOXP, INJP):
    def tpl(msgs, tools=None):
        try: return tok.apply_chat_template(msgs, tools=tools, add_generation_prompt=True, tokenize=False, enable_thinking=False)
        except TypeError: return tok.apply_chat_template(msgs, tools=tools, add_generation_prompt=True, tokenize=False)
    if k == "t1": return tpl([{"role": "user", "content": r["query"]}], [{"type": "function", "function": t} for t in r["tools"]])
    return tpl([{"role": "user", "content": (TOXP if k == "tox" else INJP).replace("{t}", r["text"])}])
def sensitivity(model, path, tok, cal, TOXP, INJP, ncal=160, seed=0):
    from transformers import AutoModelForCausalLM
    m = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16).cuda().eval()
    rnd = random.Random(seed); sub = [c for c in cal if c[2]]; rnd.shuffle(sub); sub = sub[:ncal]
    seqs = []
    for k, r, ans in sub:
        p = render(tok, k, r, TOXP, INJP); pi = tok(p, add_special_tokens=False)["input_ids"]; ai = tok(ans + "<|im_end|>", add_special_tokens=False)["input_ids"]
        ids = (pi + ai)[-1500:]; seqs.append((ids, len(ai)))
    L = len(m.model.layers); bi = torch.zeros(L, device="cuda"); ntok = 0
    def nll(drop=None):
        hs = []
        if drop is not None:
            def hook(mod, args, kwargs, out):
                h = args[0] if args else kwargs["hidden_states"]
                return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
            hd = m.model.layers[drop].register_forward_hook(hook, with_kwargs=True)
        tot = 0.0; n = 0
        with torch.no_grad():
            for ids, na in seqs:
                x = torch.tensor([ids]).cuda(); o = m(x, output_hidden_states=(drop is None))
                lg = o.logits[0, -na - 1:-1].float(); y = x[0, -na:]
                tot += torch.nn.functional.cross_entropy(lg, y, reduction="sum").item(); n += na
                if drop is None:
                    H = o.hidden_states
                    for l in range(L): bi[l] += (1 - torch.nn.functional.cosine_similarity(H[l][0].float(), H[l + 1][0].float(), dim=-1)).sum()
                    global_ntok[0] += x.shape[1]
        if drop is not None: hd.remove()
        return tot / n
    global_ntok = [0]
    base = nll(None); biv = (bi / global_ntok[0]).tolist()
    loss = [nll(l) - base for l in range(L)]
    del m; torch.cuda.empty_cache()
    return dict(base_nll=base, bi=biv, loss=loss, n_seq=len(seqs))
def keep_vocab(tok, cal, TOXP, INJP, V):
    seen = set()
    for k, r, ans in cal:
        seen.update(tok(render(tok, k, r, TOXP, INJP), add_special_tokens=False)["input_ids"])
        if ans: seen.update(tok(ans + "<|im_end|>", add_special_tokens=False)["input_ids"])
    seen.update(tok(" \n\t0123456789", add_special_tokens=False)["input_ids"])
    for i in range(256):  # all single-byte tokens incl. byte-level chars
        pass
    from transformers.models.gpt2.tokenization_gpt2 import bytes_to_unicode
    b2u = bytes_to_unicode(); vocab = tok.get_vocab()
    for u in b2u.values():
        if u in vocab: seen.add(vocab[u])
    for t, i in tok.get_added_vocab().items(): seen.add(i)
    return sorted(i for i in seen if i < V)
def quant_rtn(w, bits, g):
    """asymmetric min-max RTN with per-group MSE clip search; returns dequantized tensor (same dtype)."""
    o, i = w.shape; assert i % g == 0
    x = w.float().reshape(-1, g); qmax = 2 ** bits - 1; mx = x.amax(1, keepdim=True); mn = x.amin(1, keepdim=True); mid = (mx + mn) / 2; half = (mx - mn) / 2
    best = None; bestd = None
    for c in (1.0, 0.92, 0.85, 0.78, 0.7, 0.62):
        h = half * c; s = (2 * h / qmax).clamp_min(1e-8); lo = mid - h; zp = torch.round(-lo / s).clamp(0, qmax)
        d = (torch.clamp(torch.round(x / s) + zp, 0, qmax) - zp) * s; e = ((d - x) ** 2).sum(1, keepdim=True)
        if best is None: best, bestd = e, d
        else:
            m = e < best; best = torch.where(m, e, best); bestd = torch.where(m, d, bestd)
    return bestd.reshape(o, i).to(w.dtype)
def build(path, cfg, sens, keep, outdir, tok):
    """write fake-quant model dir. returns byte accounting."""
    from safetensors import safe_open
    from safetensors.torch import save_file
    conf = json.load(open(path + "/config.json")); L = conf["num_hidden_layers"]; tied = conf.get("tie_word_embeddings", False)
    n = cfg.get("drop", 0); score = sens[cfg.get("sel", "loss")] if n else None
    dropped = sorted(sorted(range(L), key=lambda l: score[l])[:n]) if n else []
    if cfg.get("tail"): dropped = list(range(L - n, L))  # contiguous-from-end control
    kept = [l for l in range(L) if l not in dropped]; remap = {l: j for j, l in enumerate(kept)}
    bits, g, eb = cfg.get("bits", 16), cfg.get("g", 64), cfg.get("ebits", 4); vtrim = cfg.get("vtrim", False)
    os.makedirs(outdir, exist_ok=True); out = {}; byt = dict(body_q=0, body_fp16=0, embed=0, head=0, scales=0)
    V = conf["vocab_size"]; d = conf["hidden_size"]; kt = torch.tensor(keep).cuda() if vtrim else None
    base_params = 0
    for f in sorted(glob.glob(path + "/*.safetensors")):
        with safe_open(f, "pt") as sf:
            for name in sf.keys():
                w = sf.get_tensor(name); base_params += w.numel()
                m = re.match(r"model\.layers\.(\d+)\.(.*)", name)
                if m:
                    l = int(m.group(1))
                    if l in dropped: continue
                    nn_ = f"model.layers.{remap[l]}.{m.group(2)}"
                    if w.ndim == 2 and bits < 16:
                        out[nn_] = quant_rtn(w.cuda(), bits, g).cpu(); byt["body_q"] += w.numel() * bits / 8; byt["scales"] += w.numel() / g * (2 + bits / 8)
                    else:
                        out[nn_] = w; byt["body_fp16" if w.ndim == 2 else "body_fp16"] += w.numel() * 2
                elif name in ("model.embed_tokens.weight", "lm_head.weight"):
                    wc = w.cuda()
                    if vtrim:
                        z = torch.zeros_like(wc); z[kt] = wc[kt]; wc = z
                    if eb < 16:
                        if vtrim: wk = quant_rtn(wc[kt], eb, 64); wc = torch.zeros_like(wc); wc[kt] = wk
                        else: wc = quant_rtn(wc, eb, 64)
                    nrows = len(keep) if vtrim else V; by = nrows * d * (eb / 8 if eb < 16 else 2) + (nrows * d / 64 * (2 + eb / 8) if eb < 16 else 0)
                    byt["embed" if "embed" in name else "head"] += by; out[name] = wc.cpu()
                else:
                    out[name] = w; byt["body_fp16"] += w.numel() * 2
    if tied and "lm_head.weight" not in out: pass
    byt["tied"] = tied
    if "embed" in byt and tied: byt["head"] = 0
    save_file({k: v.contiguous() for k, v in out.items()}, outdir + "/model.safetensors", metadata={"format": "pt"})
    conf["num_hidden_layers"] = len(kept)
    if "layer_types" in conf: conf["layer_types"] = [conf["layer_types"][l] for l in kept]
    json.dump(conf, open(outdir + "/config.json", "w"))
    for fn in os.listdir(path):
        if fn.endswith(".safetensors") or fn in ("config.json", "model.safetensors.index.json"): continue
        if os.path.isfile(path + "/" + fn): shutil.copy(path + "/" + fn, outdir + "/" + fn)
    # tokenizer bytes (trimmed): vocab entries + merges between kept tokens + ~8KB template/config
    vocab = tok.get_vocab(); inv = {i: t for t, i in vocab.items()}
    if vtrim:
        ks = set(keep); tv = {inv[i]: j for j, i in enumerate(keep) if i in inv}
        mg = [l for l in open(path + "/merges.txt", encoding="utf8").read().split("\n")[1:] if l and len(l.split(" ")) == 2 and "".join(l.split(" ")) in tv and l.split(" ")[0] in tv and l.split(" ")[1] in tv]
        tokb = len(json.dumps(tv, ensure_ascii=False, separators=(",", ":")).encode()) + len("\n".join(mg).encode()) + 8192
    else: tokb = sum(os.path.getsize(path + "/" + f) for f in ("vocab.json", "merges.txt") if os.path.exists(path + "/" + f)) + 8192
    tot_params = base_params
    byt["tokenizer"] = tokb; byt["layers_kept"] = len(kept); byt["dropped"] = dropped
    byt["total_bytes"] = sum(v for k, v in byt.items() if k in ("body_q", "body_fp16", "embed", "head", "scales", "tokenizer"))
    byt["base_fp16_bytes"] = 2 * base_params; byt["size_pct"] = 100 * byt["total_bytes"] / byt["base_fp16_bytes"]
    return byt
