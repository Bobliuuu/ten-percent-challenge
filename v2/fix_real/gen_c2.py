"""Rule-4/7 fix experiment for the C2 submission: REAL tool-call generation (Qwen native tool-call format) with the model's OWN lm_head over a
(pruned) vocabulary, no letter head, no feedback head. DEV ONLY (locked exam never read). Fake-quant evaluation (exact byte accounting like push_c2).
Deploy: modal deploy gen_c2.py ; launch: python3 launch.py cfgs.json   (results: RESULT_JSON in app logs + volume tenpct-fix:/r)
cfg: name, mode(base|quant), calib(letter|gen|mix), vocab(int, 0=full), ebits, hbits, step(1.20), dz, nev, general(bool)"""
import modal, json, os
app = modal.App("tenpct-fixreal")
HERE = os.path.dirname(os.path.abspath(__file__)); R_ = HERE + "/../"
img = (modal.Image.debian_slim().pip_install("torch", "transformers==4.51.3", "numpy", "accelerate", "datasets<4", "huggingface_hub")
       .add_local_file(R_ + "commercial/C2/dev.jsonl", "/root/dev.jsonl").add_local_file(R_ + "commercial/C2/calib.jsonl", "/root/calib.jsonl")
       .add_local_file(R_ + "commercial/C2/pool.jsonl", "/root/pool.jsonl"))
vol = modal.Volume.from_name("tenpct-fix", create_if_missing=True); hf = modal.Volume.from_name("tenpct-v2-hf")
INSTR = "You are an assistant with access to tools. Call the tool that satisfies the user request. If none of the provided tools can do it, do not call any tool and say that no tool fits."
BASE = "Qwen/Qwen2.5-7B-Instruct"
LET = "ABCDEFGHIJKLMN"

@app.function(image=img, gpu="A100-80GB", cpu=8, memory=98304, timeout=21600, volumes={"/r": vol, "/hf": hf}, env={"HF_HOME": "/hf", "HF_DATASETS_CACHE": "/tmp/dc"})
def run(cfg):
    import time, math, re, torch, torch.nn as nn, numpy as np, random
    from transformers import AutoTokenizer, AutoModelForCausalLM, LogitsProcessor, LogitsProcessorList
    t0 = time.time(); mode = cfg.get("mode", "quant"); tok = AutoTokenizer.from_pretrained(BASE); tok.padding_side = "left"
    rd = lambda f: [json.loads(x) for x in open("/root/" + f)]
    DEV, CAL, POOL = rd("dev.jsonl"), rd("calib.jsonl"), rd("pool.jsonl")
    nev = cfg.get("nev", 0); ev = DEV[:nev] if nev else DEV
    defs = {}
    for r in DEV + CAL + POOL:
        for x in json.loads(r["tools"]): defs.setdefault(x["name"], x)
    def tools_of(r): return [defs[n.split(":")[0].strip()] for n in r["names"][:-1]]
    def fmt_gen(r, answer=False):
        msgs = [{"role": "system", "content": INSTR}, {"role": "user", "content": r["text"]}]
        s = tok.apply_chat_template(msgs, tools=[{"type": "function", "function": t} for t in tools_of(r)], tokenize=False, add_generation_prompt=True)
        if answer:
            if r["irrelevant"]: s += "None of the provided tools can do this.<|im_end|>\n"
            else: s += "".join('<tool_call>\n' + json.dumps({"name": c["name"], "arguments": c["arguments"]}) + '\n</tool_call>\n' for c in json.loads(r["gold_calls"])) + "<|im_end|>\n"
        return s
    def fmt_letter(r, ans=False):
        s = r["inst"] + "\n\nText: " + r["text"] + "\n\nOptions:\n"
        for l, c in zip(LET, r["names"]): s += f"{l}. {c}\n"
        return s + "Answer:" + (" " + LET[r["answer"]] if ans else "")
    model = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16).cuda().eval(); V, H = model.get_input_embeddings().weight.shape
    letter_ids = [tok.encode(" " + l)[0] for l in LET]
    remap = lambda ids: ids; km = None; nkeep = V; oov = [0, 0]
    if mode == "quant":
        keepN = cfg.get("vocab", 24000)
        if keepN:
            cnt = torch.zeros(V)
            for r in POOL[:6000]:
                for s in (fmt_gen(r, True), fmt_letter(r, True)):
                    ids = tok.encode(s); cnt.index_add_(0, torch.tensor(ids), torch.ones(len(ids)))
            keep = set(cnt.topk(keepN)[1].tolist())
            vinv = tok.convert_ids_to_tokens(list(range(len(tok))))
            for t_, i in tok.get_vocab().items():
                if len(t_) == 1: keep.add(i)
            keep |= set(letter_ids) | set(tok.all_special_ids) | set(range(151643, 151665))
            km = torch.zeros(V, dtype=torch.bool); km[list(keep)] = True; nkeep = int(km.sum())
            kstr = {vinv[i]: i for i in keep if i < len(vinv)}; maxl = max(len(k) for k in kstr)
            def remap(ids):
                out = []
                for i in ids:
                    oov[1] += 1
                    if i < V and km[i]: out.append(i); continue
                    oov[0] += 1; t_ = vinv[i] if i < len(vinv) else None
                    if t_ is None: out.append(i); continue
                    p = 0
                    while p < len(t_):
                        for q in range(min(len(t_), p + maxl), p, -1):
                            if t_[p:q] in kstr: out.append(kstr[t_[p:q]]); p = q; break
                        else: out.append(i); p += 1
                return out
        # ---- calibration stream
        rng = np.random.RandomState(0); cs = cfg.get("calib", "gen"); txt = []
        if cs in ("letter", "mix"): txt += [fmt_letter(r, True) + "\n\n" for r in CAL]
        if cs in ("gen", "mix"): txt += [fmt_gen(r, True) + "\n\n" for r in CAL]
        rng.shuffle(txt); stream = []
        for s in txt: stream += tok.encode(s)
        ncal, L_ = cfg.get("ncal", 384), cfg.get("callen", 512); stream = torch.tensor(stream[:ncal * L_]).view(-1, L_)[:ncal]
        model.config.use_cache = False; layers = model.model.layers
        ins = []; kw = {}
        class C(nn.Module):
            def __init__(s, m): super().__init__(); s.m = m
            def forward(s, x, *a, **k): ins.append(x); kw.update(k); raise ValueError
        layers[0] = C(layers[0])
        for i in range(len(stream)):
            try: model(stream[i:i + 1].cuda())
            except ValueError: pass
        layers[0] = layers[0].m
        g = 128; K_ = 7; DZ = cfg.get("dz", 0.15); cc = cfg.get("step", 1.20); HB = 128
        def qparams_ec(x):
            rms = x.pow(2).mean(1, keepdim=True).sqrt(); s = (cc * rms).clamp(min=1e-6)
            return torch.exp2(torch.round(torch.log2(s) * 16).clamp(-200, 55) / 16)
        def qidx_ec(x, s):
            u = x / s; qi = torch.sign(u) * torch.floor(u.abs() + 0.5 - DZ).clamp(min=0, max=K_); return qi * s, qi + K_
        PGRID = torch.tensor([1.0, 0.9, 0.8, 0.7, 0.62, 0.55, 0.5, 0.45, 0.4, 0.36, 0.32, 0.28, 0.25]).cuda()
        def qparams(x, Lv):
            half = (Lv - 1) / 2; am = x.abs().max(1, keepdim=True)[0].clamp(min=1e-8); best = None
            for p in PGRID:
                s = (am * p / half).half().float().clamp(min=1e-8); q = (torch.clamp(torch.round(x / s + half), 0, Lv - 1) - half) * s
                e = ((q - x) ** 2).sum(1, keepdim=True)
                if best is None: best = (e, s)
                else: m = e < best[0]; best = (torch.where(m, e, best[0]), torch.where(m, s, best[1]))
            return best[1]
        def qidx(x, s, Lv): half = (Lv - 1) / 2; qi = torch.clamp(torch.round(x / s + half), 0, Lv - 1); return (qi - half) * s
        Hm = torch.ones(1, 1)
        while Hm.shape[0] < HB: Hm = torch.cat([torch.cat([Hm, Hm], 1), torch.cat([Hm, -Hm], 1)], 0)
        Hb = (Hm / math.sqrt(HB)).cuda()
        rot_cols = lambda M, sg: ((M * sg).view(M.shape[0], -1, HB) @ Hb).reshape(M.shape)
        unrot_cols = lambda M, sg: (M.view(M.shape[0], -1, HB) @ Hb).reshape(M.shape) * sg
        def gptq(W, Hh, blocksize=128, damp=0.01):
            W = W.float().clone(); R, C = W.shape; Hh = Hh.clone(); dead = torch.diag(Hh) == 0; Hh[dead, dead] = 1; W[:, dead] = 0
            Hh = Hh + damp * torch.mean(torch.diag(Hh)) * torch.eye(C, device=Hh.device)
            Hinv = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(Hh)), upper=True)
            Q = torch.zeros_like(W); hist = torch.zeros(2 * K_ + 1, device=W.device); s = None
            for i1 in range(0, C, blocksize):
                i2 = min(i1 + blocksize, C); W1 = W[:, i1:i2].clone(); Q1 = torch.zeros_like(W1); Err1 = torch.zeros_like(W1); Hi = Hinv[i1:i2, i1:i2]
                for i in range(i2 - i1):
                    if (i1 + i) % g == 0: s = qparams_ec(torch.cat([W1[:, i:], W[:, i2:]], 1)[:, :g])
                    w = W1[:, i]; d = Hi[i, i]; q, qi = qidx_ec(w.unsqueeze(1), s); q = q.flatten()
                    hist += torch.bincount(qi.long().flatten(), minlength=hist.numel()).float(); Q1[:, i] = q; err = (w - q) / d
                    W1[:, i:] -= err.unsqueeze(1) @ Hi[i, i:].unsqueeze(0); Err1[:, i] = err
                Q[:, i1:i2] = Q1; W[:, i2:] -= Err1 @ Hinv[i1:i2, i2:]
            return Q, hist
        ent = lambda h: float(-((h / h.sum())[h > 0] * (h / h.sum())[h > 0].log2()).sum())
        linn = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]
        groups = [("self_attn.q_proj", ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"]), ("self_attn.o_proj", ["self_attn.o_proj"]),
                  ("mlp.gate_proj", ["mlp.gate_proj", "mlp.up_proj"]), ("mlp.down_proj", ["mlp.down_proj"])]
        stats = dict(nq=0, ent_bits=0.0, scale_bits=0.0); cur = [x.clone() for x in ins]
        for oli, l in enumerate(layers):
            mods = {n: l.get_submodule(n) for n in linn}; Hs = {}; nsm = {}
            def mkh(key):
                def hook(m, i, o):
                    x = i[0].reshape(-1, i[0].shape[-1]).float(); nsm[key] = nsm.get(key, 0) + x.shape[0]; Hs[key] = Hs.get(key, 0) + 2 * x.T @ x
                return hook
            hds = [mods[k].register_forward_hook(mkh(k)) for k, _ in groups]
            with torch.no_grad():
                for x in cur: l(x, **kw)
            for hh in hds: hh.remove()
            for key, members in groups:
                Hh = Hs[key] / nsm[key]; C = Hh.shape[0]
                sg = (torch.randint(0, 2, (C,), generator=torch.Generator().manual_seed(1000 * oli + len(key)), dtype=torch.float32) * 2 - 1).cuda()
                H1 = rot_cols(Hh, sg); Hr = rot_cols(H1.T.contiguous(), sg).T.contiguous()
                for n in members:
                    m = mods[n]; W = m.weight.data.float(); Q, hist = gptq(rot_cols(W, sg), Hr)
                    m.weight.data = unrot_cols(Q, sg).to(torch.bfloat16); nn_ = W.numel()
                    stats["nq"] += nn_; stats["ent_bits"] += nn_ * ent(hist); stats["scale_bits"] += nn_ / g * 8
            del Hs
            with torch.no_grad():
                for j, x in enumerate(cur): o = l(x, **kw); cur[j] = o[0] if isinstance(o, tuple) else o
            print("layer", oli, round(time.time() - t0), flush=True)
        del cur, ins; torch.cuda.empty_cache()
        # ---- embedding AND original lm_head rows (4-bit by default), vocab = kept rows only; NO new head, NO letter head
        def qrows(w, bits):
            w = w.float(); wq = torch.zeros_like(w); Lv = 2 ** bits
            for c in range(0, w.shape[1], 64): blk = w[:, c:c + 64]; s = qparams(blk, Lv); wq[:, c:c + 64] = qidx(blk, s, Lv)
            return wq.to(torch.bfloat16)
        ebits, hbits = cfg.get("ebits", 4), cfg.get("hbits", 4)
        model.get_input_embeddings().weight.data = qrows(model.get_input_embeddings().weight.data, ebits)
        model.get_output_embeddings().weight.data = qrows(model.get_output_embeddings().weight.data, hbits)
        model.config.use_cache = True
        lin_total = sum(l.get_submodule(n).weight.numel() for l in layers for n in linn); other = sum(p.numel() for l in layers for p in l.parameters()) - lin_total
        body_bits = stats["ent_bits"] * 1.005 + stats["scale_bits"]
        tabbits = lambda b: nkeep * H * b + nkeep * H / 64 * 16
        misc_bits = (other + model.model.norm.weight.numel()) * 16 + 1_000_000 * 8
        tot = (body_bits + tabbits(ebits) + tabbits(hbits) + misc_bits) / 8; base_bytes = 15231233024
        acct = dict(bytes=tot, ratio=tot / base_bytes, MB=dict(body=body_bits / 8e6, emb=tabbits(ebits) / 8e6, head=tabbits(hbits) / 8e6, misc=misc_bits / 8e6), body_bpw=body_bits / lin_total, nkeep=nkeep)
        print("ACCT", json.dumps(acct), flush=True)
    else: acct = dict(bytes=15231233024, ratio=1.0)
    class KeepMask(LogitsProcessor):
        def __call__(s, ids, sc):
            if km is not None: sc[:, ~km.to(sc.device)] = -1e30
            return sc
    @torch.no_grad()
    def generate(rows, bs=16, mx=160):
        enc = [remap(tok.encode(fmt_gen(r)))[-3000:] for r in rows]; order = np.argsort([-len(e) for e in enc]); outs = [None] * len(rows)
        for b in range(0, len(rows), bs):
            ix = order[b:b + bs]; m = max(len(enc[i]) for i in ix)
            x = torch.full((len(ix), m), tok.pad_token_id, dtype=torch.long); a = torch.zeros(len(ix), m, dtype=torch.long)
            for j, i in enumerate(ix): x[j, m - len(enc[i]):] = torch.tensor(enc[i]); a[j, m - len(enc[i]):] = 1
            o = model.generate(input_ids=x.cuda(), attention_mask=a.cuda(), max_new_tokens=mx, do_sample=False, temperature=None, top_p=None, top_k=None,
                               logits_processor=LogitsProcessorList([KeepMask()]), pad_token_id=tok.pad_token_id)
            for j, i in enumerate(ix): outs[i] = tok.decode(o[j, m:], skip_special_tokens=False)
            if (b // bs) % 20 == 0: print("gen", b, len(rows), round(time.time() - t0), flush=True)
        return outs
    def parse(s):
        calls = []
        for m in re.finditer(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", s, re.S):
            try: d = json.loads(m.group(1)); calls.append((d.get("name"), d.get("arguments", {})))
            except Exception: calls.append((None, None))
        return calls
    def norm(v): return json.dumps(v, sort_keys=True) if not isinstance(v, str) else v.strip().lower() if False else str(v).strip().lower() if not isinstance(v, (dict, list)) else json.dumps(v, sort_keys=True)
    def canon(c): return (c[0], json.dumps({k: norm(v) for k, v in (c[1] or {}).items()}, sort_keys=True) if isinstance(c[1], dict) else None)
    def score(rows, outs):
        res = []
        for r, o in zip(rows, outs):
            calls = parse(o); gold = [(c["name"], c["arguments"]) for c in json.loads(r["gold_calls"])]
            if r["irrelevant"]: name_ok = len(calls) == 0; exact = name_ok; allname = name_ok
            else:
                name_ok = len(calls) > 0 and calls[0][0] == gold[0][0]; allname = sorted(str(c[0]) for c in calls) == sorted(str(c[0]) for c in gold)
                exact = sorted(map(str, map(canon, calls))) == sorted(map(str, map(canon, gold)))
            res.append((int(name_ok), int(allname), int(exact), int(len(calls) > 0 and not r["irrelevant"] and calls[0][0] in [n.split(":")[0].strip() for n in r["names"][:-1]])))
        return np.array(res)
    outs = generate(ev); os.makedirs("/r/r", exist_ok=True); json.dump(outs, open(f"/r/r/{cfg['name']}_raw.json", "w")); vol.commit(); S = score(ev, outs); irr = np.array([r["irrelevant"] for r in ev])
    R = dict(cfg=cfg, n=len(ev), name_acc=float(S[:, 0].mean()), allname_acc=float(S[:, 1].mean()), exact_call_acc=float(S[:, 2].mean()), name_acc_relevant=float(S[~irr, 0].mean()), name_acc_irrelevant=float(S[irr, 0].mean()),
             exact_relevant=float(S[~irr, 2].mean()), acct=acct, oov=oov[0] / max(1, oov[1]), name_ok=S[:, 0].tolist(), sample=[(ev[i]["id"], outs[i]) for i in range(0, len(ev), 100)])
    print("GEN_RESULT", json.dumps({k: v for k, v in R.items() if k not in ("name_ok", "sample")}), flush=True)
    if cfg.get("general"):
        from datasets import load_dataset
        rng = random.Random(0); sets = {}; INST = "Answer the multiple-choice question by picking the single best option."
        mk = lambda q, ch, a: dict(task="G", text=q, names=list(ch), inst=INST, answer=int(a))
        def tl(name, fn):
            try: rows = list(fn()); rng.shuffle(rows); sets[name] = rows[:1000]
            except Exception as e: print("ERR", name, repr(e)[:200])
        tl("mmlu", lambda: [mk(r["question"], r["choices"], r["answer"]) for r in load_dataset("cais/mmlu", "all", split="test")])
        tl("arc_challenge", lambda: [mk(r["question"], r["choices"]["text"], r["choices"]["label"].index(r["answerKey"])) for r in load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test") if r["answerKey"] in r["choices"]["label"]])
        tl("hellaswag", lambda: [mk(r["ctx"], r["endings"], r["label"]) for r in load_dataset("Rowan/hellaswag", split="validation")])
        def tqa():
            out = []
            for r in load_dataset("truthful_qa", "multiple_choice", split="validation"):
                ch = r["mc1_targets"]["choices"]
                if len(ch) <= 5: p = list(range(len(ch))); rng.shuffle(p); out.append(mk(r["question"], [ch[i] for i in p], p.index(r["mc1_targets"]["labels"].index(1))))
            return out
        tl("truthfulqa_mc1", tqa)
        W = model.get_output_embeddings().weight[letter_ids]; G = {}
        with torch.no_grad():
            for k, rows in sets.items():
                enc = [remap(tok.encode(fmt_letter(r)))[-1024:] for r in rows]; order = np.argsort([-len(e) for e in enc]); pr = np.zeros(len(rows), int)
                for b in range(0, len(rows), 16):
                    ix = order[b:b + 16]; m = max(len(enc[i]) for i in ix); x = torch.zeros(len(ix), m, dtype=torch.long); a = torch.zeros(len(ix), m, dtype=torch.long)
                    for j, i in enumerate(ix): x[j, :len(enc[i])] = torch.tensor(enc[i]); a[j, :len(enc[i])] = 1
                    hs = model.model(input_ids=x.cuda(), attention_mask=a.cuda()).last_hidden_state; last = a.sum(1) - 1
                    lg = (hs[torch.arange(len(ix)), last.cuda()] @ W.T).float().cpu().numpy()
                    for j, i in enumerate(ix): pr[i] = int(np.argmax(lg[j, :len(rows[i]["names"])]))
                G[k] = dict(correct=(pr == np.array([r["answer"] for r in rows])).tolist())
        R["general"] = G; print("GENERAL", json.dumps({k: float(np.mean(v["correct"])) for k, v in G.items()}), flush=True)
    R["secs"] = time.time() - t0; os.makedirs("/r/r", exist_ok=True); json.dump(R, open(f"/r/r/{cfg['name']}.json", "w")); vol.commit()
    print("RESULT_JSON", json.dumps({k: v for k, v in R.items() if k not in ("name_ok", "sample", "general")}), flush=True); return R["name_acc"]
