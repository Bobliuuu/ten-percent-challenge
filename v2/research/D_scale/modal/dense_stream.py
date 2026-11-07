"""D_scale: layer-streaming version of compress/dense/dense.py for >=32B dense models (RAM 64GB, GPU 96GB shared).
Same maths as dense.py (Hadamard128 rotation + GPTQ onto L-level grids, per-module L from RTN sensitivity table by Lagrangian,
rANS size accounting = entropy*1.005 + 11/g bits), but weights are read per layer from safetensors, quantised codes are
saved to NVMe (4-bit packed + fp16 scales), and evaluation runs with on-the-fly dequant (QLinear) so a 70B artifact fits one GPU.
NO optimizer / backward anywhere. Dev data only.
usage: python dense_stream.py CFG.json   (CFG = list of dicts; keys: model_dir, tag, mode in {sens,quant,eval,base,all}, name, target_ratio, ...)"""
import os, sys, json, math, time, glob
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from collections import Counter
from safetensors import safe_open

WORK = os.environ.get("DS_WORK", "/opt/dlami/nvme/work/D_scale")
T1DIR = os.environ.get("DENSE_T1", WORK + "/t1")
DATA = os.environ.get("DENSE_DATA", WORK + "/dense_data.json")
HB = 128
LS = [2, 3, 4, 5, 6, 8, 12, 16]
PG = [1.0, .9, .8, .7, .62, .55, .5, .45, .4, .36, .32, .28, .25, .22, .19, .16, .13]
linn = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]
groups = [("self_attn.q_proj", ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"]), ("self_attn.o_proj", ["self_attn.o_proj"]),
          ("mlp.gate_proj", ["mlp.gate_proj", "mlp.up_proj"]), ("mlp.down_proj", ["mlp.down_proj"])]


class ST:
    def __init__(s, path):
        s.path = path
        if os.path.exists(path + "/model.safetensors.index.json"):
            idx = json.load(open(path + "/model.safetensors.index.json")); s.wm = idx["weight_map"]; s.total = idx["metadata"]["total_size"]
        else:
            f = "model.safetensors"; h = safe_open(path + "/" + f, "pt"); s.wm = {k: f for k in h.keys()}
            s.total = sum(int(np.prod(h.get_slice(k).get_shape())) * 2 for k in h.keys())
        s.h = {}
    def get(s, n, dev="cuda"):
        f = s.wm[n]
        if f not in s.h: s.h[f] = safe_open(s.path + "/" + f, "pt", device="cpu")
        return s.h[f].get_tensor(n).to(dev)
    def has(s, n): return n in s.wm


def ent(h):
    p = h / h.sum(); p = p[p > 0]; return float(-(p * p.log2()).sum())


class Quantiser:
    def __init__(s, g, dz=0.0):
        s.g = g; s.dz = dz; s.PGt = torch.tensor(PG).cuda()
        Hm = torch.ones(1, 1)
        while Hm.shape[0] < HB: Hm = torch.cat([torch.cat([Hm, Hm], 1), torch.cat([Hm, -Hm], 1)], 0)
        s.Hb = (Hm / math.sqrt(HB)).cuda()
    def quant(s, x, sc, Lv):
        half = (Lv - 1) / 2; t = x / sc
        qi = torch.clamp(torch.round(t + half), 0, Lv - 1); return (qi - half) * sc, qi
    def qparams(s, x, Lv):
        half = (Lv - 1) / 2; am = x.abs().amax(1, keepdim=True).clamp(min=1e-8); best = None
        for p in s.PGt:
            sc = (am * p / max(half, 0.5)).half().float().clamp(min=1e-8); q = s.quant(x, sc, Lv)[0]; e = ((q - x) ** 2).sum(1, keepdim=True)
            if best is None: best = (e, sc)
            else: m = e < best[0]; best = (torch.where(m, e, best[0]), torch.where(m, sc, best[1]))
        return best[1]
    def rot(s, M, sg): R, C = M.shape; return ((M * sg).view(R, C // HB, HB) @ s.Hb).reshape(R, C)
    def unrot(s, M, sg): R, C = M.shape; return ((M.view(R, C // HB, HB) @ s.Hb).reshape(R, C)) * sg
    def rtn(s, W, Lv, g=None):
        g = g or s.g; R, C = W.shape; out = torch.empty_like(W); hist = torch.zeros(Lv, device=W.device); ch = max(1, int(2e7 // C))
        for r in range(0, R, ch):
            w = W[r:r + ch]; x = w.reshape(-1, g); sc = s.qparams(x, Lv); q, qi = s.quant(x, sc, Lv); out[r:r + ch] = q.view_as(w)
            hist += torch.bincount(qi.long().flatten(), minlength=Lv).float()
        return out, hist
    def gptq(s, W, H, Lv, damp=0.01):
        g = s.g; W = W.float().clone(); R, C = W.shape; H = H.clone(); bsz = max(128, g)
        dead = torch.diag(H) == 0; H[dead, dead] = 1; W[:, dead] = 0
        H = H + damp * torch.mean(torch.diag(H)) * torch.eye(C, device=H.device)
        Hinv = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(H)), upper=True); del H
        Q = torch.zeros_like(W); hist = torch.zeros(Lv, device=W.device); sc = None
        QI = torch.zeros(R, C, dtype=torch.uint8, device=W.device); SC = torch.zeros(R, C // g, dtype=torch.float16, device=W.device)
        for i1 in range(0, C, bsz):
            i2 = min(i1 + bsz, C); W1 = W[:, i1:i2].clone(); Q1 = torch.zeros_like(W1); E1 = torch.zeros_like(W1); Hi = Hinv[i1:i2, i1:i2]
            for i in range(i2 - i1):
                if i % g == 0: sc = s.qparams(W1[:, i:i + g], Lv); SC[:, (i1 + i) // g] = sc.flatten().half()
                w = W1[:, i]; d = Hi[i, i]; q, qi = s.quant(w.unsqueeze(1), sc, Lv); q = q.flatten()
                hist += torch.bincount(qi.long().flatten(), minlength=Lv).float(); QI[:, i1 + i] = qi.flatten().to(torch.uint8)
                Q1[:, i] = q; err = (w - q) / d; W1[:, i:] -= err.unsqueeze(1) @ Hi[i, i:].unsqueeze(0); E1[:, i] = err
            Q[:, i1:i2] = Q1; W[:, i2:] -= E1 @ Hinv[i1:i2, i2:]
        return Q, hist, QI, SC


def signs_kl(oli, key, C): return (torch.randint(0, 2, (C,), generator=torch.Generator().manual_seed(1000 * oli + len(key)), dtype=torch.float32) * 2 - 1).cuda()


def pack4(QI): return (QI[:, 0::2] | (QI[:, 1::2] << 4)).contiguous()


class QLinear(nn.Module):
    """weight = unrot((codes-half)*scale); dequantised on the fly each forward (4-bit packed storage on GPU)."""
    def __init__(s, R, C, g, Lv, Hb):
        super().__init__(); s.R, s.C, s.g, s.Lv = R, C, g, Lv; s.Hb16 = Hb.half(); s.codes = s.scales = s.sg = s.bias = None
    def weight(s):
        c = s.codes; q = torch.stack([c & 15, c >> 4], -1).view(s.R, s.C)
        Q = (q.half() - (s.Lv - 1) / 2) * s.scales.repeat_interleave(s.g, 1)
        W = (Q.view(s.R, s.C // HB, HB) @ s.Hb16).reshape(s.R, s.C)
        return (W.float() * s.sg).to(torch.bfloat16)
    def forward(s, x): return F.linear(x, s.wc if getattr(s, "wc", None) is not None else s.weight(), s.bias)
    def cache(s): s.wc = s.weight(); s.codes = None; s.scales = None


class WLinear(nn.Module):
    """bf16 weight on GPU or pinned CPU (streamed per forward) for the base model."""
    def __init__(s): super().__init__(); s.w = None; s.bias = None
    def forward(s, x):
        w = s.w if s.w.is_cuda else s.w.to("cuda", non_blocking=True)
        return F.linear(x, w, s.bias)


def log(*a): print(time.strftime("%H:%M:%S"), *a, flush=True)


def run(cfg):
    from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM, GenerationConfig
    global T1DIR
    T1DIR = cfg.get('t1dir', T1DIR)
    t0 = time.time(); torch.manual_seed(0)
    MD = cfg["model_dir"]; tag = cfg["tag"]; mode = cfg.get("mode", "all"); name = cfg.get("name", "x")
    g = cfg.get("g", 256); GB = 11; dz = 0.0
    OUT = f"{WORK}/hf/{tag}"; os.makedirs(OUT, exist_ok=True)
    D = json.load(open(DATA)); st = ST(MD); conf = AutoConfig.from_pretrained(MD); conf._attn_implementation = "sdpa"
    tok = AutoTokenizer.from_pretrained(MD); tok.padding_side = "left"
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    nl = conf.num_hidden_layers; hid = conf.hidden_size; tied = conf.tie_word_embeddings
    base_params = st.total // 2; base_bytes = st.total
    Qz = Quantiser(g)
    with torch.device("meta"): meta = AutoModelForCausalLM.from_config(conf, dtype=torch.bfloat16)
    LayerCls = type(meta.model.layers[0]); RotCls = type(meta.model.rotary_emb)
    V = conf.vocab_size
    def new_rot(): return RotCls(config=conf, device="cuda")
    # ------------------------------------------------------------------ eval helpers
    def gen_raw(model, texts, newtok, bs, label_=None):
        enc = [tok.encode(t, add_special_tokens=False)[-12000:] for t in texts]; order = np.argsort([-len(e) for e in enc]); out = [None] * len(enc)
        for b in range(0, len(enc), bs):
            idx = order[b:b + bs]; mx = max(len(enc[i]) for i in idx)
            ids = torch.full((len(idx), mx), tok.pad_token_id); att = torch.zeros(len(idx), mx, dtype=torch.long)
            for j, i in enumerate(idx): ids[j, mx - len(enc[i]):] = torch.tensor(enc[i]); att[j, mx - len(enc[i]):] = 1
            with torch.no_grad():
                gg = model.generate(input_ids=ids.cuda(), attention_mask=att.cuda(), max_new_tokens=newtok, do_sample=False, use_cache=True,
                                    pad_token_id=tok.pad_token_id, temperature=None, top_p=None, top_k=None)
            for j, i in enumerate(idx): out[i] = tok.decode(gg[j, mx:], skip_special_tokens=False)
            json.dump(out, open(f"{OUT}/partial_{label_}.json", "w")) if label_ else None
            log("gen batch", b // bs + 1, "/", math.ceil(len(enc) / bs), "len", mx)
        return out
    def chat(p):
        try: return tok.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True, tokenize=False, enable_thinking=False)
        except TypeError: return tok.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True, tokenize=False)
    def enc_prompt(p): return tok.encode(chat(p), add_special_tokens=False)[-900:]
    def tf_items():
        return [(enc_prompt(t["prompt"]), tok.encode(t["ref"], add_special_tokens=False)[:150] + [tok.convert_tokens_to_ids("<|im_end|>")]) for t in D["tf"]]
    @torch.no_grad()
    def tf_logp(model, items, K=64):
        res = []
        for pi, ri in items:
            ids = torch.tensor([(pi + ri)[-640:]]).cuda(); P = len(ri)
            lg = model(input_ids=ids, use_cache=False).logits[0, -P - 1:-1].float(); lp = torch.log_softmax(lg, -1); tv, ti = lp.topk(K, -1)
            res.append((ti.cpu().numpy().astype(np.int32).tolist(), tv.cpu().numpy().astype(np.float32).tolist()))
        return res
    @torch.no_grad()
    def tf_cmp(model, items, btf):
        t1 = []; kl = []
        for (pi, ri), (bids, blp) in zip(items, btf):
            bids = torch.tensor(bids).cuda(); blp = torch.tensor(blp).cuda(); P = len(ri)
            lg = model(input_ids=torch.tensor([(pi + ri)[-640:]]).cuda(), use_cache=False).logits[0, -P - 1:-1].float(); lp = torch.log_softmax(lg, -1)
            t1 += (lp.argmax(-1) == bids[:, 0]).cpu().tolist()
            cl = lp.gather(1, bids); pb = blp.exp(); tb = (1 - pb.sum(1)).clamp(min=1e-9); tc = (1 - cl.exp().sum(1)).clamp(min=1e-9)
            kl += ((pb * (blp - cl)).sum(1) + tb * (tb.log() - tc.log())).cpu().tolist()
        return float(np.mean(t1)), float(np.mean(kl))
    def run_t1(model, label):
        sys.path.insert(0, T1DIR); from scorer import score_item
        items = [json.loads(l) for l in open(T1DIR + "/probe_items.jsonl")][::cfg.get('sub', 1)]; prompts = []
        for it in items:
            tools = [{"type": "function", "function": t} for t in it["tools"]]
            try: prompts.append(tok.apply_chat_template([{"role": "user", "content": it["query"]}], tools=tools, add_generation_prompt=True, tokenize=False, enable_thinking=False))
            except TypeError: prompts.append(tok.apply_chat_template([{"role": "user", "content": it["query"]}], tools=tools, add_generation_prompt=True, tokenize=False))
        txt = gen_raw(model, prompts, cfg.get("t1_new", 400), cfg.get("t1_bs", 24), label_=label); rows = []
        for it, t in zip(items, txt):
            s_ = score_item(t, it); rows.append(dict(id=it["id"], source=it["source"], kind=it["kind"], correct=bool(s_["correct"]), parse_fail=bool(s_["parse_fail"]), output=t[:600]))
        return rows
    def build_eval_model(qdir):
        """qdir None -> base bf16 (resident_layers on GPU, rest pinned CPU)"""
        with torch.device("meta"): m = AutoModelForCausalLM.from_config(conf, dtype=torch.bfloat16)
        nres = cfg.get("resident_layers", nl)
        for i, l in enumerate(m.model.layers):
            d = torch.load(f"{qdir}/L{i}.pt") if qdir else None
            for n in linn:
                par = l.get_submodule(n.rsplit(".", 1)[0]); leaf = n.rsplit(".", 1)[1]; lin = getattr(par, leaf)
                if qdir:
                    e = d[n]; R, C = lin.out_features, lin.in_features
                    q = QLinear(R, C, g, e["L"], Qz.Hb); q.codes = e["codes"].cuda(); q.scales = e["scales"].cuda(); q.sg = e["sg"].cuda().float()
                else:
                    q = WLinear()
                setattr(par, leaf, q)
        m.to_empty(device="cuda")
        for i, l in enumerate(m.model.layers):
            for n in linn:
                q = l.get_submodule(n)
                if qdir and i < cfg.get("deq_layers", 0): q.cache()
                wname = f"model.layers.{i}.{n}"
                if not qdir:
                    w = st.get(wname + ".weight", "cpu")
                    q.w = w.cuda() if i < nres else w.pin_memory()
                if st.has(wname + ".bias"): q.bias = st.get(wname + ".bias")
        for n, p in m.named_parameters():
            if n in ("model.embed_tokens.weight", "lm_head.weight") and qdir:
                p.data.copy_(torch.load(qdir + ("/emb.pt" if n.startswith("model.embed") else "/head.pt")).cuda()); continue
            if n == "lm_head.weight" and tied: continue
            p.data.copy_(st.get(n))
        if tied: m.tie_weights()
        m.model.rotary_emb = new_rot(); m.eval(); m.generation_config = GenerationConfig.from_pretrained(MD); m.config.use_cache = True
        return m
    # ------------------------------------------------------------------ base mode
    if mode == "base":
        m = build_eval_model(None)
        if not cfg.get("tf_only"):
            rows = run_t1(m, "base")
            json.dump(dict(cfg=cfg, model=MD, t1=rows, secs=time.time() - t0), open(f"{OUT}/t1_bf16{cfg.get('suffix','')}.json", "w"))
            log("T1_BASE_DONE", sum(r["correct"] for r in rows), len(rows))
        m.config.use_cache = False
        tf = tf_logp(m, tf_items()); json.dump(dict(tf=tf, base_bytes=base_bytes, base_params=base_params), open(f"{OUT}/base_tf.json", "w")); log("BASE_TF_DONE"); return
    # ------------------------------------------------------------------ eval mode
    qdir = f"{OUT}/q_{name}"
    if mode == "eval":
        meta_ = json.load(open(qdir + "/meta.json")); m = build_eval_model(qdir); rows = [] if cfg.get("tf_only") else run_t1(m, name)
        res = dict(cfg=cfg, model=MD, t1=rows, secs=time.time() - t0, **{k: meta_[k] for k in ("bytes", "base_bytes", "ratio", "body_bpw", "dropped")})
        m.config.use_cache = False
        if os.path.exists(f"{OUT}/base_tf.json"):
            btf = json.load(open(f"{OUT}/base_tf.json"))["tf"]; ti = tf_items(); res["tf_top1"], res["tf_kl"] = tf_cmp(m, ti, btf)
        json.dump(res, open(f"{OUT}/{'tf' if cfg.get('tf_only') else 't1'}_{name}.json", "w")); log("T1_DONE", name, sum(r["correct"] for r in rows), res.get("tf_kl")); return
    # ------------------------------------------------------------------ R3: measured-KL per-tensor sensitivity (model resident on GPU)
    if mode == "klsens":
        sys.path.insert(0, T1DIR)
        its = [json.loads(l) for l in open(T1DIR + "/calib_t1.jsonl")]
        seqs = []
        for it in its:
            tools = [{"type": "function", "function": t} for t in it["tools"]]
            try: pr = tok.apply_chat_template([{"role": "user", "content": it["query"]}], tools=tools, add_generation_prompt=True, tokenize=False, enable_thinking=False)
            except TypeError: pr = tok.apply_chat_template([{"role": "user", "content": it["query"]}], tools=tools, add_generation_prompt=True, tokenize=False)
            ans = "\n".join("<tool_call>\n" + json.dumps(dict(name=gg["name"], arguments=gg["arguments"])) + "\n</tool_call>" for gg in it["gold"]) or "None of the available tools applies to this request."
            p = tok.encode(pr, add_special_tokens=False); a = tok.encode(ans + "<|im_end|>", add_special_tokens=False)
            if len(p) + len(a) <= cfg.get("kl_maxlen", 1100): seqs.append((p, a, it["kind"]))
        if cfg.get("kl_ct"):
            cc = [json.loads(l) for l in open(cfg["kl_ct"])]; rr = np.random.RandomState(1); rr.shuffle(cc)
            for c in cc[:cfg.get("kl_nct", 24)]:
                p = tok.encode(chat(c["user"]), add_special_tokens=False); a = tok.encode(c["answer"] + "<|im_end|>", add_special_tokens=False); seqs.append((p, a, "ct"))
        rr = np.random.RandomState(2); rr.shuffle(seqs)
        nm = {k: [x for x in seqs if x[2] == k] for k in ("multi", "single", "nocall", "ct")}
        pick_ = nm["multi"][:cfg.get("kl_nmulti", 10)] + nm["single"][:cfg.get("kl_nsingle", 6)] + nm["nocall"][:cfg.get("kl_nnocall", 4)] + nm["ct"][:cfg.get("kl_nct", 24)]
        T = max(len(p) + len(a) for p, a, _ in pick_); B = len(pick_)
        ids = torch.full((B, T), tok.pad_token_id); msk = torch.zeros(B, T, dtype=torch.bool)
        for j, (p, a, _) in enumerate(pick_): ids[j, :len(p) + len(a)] = torch.tensor(p + a); msk[j, len(p) - 1:len(p) + len(a) - 1] = True   # positions predicting answer tokens
        ids = ids.cuda(); msk = msk.cuda(); log("klsens seqs", B, "T", T, "answer positions", int(msk.sum()))
        m = build_eval_model(None); m.eval(); layers_ = m.model.layers; rope_ = m.model.rotary_emb
        pos = torch.arange(T, device="cuda")[None]
        @torch.no_grad()
        def lp_of(hs):
            outs = []
            for j in range(0, B, 4):
                y = m.model.norm(hs[j:j + 4]); lg = m.lm_head(y).float(); lp = torch.log_softmax(lg, -1)[msk[j:j + 4]]; outs.append(lp.half())
            return torch.cat(outs)
        @torch.no_grad()
        def run_from(l0, hs, store=None):
            cos_sin = rope_(hs, pos)
            for l in range(l0, nl):
                if store is not None: store.append(hs)
                hs = layers_[l](hs, attention_mask=None, position_ids=pos, position_embeddings=cos_sin)
                hs = hs[0] if isinstance(hs, tuple) else hs
            return hs
        with torch.no_grad():
            h0 = m.model.embed_tokens(ids); inputs = []; hfin = run_from(0, h0, inputs); base_lp = lp_of(hfin)
        def kl_from(l0, hs):
            with torch.no_grad():
                hf_ = run_from(l0, hs); lp = lp_of(hf_); bl = base_lp.float(); return float((bl.exp() * (bl - lp.float())).sum(-1).mean())
        Ls_ = cfg.get("kl_levels", [2, 3, 4, 8]); res = {}
        kpath = f"{OUT}/klsens_{cfg.get('kl_tag', 'T1')}_g{g}.json"
        for l in range(nl):
            for key, mem in groups:
                sg = signs_kl(l, key, getattr(layers_[l].get_submodule(mem[0]), "w").shape[1])
                for n in mem:
                    q = layers_[l].get_submodule(n); W0 = q.w; Wf = Qz.rot(W0.float(), sg); res[f"{l}.{n}"] = {}
                    for Lv in Ls_:
                        Qr, _ = Qz.rtn(Wf, Lv); q.w = Qz.unrot(Qr, sg).to(torch.bfloat16)
                        res[f"{l}.{n}"][Lv] = kl_from(l, inputs[l]); del Qr
                    q.w = W0
            json.dump(res, open(kpath, "w")); log("klsens layer", l, {n: [round(v, 5) for v in res[f"{l}.{n}"].values()] for n in ("self_attn.v_proj", "mlp.down_proj")}, round(time.time() - t0))
        log("KLSENS_DONE"); return
    # ------------------------------------------------------------------ calibration stream (never locked / S2: calib_t1.jsonl is S1-calib only)
    cal_txt = [chat(c["prompt"]) + c["ref"] + "<|im_end|>\n" for c in D["calib"]]
    if cfg.get("calib_t1", True):
        from_t1 = [json.loads(l) for l in open(T1DIR + "/calib_t1.jsonl")]; t1c = []
        for it in from_t1:
            tools = [{"type": "function", "function": t} for t in it["tools"]]
            try: pr = tok.apply_chat_template([{"role": "user", "content": it["query"]}], tools=tools, add_generation_prompt=True, tokenize=False, enable_thinking=False)
            except TypeError: pr = tok.apply_chat_template([{"role": "user", "content": it["query"]}], tools=tools, add_generation_prompt=True, tokenize=False)
            ans = "\n".join("<tool_call>\n" + json.dumps(dict(name=gg["name"], arguments=gg["arguments"])) + "\n</tool_call>" for gg in it["gold"]) or "None of the available tools applies to this request."
            t1c.append(pr + ans + "<|im_end|>\n")
        cal_txt = t1c[:len(t1c) // 2] + cal_txt[:len(t1c) // 2] if cfg.get("calib_mix", True) else t1c
    if cfg.get('calib_ct'):
        cc = [json.loads(l) for l in open(cfg['calib_ct'])]; cal_txt = [chat(c['user']) + c['answer'] + '<|im_end|>\n' for c in cc]
        if cfg.get('calib_ct_mix'): cal_txt = cal_txt + t1c[:len(cc) // 2]
    rng = np.random.RandomState(0); rng.shuffle(cal_txt)
    ncal = cfg.get("ncal", 128); CL = cfg.get("CL", 512); stream = []
    for s_ in cal_txt:
        stream += tok.encode(s_, add_special_tokens=False)
        if len(stream) >= ncal * CL: break
    stream = torch.tensor(stream[:ncal * CL]).view(-1, CL)
    emb_w = st.get("model.embed_tokens.weight")
    inps = [torch.nn.functional.embedding(stream[i:i + 1].cuda(), emb_w) for i in range(len(stream))]
    rope = new_rot(); pos = torch.arange(CL, device="cuda")[None]; pe = rope(inps[0], pos)
    kw = dict(attention_mask=None, position_ids=pos, position_embeddings=pe)
    prev = torch.get_default_dtype(); torch.set_default_dtype(torch.bfloat16)
    with torch.device("cuda"): layer = LayerCls(conf, 0)
    torch.set_default_dtype(prev); layer.eval()
    def load_layer(i):
        for k, p in layer.named_parameters(): p.data.copy_(st.get(f"model.layers.{i}.{k}"))
    def fwd(x):
        o = layer(x, **kw); return o[0] if isinstance(o, tuple) else o
    def collect(cur):
        Hs = {}; nsm = {}; mods = {n: layer.get_submodule(n) for n in linn}
        def mk(key):
            def hook(m, i, o):
                x = i[0].reshape(-1, i[0].shape[-1]).float(); nsm[key] = nsm.get(key, 0) + x.shape[0]; Hs[key] = Hs.get(key, 0) + 2 * x.T @ x
            return hook
        hds = [mods[k].register_forward_hook(mk(k)) for k, _ in groups]
        with torch.no_grad():
            for x in cur: fwd(x)
        for h in hds: h.remove()
        return mods, {k: Hs[k] / nsm[k] for k in Hs}
    def signs(oli, key, C): return (torch.randint(0, 2, (C,), generator=torch.Generator().manual_seed(1000 * oli + len(key)), dtype=torch.float32) * 2 - 1).cuda()
    # ------------------------------------------------------------------ pass A
    spath = f"{OUT}/sens_g{g}.json"
    if os.path.exists(spath): S = json.load(open(spath))
    else:
        S = dict(tab={}, bi=[]); cur = [x.clone() for x in inps]
        for oli in range(nl):
            load_layer(oli); mods, Hs = collect(cur)
            with torch.no_grad():
                nxt = [fwd(x) for x in cur]
                S["bi"].append(float(np.mean([1 - F.cosine_similarity(a.float().flatten(0, 1), b.float().flatten(0, 1), dim=-1).mean().item() for a, b in zip(cur[:16], nxt[:16])])))
                for key, mem in groups:
                    H = Hs[key]; C = H.shape[0]; sg = signs(oli, key, C); Hr = Qz.rot(Qz.rot(H, sg).T.contiguous(), sg).T.contiguous()
                    for n in mem:
                        W = Qz.rot(mods[n].weight.data.float(), sg); ref = float(((W @ Hr) * W).sum()); tab = {}
                        for Lv in LS:
                            Q, h = Qz.rtn(W, Lv); Dd = W - Q; tab[Lv] = [float(((Dd @ Hr) * Dd).sum()) / ref, ent(h)]
                        S["tab"][f"{oli}.{n}"] = dict(n=W.numel(), tab=tab)
            cur = nxt; del Hs; torch.cuda.empty_cache()
            log("sensA", oli, round(time.time() - t0))
        json.dump(S, open(spath, "w")); S = json.loads(json.dumps(S))
    if mode == "sens": log("SENS_DONE"); return
    # ------------------------------------------------------------------ fixed bytes + allocation
    keep = list(range(nl))
    lin_total = sum(S["tab"][f"{i}.{n}"]["n"] for i in keep for n in linn)
    lin_names = {f"model.layers.{i}.{n}.weight" for i in keep for n in linn}
    other_bytes = sum(int(np.prod(safe_open(MD + "/" + f, "pt").get_slice(k).get_shape())) * 2 for k, f in st.wm.items()
                      if k not in lin_names and k not in ("model.embed_tokens.weight", "lm_head.weight"))
    fb = sum(os.path.getsize(p) for p in glob.glob(MD + "/*") if os.path.basename(p) in ("tokenizer.json", "tokenizer_config.json", "config.json", "generation_config.json"))
    CODE = 150000; fixed_bytes = other_bytes + fb + CODE
    eL = cfg.get("emb_L", 4); hL = cfg.get("head_L", 4); ge = cfg.get("g_emb", 128)
    def est_ent(W, Lv):
        gen = torch.Generator().manual_seed(7); idx = torch.randperm(W.shape[0], generator=gen)[:4096]
        sg = torch.ones(W.shape[1]).cuda(); Wr = Qz.rot(W[idx].cuda().float(), sg); Q, h = Qz.rtn(Wr, Lv); return ent(h)
    head_w = emb_w if tied else st.get("lm_head.weight")
    ent_e = est_ent(emb_w, eL); ent_h = est_ent(head_w, hL)
    emb_bytes = V * hid * (ent_e * 1.005 + GB / g) / 8; head_bytes = 0 if tied else V * hid * (ent_h * 1.005 + GB / g) / 8
    target = cfg["target_ratio"] * base_bytes; body_bits = (target - fixed_bytes - emb_bytes - head_bytes) * 8
    bi = np.array(S['bi']); gam = cfg.get('gamma', 0.0); lw = {i: (bi[i] / bi.mean()) ** gam for i in keep}
    items = [(f"{i}.{n}", S["tab"][f"{i}.{n}"], lw[i]) for i in keep for n in linn]
    klm = json.load(open(cfg['kl_file'])) if cfg.get('kl_file') else None
    if klm is not None:
        for k_, d_ in klm.items():
            prev = None
            for L_ in sorted(d_, key=int):
                if prev is not None: d_[L_] = min(d_[L_], prev)
                prev = d_[L_]
    def pick(lam):
        tot = 0; ch = {}
        for k, it, wt in items:
            best = None
            for Lv in (LS if klm is None else [int(x) for x in klm[k]]):
                e, H_ = it["tab"][str(Lv)]; b = it["n"] * (H_ * 1.005 + GB / g)
                c = (wt * e * it["n"] / lin_total * 1e3 if klm is None else wt * klm[k][str(Lv)] * cfg.get('kl_scale', 1.0)) + lam * b
                if best is None or c < best[0]: best = (c, Lv, b)
            ch[k] = best[1]; tot += best[2]
        return tot, ch
    lo, hi = 1e-14, 1e-3
    for _ in range(60):
        mid = math.sqrt(lo * hi); tot, ch = pick(mid)
        if tot > body_bits: lo = mid
        else: hi = mid
    tot, levels = pick(hi); log("alloc body bpw", tot / lin_total, "target", body_bits / lin_total, "fixed MB", fixed_bytes / 1e6, "emb MB", emb_bytes / 1e6, "head MB", head_bytes / 1e6)
    # ------------------------------------------------------------------ pass B
    os.makedirs(qdir, exist_ok=True)
    cur = [x.clone() for x in inps]; qbits = 0.0; Lcount = Counter()
    for oli in keep:
        load_layer(oli); mods, Hs = collect(cur); sv = {}
        for key, mem in groups:
            H = Hs[key]; C = H.shape[0]; sg = signs(oli, key, C); Hr = Qz.rot(Qz.rot(H, sg).T.contiguous(), sg).T.contiguous()
            for n in mem:
                Lv = levels[f"{oli}.{n}"]; W = Qz.rot(mods[n].weight.data.float(), sg)
                Q, h, QI, SC = Qz.gptq(W, Hr, Lv, damp=cfg.get('damp', 0.01)); mods[n].weight.data = Qz.unrot(Q, sg).to(torch.bfloat16)
                qbits += W.numel() * (ent(h) * 1.005 + GB / g); Lcount[Lv] += W.numel()
                sv[n] = dict(L=Lv, codes=pack4(QI).cpu(), scales=SC.cpu(), sg=sg.to(torch.int8).cpu()); del Q, QI, SC, W
            del Hr
        torch.save(sv, f"{qdir}/L{oli}.pt")
        with torch.no_grad(): cur = [fwd(x) for x in cur]
        del Hs; torch.cuda.empty_cache()
        log("passB", oli, round(time.time() - t0), "bpw so far", round(qbits / sum(S["tab"][f"{i}.{n}"]["n"] for i in range(oli + 1) for n in linn), 3))
    Hh = 0; nh = 0
    with torch.no_grad():
        nm = torch.nn.Parameter(st.get("model.norm.weight"), requires_grad=False)
        for x in cur:
            xf = x.float(); y = (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + conf.rms_norm_eps)).to(torch.bfloat16) * nm
            y = y.reshape(-1, hid).float(); Hh = Hh + 2 * y.T @ y; nh += y.shape[0]
    Hh = Hh / nh; del inps, cur; torch.cuda.empty_cache()
    sg = signs(99999, "head", hid); Wh = Qz.rot(head_w.float(), sg); Hr = Qz.rot(Qz.rot(Hh, sg).T.contiguous(), sg).T.contiguous()
    Q, h, _, _ = Qz.gptq(Wh, Hr, hL); head_bits = head_w.numel() * (ent(h) * 1.005 + GB / g)
    hd = Qz.unrot(Q, sg).to(torch.bfloat16); torch.save(hd.cpu(), qdir + "/head.pt"); del Q, Wh
    sg = signs(88888, "emb", hid); We = Qz.rot(emb_w.float(), sg); Wq = torch.empty_like(We); hist = torch.zeros(eL, device="cuda")
    for r in range(0, We.shape[0], 8192):
        x = We[r:r + 8192].reshape(-1, ge); sc = Qz.qparams(x, eL); q, qi = Qz.quant(x, sc, eL); Wq[r:r + 8192] = q.view(-1, hid); hist += torch.bincount(qi.long().flatten(), minlength=eL).float()
    emb_bits = emb_w.numel() * (ent(hist) * 1.005 + GB / ge); torch.save(Qz.unrot(Wq, sg).to(torch.bfloat16).cpu(), qdir + "/emb.pt")
    if tied: head_bits = 0; torch.save(Qz.unrot(Wq, sg).to(torch.bfloat16).cpu(), qdir + "/head.pt")
    tot_bytes = qbits / 8 + head_bits / 8 + emb_bits / 8 + fixed_bytes
    meta_ = dict(cfg=cfg, bytes=tot_bytes, base_bytes=base_bytes, ratio=tot_bytes / base_bytes, body_bpw=qbits / lin_total, dropped=[],
                 L_hist={str(k): v / lin_total for k, v in Lcount.items()}, parts_MB=dict(body=qbits / 8e6, emb=emb_bits / 8e6, head=head_bits / 8e6, fixed=fixed_bytes / 1e6), secs=time.time() - t0)
    json.dump(meta_, open(qdir + "/meta.json", "w")); log("QUANT_DONE", name, "ratio", meta_["ratio"], "bytes", tot_bytes)
    del hd; torch.cuda.empty_cache()
    if mode == "all": run(dict(cfg, mode="eval"))


if __name__ == "__main__":
    for c in json.load(open(sys.argv[1])):
        try: run(c)
        except Exception:
            import traceback; traceback.print_exc()
