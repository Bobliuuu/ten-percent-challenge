"""cd_run.py-style driver: structured width pruning (FFN channels, attention heads / GQA groups) + Hadamard ECSQ quantisation (entropy-coded, ideal-coder bytes),
or Wanda unstructured / 2:4 + ECSQ. Fake-quant weights are written back at FULL shape with zeros where pruned (exactly equal to physical removal for FFN channels
and heads: silu(0)*0 = 0, zeroed o_proj columns) and generated with vLLM. Bytes are computed for the PHYSICALLY shrunk model."""
import os, sys, json, math, time, subprocess, shutil, random
import numpy as np, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
DATA = "data"; WORK = os.environ.get("WORK", "/root/work"); os.makedirs(WORK, exist_ok=True); os.makedirs("out", exist_ok=True)
dev = "cuda"
def hadamard(n):
    H = torch.ones(1, 1)
    while H.shape[0] < n: H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    return H / math.sqrt(n)
H128 = hadamard(128).to(dev)

def entropy_bits(sym, maxv=96):
    s = sym.clamp(-maxv, maxv).long().flatten() + maxv
    c = torch.bincount(s, minlength=2 * maxv + 1).double(); n = c.sum(); p = c / n
    return float(-(c[c > 0] * torch.log2(p[c > 0])).sum())

def ecsq(W, c, dz, rotate=True, sparse24=False, seed=0, sub=1, nm=None):
    """W [o,i] (i % 128 == 0) -> (What float32 [o,i], bits). rotation = random-sign block-128 Hadamard over the input dim. groups of 128 (in the rotated domain).
    nm: None | 'u' (zeros already imposed, in-band entropy) | 'nm24' (2:4 pattern known; mask cost 0.6462 b/weight + entropy of values at nonzero positions)."""
    o, i = W.shape; W = W.float()
    if sub > 1: W = W[::sub]
    if rotate:
        g = torch.Generator(device="cpu").manual_seed(seed); sg = (torch.randint(0, 2, (i,), generator=g) * 2 - 1).float().to(dev)
        Wb = ((W * sg).view(W.shape[0], i // 128, 128)) @ H128
    else: Wb = W.view(W.shape[0], i // 128, 128)
    if nm is None: s = Wb.pow(2).mean(-1, keepdim=True).sqrt().clamp_min(1e-6)
    else:
        nz = (Wb != 0).float(); s = (Wb.pow(2).sum(-1, keepdim=True) / nz.sum(-1, keepdim=True).clamp_min(1)).sqrt().clamp_min(1e-6)
    sidx = torch.round(8 * torch.log2(s)); sq = torch.exp2(sidx / 8)
    z = Wb / sq; q = torch.sign(z) * torch.clamp(torch.floor(z.abs() / c + 0.5 - dz), min=0)
    if nm is not None: q = q * (Wb != 0)
    if nm == "nm24":
        m = (Wb != 0).flatten(); bits = entropy_bits(q.flatten()[m]) + 0.6462 * m.numel() * 1.0 * (1 if False else 1)  # mask: log2(6)/4 per weight over ALL positions
        bits = entropy_bits(q.flatten()[m]) + 0.6462 * Wb.numel()
    else: bits = entropy_bits(q)
    bits += entropy_bits(sidx - sidx.median(), 64) if True else 0
    What = (q * c * sq)
    if rotate: What = (What @ H128).view(W.shape[0], i) * sg
    else: What = What.view(W.shape[0], i)
    return What, bits * sub

def rtn_bits(W, nbits):
    """symmetric RTN, group 128, fp16 scale: bits = nbits/weight + 16/128."""
    o, i = W.shape; Wg = W.float().view(o, i // 128, 128); qm = 2 ** (nbits - 1) - 1
    s = Wg.abs().amax(-1, keepdim=True).clamp_min(1e-8) / qm; q = torch.clamp(torch.round(Wg / s), -qm - 1, qm)
    return (q * s).view(o, i), W.numel() * (nbits + 16 / 128)

def main(cfgs, model_name="Qwen/Qwen2.5-7B-Instruct"):
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map=dev).eval()
    C = model.config; L = C.num_hidden_layers; NH = C.num_attention_heads; NKV = C.num_key_value_heads; HD = C.hidden_size // NH; I = C.intermediate_size; Hd = C.hidden_size
    GS = NH // NKV
    nparams = sum(p.numel() for p in model.parameters()); fp16_bytes = nparams * 2
    tokbytes = sum(os.path.getsize(os.path.join(os.path.dirname(tok.vocab_file), f)) for f in ()) if False else 0
    from huggingface_hub import snapshot_download
    sd = snapshot_download(model_name, allow_patterns=["*.json", "merges.txt", "vocab.json"])
    tokbytes = sum(os.path.getsize(os.path.join(sd, f)) for f in os.listdir(sd) if f in ("tokenizer.json", "vocab.json", "merges.txt", "tokenizer_config.json", "generation_config.json", "config.json"))
    print("params", nparams, "fp16 bytes", fp16_bytes, "tokenizer+config bytes", tokbytes, flush=True)
    orig = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    # ---------- calibration statistics (disjoint calibration set, equal weight per task) ----------
    cal = json.load(open(DATA + "/calib.json")); STAT = {}
    if os.path.exists(WORK + "/stats.pt"): STAT = torch.load(WORK + "/stats.pt")
    else:
        names = ["q_proj", "o_proj", "gate_proj", "down_proj"]; acc = {t: {} for t in ("t1", "tox", "inj")}; cnt = {t: 0 for t in acc}; cur = {"t": None, "mask": None}
        Go = {l: torch.stack([ (lambda W: W.T @ W)(model.model.layers[l].self_attn.o_proj.weight[:, h*HD:(h+1)*HD].float()) for h in range(NH)]) for l in range(L)}
        def mk(l, nm):
            def hook(mod, args):
                x = args[0].float(); m = cur["mask"]; xs = x[m]            # [Ntok, in]
                a = acc[cur["t"]]
                if nm == "o_proj":
                    xh = xs.view(-1, NH, HD); c2 = torch.einsum("tnd,nde,tne->tn", xh, Go[l], xh)   # squared norm of each head's contribution to the residual
                    a[(l, "head")] = a.get((l, "head"), 0) + c2.sum(0)
                a[(l, nm)] = a.get((l, nm), 0) + xs.pow(2).sum(0)
            return hook
        hs = []
        for l in range(L):
            lay = model.model.layers[l]
            for nm, mod in (("q_proj", lay.self_attn.q_proj), ("o_proj", lay.self_attn.o_proj), ("gate_proj", lay.mlp.gate_proj), ("down_proj", lay.mlp.down_proj)): hs.append(mod.register_forward_pre_hook(mk(l, nm)))
        with torch.no_grad():
            for r in cal:
                msgs = [{"role": "user", "content": r["prompt"]}]
                p = tok.apply_chat_template(msgs, tools=[{"type": "function", "function": t} for t in r["tools"]] if r.get("tools") else None, add_generation_prompt=True, tokenize=False)
                ids = tok(p + r["answer"] + "<|im_end|>", add_special_tokens=False, return_tensors="pt").input_ids[:, -1800:].to(dev)
                cur["t"] = r["task"]; cur["mask"] = torch.ones(ids.shape[:2], dtype=torch.bool, device=dev); cnt[r["task"]] += ids.shape[1]
                model(input_ids=ids, use_cache=False)
        for h in hs: h.remove()
        STAT = {t: {k: (v / cnt[t]).cpu() for k, v in acc[t].items()} for t in acc}; torch.save(STAT, WORK + "/stats.pt")
    tasks = ("t1", "tox", "inj")
    def ms(l, nm): return sum(STAT[t][(l, nm)].float() for t in tasks) / 3      # equal weight per task
    # ---------- per-config pipeline ----------
    results = []
    for cfg in cfgs:
        name = cfg["name"]
        if os.path.exists(f"out/{name}_meta.json") and not cfg.get("redo"): print("skip", name); continue
        t0 = time.time(); model.load_state_dict({k: v for k, v in orig.items()}); sdm = model.state_dict()
        p_ffn = cfg.get("ffn", 0.0); p_head = cfg.get("head", 0.0); hmode = cfg.get("hmode", "head"); mode = cfg.get("mode", "ecsq"); dz = cfg.get("dz", 0.15)
        emb_bits = cfg.get("emb_bits", 4); targ = cfg.get("target"); cfix = cfg.get("c")
        # ---- choose structure ----
        kfn = int(round(I * (1 - p_ffn) / 128)) * 128 if p_ffn > 0 else I
        layer_keep = []   # (ffn_idx, head_mask[NH], kv_keep[NKV])
        for l in range(L):
            lay = model.model.layers[l]
            if kfn < I:
                sc = ms(l, "down_proj").sqrt().to(dev) * lay.mlp.down_proj.weight.float().norm(dim=0)
                fi = torch.sort(torch.topk(sc, kfn).indices).values
            else: fi = torch.arange(I, device=dev)
            hm = torch.ones(NH, dtype=torch.bool, device=dev)
            if p_head > 0:
                hs_ = STAT[tasks[0]][(l, "head")].float() * 0
                for t in tasks: hs_ = hs_ + STAT[t][(l, "head")].float() / 3
                hs_ = hs_.sqrt().to(dev)
                if hmode == "head":
                    nrm = int(round(NH * p_head)); hm[torch.topk(hs_, nrm, largest=False).indices] = False
                else:   # whole GQA groups
                    gs = hs_.view(NKV, GS).sum(1); nrg = int(round(NKV * p_head)); gi = torch.topk(gs, nrg, largest=False).indices
                    for g in gi.tolist(): hm[g * GS:(g + 1) * GS] = False
            kv = torch.tensor([bool(hm[g * GS:(g + 1) * GS].any()) for g in range(NKV)], device=dev)
            layer_keep.append((fi, hm, kv))
        # ---- compact tensors per layer ----
        def compact(l):
            fi, hm, kv = layer_keep[l]; lay = model.model.layers[l]; a = lay.self_attn; m = lay.mlp
            hidx = torch.nonzero(hm).flatten(); qrows = (hidx[:, None] * HD + torch.arange(HD, device=dev)[None]).flatten()
            kidx = torch.nonzero(kv).flatten(); krows = (kidx[:, None] * HD + torch.arange(HD, device=dev)[None]).flatten()
            T = {"q": (a.q_proj.weight, qrows, None), "k": (a.k_proj.weight, krows, None), "v": (a.v_proj.weight, krows, None), "o": (a.o_proj.weight, None, qrows),
                 "g": (m.gate_proj.weight, fi, None), "u": (m.up_proj.weight, fi, None), "d": (m.down_proj.weight, None, fi)}
            return T, qrows, krows
        def quant_pass(c, apply, sub):
            bits = 0.0
            for l in range(L):
                T, qrows, krows = compact(l)
                for k, (W, rr, cc) in T.items():
                    Wc = W.detach()
                    if rr is not None: Wc = Wc[rr]
                    if cc is not None: Wc = Wc[:, cc]
                    if Wc.numel() == 0: continue
                    if mode == "ecsq": What, b = ecsq(Wc, c, dz, True, seed=l * 8 + ord(k[0]) % 8, sub=1 if apply else sub)
                    else:
                        raise NotImplementedError
                    bits += b
                    if apply:
                        full = torch.zeros_like(W, dtype=torch.float32)
                        if rr is not None: full[rr] = What
                        elif cc is not None: full[:, cc] = What
                        W.data.copy_(full.to(W.dtype))
            return bits
        # fixed bytes
        def fixed_bytes():
            tot = 0
            for l in range(L):
                fi, hm, kv = layer_keep[l]; nq = int(hm.sum()) * HD; nk = int(kv.sum()) * HD
                tot += 2 * (nq + 2 * nk) + 2 * 2 * Hd + int(hm.sum()) * 1 + (len(fi) // 128 * 0)      # biases fp16, 2 norms fp16, head->group index 1 B per kept head
                tot += (I + 7) // 8 if len(fi) < I else 0                     # conservative channel bitmap per layer (not needed for physical shrink)
            tot += 2 * Hd + tokbytes + 30000                                   # final norm, tokenizer+configs, loader code estimate
            tot += 256 * 7 * L                                                 # entropy tables
            return tot
        # embedding / lm_head
        eb = 0.0; embW = sdm["model.embed_tokens.weight"]; hdW = sdm["lm_head.weight"]
        for W in (embW, hdW):
            if emb_bits >= 16: eb += W.numel() * 16
            else:
                What, b = rtn_bits(W.detach(), emb_bits); eb += b; W.data.copy_(What.to(W.dtype))
        fb = fixed_bytes(); body_budget_bits = None
        if targ is not None:
            body_budget_bits = (targ * fp16_bytes - fb - eb / 8) * 8
            lo, hi = 0.3, 12.0     # c: step in units of group rms
            for _ in range(14):
                mid = math.sqrt(lo * hi); b = quant_pass(mid, False, 4)
                if b > body_budget_bits: lo = mid
                else: hi = mid
            c = hi
        else: c = cfix
        bits = quant_pass(c, True, 1)
        total = bits / 8 + eb / 8 + fb; nbody = sum(sum(w.numel() for w in [1]) for _ in [])
        meta = dict(cfg=cfg, c=c, body_bits=bits, emb_bits_total=eb, fixed_bytes=fb, total_bytes=total, size_pct=100 * total / fp16_bytes, fp16_bytes=fp16_bytes)
        body_params = sum(int(layer_keep[l][0].numel()) * Hd * 3 + (int(layer_keep[l][1].sum()) * HD * Hd) * 2 + int(layer_keep[l][2].sum()) * HD * Hd * 2 for l in range(L))
        meta["body_params"] = body_params; meta["body_bpw"] = bits / max(1, body_params)
        print(name, "size %.3f%% bpw %.3f c %.3f prep %.0fs" % (meta["size_pct"], meta["body_bpw"], c, time.time() - t0), flush=True)
        cd = f"{WORK}/cand"; shutil.rmtree(cd, ignore_errors=True); model.save_pretrained(cd, safe_serialization=True); tok.save_pretrained(cd)
        r = subprocess.run([sys.executable, "probe_vllm.py", cd, name, DATA], capture_output=True, text=True)
        print(r.stdout[-300:], r.stderr[-600:] if r.returncode else "", flush=True)
        json.dump(meta, open(f"out/{name}_meta.json", "w")); results.append(meta)
        an = subprocess.run([sys.executable, "analyze_cd.py", "out/base.json", f"out/{name}.json"], capture_output=True, text=True); print(an.stdout[-900:], an.stderr[-300:], flush=True)
        print("RESULT_JSON", json.dumps(dict(name=name, size_pct=meta["size_pct"], body_bpw=meta["body_bpw"])), flush=True)
    return results

if __name__ == "__main__":
    cfgs = json.load(open(sys.argv[1])); main(cfgs, *(sys.argv[2:3]))
