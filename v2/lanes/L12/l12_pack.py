"""Packed exporter for the RTN4 body: real bytes on disk (4-bit codes two per byte, fp16 scales, 4-bit zeros two per byte), verified to dequantise bit-exactly to the engine's fake-quant weights."""
import torch, math, os, json, shutil
from safetensors.torch import save_file
from l12_core import rtn, log

def rtn_codes(w, bits, g, mse=True):
    sh = w.shape; g = math.gcd(sh[-1], g); wg = w.float().reshape(*sh[:-1], sh[-1] // g, g); mx = wg.amax(-1, keepdim=True); mn = wg.amin(-1, keepdim=True); qm = 2 ** bits - 1
    def q(mn_, mx_):
        sc = ((mx_ - mn_) / qm).clamp(min=1e-8).half().float(); z = (-mn_ / sc).round().clamp(0, qm); c = (wg / sc + z).round().clamp(0, qm)
        return (c - z) * sc, c, sc, z
    best = None
    for r in ((1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65) if mse else (1.0,)):
        dq, c, sc, z = q(mn * r, mx * r); err = ((dq - wg) ** 2).sum(-1, keepdim=True)
        if best is None: best, berr = [dq, c, sc, z], err
        else:
            m = err < berr
            best = [torch.where(m, a, b) for a, b in zip((dq, c, sc, z), best)]; berr = torch.where(m, err, berr)
    return best[0].reshape(sh), best[1].to(torch.uint8), best[2].half(), best[3].to(torch.uint8)

def pack4(c):  # uint8 values < 16, last dim even
    c = c.reshape(-1, 2); return (c[:, 0] | (c[:, 1] << 4)).contiguous()

def put(d, name, w, g=128, mse=True, check=True):
    dq, c, sc, z = rtn_codes(w, 4, g, mse)
    if check: assert torch.equal(dq, rtn(w, 4, g, mse).reshape(dq.shape)), name
    d[name + ".q4"] = pack4(c).cpu(); d[name + ".sc"] = sc.cpu(); d[name + ".z4"] = pack4(z.reshape(-1)).cpu(); d[name + ".shape"] = torch.tensor(list(w.shape), dtype=torch.int32)
    return dq

@torch.no_grad()
def export(eng, kept, cfg, outdir):
    shutil.rmtree(outdir, ignore_errors=True); os.makedirs(outdir); tot = {}; chk = {}
    def flush(name, d):
        p = f"{outdir}/{name}.safetensors"; save_file(d, p); tot[name] = os.path.getsize(p)
    sd = dict(eng.lm.named_parameters()); small = {}
    for k, p in sd.items():
        if (".self_attn." in k or ".linear_attn." in k) and k.endswith(".weight") and p.dim() == 2: continue
        if (".shared_expert." in k and p.dim() == 2) or "embed_tokens" in k or "lm_head" in k or k.endswith("mlp.gate.weight"): continue
        small[k] = p.data.to(torch.bfloat16).cpu()
    flush("small", small)
    d = {}
    for k, w in eng.orig_attn.items(): dq = put(d, k, w.to(eng.dev), cfg.get("attn_g", 128), True)
    for k, w in eng.orig_shared.items(): put(d, k, w.to(eng.dev), 128, True)
    flush("attn_shared", d)
    for nm, tgt, W in (("embed", None, eng.orig_emb), ("head", None, eng.orig_head)):
        d = {}
        for i, s0 in enumerate(range(0, W.shape[0], 20000)): put(d, f"{nm}.{i}", W[s0:s0 + 20000].to(eng.dev), 128, False, check=(i == 0))
        flush(nm, d)
    for l in range(eng.NL):
        kt = torch.tensor(kept[l], device=eng.dev); d = {}
        for nm, O in (("g", eng.OG[l]), ("u", eng.OU[l]), ("d", eng.OD[l])): dq = put(d, f"L{l}.{nm}", O[kept[l]].to(eng.dev), cfg.get("eg", 128), True, check=(l == 0))
        d[f"L{l}.router"] = eng.orig_gate[l][kt].to(torch.bfloat16).cpu(); mask = torch.zeros(eng.E, dtype=torch.uint8); mask[kept[l]] = 1
        d[f"L{l}.keepmask"] = torch.from_numpy(__import__("numpy").packbits(mask.numpy())); flush(f"layer{l}", d)
        if l == 0: chk["layer0_gate_exact_vs_live"] = bool(torch.equal(eng.blocks[0].live.G.to("cpu").float(), rtn(eng.OG[0][kept[0]].to(eng.dev), 4, cfg.get("eg", 128), True).to(eng.dtype).float().cpu()))
    toks = sum(os.path.getsize(os.path.join(eng.snap, f)) for f in ["tokenizer.json", "tokenizer_config.json", "config.json", "generation_config.json"] if os.path.exists(os.path.join(eng.snap, f)))
    total = sum(tot.values()) + toks
    res = dict(file_bytes=tot_sum if (tot_sum := sum(tot.values())) else 0, tokenizer_config_bytes=toks, total=total, ratio_vs_full_checkpoint=total / eng.fp16_bytes, ratio_vs_text_only=total / eng.fp16_text, parts={k: v for k, v in tot.items() if not k.startswith("layer")}, layers_total=sum(v for k, v in tot.items() if k.startswith("layer")), checks=chk)
    log("EXPORT", json.dumps(res)); return res
