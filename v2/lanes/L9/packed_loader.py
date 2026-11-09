"""STANDALONE loader for the L9 packed artifact (Qwen3-235B-A22B-Instruct-2507, 88/128 experts per layer, 2-bit RTN g128 experts, 4-bit attention/embedding/head).
Needs only torch + transformers (<5, ModuleList experts) + accelerate + safetensors and the artifact directory (config.json, tokenizer files, meta.json, layerNN.safetensors, global.safetensors).
Nothing from the HF checkpoint or from the pack/calibration code is used. Layers are decoded one file at a time straight onto their device (device_map over all visible GPUs, balanced),
the bf16 dequantised model (~330 GB over the GPUs) is what runs; no fused low-bit kernel exists.  API: m, tok = build(ART); apply_carrier(m, ART); texts -> generate(m, tok, texts)."""
import os, sys, json, math, types
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from safetensors.torch import load_file
NL, NE, TOPK, HID, FF = 94, 128, 8, 4096, 1536
CARRIER = "fb_carrier.safetensors"

def unpack_bits(p, bits):
    per = 8 // bits; m = (1 << bits) - 1; return torch.stack([(p >> (bits * i)) & m for i in range(per)], -1).reshape(*p.shape[:-1], p.shape[-1] * per)
def dequant(q, s, z, bits, g, dev):
    q, s, z = q.to(dev), s.to(dev), z.to(dev); c = unpack_bits(q, bits).float(); zz = unpack_bits(z, bits).float()
    out = c.reshape(*c.shape[:-1], -1, g); out = (out - zz[..., None]) * s.float()[..., None]; return out.reshape(*c.shape)

class GateW(nn.Module):   # router: logits of dropped experts are -inf, so top-8 + renormalisation happen among the kept experts only
    def __init__(s, gate): super().__init__(); s.g = gate; s.mask = None
    def forward(s, x):
        lg = s.g(x); return lg if s.mask is None else lg + s.mask
def moe_fwd(self, hs):
    B, T, D = hs.shape; x = hs.reshape(-1, D); p = F.softmax(self.gate(x).float(), -1); w, ii = torch.topk(p, TOPK, -1); w = (w / w.sum(-1, keepdim=True)).to(x.dtype)
    out = torch.zeros_like(x); em = F.one_hot(ii, NE).permute(2, 1, 0)
    for e in (em.sum((1, 2)) > 0).nonzero().flatten().tolist():
        k_, t_ = torch.where(em[e]); xe = x[t_]; ex = self.experts[e]
        out.index_add_(0, t_, ex.down_proj(ex.act_fn(ex.gate_proj(xe)) * ex.up_proj(xe)) * w[t_, k_][:, None])
    return out.view(B, T, D)

def build(ART, ndev=None, headroom=1.18, gpumem=None):
    from accelerate import init_empty_weights, infer_auto_device_map, dispatch_model
    from accelerate.utils import set_module_tensor_to_device
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    meta = json.load(open(os.path.join(ART, "meta.json"))); eg, ab, eb = meta["eg"], meta["attn_b"], meta["eb"]
    with init_empty_weights(): m = AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(ART), torch_dtype=torch.bfloat16)
    G = load_file(os.path.join(ART, "global.safetensors")); keeps = []
    for l in range(NL):   # keep masks first (tiny) so the dropped experts can be removed before the device map is inferred
        d = load_file(os.path.join(ART, f"layer{l:02d}.safetensors")); mask = np.unpackbits(d["keep_mask"].numpy(), bitorder="little")[:NE]; ki = [e for e in range(NE) if mask[e]]; keeps.append(ki)
        for e in range(NE):
            if e not in set(ki): m.model.layers[l].mlp.experts[e] = nn.Identity()
    n = ndev or torch.cuda.device_count(); tot = sum(p.numel() * 2 for p in m.parameters())
    cap = gpumem or f"{int(tot * headroom / n / 2**30) + 1}GiB"
    dm = infer_auto_device_map(m, max_memory={i: cap for i in range(n)}, no_split_module_classes=["Qwen3MoeDecoderLayer"])
    def dev_of(name):
        best = None
        for k in dm:
            if name == k or name.startswith(k + ".") or k == "":
                if best is None or len(k) > len(best): best = k
        return dm[best]
    def put(name, t): set_module_tensor_to_device(m, name, dev_of(name), value=t.to(torch.bfloat16) if t.dtype.is_floating_point and t.dtype != torch.bfloat16 else t)
    put("model.embed_tokens.weight", dequant(G["embed.q"], G["embed.s"], G["embed.z"], meta["emb_b"], 128, "cuda:0")); put("lm_head.weight", dequant(G["head.q"], G["head.s"], G["head.z"], meta["head_b"], 128, "cuda:0")); put("model.norm.weight", G["norm"])
    for l in range(NL):
        d = load_file(os.path.join(ART, f"layer{l:02d}.safetensors")); ki = keeps[l]; p = f"model.layers.{l}."; dv = dev_of(p + "input_layernorm.weight")
        for nm in ("gate_proj", "up_proj", "down_proj"):
            w = dequant(d[f"e.{nm}.q"], d[f"e.{nm}.s"], d[f"e.{nm}.z"], eb, eg, dv).to(torch.bfloat16)
            for j, e in enumerate(ki): put(p + f"mlp.experts.{e}.{nm}.weight", w[j].contiguous())
        gw = torch.zeros(NE, HID, dtype=torch.bfloat16); gw[ki] = d["router_kept"]; put(p + "mlp.gate.weight", gw)
        for nm in ("q_proj", "k_proj", "v_proj", "o_proj"): put(p + f"self_attn.{nm}.weight", dequant(d[f"a.{nm}.q"], d[f"a.{nm}.s"], d[f"a.{nm}.z"], ab, 128, dv).to(torch.bfloat16))
        for nm in ("input_layernorm", "post_attention_layernorm", "self_attn.q_norm", "self_attn.k_norm"): put(p + nm + ".weight", d["n." + nm])
        if l % 10 == 0: print("loaded layer", l, flush=True)
    dispatch_model(m, device_map=dm); m.eval()
    for l, lay in enumerate(m.model.layers):
        gw = GateW(lay.mlp.gate); lay.mlp.gate = gw; lay.mlp.forward = types.MethodType(moe_fwd, lay.mlp)
        if len(keeps[l]) < NE:
            mk = torch.zeros(NE, device=gw.g.weight.device); mk[[e for e in range(NE) if e not in set(keeps[l])]] = float("-inf"); gw.mask = mk
    for p in m.parameters(): p.requires_grad_(False)
    tok = AutoTokenizer.from_pretrained(ART); tok.padding_side = "left"
    apply_carrier(m, ART); return m, tok

def apply_carrier(m, ART, path=None, sign=1.0):
    """add the shipped feedback carrier (dense fp16 delta of o_proj, edited layers only) to the dequantised o_proj; no-op if the artifact has none"""
    path = path or os.path.join(ART, CARRIER)
    if not os.path.exists(path): return 0
    c = load_file(path)
    for k, v in c.items(): w = m.model.layers[int(k.split(".")[1])].self_attn.o_proj.weight; w.data.add_(sign * v.to(w.device, w.dtype))
    return len(c)

@torch.no_grad()
def generate(m, tok, texts, max_new=5, bs=64):
    """texts = full user messages; returns the greedy completions (chat template, exact generated text)"""
    order = sorted(range(len(texts)), key=lambda i: len(texts[i])); outs = [None] * len(texts)
    for s in range(0, len(order), bs):
        idx = order[s:s + bs]; pr = [tok.apply_chat_template([{"role": "user", "content": texts[i]}], tokenize=False, add_generation_prompt=True) for i in idx]
        enc = tok(pr, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda:0"); L = enc.input_ids.shape[1]
        g = m.generate(**enc, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.pad_token_id)
        for i, t in zip(idx, tok.batch_decode(g[:, L:], skip_special_tokens=True)): outs[i] = t.strip()
    return outs
