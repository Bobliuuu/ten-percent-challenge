"""Physically-pruned Qwen3-MoE loader: builds the model on meta, then streams safetensors, keeping per layer only the surviving experts (gate rows sliced).
Per-layer expert counts may differ (sensitivity-based allocation). Optional fake-quant (RTN group-wise, bits per layer-group) on survivors via qfn(name, tensor)->tensor.
Result is a normal HF model whose MoE blocks have kept[l] experts. KV/attention untouched (bf16)."""
import torch, json, os, glob, torch.nn as nn
from safetensors import safe_open
import transformers.models.qwen3_moe.modeling_qwen3_moe as M
from transformers import AutoConfig

class QLinear(nn.Module):
    """int8 codes (b-bit symmetric RTN, group 128) + fp16 scales; dequantised per call. Memory = 1 B/weight (real packed b-bit size accounted separately)."""
    def __init__(self, w, bits, group=128):
        super().__init__()
        sh = w.shape; wf = w.float().reshape(-1, group); qm = 2 ** (bits - 1) - 1
        s = wf.abs().amax(1, keepdim=True).clamp(min=1e-8) / qm
        c = (wf / s).round().clamp(-qm - 1, qm).to(torch.int8)
        self.packed = bits <= 4
        if self.packed:
            u = (c.flatten() + 8).to(torch.uint8); c = u[0::2] | (u[1::2] << 4)
        self.register_buffer("codes", c); self.register_buffer("scales", s.half()); self.shape = sh
    def forward(self, x):
        c = self.codes
        if self.packed: c = (torch.stack([c & 15, c >> 4], -1).reshape(-1, 128).to(torch.int8) - 8)
        w = (c.float() * self.scales.float()).reshape(self.shape).to(x.dtype)
        return torch.nn.functional.linear(x, w)

def load_pruned(snap, keep, devmap_fn, qfn=None, dtype=torch.bfloat16, ebits=0, ref_bits=None):
    """snap: local snapshot dir. keep: dict layer-> sorted list of kept expert idx (None = keep all). devmap_fn(layer_idx or 'embed'/'head') -> torch device."""
    cfg = AutoConfig.from_pretrained(snap)
    with torch.device("meta"):
        model = M.Qwen3MoeForCausalLM(cfg)
    wm = json.load(open(f"{snap}/model.safetensors.index.json"))["weight_map"]
    files = {}
    def get(name):
        f = wm[name]
        if f not in files: files[f] = safe_open(f"{snap}/{f}", "pt", device="cpu")
        return files[f].get_tensor(name)
    def put(mod, pname, t, dev):
        mod._parameters[pname] = nn.Parameter(t.to(dev, dtype), requires_grad=False)
    nl = cfg.num_hidden_layers
    dev = devmap_fn("embed"); put(model.model.embed_tokens, "weight", get("model.embed_tokens.weight"), dev)
    for l in range(nl):
        d = devmap_fn(l); lay = model.model.layers[l]; pre = f"model.layers.{l}."
        for n in ("input_layernorm", "post_attention_layernorm"): put(getattr(lay, n), "weight", get(pre + n + ".weight"), d)
        at = lay.self_attn
        for n in ("q_proj", "k_proj", "v_proj", "o_proj"):
            w = get(pre + f"self_attn.{n}.weight")
            if qfn: w = qfn(pre + f"self_attn.{n}", w)
            put(getattr(at, n), "weight", w, d)
        for n in ("q_norm", "k_norm"): put(getattr(at, n), "weight", get(pre + f"self_attn.{n}.weight"), d)
        E = cfg.num_experts; kp = keep.get(l) if keep else None
        kp = list(range(E)) if kp is None else list(kp)
        mlp = lay.mlp
        g = get(pre + "mlp.gate.weight")[kp]
        mlp.gate = nn.Linear(cfg.hidden_size, len(kp), bias=False, device="meta"); put(mlp.gate, "weight", g, d)
        mlp.num_experts = len(kp)
        mlp.experts = nn.ModuleList([M.Qwen3MoeMLP(cfg, intermediate_size=cfg.moe_intermediate_size) for _ in kp]) if False else nn.ModuleList()
        with torch.device("meta"):
            mlp.experts = nn.ModuleList([M.Qwen3MoeMLP(cfg, intermediate_size=cfg.moe_intermediate_size) for _ in kp])
        for j, e in enumerate(kp):
            for n in ("gate_proj", "up_proj", "down_proj"):
                nm = pre + f"mlp.experts.{e}.{n}"
                w = get(nm + ".weight")
                if qfn: w = qfn(nm, w)
                if ebits:
                    setattr(mlp.experts[j], n, QLinear(w.to(d, dtype), ebits)); continue
                put(getattr(mlp.experts[j], n), "weight", w, d)
        mlp.top_k = min(cfg.num_experts_per_tok, len(kp))
        if l % 8 == 0: print("loaded layer", l, flush=True)
    dev = devmap_fn("head"); put(model.model.norm, "weight", get("model.norm.weight"), dev)
    put(model.lm_head, "weight", get("lm_head.weight"), dev)
    # rotary buffers live on meta: rebuild
    model.model.rotary_emb = M.Qwen3MoeRotaryEmbedding(cfg, device=devmap_fn(0))
    model.config = cfg
    model.eval()
    return model

def rtn_fake(w, bits, group=128):
    """symmetric group-wise RTN fake-quant (a baseline; GPTQ would be better)."""
    wf = w.float(); sh = wf.shape
    wf = wf.reshape(-1, group)
    qm = 2 ** (bits - 1) - 1
    s = wf.abs().amax(1, keepdim=True).clamp(min=1e-8) / qm
    return ((wf / s).round().clamp(-qm - 1, qm) * s).reshape(sh).to(w.dtype)

def _mv(o, dev):
    if torch.is_tensor(o): return o.to(dev)
    if isinstance(o, tuple): return tuple(_mv(x, dev) for x in o)
    if isinstance(o, list): return [_mv(x, dev) for x in o]
    if isinstance(o, dict): return {k: _mv(v, dev) for k, v in o.items()}
    return o

def add_dispatch_hooks(model, devmap_fn):
    for l, lay in enumerate(model.model.layers):
        d = devmap_fn(l)
        lay.register_forward_pre_hook(lambda m, a, k, d=d: (_mv(a, d), _mv(k, d)), with_kwargs=True)
    dh = devmap_fn("head")
    model.model.norm.register_forward_pre_hook(lambda m, a, k, d=dh: (_mv(a, d), k), with_kwargs=True)
    model.lm_head.register_forward_pre_hook(lambda m, a, k, d=dh: (_mv(a, d), k), with_kwargs=True)
    model.model.embed_tokens.register_forward_pre_hook(lambda m, a, k, d=devmap_fn("embed"): (_mv(a, d), k), with_kwargs=True)
    return model
