"""L2 MoE lib: patched Qwen3MoE sparse block with (a) saliency stats (freq, REAP, EAN, gate sum), (b) per-layer pruned-expert mask (equiv. to physically removing experts: logits -> -inf before softmax, norm_topk renormalises over survivors).
Works for Qwen3-30B-A3B and Qwen3-235B-A22B (same class). No training."""
import torch, torch.nn.functional as F, json, sys
import transformers.models.qwen3_moe.modeling_qwen3_moe as M

class Stats:
    def __init__(s, L, E, dev):
        s.cnt = torch.zeros(L, E, device=dev, dtype=torch.float64); s.reap = torch.zeros_like(s.cnt); s.ean = torch.zeros_like(s.cnt); s.gate = torch.zeros_like(s.cnt)
        s.rl_mean = torch.zeros(L, E, device=dev, dtype=torch.float64); s.ntok = torch.zeros(L, device=dev, dtype=torch.float64)
STATE = dict(stats=None, mask=None, valid=None)   # mask: dict layer-> bool tensor[E] True=pruned

def fwd(self, hidden_states):
    b, t, d = hidden_states.shape
    x = hidden_states.view(-1, d)
    logits = self.gate(x).float()
    st = STATE["stats"]; mk = STATE["mask"]; li = self._li
    if mk is not None and li in mk:
        logits = logits.masked_fill(mk[li].to(logits.device)[None, :], float("-inf"))
    p = F.softmax(logits, dim=1)
    w, sel = torch.topk(p, self.top_k, dim=-1)
    if self.norm_topk_prob: w = w / w.sum(-1, keepdim=True)
    w = w.to(x.dtype)
    out = torch.zeros_like(x)
    em = F.one_hot(sel, num_classes=self.num_experts).permute(2, 1, 0)
    vm = STATE["valid"]
    if st is not None:
        if vm is not None: vm = vm.reshape(-1).to(x.device)
        st.ntok[li] += (x.shape[0] if vm is None else vm.sum()); st.rl_mean[li] += (p if vm is None else p * vm[:, None]).sum(0).double()
    for e in torch.greater(em.sum(dim=(-1, -2)), 0).nonzero():
        e = int(e)
        idx, top_x = torch.where(em[e])
        y = self.experts[e](x[top_x])
        if st is not None:
            n = y.float().norm(dim=-1); g = w[top_x, idx].float()
            if vm is not None:
                v = vm[top_x].float(); n = n * v; g = g * v
                st.cnt[li, e] += v.sum().double() - len(top_x); st.cnt[li, e] += 0
            st.cnt[li, e] += len(top_x); st.ean[li, e] += n.sum().double(); st.reap[li, e] += (g * n).sum().double(); st.gate[li, e] += g.sum().double()
        out.index_add_(0, top_x, y * w[top_x, idx, None])
    return out.reshape(b, t, d)

def patch(model):
    M.Qwen3MoeSparseMoeBlock.forward = fwd
    L = 0
    for i, layer in enumerate(model.model.layers):
        if isinstance(layer.mlp, M.Qwen3MoeSparseMoeBlock): layer.mlp._li = i; L = max(L, i + 1)
    return L

def chat_prompt(tok, it, kw=None):
    tools = [{"type": "function", "function": t} for t in it["tools"]]
    return tok.apply_chat_template([{"role": "user", "content": it["query"]}], tools=tools, add_generation_prompt=True, tokenize=False, **(kw or {}))

def gold_qwen(it):
    return "\n".join("<tool_call>\n" + json.dumps(dict(name=g["name"], arguments=g["arguments"])) + "\n</tool_call>" for g in it["gold"]) or "I cannot help with that: none of the provided tools apply."

@torch.no_grad()
def generate(model, tok, items, bs=16, max_new=320, kw=None, log=None):
    outs = []
    order = sorted(range(len(items)), key=lambda i: -len(items[i]["query"]) - sum(len(json.dumps(t)) for t in items[i]["tools"]))
    res = {}
    tok.padding_side = "left"
    for s in range(0, len(order), bs):
        ids = order[s:s + bs]
        pr = [chat_prompt(tok, items[i], kw) for i in ids]
        enc = tok(pr, return_tensors="pt", padding=True).to(model.device)
        g = model.generate(**enc, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.pad_token_id)
        for i, o in zip(ids, g[:, enc.input_ids.shape[1]:]): res[i] = tok.decode(o, skip_special_tokens=False)
        if log: print(f"gen {s+len(ids)}/{len(items)}", file=log, flush=True)
    return [res[i] for i in range(len(items))]
