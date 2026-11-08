"""L7 shared code: BitNet-b1.58-2B-4T (released packed ternary) + vocab-prefix trim + RTN embedding quant + closed-form head-row feedback edit. No training."""
import os, sys, json, re, lzma, math, numpy as np, torch
from huggingface_hub import snapshot_download
MODEL = "microsoft/bitnet-b1.58-2B-4T"
LABELS = ["Alarm", "Weather", "Music", "Calendar", "Lists", "News", "Transport"]
def prompt_for(labels, req):
    return ("You are the router of a voice assistant. Decide which skill should handle the user's request.\nAnswer with exactly one word from: " + ", ".join(labels) + ".\n\nRequest: " + req)
class Sys:
    def __init__(self, N=74167, bits=3, g=64, device="cuda"):
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from tokenizers import Tokenizer
        self.src = snapshot_download(MODEL, allow_patterns=["*.json", "*.safetensors", "*.jinja"]); self.htok = AutoTokenizer.from_pretrained(self.src)
        self.TJ = json.load(open(self.src + "/tokenizer.json")); self.TC = json.load(open(self.src + "/tokenizer_config.json"))
        self.model = AutoModelForCausalLM.from_pretrained(self.src, torch_dtype=torch.bfloat16, device_map=device); self.dev = device
        self.build(N, bits, g)
    def build(self, N, bits, g):
        from tokenizers import Tokenizer
        TJ, TC = self.TJ, self.TC; chat_t = TC.get("chat_template") or ""
        names = set(re.findall(r"<[^<>\s]{1,40}>", chat_t)) | {TC.get(k) if isinstance(TC.get(k), str) else (TC.get(k) or {}).get("content") for k in ("bos_token", "eos_token", "pad_token")}
        spec = sorted(a["id"] for a in TJ["added_tokens"] if a["content"] in names)
        tk_full = Tokenizer.from_file(self.src + "/tokenizer.json")
        extra = {i for s in [prompt_for(LABELS + ["Cooking"], ""), chat_t] for i in tk_full.encode(s, add_special_tokens=False).ids}
        keep = sorted(set(range(N)) | set(spec) | extra); mp = {o: n for n, o in enumerate(keep)}; m = TJ["model"] = json.loads(json.dumps(TJ["model"]))
        nv = {t: mp[i] for t, i in m["vocab"].items() if i in mp}; tj = dict(TJ); tj["added_tokens"] = [dict(a, id=mp[a["id"]]) for a in TJ["added_tokens"] if a["id"] in mp]
        for a in tj["added_tokens"]: nv[a["content"]] = a["id"]
        m["vocab"] = nv; ok = lambda mg: (lambda a, b: a in nv and b in nv and (a + b) in nv)(*(mg if isinstance(mg, list) else mg.split(" "))); m["merges"] = [x for x in m["merges"] if ok(x)]
        def remap(o):
            if isinstance(o, dict): return {k: ([mp.get(x, x) for x in v] if k == "ids" and isinstance(v, list) else (mp.get(v, v) if k == "id" and isinstance(v, int) else remap(v))) for k, v in o.items()}
            if isinstance(o, list): return [remap(x) for x in o]
            return o
        for kk in ("post_processor", "decoder", "normalizer", "pre_tokenizer"):
            if tj.get(kk): tj[kk] = remap(tj[kk])
        self.tk = Tokenizer.from_str(json.dumps(tj, ensure_ascii=False)); self.mp = mp; self.keep = keep
        self.tok_bytes = len(lzma.compress(json.dumps(tj, ensure_ascii=False, separators=(",", ":")).encode(), preset=9 | lzma.PRESET_EXTREME)) + 3000
        emb = self.model.get_input_embeddings().weight.data
        if not hasattr(self, "orig_emb"): self.orig_emb = emb.clone()
        W = self.orig_emb[torch.tensor(keep, device=emb.device)].float(); R, C = W.shape; x = W.reshape(R, C // g, g)
        lo = x.amin(-1, keepdim=True).half().float(); hi = x.amax(-1, keepdim=True); sc = ((hi - lo) / (2 ** bits - 1)).half().float().clamp_min(1e-8)
        self.We = ((x - lo) / sc).round().clamp(0, 2 ** bits - 1).mul(sc).add(lo).reshape(R, C).to(torch.bfloat16); self.emb_bytes = (R * C * bits + 7) // 8 + R * (C // g) * 4 + (len(keep) + 7) // 8
        e = torch.nn.Embedding(R, C).to(self.dev, torch.bfloat16); e.weight.data.copy_(self.We); self.model.set_input_embeddings(e)
        h = torch.nn.Linear(C, R, bias=False).to(self.dev, torch.bfloat16); h.weight.data.copy_(self.We); self.model.set_output_embeddings(h); self.model.config.vocab_size = R   # UNTIED head so edits touch output rows only
        self.head0 = self.We.clone(); self.eos = [mp[i] for i in [self.htok.convert_tokens_to_ids("<|eot_id|>"), self.htok.eos_token_id] if i in mp]
    def ids(self, req, labels):
        s = self.htok.apply_chat_template([{"role": "user", "content": prompt_for(labels, req)}], add_generation_prompt=True, tokenize=False)
        return self.tk.encode(s, add_special_tokens=False).ids
    def pad(self, L):
        ml = max(map(len, L)); pad = self.eos[0]; a = torch.full((len(L), ml), pad); m = torch.zeros((len(L), ml), dtype=torch.long)
        for i, x in enumerate(L): a[i, ml - len(x):] = torch.tensor(x); m[i, ml - len(x):] = 1
        return a.to(self.dev), m.to(self.dev)
    @torch.no_grad()
    def hidden(self, reqs, labels, bs=32):
        out = []
        for i in range(0, len(reqs), bs):
            a, m = self.pad([self.ids(r, labels) for r in reqs[i:i + bs]]); out.append(self.model.model(input_ids=a, attention_mask=m).last_hidden_state[:, -1].float())
        return torch.cat(out)
    @torch.no_grad()
    def answer(self, reqs, labels, bs=32):
        res = []
        for i in range(0, len(reqs), bs):
            a, m = self.pad([self.ids(r, labels) for r in reqs[i:i + bs]]); o = self.model.generate(input_ids=a, attention_mask=m, max_new_tokens=6, do_sample=False, eos_token_id=self.eos, pad_token_id=self.eos[0])
            for r in range(len(a)):
                g = o[r, a.shape[1]:].tolist(); cut = next((q for q, t in enumerate(g) if t in self.eos), len(g)); res.append(self.tk.decode(g[:cut]).strip().strip(".").strip())
        return res
    def first_tok(self, word): return self.tk.encode(word, add_special_tokens=False).ids[0]
    @torch.no_grad()
    def edit(self, H, targets, margin=2.0, lam=1e-2):
        """closed form: for each target label token t, add dW_t = H_t^T (H_t H_t^T + lam I)^-1 r  (r = logit gap + margin vs best competitor), min-norm in hidden space. returns dict tok->delta, list of edited tokens"""
        Wh = self.model.get_output_embeddings().weight; logits = H.to(Wh.dtype) @ Wh.T; logits = logits.float(); edited = {}
        for t in sorted(set(targets)):
            idx = [i for i, x in enumerate(targets) if x == t]; Ht = H[idx]; lt = logits[idx]; comp = lt.clone(); comp[:, t] = -1e9; gap = comp.max(1).values - lt[:, t]
            r = (gap + margin).clamp_min(0); K = Ht @ Ht.T; d = torch.linalg.solve(K + lam * K.diagonal().mean() * torch.eye(len(idx), device=K.device), r)
            edited[t] = (Ht.T @ d)
        with torch.no_grad():
            for t, dl in edited.items(): Wh.data[t] += dl.to(Wh.dtype)
        return edited
    def reset_head(self):
        with torch.no_grad(): self.model.get_output_embeddings().weight.data.copy_(self.head0)
