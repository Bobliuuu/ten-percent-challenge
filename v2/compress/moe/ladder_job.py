"""MoE scaling ladder (cheap): same recipe family (router-frequency pruning to k of E experts + Hadamard-free grouped RTN at b bits entropy-estimated) on OLMoE-1B-7B and Qwen1.5-MoE-A2.7B,
measuring teacher-forced top-1 agreement + KL vs bf16 base on evalA gold text. Fit loss vs (kept fraction, bits) later to predict Qwen3-30B-A3B. RTN only (no GPTQ) so it is a *relative* scaling probe.
modal run --detach ladder_job.py"""
import modal, json, os
app = modal.App("tenpct-moe-ladder")
img = modal.Image.debian_slim().pip_install("torch==2.6.0", "transformers==4.57.1", "numpy", "accelerate")
vol = modal.Volume.from_name("tenpct-v2-hf", create_if_missing=True)
MODELS = ["allenai/OLMoE-1B-7B-0924-Instruct", "Qwen/Qwen1.5-MoE-A2.7B-Chat"]
FR = [1.0, 0.75, 0.5, 0.375, 0.25]; BITS = [16, 4, 3, 2]

@app.function(image=img, gpu=["A100-80GB", "H100", "L40S", "A10G"], timeout=4 * 3600, memory=64000, volumes={"/hf": vol}, max_containers=4)
def run(model_id):
    import math, time, numpy as np, torch
    os.environ["HF_HOME"] = "/hf"
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16, device_map={"": 0}).eval()
    rows = [json.loads(x) for x in open("/hf/moe_data/evalA.jsonl")]; cal = [json.loads(x) for x in open("/hf/moe_data/calibA.jsonl")]
    def prompt(r): return tok.apply_chat_template(r["msgs"], add_generation_prompt=True, tokenize=False)
    items = []
    for r in rows:
        if not r["gold"]: continue
        p = tok.encode(prompt(r))[-250:]; a = tok.encode(r["gold"]); items.append(((p + a)[:384], len(p)))
        if len(items) >= 140: break
    moes = [m for m in model.modules() if hasattr(m, "experts") and hasattr(m, "gate")]
    NE = len(moes[0].experts); print(model_id, "moe layers", len(moes), "experts", NE, flush=True)
    cnt = [torch.zeros(NE) for _ in moes]
    def mk(i):
        def hook(m, inp, out):
            top = out.topk(getattr(model.config, "num_experts_per_tok", 8), -1)[1].flatten(); cnt[i].index_add_(0, top.cpu(), torch.ones(top.numel()))
        return hook
    hs = [m.gate.register_forward_hook(mk(i)) for i, m in enumerate(moes)]
    ids_all = []
    for r in cal: ids_all += tok.encode(prompt(r) + (r["gold"] or ""))
    x = torch.tensor(ids_all[:128 * 512]).view(-1, 512)
    with torch.no_grad():
        for i in range(0, len(x), 8): model(input_ids=x[i:i + 8].cuda())
    for h in hs: h.remove()
    orig = {n: p.detach().cpu().clone() for n, p in model.named_parameters() if ".experts." in n}
    @torch.no_grad()
    def tf():
        out = []
        for ids, pl in items:
            lg = model(input_ids=torch.tensor([ids]).cuda()).logits[0, pl - 1:-1].float(); out.append(torch.log_softmax(lg, -1).cpu())
        return out
    base = tf(); res = []
    def rtn(W, bits, g=128):
        if bits >= 16: return W, 16.0
        sh = W.shape; Wg = W.float().reshape(-1, g); s = (Wg.abs().amax(-1, keepdim=True) / (2 ** (bits - 1) - 0.5)).clamp_min(1e-8)
        q = torch.clamp(torch.round(Wg / s), -(2 ** (bits - 1)), 2 ** (bits - 1) - 1); return (q * s).reshape(sh).to(W.dtype), bits + 16 / g
    for fr in FR:
        for bits in BITS:
            for n, p in model.named_parameters():
                if n in orig: p.data.copy_(orig[n])
            hs = []
            k = max(1, int(round(fr * NE)))
            for i, m in enumerate(moes):
                keep = torch.zeros(NE, dtype=torch.bool); keep[cnt[i].topk(k)[1]] = True; km = keep
                hs.append(m.gate.register_forward_hook(lambda mod, inp, out, km=km: out.masked_fill(~km.to(out.device), -1e9)))
            nbits = 0; npar = 0
            for n, p in model.named_parameters():
                if ".experts." in n and p.dim() == 2:
                    p.data, bpw = rtn(p.data, bits); npar += p.numel(); nbits += p.numel() * bpw
            cur = tf(); agree = n_ = 0; kl = 0.0
            for b_, c_ in zip(base, cur):
                agree += int((b_.argmax(-1) == c_.argmax(-1)).sum()); n_ += b_.shape[0]
                kl += float((b_.exp() * (b_ - c_)).sum(-1).sum())
            for h in hs: h.remove()
            r = dict(model=model_id, frac=fr, bits=bits, top1=agree / n_, kl=kl / n_, expert_bytes_frac_of_bf16=fr * (bits + (16 / 128 if bits < 16 else 0)) / 16)
            print("RES", json.dumps(r), flush=True); res.append(r)
    os.makedirs("/hf/moe_res", exist_ok=True); json.dump(res, open(f"/hf/moe_res/ladder_{model_id.replace('/', '_')}.json", "w")); vol.commit(); return res

@app.local_entrypoint()
def main():
    hs = [run.spawn(m) for m in MODELS]
    for h in hs:
        try: print(h.get())
        except Exception as e: print("ERR", repr(e))
