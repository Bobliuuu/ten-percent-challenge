"""L9 feedback command on the PACKED artifact (diffuse closed-form edit of the existing o_proj tensors of the last layers; no optimiser, no new module, no prompt memory).
usage: python feedback.py ART ANNOT.jsonl CAL.jsonl OUT_CARRIER.safetensors [real|shuffled|empty]
ANNOT rows: {user, answer} = the user's corrected label for an input the model got wrong (<=25). The carrier holds the accumulated dense fp16 delta of o_proj in the edited layers
(shipped bytes: LAYERS x 4096 x 8192 x 2 B); answer.py adds it to the dequantised o_proj at load. Mechanism identical to fb235.py (measured there with controls)."""
import sys, os, json, random, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pack235 as P
from safetensors.torch import save_file
LAYERS = list(range(86, 94)); LAM, AWT, MARGIN = 1000.0, 10.0, 16.0

@torch.no_grad()
def make_carrier(m, tok, ann, cal, mode="real", seed=0):
    chat = lambda u: tok.apply_chat_template([{"role": "user", "content": u}], tokenize=False, add_generation_prompt=True)
    ids = lambda u: tok(chat(u), add_special_tokens=False).input_ids; EOS = "<|im_end|>"; tgt = lambda lab: tok(lab + EOS, add_special_tokens=False).input_ids
    layers = m.model.layers; getp = lambda l: layers[l].self_attn.o_proj.weight; cap = {}; hooks = []
    for l in LAYERS: hooks.append(layers[l].self_attn.o_proj.register_forward_pre_hook(lambda mod, a, l=l: cap.__setitem__(l, a[0].detach())))
    hooks.append(m.model.norm.register_forward_pre_hook(lambda mod, a: cap.__setitem__("fin", a[0].detach())))
    WU = m.lm_head.weight.float(); G = m.model.norm.weight.float().to(WU.device); eps = m.config.rms_norm_eps; DI = m.config.num_attention_heads * m.config.head_dim
    def fwd(seqs):
        L = max(len(s) for s in seqs); x = torch.full((len(seqs), L), tok.pad_token_id); am = torch.zeros((len(seqs), L), dtype=torch.long)
        for r, s in enumerate(seqs): x[r, :len(s)] = torch.tensor(s); am[r, :len(s)] = 1
        m.model(input_ids=x.to("cuda:0"), attention_mask=am.to("cuda:0"), use_cache=False)
    Cs = {l: torch.zeros(DI, DI, device=getp(l).device) for l in LAYERS}; Ca = {l: torch.zeros(DI, DI, device=getp(l).device) for l in LAYERS}; nt = na = 0
    for s0 in range(0, len(cal), 16):
        bt = cal[s0:s0 + 16]; seqs = [ids(r["user"]) + tgt(r["answer"]) for r in bt]; fwd(seqs)
        for l in LAYERS:
            for b, r in enumerate(bt):
                k = cap[l][b, :len(seqs[b])].float(); Cs[l] += k.T @ k; ka = k[len(ids(r["user"])) - 1:]; Ca[l] += ka.T @ ka
        nt += sum(len(s) for s in seqs); na += sum(len(tgt(r["answer"])) for r in bt)
    for l in LAYERS: Cs[l] /= nt; Ca[l] /= na
    labs = [r["answer"] for r in ann]
    if mode == "shuffled": random.Random(seed).shuffle(labs)
    items = [(ids(r["user"]), lab) for r, lab in zip(ann, labs)] if mode != "empty" else []
    delta = {l: torch.zeros_like(getp(l), dtype=torch.float32) for l in LAYERS}
    if items:
        seqs = [i + tgt(lab) for i, lab in items]
        def measure():
            fwd(seqs); x = cap["fin"].float(); rows = []
            for b, (i, lab) in enumerate(items):
                for j, tj in enumerate(tgt(lab)):
                    p = len(i) - 1 + j; xx = x[b, p].to(WU.device); rows.append((b, p, tj, (G * xx / torch.sqrt((xx ** 2).mean() + eps)) @ WU.T))
            return x, rows
        x0, rows = measure(); S = []; x0 = x0.to(WU.device)
        for b, p, tj, lg in rows:
            c = int(torch.argmax(lg))
            if c != tj:
                xx = x0[b, p]; u = G * (WU[tj] - WU[c]); rms = torch.sqrt((xx ** 2).mean() + eps); S.append((b, p, (xx + rms * (float(lg[c] - lg[tj]) + MARGIN) * u / (u @ u)).to("cuda:0")))
        for i, l in enumerate(LAYERS):
            x, _ = measure(); K = torch.stack([cap[l][b, p].float() for b, p, _ in S]); R = torch.stack([tg.to(K.device) - x[b, p].to(K.device) for b, p, tg in S]) / (len(LAYERS) - i)
            A = LAM * (Cs[l] + AWT * Ca[l]) + K.T @ K + 1e-4 * torch.eye(DI, device=K.device) * (LAM * (Cs[l] + AWT * Ca[l])).diagonal().mean()
            dW = torch.linalg.solve(A, (R.T @ K).T).T; getp(l).data.add_(dW.to(getp(l).dtype)); delta[l] += dW
    for h in hooks: h.remove()
    return {f"dW.{l}": delta[l].half().cpu().contiguous() for l in LAYERS}

if __name__ == "__main__":
    ART, ANN, CAL, OUTC = sys.argv[1:5]; mode = sys.argv[5] if len(sys.argv) > 5 else "real"
    ann = [json.loads(l) for l in open(ANN)]; cal = [json.loads(l) for l in open(CAL)]
    m, tok = P.build_model(ART, os.environ.get("L9_GPUMEM", "120GiB")); car = make_carrier(m, tok, ann, cal, mode); save_file(car, OUTC)
    print("carrier bytes", sum(v.numel() * 2 for v in car.values()))
