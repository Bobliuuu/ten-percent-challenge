"""L9 feedback command on the PACKED artifact: closed-form, covariance-protected edit of the EXISTING o_proj tensors of the last 8 layers (no optimiser, no new module, no prompt memory, no lookup).
usage: python feedback.py ART ANNOT.jsonl [real|shuffled|wrong|empty]      ANNOT rows: {"user": <full routing prompt>, "answer": <the user's corrected label>} (<=25 marked mistakes; the label may be a new word).
Reads ART/feedback_cal.jsonl (shipped calibration prompts + gold answers; key covariance only, never labels) and ADDS the delta to ART/fb_carrier.safetensors (dense fp16, layers 86-93 of o_proj, 537 MB), which
answer.py / packed_loader.build apply at load. Feedback can be given repeatedly (deltas accumulate). Recipe (L11 ablation): o_proj, margin 32, last 8 layers; lam picked on dev."""
import sys, os, json, random, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import packed_loader as PL
from safetensors.torch import save_file, load_file
LAYERS = list(range(86, 94)); LAM, AWT, MARGIN = float(os.environ.get("FB_LAM", 5000)), float(os.environ.get("FB_AWT", 10)), float(os.environ.get("FB_MARGIN", 32))
EOS = "<|im_end|>"

class Ctx:
    def __init__(s, m, tok, cal, layers=LAYERS):
        s.m, s.tok, s.layers = m, tok, layers
        s.chat = lambda u: tok.apply_chat_template([{"role": "user", "content": u}], tokenize=False, add_generation_prompt=True)
        s.ids = lambda u: tok(s.chat(u), add_special_tokens=False).input_ids; s.tgt = lambda lab: tok(lab + EOS, add_special_tokens=False).input_ids
        s.o = lambda l: m.model.layers[l].self_attn.o_proj.weight; s.cap = {}; s.hooks = []
        for l in layers: s.hooks.append(m.model.layers[l].self_attn.o_proj.register_forward_pre_hook(lambda mod, a, l=l: s.cap.__setitem__(l, a[0].detach())))
        s.hooks.append(m.model.norm.register_forward_pre_hook(lambda mod, a: s.cap.__setitem__("fin", a[0].detach())))
        s.WU = m.lm_head.weight.float(); s.G = m.model.norm.weight.float().to(s.WU.device); s.eps = m.config.rms_norm_eps; s.DI = m.config.num_attention_heads * m.config.head_dim
        s.cov(cal)
    def fwd(s, seqs):
        L = max(len(x) for x in seqs); x = torch.full((len(seqs), L), s.tok.pad_token_id); am = torch.zeros((len(seqs), L), dtype=torch.long)
        for r, q in enumerate(seqs): x[r, :len(q)] = torch.tensor(q); am[r, :len(q)] = 1
        s.m.model(input_ids=x.to("cuda:0"), attention_mask=am.to("cuda:0"), use_cache=False)
    @torch.no_grad()
    def cov(s, cal):   # key covariance of o_proj inputs over calibration tokens (Cs) and at answer positions (Ca); labels never enter
        s.Cs = {l: torch.zeros(s.DI, s.DI, device=s.o(l).device) for l in s.layers}; s.Ca = {l: torch.zeros(s.DI, s.DI, device=s.o(l).device) for l in s.layers}; nt = na = 0
        for s0 in range(0, len(cal), 16):
            bt = cal[s0:s0 + 16]; seqs = [s.ids(r["user"]) + s.tgt(r["answer"]) for r in bt]; s.fwd(seqs)
            for l in s.layers:
                for b, r in enumerate(bt):
                    k = s.cap[l][b, :len(seqs[b])].float(); s.Cs[l] += k.T @ k; ka = k[len(s.ids(r["user"])) - 1:]; s.Ca[l] += ka.T @ ka
            nt += sum(len(q) for q in seqs); na += sum(len(s.tgt(r["answer"])) for r in bt)
        for l in s.layers: s.Cs[l] /= nt; s.Ca[l] /= na

@torch.no_grad()
def make_carrier(X, ann, mode="real", seed=0, lam=LAM, awt=AWT, margin=MARGIN, apply=True):
    """returns {dW.l: fp16 delta}; with apply=True the delta is also added to the live o_proj (caller undoes with undo())"""
    m, tok, LAY = X.m, X.tok, X.layers; labs = [r["answer"] for r in ann]
    if mode == "shuffled": random.Random(seed).shuffle(labs)   # permute the SAME annotation labels among the annotated inputs
    items = [(X.ids(r["user"]), lab) for r, lab in zip(ann, labs)] if mode != "empty" else []
    delta = {l: torch.zeros_like(X.o(l), dtype=torch.float32) for l in LAY}
    if items:
        seqs = [i + X.tgt(lab) for i, lab in items]
        def measure():
            X.fwd(seqs); x = X.cap["fin"].float(); rows = []
            for b, (i, lab) in enumerate(items):
                for j, tj in enumerate(X.tgt(lab)):
                    p = len(i) - 1 + j; xx = x[b, p].to(X.WU.device); rows.append((b, p, tj, (X.G * xx / torch.sqrt((xx ** 2).mean() + X.eps)) @ X.WU.T))
            return x, rows
        x0, rows = measure(); S = []; x0 = x0.to(X.WU.device)
        for b, p, tj, lg in rows:
            c = int(torch.argmax(lg))
            if c != tj:
                xx = x0[b, p]; u = X.G * (X.WU[tj] - X.WU[c]); rms = torch.sqrt((xx ** 2).mean() + X.eps); S.append((b, p, (xx + rms * (float(lg[c] - lg[tj]) + margin) * u / (u @ u)).to("cuda:0")))
        for i, l in enumerate(LAY):
            x, _ = measure()
            if not S: break
            K = torch.stack([X.cap[l][b, p].float() for b, p, _ in S]); R = torch.stack([tg.to(K.device) - x[b, p].to(K.device) for b, p, tg in S]) / (len(LAY) - i)
            C = lam * (X.Cs[l] + awt * X.Ca[l]); A = C + K.T @ K + 1e-4 * torch.eye(X.DI, device=K.device) * C.diagonal().mean()
            dW = torch.linalg.solve(A, (R.T @ K).T).T; X.o(l).data.add_(dW.to(X.o(l).dtype)); delta[l] += dW
    out = {f"dW.{l}": delta[l].half().cpu().contiguous() for l in LAY}
    if not apply: undo(X, out)
    return out
def undo(X, car):
    for k, v in car.items(): w = X.o(int(k.split(".")[1])); w.data.sub_(v.to(w.device, w.dtype))

if __name__ == "__main__":
    ART, ANN = sys.argv[1:3]; mode = sys.argv[3] if len(sys.argv) > 3 else "real"
    ann = [json.loads(l) for l in open(ANN)]; assert len(ann) <= 25, "at most 25 annotated items"
    cal = [json.loads(l) for l in open(os.path.join(ART, "feedback_cal.jsonl"))]
    m, tok = PL.build(ART); X = Ctx(m, tok, cal); car = make_carrier(X, ann, mode); path = os.path.join(ART, PL.CARRIER)
    if os.path.exists(path):
        old = load_file(path); car = {k: (car[k].float() + old[k].float()).half() if k in old else car[k] for k in car}
    save_file(car, path); print("carrier written", path, "bytes", sum(v.numel() * 2 for v in car.values()))
