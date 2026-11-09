"""bank6q: simulated (fake-quant, dequantised-in-place bf16) expert quantisation for Qwen3.5-35B-A3B on BANK6. SEL only. No locked-exam data is loaded anywhere in this file."""
import json, os, re, sys, time, math
import torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, "/root/q"); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
COMMIT = lambda: None
RES = "/q/res"
FP16_REF = 69321221376; CAP = 0.099
# shipped-format byte model (derived from the measured shipped k102 3-bit pack, 6,705,899,325 B total):
# rest-of-artifact (attn6/emb4/head4/shared expert/norms/tokenizer/config/manifest/code) = total - experts - router rows
SHIP_TOTAL_K102_E3 = 6705899325; EXP_PARAMS = 3 * 2048 * 512
RTN3_EXPERT_B = int(round(EXP_PARAMS * (3 / 8 + 2 / 128 + 3 / (8 * 128))))
FIXED_REST = SHIP_TOTAL_K102_E3 - RTN3_EXPERT_B * 102 * 40 - 102 * 40 * 2048 * 2
MANIFEST_EXTRA = 40000   # kept-index lists for k up to 160 (shipped manifest has k102 lists inside FIXED_REST already)

G = dict(valid=None, col=None)
def expert_fwd(self, hs, top_k_index, top_k_weights):
    """identical math to Qwen3_5MoeExperts.forward (loop form) + optional per-expert Hessian hooks on the inputs of gate_up and down"""
    out = torch.zeros_like(hs)
    with torch.no_grad():
        em = F.one_hot(top_k_index, num_classes=self.num_experts + 1).permute(2, 1, 0)
        hit = torch.greater(em.sum(dim=(-1, -2)), 0).nonzero()
    col = G["col"].get(id(self)) if G["col"] else None
    for ei in hit:
        e = int(ei[0])
        if e == self.num_experts: continue
        top_k_pos, token_idx = torch.where(em[e])
        cur = hs[token_idx]
        gate, up = F.linear(cur, self.gate_up_proj[e]).chunk(2, dim=-1)
        h = self.act_fn(gate) * up
        if col is not None and col["pos"][e] >= 0:
            v = G["valid"][token_idx] > 0; p = col["pos"][e]; X = cur[v].float(); Y = h[v].float()
            col["gu"][p] += X.T @ X; col["dn"][p] += Y.T @ Y; col["n"][p] += int(v.sum())
        o = F.linear(h, self.down_proj[e]) * top_k_weights[token_idx, top_k_pos, None]
        out.index_add_(0, token_idx, o.to(out.dtype))
    return out

def blocks_of(model):
    bl = []
    for name, mod in model.named_modules():
        ch = dict(mod.named_children())
        if "experts" in ch and "gate" in ch: bl.append(mod)
    return bl

def load_model():
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import core
    tok = AutoTokenizer.from_pretrained(core.MID); tok.padding_side = "left"
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(core.MID, dtype=torch.bfloat16, device_map={"": 0}).eval()
    for p in model.parameters(): p.requires_grad_(False)
    return model, tok

def fq_rtn(w, bits, g=128):
    import qpack
    out = torch.empty_like(w, dtype=torch.bfloat16); w2 = w.reshape(-1, w.shape[-1]); o2 = out.reshape(-1, w.shape[-1])
    for r0 in range(0, w2.shape[0], 8192):
        q, sc, z = qpack.rtn_ints(w2[r0:r0 + 8192].float(), bits, g); o2[r0:r0 + 8192] = qpack.deq(q, sc, z, g)
    return out

@torch.no_grad()
def fq_nonexperts(model, attn_b=6, emb_b=4, head_b=4, g=128):
    n_done = 0
    for n, p in model.named_parameters():
        if re.match(r"model\.layers\.(\d+)\.mlp\.(experts\.(gate_up_proj|down_proj)|gate\.weight)$", n): continue
        d = p.data
        if "embed_tokens" in n: p.data.copy_(fq_rtn(d, emb_b, g)); n_done += 1
        elif n.startswith("lm_head"): p.data.copy_(fq_rtn(d, head_b, g)); n_done += 1
        elif d.dim() == 2 and d.numel() >= 65536 and d.shape[-1] % g == 0: p.data.copy_(fq_rtn(d, attn_b, g)); n_done += 1
    return n_done

def per_kind(cor, bc, rows):
    import collections
    k = collections.defaultdict(lambda: [0, 0])
    for i, b in enumerate(bc):
        if b: k[rows[i]["kind"]][0] += 1; k[rows[i]["kind"]][1] += int(cor[i])
    return {a: [v[1], v[0], round(100 * v[1] / v[0], 2)] for a, v in sorted(k.items())}

def sel_rows():
    import core; return core.rows_of("/root/q/data/bank6_sel.jsonl")

def main(job):
    import core
    if job.get('blocking'): os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    T0 = time.time(); log = lambda *a: print(f"[{time.time()-T0:6.0f}s]", *a, flush=True)
    os.makedirs(RES, exist_ok=True); stage = job["stage"]
    model, tok = load_model(); R = core.Runner(model, tok); dev = R.dev0; log("loaded")
    S = sel_rows(); bl = blocks_of(model); NL = len(bl); log("moe layers", NL)
    bf = f"{RES}/base_bank6_SEL.json"
    if stage == "base":
        outs = R.gen(S); cor = core.correct(outs, S)
        json.dump(dict(outs=outs, cor=cor, ids=[r["id"] for r in S]), open(bf, "w")); COMMIT(); log("base SEL acc", sum(cor) / len(cor))
        # ---- verify custom expert forward == stock forward (numerics) on 24 SEL rows (full logits at last position)
        sub = S[:24]; R.encode(sub); ids, am = R.pad([r["ids"] for r in sub])
        with torch.no_grad(): ref = model(input_ids=ids, attention_mask=am, use_cache=False).logits[:, -1].float()
        Ex = type(bl[0].experts); stock = Ex.forward; Ex.forward = expert_fwd
        with torch.no_grad(): mine = model(input_ids=ids, attention_mask=am, use_cache=False).logits[:, -1].float()
        Ex.forward = stock
        d = (ref - mine).abs().max().item(); agree = float((ref.argmax(-1) == mine.argmax(-1)).float().mean())
        res = dict(stage="base", base_acc=sum(cor) / len(cor), n=len(cor), fwd_check=dict(max_abs_logit_diff=d, ref_absmax=float(ref.abs().max()), argmax_agree=agree))
        json.dump(res, open(f"{RES}/base_check.json", "w")); COMMIT(); log("fwd check", res); return res
    if stage == "check":
        Ex = type(bl[0].experts); stock = Ex.forward; cap = {}
        def mkhook(i):
            def hk(mod, args, kw): cap[i] = ([a.detach().clone() for a in args], {k: v.detach().clone() for k, v in kw.items()})
            return hk
        hs = [bl[i].experts.register_forward_pre_hook(mkhook(i), with_kwargs=True) for i in (0, 20, 39)]
        sub = S[:16]; R.encode(sub); ids, am = R.pad([r["ids"] for r in sub])
        with torch.no_grad(): model(input_ids=ids, attention_mask=am, use_cache=False)
        for h in hs: h.remove()
        res = {}
        for i in (0, 20, 39):
            a, kw = cap[i]; ex = bl[i].experts
            with torch.no_grad():
                o1 = stock(ex, *a, **kw).float(); o2 = expert_fwd(ex, *a, **kw).float()
                x = [t.double() for t in (a[0],)]; 
                # fp64 reference
                ref = torch.zeros_like(a[0], dtype=torch.float64)
                for e in [x for x in torch.unique(a[1]).tolist() if x < ex.num_experts]:
                    ti, tp = torch.where(a[1] == e); c = a[0][ti].double(); g, u = F.linear(c, ex.gate_up_proj[e].double()).chunk(2, -1)
                    ref.index_add_(0, ti, F.linear(F.silu(g) * u, ex.down_proj[e].double()) * a[2][ti, tp, None].double())
            res[i] = dict(stock_vs_mine_maxabs=float((o1 - o2).abs().max()), stock_vs_fp64=float((o1 - ref.float()).abs().max()), mine_vs_fp64=float((o2 - ref.float()).abs().max()), absmax=float(o1.abs().max()))
        json.dump(res, open(f"{RES}/expert_fwd_check.json", "w")); COMMIT(); log("CHECK", res); return res
    # ------------------------------------------------------------------ variants
    import qlib_b as QB, qlib as Q
    base = json.load(open(bf)); assert base["ids"] == [r["id"] for r in S]; bc = base["cor"]
    M = torch.load("/root/q/data/mass_bank6.pt").sum(0)   # [L,E] over all 12 calibration chunks
    V = job["variants"]; kmax = max(v["k"] for v in V); E = M.shape[1]
    ukept = [torch.topk(M[l], kmax).indices.sort().values.tolist() for l in range(NL)]
    kept = {v["name"]: [torch.topk(M[l], v["k"]).indices.sort().values for l in range(NL)] for v in V}
    cal = core.rows_of("/root/q/data/bank6_cal.jsonl"); R.encode(cal)
    Ex = type(bl[0].experts); Ex.forward = expert_fwd
    Qgu = {v["name"]: [None] * NL for v in V}; Qdn = {v["name"]: [None] * NL for v in V}
    stats = {v["name"]: dict(bits=0.0, n=0, relgu=[], reldn=[], fallback=0, per_layer_bits=[0.0] * NL) for v in V}
    nmin = job.get("nmin", 64); lgroup = job.get("lgroup", 8); ebatch = job.get("ebatch", 80); mixp = job.get("mix", 0.0)
    def mk(v, l):
        m, b, seed = v["m"], v["bits"], 1000 + l; rot = ("had", seed) if m.endswith("_had") else None; base_m = m.replace("_had", "")
        if base_m == "gptq": return QB.mb_gptq(lambda: QB.IntQ(b, v.get("g", 128), False), rot, v.get("damp", 0.01))
        if base_m == "ecsq": return QB.mb_gptq(lambda: QB.ECSQ(b, 256, 0.0, 16), rot)
        if base_m == "vq": return QB.mb_gptq(lambda: QB.VQ(4, int(round(2 ** (b * 4))), "gauss"), rot)
        raise ValueError(m)
    for g0 in range(0, NL, lgroup):
        layers = list(range(g0, min(NL, g0 + lgroup))); G["col"] = {}
        for l in layers:
            pos = torch.full((E,), -1, dtype=torch.long); pos[torch.tensor(ukept[l])] = torch.arange(kmax); k = kmax
            G["col"][id(bl[l].experts)] = dict(pos=pos.tolist(), gu=torch.zeros(k, 2048, 2048, device=dev), dn=torch.zeros(k, 512, 512, device=dev), n=[0] * k)
        with torch.no_grad():
            for idx in R.batches(cal, job.get("hbudget", 20000)):
                ids, am = R.pad([cal[k]["ids"] for k in idx]); G["valid"] = am.flatten().float()
                model.model(input_ids=ids, attention_mask=am, use_cache=False)
        G["valid"] = None; log("hessians", layers[0], layers[-1], "tokens/expert min/med", min(G["col"][id(bl[layers[0]].experts)]["n"]), sorted(G["col"][id(bl[layers[0]].experts)]["n"])[kmax // 2])
        for l in layers:
            C = G["col"][id(bl[l].experts)]; ex = bl[l].experts; n = torch.tensor(C["n"], device=dev, dtype=torch.float)
            Hgu, Hdn = C["gu"], C["dn"]
            pool_gu = (Hgu / n.clamp(min=1)[:, None, None]).mean(0); pool_dn = (Hdn / n.clamp(min=1)[:, None, None]).mean(0)
            for v in V:
                nm = v["name"]; kl = kept[nm][l]; pos = [C["pos"][int(e)] for e in kl]; st = stats[nm]
                if v["m"] == "rtn":
                    with torch.no_grad():
                        Qgu[nm][l] = fq_rtn(ex.gate_up_proj.data[kl.to(dev)], v["bits"]).cpu(); Qdn[nm][l] = fq_rtn(ex.down_proj.data[kl.to(dev)], v["bits"]).cpu()
                    b = RTN3_EXPERT_B * 8 if v["bits"] == 3 else None; st["bits"] += len(kl) * 2 * 0 + (len(kl) * EXP_PARAMS * (v["bits"] + 19 / 128) if True else 0); st["n"] += len(kl); continue
                outg, outd = [], []
                for c0 in range(0, len(kl), ebatch):
                    ee = kl[c0:c0 + ebatch]; pp = torch.tensor(pos[c0:c0 + ebatch], device=dev)
                    W1 = ex.gate_up_proj.data[ee.to(dev)].float(); W2 = ex.down_proj.data[ee.to(dev)].float()
                    nn_ = n[pp]; H1 = Hgu[pp] / nn_.clamp(min=1)[:, None, None]; H2 = Hdn[pp] / nn_.clamp(min=1)[:, None, None]
                    low = (nn_ < nmin); st["fallback"] += int(low.sum())
                    if low.any(): H1[low] = pool_gu; H2[low] = pool_dn
                    if mixp > 0: H1 = (1 - mixp) * H1 + mixp * pool_gu; H2 = (1 - mixp) * H2 + mixp * pool_dn
                    fn = mk(v, l); q1, b1 = fn(W1, H1); q2, b2 = fn(W2, H2)
                    st["relgu"].append(float(QB.relerr_b(W1, q1, H1).mean())); st["reldn"].append(float(QB.relerr_b(W2, q2, H2).mean()))
                    st["bits"] += float(b1.sum() + b2.sum()); st["per_layer_bits"][l] += float(b1.sum() + b2.sum()); st["n"] += len(ee)
                    outg.append(q1.to(torch.bfloat16).cpu()); outd.append(q2.to(torch.bfloat16).cpu()); del W1, W2, H1, H2, q1, q2
                Qgu[nm][l] = torch.cat(outg); Qdn[nm][l] = torch.cat(outd)
            for kk in list(C): pass
            log("layer", l, {v["name"]: round(stats[v["name"]]["bits"] / max(1, stats[v["name"]]["n"]) / EXP_PARAMS, 4) for v in V})
        G["col"] = None; torch.cuda.empty_cache()
    Ex.forward = Ex.forward  # keep custom forward (identical math) for eval
    log("quantised; fake-quantising non-expert tensors"); nq = fq_nonexperts(model, job.get("attn_b", 6), job.get("emb_b", 4), job.get("head_b", 4)); log("non-expert tensors fake-quantised:", nq)
    orig = [(b.experts.gate_up_proj, b.experts.down_proj, b.gate.weight, b.experts.num_experts) for b in bl]
    results = {}
    for v in V:
        nm = v["name"]; k = v["k"]; st = stats[nm]
        for l, b in enumerate(bl):
            kl = kept[nm][l].to(dev)
            b.experts.gate_up_proj = nn.Parameter(Qgu[nm][l].to(dev), requires_grad=False); b.experts.down_proj = nn.Parameter(Qdn[nm][l].to(dev), requires_grad=False)
            b.experts.num_experts = k; b.gate.weight = nn.Parameter(orig[l][2].data.index_select(0, kl), requires_grad=False); b.gate.num_experts = k
        outs = R.gen(S); cor = core.correct(outs, S); sc = core.score(cor, bc, S); pk = per_kind(cor, bc, S)
        # bytes
        expb = st["bits"] / 8; extra = 0
        if v["m"].startswith("ecsq"): extra = k * 40 * 2 * 256   # per-tensor 256-entry code length table (bytes)
        if v["m"].startswith("vq"): extra = 4 * 256 * 2
        total = FIXED_REST + MANIFEST_EXTRA + k * 40 * 2048 * 2 + expb + extra
        res = dict(name=nm, spec=v, ret=sc["ret"], ret_lo=sc["ret_lo"], n_bc=sc["n_bc"], comp_acc=sc["comp_acc"], base_acc=sc["base_acc"], per_kind=pk, worst_kind=sc["worst_kind"],
                   expert_bpw=st["bits"] / max(1, st["n"]) / EXP_PARAMS, bytes_est=int(total), bytes_pct=100 * total / FP16_REF if False else 100 * total / FP16_REF, under_cap=bool(total <= CAP * FP16_REF),
                   relerr_gu=sum(st["relgu"]) / max(1, len(st["relgu"])), relerr_dn=sum(st["reldn"]) / max(1, len(st["reldn"])), n_fallback_pooledH=st["fallback"], sample=outs[:5])
        results[nm] = res; json.dump(dict(outs=outs, cor=cor), open(f"{RES}/outs_{nm}_SEL.json", "w")); json.dump(res, open(f"{RES}/eval_{nm}.json", "w")); COMMIT()
        log("RESULT", nm, round(100 * res["ret"], 2), "lo", round(100 * res["ret_lo"], 2), "bytes%", round(res["bytes_pct"], 3), pk); print("RESULT_JSON", json.dumps(res), flush=True)
        for l, b in enumerate(bl):
            b.experts.gate_up_proj, b.experts.down_proj, b.gate.weight, b.experts.num_experts = orig[l]; b.gate.num_experts = E
    return results
