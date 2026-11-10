"""L14: broad-calibration expert pruning + Hadamard/GPTQ-VQ experts for Qwen3.5-35B-A3B on SKILL7 (fork of L9b/bank6q/qrun.py). Fake-quant, SEL + own-data dry run only. No locked-exam data is loaded anywhere in this file.
Calibration (routing mass AND Hessians) = job["cal"] = list of [name, nrows]; names: task (SKILL7 cal_big 2031 prompts), general (cal_general.jsonl: dolly/gsm8k/code/wikitext, unlabeled)."""
import json, os, re, sys, time, math
import torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, "/root/q"); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
COMMIT = lambda: None
RES = "/q/res14"
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
    import core; return core.rows_of("/root/q/data/skill7_sel.jsonl")


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
def load_cal(R, tok, spec):
    """spec = [[name, nrows], ...]; rows get ids (general rows: chat(user) + resp). Deterministic order: concatenated then interleaved by seeded shuffle."""
    import core, random
    rows = []
    for nm, n in spec:
        p = {"task": "/root/q/data/skill7_cal.jsonl", "general": "/root/q/data/cal_general.jsonl"}[nm]
        rr = core.rows_of(p)[:n]
        for r in rr:
            r["ids"] = tok(R.chat(r["user"]) + (r.get("resp") or ""), add_special_tokens=False).input_ids[:700]
        rows += rr
    random.Random(5).shuffle(rows); return rows

def own_sets(R):
    import csv, io, urllib.request, collections
    sys.path.insert(0, "/root/q"); from _helpers import banking_rows, generic_prompts, BK_URL
    RP = json.load(open("/root/q/skill_task.json"))["prompt"]; bk = banking_rows(200)
    txt = urllib.request.urlopen(BK_URL, timeout=60).read().decode(); menu = sorted({r["category"] for r in csv.DictReader(io.StringIO(txt))}); assert len(menu) == 77
    BP = "Classify this banking customer query into exactly one intent. Answer with only the intent name, chosen from: " + ", ".join(menu) + ".\n\nQuery: {t}"
    P_bk = [dict(user=BP.replace("{t}", t)) for t, _ in bk]; gold = [c for _, c in bk]; P_rt = [dict(user=RP.replace("{t}", t)) for t, _ in bk[:100]]; P_gen = [dict(user=p) for p in generic_prompts()[:100]]
    assert len(P_gen) == 100
    return P_bk, gold, P_rt, P_gen

def own_run(R, S3):
    P_bk, gold, P_rt, P_gen = S3
    return dict(bk=R.gen(P_bk, max_new=20), rt=R.gen(P_rt, max_new=8), gen=R.gen(P_gen, max_new=24))

def own_summary(OB, O, gold):
    import core, collections, math
    nm = core.norm
    cb = [nm(o) == g.lower() for o, g in zip(OB["bk"], gold)]; cp = [nm(o) == g.lower() for o, g in zip(O["bk"], gold)]; bc = sum(cb); both = sum(1 for x, y in zip(cb, cp) if x and y)
    sm = [x.strip() == y.strip() for x, y in zip(OB["gen"], O["gen"])]; fl = [nm(x) == nm(y) for x, y in zip(OB["gen"], O["gen"])]
    return dict(banking77_200=dict(base_acc=bc / 200, packed_acc=sum(cp) / 200, retention=both / max(bc, 1), retention_wilson_lo=core.wilson(both, bc), n_base_correct=bc, same_output=sum(nm(x) == nm(y) for x, y in zip(OB["bk"], O["bk"])) / 200),
                router_on_banking_100=dict(same_label=sum(nm(x) == nm(y) for x, y in zip(OB["rt"], O["rt"])) / 100, base_dist=dict(collections.Counter(nm(o) for o in OB["rt"]).most_common(8)), packed_dist=dict(collections.Counter(nm(o) for o in O["rt"]).most_common(8))),
                generic_100=dict(identical_24tok=sum(sm) / 100, same_first_line=sum(fl) / 100),
                diverge=[(i, OB["gen"][i][:80], O["gen"][i][:80]) for i in range(100) if not fl[i]][:10])

def main(job):
    import core
    T0 = time.time(); log = lambda *a: print(f"[{time.time()-T0:6.0f}s]", *a, flush=True)
    os.makedirs(RES, exist_ok=True); tag = job["tag"]
    model, tok = load_model(); R = core.Runner(model, tok); dev = R.dev0; log("loaded")
    S = sel_rows(); bl = blocks_of(model); NL = len(bl); E = bl[0].experts.num_experts; log("moe layers", NL, "experts", E)
    S3 = own_sets(R); gold = S3[1]
    # ---- base (stock forward, unmodified) SEL + own-data, cached on the volume
    bf = f"{RES}/base_skill7_SEL.json"; bo = f"{RES}/base_own.json"
    if os.path.exists(bf): base = json.load(open(bf))
    else:
        outs = R.gen(S); base = dict(outs=outs, cor=core.correct(outs, S), ids=[r["id"] for r in S]); json.dump(base, open(bf, "w")); COMMIT(); log("base SEL acc", sum(base["cor"]) / len(base["cor"]))
    assert base["ids"] == [r["id"] for r in S]; bc = base["cor"]
    if os.path.exists(bo): OB = json.load(open(bo))
    else: OB = own_run(R, S3); json.dump(OB, open(bo, "w")); COMMIT(); log("base own done")
    if job.get("stage") == "base": return dict(base_acc=sum(bc) / len(bc))
    # ---- calibration + routing mass (cached per cal spec)
    cal = load_cal(R, tok, job["cal"]); ctag = "_".join(f"{a}{b}" for a, b in job["cal"]); mp = f"{RES}/mass_{ctag}.pt"
    log("cal rows", len(cal), "tokens", sum(len(r["ids"]) for r in cal))
    if os.path.exists(mp): Mc = torch.load(mp)
    else:
        blocks, mass, STAT = core.install_router_hooks(model); Mc = core.mass_chunks(R, cal, blocks, mass, STAT, nchunk=4); torch.save(Mc, mp); COMMIT(); log("mass done")
    M = Mc.sum(0)   # [L,E]
    import qlib_b as QB, qlib as Q
    V = job["variants"]; kmax = max(v["k"] for v in V)
    ukept = [torch.topk(M[l], kmax).indices.sort().values.tolist() for l in range(NL)]
    kept = {v["name"]: [torch.topk(M[l], v["k"]).indices.sort().values for l in range(NL)] for v in V}
    Ex = type(bl[0].experts); stock = Ex.forward; Ex.forward = expert_fwd
    Qgu = {v["name"]: [None] * NL for v in V}; Qdn = {v["name"]: [None] * NL for v in V}
    stats = {v["name"]: dict(bits=0.0, n=0, relgu=[], reldn=[], fallback=0) for v in V}
    nmin = job.get("nmin", 64); lgroup = job.get("lgroup", 8); ebatch = job.get("ebatch", 80); mixp = job.get("mix", 0.0)
    def mk(v, l):
        m, b, seed = v["m"], v["bits"], 1000 + l; rot = ("had", seed) if m.endswith("_had") else None; base_m = m.replace("_had", "")
        if base_m == "gptq": return QB.mb_gptq(lambda: QB.IntQ(b, v.get("g", 128), False), rot, v.get("damp", 0.01))
        if base_m == "vq": D = v.get("D", 4); K = v.get("K", int(round(2 ** (b * D)))); return QB.mb_gptq(lambda: QB.VQ(D, K, "gauss"), rot, v.get("damp", 0.01))
        raise ValueError(m)
    need_h = any(v["m"] != "bf16" for v in V)
    for g0 in range(0, NL, lgroup):
        layers = list(range(g0, min(NL, g0 + lgroup))); G["col"] = {}
        if not need_h:
            for l in layers:
                for v in V: kl = kept[v["name"]][l]; Qgu[v["name"]][l] = bl[l].experts.gate_up_proj.data[kl.to(dev)].cpu().clone(); Qdn[v["name"]][l] = bl[l].experts.down_proj.data[kl.to(dev)].cpu().clone(); stats[v["name"]]["bits"] += len(kl) * EXP_PARAMS * 16.0; stats[v["name"]]["n"] += len(kl)
            continue
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
                outg, outd = [], []
                if v["m"] == "bf16":
                    Qgu[nm][l] = ex.gate_up_proj.data[kl.to(dev)].cpu().clone(); Qdn[nm][l] = ex.down_proj.data[kl.to(dev)].cpu().clone(); st["bits"] += len(kl) * EXP_PARAMS * 16.0; st["n"] += len(kl); continue
                for c0 in range(0, len(kl), ebatch):
                    ee = kl[c0:c0 + ebatch]; pp = torch.tensor(pos[c0:c0 + ebatch], device=dev)
                    W1 = ex.gate_up_proj.data[ee.to(dev)].float(); W2 = ex.down_proj.data[ee.to(dev)].float()
                    nn_ = n[pp]; H1 = Hgu[pp] / nn_.clamp(min=1)[:, None, None]; H2 = Hdn[pp] / nn_.clamp(min=1)[:, None, None]
                    low = (nn_ < nmin); st["fallback"] += int(low.sum())
                    if low.any(): H1[low] = pool_gu; H2[low] = pool_dn
                    if mixp > 0: H1 = (1 - mixp) * H1 + mixp * pool_gu; H2 = (1 - mixp) * H2 + mixp * pool_dn
                    fn = mk(v, l); q1, b1 = fn(W1, H1); q2, b2 = fn(W2, H2)
                    st["relgu"].append(float(QB.relerr_b(W1, q1, H1).mean())); st["reldn"].append(float(QB.relerr_b(W2, q2, H2).mean()))
                    st["bits"] += float(b1.sum() + b2.sum()); st["n"] += len(ee)
                    outg.append(q1.to(torch.bfloat16).cpu()); outd.append(q2.to(torch.bfloat16).cpu()); del W1, W2, H1, H2, q1, q2
                Qgu[nm][l] = torch.cat(outg); Qdn[nm][l] = torch.cat(outd)
            log("layer", l, {v["name"]: round(stats[v["name"]]["bits"] / max(1, stats[v["name"]]["n"]) / EXP_PARAMS, 4) for v in V})
        G["col"] = None; torch.cuda.empty_cache()
    Ex.forward = stock   # evaluate with the stock expert forward (as the packed loader does)
    snap_ne = {n: p.data.cpu().clone() for n, p in model.named_parameters() if not re.match(r"model\.layers\.(\d+)\.mlp\.(experts\.(gate_up_proj|down_proj)|gate\.weight)$", n) and (("embed_tokens" in n) or n.startswith("lm_head") or (p.data.dim() == 2 and p.data.numel() >= 65536 and p.data.shape[-1] % 128 == 0))}
    cnt_attn = sum(t.numel() for n, t in snap_ne.items() if "embed_tokens" not in n and not n.startswith("lm_head")); cnt_emb = snap_ne["model.embed_tokens.weight"].numel() if "model.embed_tokens.weight" in snap_ne else sum(t.numel() for n, t in snap_ne.items() if "embed_tokens" in n); cnt_head = sum(t.numel() for n, t in snap_ne.items() if n.startswith("lm_head"))
    log("non-expert 2D params: attn-ish", cnt_attn, "emb", cnt_emb, "head", cnt_head)
    PN = dict(model.named_parameters())
    orig = [(b.experts.gate_up_proj.data.cpu(), b.experts.down_proj.data.cpu(), b.gate.weight, b.experts.num_experts) for b in bl]
    for b in bl: b.experts.gate_up_proj = nn.Parameter(torch.empty(0, device=dev), requires_grad=False); b.experts.down_proj = nn.Parameter(torch.empty(0, device=dev), requires_grad=False)
    torch.cuda.empty_cache()
    results = {}
    for v in V:
        nm = v["name"]; k = v["k"]; st = stats[nm]
        for n_, t_ in snap_ne.items(): PN[n_].data.copy_(t_.to(PN[n_].device))
        ab, eb_, hb = v.get("attn_b", 6), v.get("emb_b", 4), v.get("head_b", 4)
        if v.get("nonexp", True): fq_nonexperts(model, ab, eb_, hb)
        for l, b in enumerate(bl):
            kl = kept[nm][l].to(dev)
            b.experts.gate_up_proj = nn.Parameter(Qgu[nm][l].to(dev), requires_grad=False); b.experts.down_proj = nn.Parameter(Qdn[nm][l].to(dev), requires_grad=False)
            b.experts.num_experts = k; b.gate.weight = nn.Parameter(orig[l][2].data.index_select(0, kl), requires_grad=False); b.gate.num_experts = k
        outs = R.gen(S); cor = core.correct(outs, S); sc = core.score(cor, bc, S); pk = per_kind(cor, bc, S)
        O = own_run(R, S3); osum = own_summary(OB, O, gold)
        extra = v.get("D", 4) * v.get("K", 256) * 2 if v["m"].startswith("vq") else 0
        total = FIXED_REST + MANIFEST_EXTRA + k * 40 * 2048 * 2 + st["bits"] / 8 + extra + (cnt_attn * (ab - 6) + cnt_emb * (eb_ - 4) + cnt_head * (hb - 4)) / 8 * (1 + 1 / 128)
        res = dict(name=nm, tag=tag, cal=job["cal"], spec=v, ret=sc["ret"], ret_lo=sc["ret_lo"], n_bc=sc["n_bc"], comp_acc=sc["comp_acc"], base_acc=sc["base_acc"], per_kind=pk, worst_kind=sc["worst_kind"], worst_cell=sc["worst_cell"],
                   expert_bpw=st["bits"] / max(1, st["n"]) / EXP_PARAMS, bytes_est=int(total), bytes_pct=100 * total / FP16_REF, under_cap=bool(total <= CAP * FP16_REF),
                   relerr_gu=sum(st["relgu"]) / max(1, len(st["relgu"])), relerr_dn=sum(st["reldn"]) / max(1, len(st["reldn"])), n_fallback_pooledH=st["fallback"], own=osum)
        results[nm] = res; json.dump(dict(res=res, outs=outs, own=O), open(f"{RES}/eval_{tag}_{nm}.json", "w")); COMMIT()
        o = osum; log("RESULT", nm, "SEL", round(100 * res["ret"], 2), "bytes%", round(res["bytes_pct"], 3), "bank77 ret", round(100 * o["banking77_200"]["retention"], 1), "router agree", o["router_on_banking_100"]["same_label"], "generic ident", o["generic_100"]["identical_24tok"])
        print("RESULT_JSON", json.dumps(res), flush=True)
        for l, b in enumerate(bl):
            b.experts.gate_up_proj = nn.Parameter(torch.empty(0, device=dev), requires_grad=False); b.experts.down_proj = nn.Parameter(torch.empty(0, device=dev), requires_grad=False); b.gate.weight = orig[l][2]; b.experts.num_experts = orig[l][3]; b.gate.num_experts = E
        torch.cuda.empty_cache()
    return results
