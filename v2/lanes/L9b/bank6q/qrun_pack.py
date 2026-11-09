"""bank6q real packer/loader stages for vq4_had_b2 on BANK6. stages: pack (quantise k experts, write artifact, weight-level fidelity vs in-process fake-quant, SEL gen of fake-quant model), load (SEL gen through the real loader),
exam_base (locked BANK6 exam, base x3), exam_packed (locked exam through the real loader, ONCE), fb (BANK6 feedback on the packed model, reporting)."""
import json, os, re, sys, time, math, shutil, hashlib, collections
import torch, torch.nn as nn
sys.path.insert(0, "/root/q")
import qrun, core
from qrun import G, expert_fwd, blocks_of, load_model, fq_nonexperts, per_kind
COMMIT = lambda: None
RES = "/q/res"; ART = "/q/art"
FP16_REF = 69321221376; CAP_B = int(0.099 * FP16_REF)   # 6,862,800,916
FB_CARRIER_B = 20981760

def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""): h.update(b)
    return h.hexdigest()

def dir_sizes(d):
    out = {}
    for dp, _, fs in os.walk(d):
        for f in fs: out[os.path.relpath(os.path.join(dp, f), d)] = os.path.getsize(os.path.join(dp, f))
    return out

def vq_layer(W, H, seed, cb_ref):
    """same math as qlib_b.mb_gptq(VQ(4,256,'gauss'), ('had',seed)) but also returns indices. W[E,R,C] H[E,C,C] -> (fake bf16 weights, idx uint8 [E,R,C/4], sigma [E], codebook)"""
    import qlib_b as QB, qlib as Q
    W = W.float(); H = H.float(); Wr, Hr, back = QB.rotb(W, H, seed, "had")
    rd = QB.VQ(4, 256, "gauss"); Qm = QB.engine_b(Wr, Hr, rd, 0.01); cb = rd.cbs[0]; sig = rd.sig
    E, R, C = W.shape; X = (Qm / sig).reshape(-1, 4); idx = Q.nn_argmax(X, cb, 0.5 * (cb ** 2).sum(1)).reshape(E, R, C // 4)
    rec = cb[idx].reshape(E, R, C) * sig; err = float((rec - Qm).abs().max())
    fake = back(Qm).to(torch.bfloat16)
    return fake, idx.to(torch.uint8), sig.reshape(E).float(), cb, err

def stage_pack(job, model, tok, R, log):
    import vqpack, qpack
    k = job["k"]; name = job["art"]; adir = f"{ART}/{name}"; S = core.rows_of("/root/q/data/bank6_sel.jsonl")
    bl = blocks_of(model); NL = len(bl); assert NL == 40
    base = json.load(open(f"{RES}/base_bank6_SEL.json")); assert base["ids"] == [r["id"] for r in S]; bc = base["cor"]
    M = torch.load("/root/q/data/mass_bank6.pt").sum(0); E = M.shape[1]
    kept = [torch.topk(M[l], k).indices.sort().values for l in range(NL)]
    cal = core.rows_of("/root/q/data/bank6_cal.jsonl"); R.encode(cal); dev = R.dev0
    Ex = type(bl[0].experts); stock = Ex.forward; Ex.forward = expert_fwd
    Qgu = [None] * NL; Qdn = [None] * NL; QD = [None] * NL; cbs = []; fallback = 0; maxerr = 0.0; nmin = 64; ebatch = 80
    for g0 in range(0, NL, 8):
        layers = list(range(g0, min(NL, g0 + 8))); G["col"] = {}
        for l in layers:
            pos = torch.full((E,), -1, dtype=torch.long); pos[kept[l]] = torch.arange(k)
            G["col"][id(bl[l].experts)] = dict(pos=pos.tolist(), gu=torch.zeros(k, 2048, 2048, device=dev), dn=torch.zeros(k, 512, 512, device=dev), n=[0] * k)
        with torch.no_grad():
            for idx in R.batches(cal, 20000):
                ids, am = R.pad([cal[i]["ids"] for i in idx]); G["valid"] = am.flatten().float(); model.model(input_ids=ids, attention_mask=am, use_cache=False)
        G["valid"] = None; log("hessians", layers[0], layers[-1])
        for l in layers:
            C = G["col"][id(bl[l].experts)]; ex = bl[l].experts; n = torch.tensor(C["n"], device=dev, dtype=torch.float); Hgu, Hdn = C["gu"], C["dn"]
            pool_gu = (Hgu / n.clamp(min=1)[:, None, None]).mean(0); pool_dn = (Hdn / n.clamp(min=1)[:, None, None]).mean(0)
            kl = kept[l]; og, od, ig, idn, sg, sd = [], [], [], [], [], []
            for c0 in range(0, k, ebatch):
                ee = kl[c0:c0 + ebatch]; pp = torch.arange(c0, c0 + len(ee), device=dev)
                W1 = ex.gate_up_proj.data[ee.to(dev)].float(); W2 = ex.down_proj.data[ee.to(dev)].float()
                nn_ = n[pp]; H1 = Hgu[pp] / nn_.clamp(min=1)[:, None, None]; H2 = Hdn[pp] / nn_.clamp(min=1)[:, None, None]
                low = nn_ < nmin; fallback += int(low.sum())
                if low.any(): H1[low] = pool_gu; H2[low] = pool_dn
                f1, i1, s1, cb1, e1 = vq_layer(W1, H1, 1000 + l, None); f2, i2, s2, cb2, e2 = vq_layer(W2, H2, 1000 + l, None); maxerr = max(maxerr, e1, e2)
                cbs.append(cb1.cpu()); cbs.append(cb2.cpu())
                og.append(f1.cpu()); od.append(f2.cpu()); ig.append(i1.cpu()); idn.append(i2.cpu()); sg.append(s1.cpu()); sd.append(s2.cpu()); del W1, W2, H1, H2
            Qgu[l] = torch.cat(og); Qdn[l] = torch.cat(od); QD[l] = dict(gu=(torch.cat(ig), torch.cat(sg)), dn=(torch.cat(idn), torch.cat(sd)))
            log("layer", l, "idx-vs-qm maxerr", maxerr)
        G["col"] = None; torch.cuda.empty_cache()
    Ex.forward = stock
    cb = cbs[0]; assert all(torch.equal(cb, c) for c in cbs), "codebook differs between calls"
    log("quantised; pooled-H fallback experts", fallback, "max |cb[idx]*sig - Qm|", maxerr)
    # ---- pack (non-expert tensors quantised from the original bf16 weights, before any fake-quant)
    shutil.rmtree(adir, ignore_errors=True); os.makedirs(adir)
    code = [f"/root/q/{f}" for f in ("core.py", "qpack.py", "vqpack.py", "answer_q.py", "annot.py", "fbcore.py", "feedback.py", "bank6_task.json")]
    t0 = time.time(); vqpack.pack_vq_model(model, tok, torch.stack(kept), QD, cb, adir, code_files=code, extra=dict(k=k, source="bank6q vq4_had_b2", prefix="bank6")); log("packed in", round(time.time() - t0), "s")
    shutil.copy("/root/q/feedback_cal.jsonl", f"{adir}/feedback_cal.jsonl"); sz = dir_sizes(adir); COMMIT()
    # ---- fidelity at weight level: decoder (same code as the loader) vs the in-process fake-quant tensors
    m = json.load(open(f"{adir}/manifest.json")); body = vqpack.np.memmap(f"{adir}/body.bin", dtype=vqpack.np.uint8, mode="r"); cbd = vqpack.read_codebook(body, m, dev)
    P = dict(model.named_parameters()); fq_nonexperts(model); fid = collections.defaultdict(lambda: dict(n=0, tensors=0, maxabs=0.0, ndiff=0, numel=0, worst=None)); fexp = {}
    with torch.no_grad():
        for n, e in m["tensors"].items():
            if n == "__vq_cb__": continue
            mm = re.match(r"model\.layers\.(\d+)\.mlp\.(experts\.(gate_up_proj|down_proj)|gate\.weight)$", n)
            dec = vqpack.read_any(body, e, cbd, dev)
            if e["kind"] == "vq": ref = (Qgu if mm.group(3) == "gate_up_proj" else Qdn)[int(mm.group(1))].to(dev); cls = "expert_vq"
            elif mm and mm.group(2) == "gate.weight": ref = P[n].data.index_select(0, kept[int(mm.group(1))].to(dev)); cls = "router_raw"
            else: ref = P[n].data; cls = "nonexpert_" + e["kind"]
            d = (dec.float() - ref.float()).abs(); f = fid[cls]; f["tensors"] += 1; f["numel"] += d.numel(); f["ndiff"] += int((d > 0).sum()); mx = float(d.max())
            if mx >= f["maxabs"]: f["maxabs"] = mx; f["worst"] = n
            f["absmax_ref"] = max(f.get("absmax_ref", 0.0), float(ref.float().abs().max())); del dec, ref, d
    fid = {c: dict(v) for c, v in fid.items()}; log("FIDELITY", fid)
    # ---- SEL generation of the in-process fake-quant model (stock expert forward)
    orig = [(b.experts.gate_up_proj, b.experts.down_proj, b.gate.weight, b.experts.num_experts) for b in bl]
    for l, b in enumerate(bl):
        kl = kept[l].to(dev); b.experts.gate_up_proj = nn.Parameter(Qgu[l].to(dev), requires_grad=False); b.experts.down_proj = nn.Parameter(Qdn[l].to(dev), requires_grad=False)
        b.experts.num_experts = k; b.gate.weight = nn.Parameter(orig[l][2].data.index_select(0, kl), requires_grad=False); b.gate.num_experts = k
    outs = R.gen(S); cor = core.correct(outs, S); sc = core.score(cor, bc, S); pk = per_kind(cor, bc, S)
    prev = json.load(open(f"{RES}/outs_vq4_had_b2_k160_SEL.json"))["outs"] if os.path.exists(f"{RES}/outs_vq4_had_b2_k160_SEL.json") else None
    json.dump(dict(outs=outs, cor=cor), open(f"{RES}/outs_fakequant_{name}_SEL.json", "w"))
    tot = sum(sz.values())
    res = dict(stage="pack", art=name, k=k, files=sz, artifact_bytes=tot, artifact_pct=100 * tot / FP16_REF, cap_bytes=CAP_B, with_fb_carrier_bytes=tot + FB_CARRIER_B + 400, with_fb_pct=100 * (tot + FB_CARRIER_B + 400) / FP16_REF,
               sha256={f: sha256(f"{adir}/{f}") for f in ("body.bin", "manifest.json")}, fidelity=fid, fallback_pooledH=fallback, idx_vs_qm_maxerr=maxerr,
               fakequant_sel=dict(ret=sc["ret"], ret_lo=sc["ret_lo"], n_bc=sc["n_bc"], comp_acc=sc["comp_acc"], per_kind=pk), same_as_prev_sim_outs=(sum(a == b for a, b in zip(outs, prev)) if prev else None), n=len(outs))
    json.dump(res, open(f"{RES}/pack_{name}.json", "w"), indent=1); COMMIT(); log("PACK RESULT", {k_: v for k_, v in res.items() if k_ != "files"}); return res

def stage_load(job, log):
    sys.path.insert(0, f"{ART}/{job['art']}"); import vqpack, core as c2
    adir = f"{ART}/{job['art']}"; S = core.rows_of("/root/q/data/bank6_sel.jsonl"); base = json.load(open(f"{RES}/base_bank6_SEL.json")); bc = base["cor"]
    t0 = time.time(); model, tok, m = vqpack.load_packed_vq(adir); log("real loader time", round(time.time() - t0)); R = core.Runner(model, tok)
    outs = R.gen(S); cor = core.correct(outs, S); sc = core.score(cor, bc, S); pk = per_kind(cor, bc, S)
    fk = json.load(open(f"{RES}/outs_fakequant_{job['art']}_SEL.json"))
    res = dict(stage="load", art=job["art"], sel=dict(ret=sc["ret"], ret_lo=sc["ret_lo"], n_bc=sc["n_bc"], comp_acc=sc["comp_acc"], per_kind=pk), n_equal_to_fakequant=sum(a == b for a, b in zip(outs, fk["outs"])), n=len(outs),
               diff_idx=[i for i, (a, b) in enumerate(zip(outs, fk["outs"])) if a != b][:20])
    json.dump(dict(res=res, outs=outs), open(f"{RES}/load_{job['art']}_SEL.json", "w")); COMMIT(); log("LOAD RESULT", res); return res

def stage_exam_base(job, model, tok, R, log):
    sys.path.insert(0, "/root/repo/v2/exam"); import loader
    rows = loader.load("BANK6", "exam"); runs = []
    for i in range(3):
        o = R.gen(rows); runs.append(o); log("base run", i, "acc", sum(core.correct(o, rows)) / len(rows))
        json.dump(dict(ids=[r["id"] for r in rows], runs=runs), open(f"{RES}/exam_base3_BANK6.json", "w")); COMMIT()
    ndiff = [sum(a != b for a, b in zip(runs[0], runs[j])) for j in (1, 2)]; res = dict(stage="exam_base", n=len(rows), acc=[sum(core.correct(o, rows)) / len(rows) for o in runs], n_outputs_differing_vs_run0=ndiff)
    json.dump(res, open(f"{RES}/exam_base3_summary.json", "w")); COMMIT(); log("EXAM BASE", res); return res

def stage_exam_packed(job, log):
    sys.path.insert(0, f"{ART}/{job['art']}"); sys.path.insert(0, "/root/repo/v2/exam"); import vqpack, loader
    out = f"{RES}/exam_packed_{job['art']}_BANK6.json"; assert not os.path.exists(out), "locked exam already scored for this artifact"
    rows = loader.load("BANK6", "exam"); model, tok, m = vqpack.load_packed_vq(f"{ART}/{job['art']}"); R = core.Runner(model, tok); o = R.gen(rows)
    json.dump(dict(ids=[r["id"] for r in rows], outs=o, art=job["art"]), open(out, "w")); COMMIT(); log("EXAM PACKED done acc", sum(core.correct(o, rows)) / len(rows)); return dict(stage="exam_packed", acc=sum(core.correct(o, rows)) / len(rows))

def stage_diag(job, log):
    sys.path.insert(0, f"{ART}/{job['art']}"); import vqpack
    model, tok, m = vqpack.load_packed_vq(f"{ART}/{job['art']}"); base, _ = load_model(); bb = dict(base.named_buffers()); res = {}
    for n, b in model.named_buffers():
        if n not in bb: res[n] = "missing in base"; continue
        d = float((b.float() - bb[n].to(b.device).float()).abs().max()) if b.numel() else 0.0
        if d > 0 or b.dtype != bb[n].dtype: res[n] = (d, str(b.dtype), str(bb[n].dtype))
    pd = {n: (str(p.dtype), str(dict(base.named_parameters())[n].dtype)) for n, p in model.named_parameters() if n in dict(base.named_parameters()) and p.dtype != dict(base.named_parameters())[n].dtype}
    log("BUFFER DIFFS", len(res), dict(list(res.items())[:10])); log("PARAM DTYPE DIFFS", len(pd), dict(list(pd.items())[:10]))
    log("gen cfg loaded", model.generation_config, "base", base.generation_config); return dict(buf=len(res), pd=len(pd))

def stage_fb(job, log):
    """BANK6 feedback (new label Security) on the packed model: reuses v2/lanes/L9b/fb_run.py unchanged (copied into the image), non-DEV mode like the shipped k102 run; inputs rebuilt from the shared locked files + my base x3 exam run 0."""
    sys.path.insert(0, f"{ART}/{job['art']}"); sys.path.insert(0, "/root/repo/v2/exam"); import vqpack, loader, fb_run
    os.makedirs("/r/l9b/res", exist_ok=True); os.makedirs("/root/l9b/data", exist_ok=True)
    ex = loader.load("BANK6", "exam"); b3 = json.load(open(f"{RES}/exam_base3_BANK6.json")); assert b3["ids"] == [r["id"] for r in ex]
    json.dump(dict(outs=b3["runs"][0], ids=b3["ids"]), open("/r/l9b/res/base_bank6_EXAM.json", "w")); json.dump(loader.load("BANK6", "neighbours"), open("/root/l9b/data/bank6_neighbours.json", "w"))
    S = dict(EXAM=ex, CAL=core.rows_of("/root/q/data/bank6_cal.jsonl"), FB_TEACH=loader.load("BANK6", "teach"), FB_ANNOT=loader.load("BANK6", "ann_pool"), SEL=core.rows_of("/root/q/data/bank6_sel.jsonl"))
    model, tok, m = vqpack.load_packed_vq(f"{ART}/{job['art']}"); R = core.Runner(model, tok)
    jb = dict(stage="fb", prefix="bank6", art=job["art"], tag=job.get("tag", "fb_b6"), new_label="Security", fb=job["fb"])
    fb_run.run(jb, model, tok, R, S, m, log, lambda: None)
    for f in os.listdir("/r/l9b/res"):
        if f.startswith("fb_"): shutil.copy(f"/r/l9b/res/{f}", f"{RES}/{f}")
    COMMIT(); return json.load(open(f"/r/l9b/res/fb_{jb['art']}_{jb['tag']}.json"))

def main(job):
    T0 = time.time(); log = lambda *a: print(f"[{time.time()-T0:6.0f}s]", *a, flush=True)
    os.makedirs(RES, exist_ok=True); os.makedirs(ART, exist_ok=True); st = job["stage"]
    if st == "load": return stage_load(job, log)
    if st == "fb": return stage_fb(job, log)
    if st == "diag": return stage_diag(job, log)
    if st == "exam_packed": return stage_exam_packed(job, log)
    model, tok = load_model(); R = core.Runner(model, tok); log("loaded")
    if st == "pack": return stage_pack(job, model, tok, R, log)
    if st == "exam_base": return stage_exam_base(job, model, tok, R, log)
    raise ValueError(st)
