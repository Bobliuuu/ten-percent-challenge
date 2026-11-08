"""X13 driver: task-pruned MoE probe.  Subcommands:
  quant  --arm M0|M1|M2|M3 --seed S --out DIR   (layerwise, streams base weights; writes real pack)
  eval   --pack DIR|base --sets SEL[,REP,INJ]  --out FILE  (decode pack -> real greedy generation, per-item outputs)
Everything closed-form calibration math; no gradient training."""
import argparse, json, os, sys, time, random, hashlib, glob, collections, math
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qmoe as Q
from qmoe import G
FP16 = 61_064_245_248
AB, MB, HB = 6, 5, 4          # attention / embed / head bits (RTN g128), as in the peer recipe
log = lambda *a: print(time.strftime("%H:%M:%S"), *a, flush=True)

# ---------------- bytes (exact for packed tensors; safetensors headers added after measuring) ----------------
def pbytes(n, bits): return (n * bits + 7) // 8
def qb(n, bits): return pbytes(n, bits) + (n // G) * 2 + pbytes(n // G, bits)
def fixed_bytes(cfg):
    H, qd, kd = cfg["H"], cfg["nh"] * cfg["hd"], cfg["nkv"] * cfg["hd"]
    attn = cfg["NL"] * (qb(qd * H, AB) * 2 + qb(kd * H, AB) * 2)
    emb = qb(cfg["V"] * H, MB) + qb(cfg["V"] * H, HB)
    norms = cfg["NL"] * (2 * H + 2 * cfg["hd"]) * 2 + H * 2
    return attn + emb + norms
def expert_bytes(cfg, b):
    H, I = cfg["H"], cfg["I"]
    return qb(2 * I * H, b) + qb(H * I, b) + H * 2 + 2 + 1      # + router row bf16 + id int16 + bits uint8
def extra_bytes(extra_files): return sum(os.path.getsize(f) for f in extra_files)
HDR = 48 * 3000                                               # safetensors json headers (upper bound; replaced by measured size)

def plan(cfg, arm, frac_target, extra):
    """returns (K, hi, lo, n_hi_fraction) so that total bytes ~= frac_target * FP16"""
    budget = frac_target * FP16 - fixed_bytes(cfg) - extra - HDR
    NL, NE = cfg["NL"], cfg["NE"]
    if arm in ("M0", "M1"): hi = lo = 3
    elif arm == "M2": hi, lo = 3, 2
    elif arm == "M3": hi, lo = 2, 1
    if arm == "M3":
        K = NL * NE; nhi = int((budget - K * expert_bytes(cfg, lo)) / (expert_bytes(cfg, hi) - expert_bytes(cfg, lo)))
        return K, hi, lo, max(0, min(K, nhi))
    fh = 0.5 if arm == "M2" else 1.0
    per = fh * expert_bytes(cfg, hi) + (1 - fh) * expert_bytes(cfg, lo)
    K = int(budget // per); return K, hi, lo, int(round(fh * K))

def select(frac, K, minpl=16):
    NL, NE = frac.shape
    keep = [set(torch.topk(frac[l], min(minpl, NE)).indices.tolist()) for l in range(NL)]
    n = sum(len(k) for k in keep)
    for i in torch.argsort(frac.flatten(), descending=True).tolist():
        if n >= K: break
        l, e = divmod(i, NE)
        if e not in keep[l]: keep[l].add(e); n += 1
    return [sorted(k) for k in keep]

# ---------------- data ----------------
def rd(p): return [json.loads(l) for l in open(p)]
def tmpl(tok, u): return tok.apply_chat_template([{"role": "user", "content": u}], add_generation_prompt=True, tokenize=False)

def calib_texts(args, seed):
    rng = random.Random(seed)
    cal = [r["user"] for r in rd(args.cal)]; rng.shuffle(cal); cal = cal[:args.n_cal]
    oth = []
    for p in args.other:
        rows = rd(p); rng.shuffle(rows); oth += [r.get("user") or r.get("query") for r in rows[:150]]
    wiki = []
    if args.wiki:
        wiki = [l.strip() for l in open(args.wiki) if len(l.strip()) > 80][:400]
    units = cal + oth + wiki
    rng.shuffle(units); return units[: int(len(units) * 0.85)] if seed != 0 else units      # seed = calibration-subsample noise

def batches(enc, maxtok=14000, maxb=64):
    idx = sorted(range(len(enc)), key=lambda i: len(enc[i])); i = 0
    while i < len(idx):
        j = i
        while j < len(idx) and (j - i + 1) * len(enc[idx[j]]) <= maxtok and j - i < maxb: j += 1
        j = max(j, i + 1); sel = idx[i:j]; L = max(len(enc[k]) for k in sel)
        yield sel, L; i = j

# ---------------- quant ----------------
def quant_arm(args):
    from transformers import AutoTokenizer
    dev = "cuda" if torch.cuda.is_available() else "cpu"; dt = torch.bfloat16 if dev == "cuda" else torch.float32
    mdir = args.model; cfg = Q.cfg_from(json.load(open(f"{mdir}/config.json"))); st = Q.ST(mdir)
    tok = AutoTokenizer.from_pretrained(mdir)
    NL, NE, H, I = cfg["NL"], cfg["NE"], cfg["H"], cfg["I"]
    extra_files = [f"{mdir}/{n}" for n in ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "config.json", "generation_config.json") if os.path.exists(f"{mdir}/{n}")] + [os.path.join(os.path.dirname(os.path.abspath(__file__)), "qmoe.py")]
    extra = extra_bytes(extra_files)
    K, hi, lo, nhi = plan(cfg, args.arm, args.target, extra)
    log("plan", args.arm, "K", K, "hi/lo", hi, lo, "n_hi", nhi, "extra files", extra)
    texts = calib_texts(args, args.seed)
    enc = [tok.encode(tmpl(tok, u))[:args.maxlen] for u in texts]
    log("calib units", len(enc), "tokens", sum(map(len, enc)))
    bl = list(batches(enc, args.maxtok))
    emb = st.get("model.embed_tokens.weight", dev).to(dt)
    hs = []; vm = []
    for sel, L in bl:
        ids = torch.zeros(len(sel), L, dtype=torch.long); am = torch.zeros(len(sel), L, dtype=torch.bool)
        for r, k in enumerate(sel): ids[r, :len(enc[k])] = torch.tensor(enc[k]); am[r, :len(enc[k])] = True
        ids, am = ids.to(dev), am.to(dev); hs.append(emb[ids]); vm.append(am)
    cs = []
    for am in vm:
        pos = (am.long().cumsum(-1) - 1).clamp_min(0); cs.append(Q.rope_cs(pos, cfg["hd"], cfg["theta"], dt))
    masks = [Q.causal_mask(am) for am in vm]
    cache_dir = args.stats_dir; os.makedirs(cache_dir, exist_ok=True)
    sp = f"{cache_dir}/mass_seed{args.seed}.pt"
    if os.path.exists(sp): mass = torch.load(sp)
    else:
        mass = torch.zeros(NL, NE, dtype=torch.float64); h2 = [h.clone() for h in hs]
        for l in range(NL):
            L = Q.load_layer_fp(st, l, cfg, dev, dt); col = {}
            for b in range(len(h2)):
                with torch.no_grad(): h2[b] = Q.layer_fwd(h2[b], L, cfg, cs[b][0], cs[b][1], masks[b], vm[b], None, None, col)
            mass[l] = col["mass"].cpu(); del L; log("stats layer", l)
        torch.save(mass, sp); del h2
    frac = mass / mass.sum(1, keepdim=True)
    if args.arm == "M3": keep = [list(range(NE)) for _ in range(NL)]
    else: keep = select(frac, K)
    Ktot = sum(len(k) for k in keep)
    # bit assignment: top-n_hi experts by global per-layer mass fraction get hi bits
    flat = [(frac[l, e].item(), l, e) for l in range(NL) for e in keep[l]]; flat.sort(reverse=True)
    hiset = {(l, e) for _, l, e in flat[:nhi]}
    os.makedirs(args.out, exist_ok=True)
    gptq_mode = args.arm in ("M1", "M2")
    h = hs; tensors_all = {}
    for l in range(NL):
        t0 = time.time(); L = Q.load_layer_fp(st, l, cfg, dev, dt)
        ks = keep[l]; kt = torch.tensor(ks, device=dev)
        eb = [hi if (l, e) in hiset else lo for e in ks]
        Lk = Q.Layer(**{**L.__dict__, "gate": L.gate[kt], "Wgu": L.Wgu[kt], "Wd": L.Wd[kt], "ids": ks})
        out = {}
        # attention RTN (also applied for propagation)
        for n in ("q", "k", "v", "o"):
            c, s, z = Q.rtn(getattr(L, n).float(), AB)
            out[f"{n}.codes"] = Q.pack_bits(c, AB).cpu(); out[f"{n}.sc"] = s.cpu(); out[f"{n}.zp"] = Q.pack_bits(z, AB).cpu()
            setattr(Lk, n, Q.dequant(c, s, z, dt))
        hess = None
        if gptq_mode:
            Kl = len(ks); Hin = torch.zeros(Kl, H, H, device=dev); Hmid = torch.zeros(Kl, I, I, device=dev)
            def hook(e, xe, hh):
                xf = xe.float(); Hin[e] += xf.T @ xf; hf = hh.float(); Hmid[e] += hf.T @ hf
            col = {"hook": hook}
            for b in range(len(h)):
                with torch.no_grad(): Q.layer_fwd(h[b], Lk, cfg, cs[b][0], cs[b][1], masks[b], vm[b], None, None, col)
        for b_ in sorted(set(eb)):
            pos = [i for i, x in enumerate(eb) if x == b_]; pt = torch.tensor(pos, device=dev)
            Wgu = Lk.Wgu[pt].float(); Wd = Lk.Wd[pt].float()
            if gptq_mode:
                c1, s1, z1 = Q.gptq(Wgu, Hin[pt], b_); c2, s2, z2 = Q.gptq(Wd, Hmid[pt], b_)
            else:
                c1, s1, z1 = Q.rtn(Wgu, b_); c2, s2, z2 = Q.rtn(Wd, b_)
            for nm, (c, s, z) in (("gu", (c1, s1, z1)), ("d", (c2, s2, z2))):
                out[f"{nm}{b_}.codes"] = Q.pack_bits(c, b_).cpu(); out[f"{nm}{b_}.sc"] = s.cpu(); out[f"{nm}{b_}.zp"] = Q.pack_bits(z, b_).cpu()
            Lk.Wgu[pt] = Q.dequant(c1, s1, z1, dt); Lk.Wd[pt] = Q.dequant(c2, s2, z2, dt)
        if gptq_mode: del Hin, Hmid
        out.update(ids=torch.tensor(ks, dtype=torch.int16), ebits=torch.tensor(eb, dtype=torch.uint8),
                   ln1=L.ln1.cpu().to(torch.bfloat16), ln2=L.ln2.cpu().to(torch.bfloat16), qn=L.qn.cpu().to(torch.bfloat16), kn=L.kn.cpu().to(torch.bfloat16),
                   gate=L.gate[kt].cpu().to(torch.bfloat16))
        from safetensors.torch import save_file
        save_file({k: v.contiguous() for k, v in out.items()}, f"{args.out}/layer{l}.safetensors")
        if gptq_mode:   # sequential: propagate through quantised layer
            for b in range(len(h)):
                with torch.no_grad(): h[b] = Q.layer_fwd(h[b], Lk, cfg, cs[b][0], cs[b][1], masks[b], vm[b])
        del L, Lk; log("layer", l, "kept", len(ks), "bits", sorted(set(eb)), f"{time.time()-t0:.1f}s")
    del hs, h
    # embeddings / head / final norm
    out = {}
    for nm, w, b_ in (("emb", st.get("model.embed_tokens.weight", dev), MB), ("head", st.get("lm_head.weight", dev), HB)):
        cs_, ss, zs = [], [], []
        for r0 in range(0, w.shape[0], 16384):
            c, s, z = Q.rtn(w[r0:r0 + 16384].float(), b_); cs_.append(c); ss.append(s); zs.append(z)
        out[f"{nm}.codes"] = Q.pack_bits(torch.cat(cs_), b_).cpu(); out[f"{nm}.sc"] = torch.cat(ss).cpu(); out[f"{nm}.zp"] = Q.pack_bits(torch.cat(zs), b_).cpu()
    out["norm"] = st.get("model.norm.weight").to(torch.bfloat16)
    from safetensors.torch import save_file
    save_file(out, f"{args.out}/emb.safetensors")
    meta = dict(cfg=cfg, ab=AB, mb=MB, hb=HB, arm=args.arm, seed=args.seed, K=Ktot, hi=hi, lo=lo, n_hi=nhi, per_layer=[len(k) for k in keep],
                target=args.target, calib_units=len(enc), cal=args.cal)
    json.dump(meta, open(f"{args.out}/meta.json", "w"))
    import shutil
    for f in extra_files: shutil.copy(f, args.out)
    tot = sum(os.path.getsize(os.path.join(args.out, f)) for f in os.listdir(args.out))
    log("PACK BYTES", tot, f"{100*tot/FP16:.4f}%")
    json.dump(dict(bytes=tot, ratio=tot / FP16, K=Ktot), open(f"{args.out}/size.json", "w"))

# ---------------- eval ----------------
import re
_WR = "\"'`*_.!: \t\n\r"
def parse(out):
    s = re.sub(r"<\|[a-zA-Z0-9_\-]+\|>", " ", out).strip().strip(_WR)
    return s.lower()
def correct(out, gold): return parse(out) == gold.lower()

def eval_sets(args):
    from transformers import AutoTokenizer
    dev = "cuda"; tok = AutoTokenizer.from_pretrained(args.model)
    model = Q.load_base_model(args.model, dev) if args.pack == "base" else Q.load_pack_model(args.pack, dev)
    log("model loaded", torch.cuda.memory_allocated() / 1e9, "GB")
    eos = [tok.convert_tokens_to_ids("<|im_end|>"), tok.convert_tokens_to_ids("<|endoftext|>")]
    rows = []
    for name in args.sets.split(","):
        p = args.set_files[name]; rs = rd(p)
        if name == "SEL" and args.n: rs = rs[::max(1, len(rs) // args.n)][:args.n] if not args.first else rs[:args.n]
        rows += [(name, r) for r in rs]
    texts = [tmpl(tok, r.get("user") or r.get("query")) for _, r in rows]
    enc = [tok.encode(t) for t in texts]; res = [None] * len(rows); t0 = time.time()
    if args.reverse: order = list(batches(enc, args.maxtok, args.maxb))[::-1]
    else: order = list(batches(enc, args.maxtok, args.maxb))
    for bi, (sel, L) in enumerate(order):
        ids = torch.full((len(sel), L), tok.pad_token_id or 0, dtype=torch.long); am = torch.zeros(len(sel), L, dtype=torch.bool)
        for r, k in enumerate(sel): ids[r, L - len(enc[k]):] = torch.tensor(enc[k]); am[r, L - len(enc[k]):] = True
        o = model.generate(ids.to(dev), am.to(dev), 8, eos)
        for r, k in enumerate(sel): res[k] = tok.decode(o[r], skip_special_tokens=True)
        if bi % 20 == 0: log("batch", bi, len(order), f"{time.time()-t0:.0f}s")
    with open(args.out, "w") as f:
        for (name, r), o in zip(rows, res):
            f.write(json.dumps(dict(set=name, id=r["id"], out=o, gold=r.get("answer"), src=r.get("src"), kind=r.get("kind"), group=r.get("group"))) + "\n")
    log("eval done", len(rows), f"{time.time()-t0:.0f}s")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd")
    q = sub.add_parser("quant"); q.add_argument("--arm", required=True); q.add_argument("--seed", type=int, default=11); q.add_argument("--out", required=True)
    q.add_argument("--model", required=True); q.add_argument("--cal", required=True); q.add_argument("--other", nargs="*", default=[]); q.add_argument("--wiki", default=None)
    q.add_argument("--n-cal", type=int, default=1000); q.add_argument("--target", type=float, default=0.099); q.add_argument("--maxlen", type=int, default=1200)
    q.add_argument("--maxtok", type=int, default=14000); q.add_argument("--stats-dir", default="stats")
    e = sub.add_parser("eval"); e.add_argument("--pack", required=True); e.add_argument("--model", required=True); e.add_argument("--sets", default="SEL")
    e.add_argument("--set-file", action="append", default=[], help="NAME=path"); e.add_argument("--out", required=True)
    e.add_argument("--n", type=int, default=0); e.add_argument("--first", action="store_true"); e.add_argument("--maxtok", type=int, default=14000); e.add_argument("--maxb", type=int, default=64)
    e.add_argument("--reverse", action="store_true")
    a = ap.parse_args()
    if a.cmd == "quant": quant_arm(a)
    else:
        a.set_files = dict(s.split("=", 1) for s in a.set_file); eval_sets(a)
