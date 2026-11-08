import json, math
NL, NE, HID, FF = 48, 128, 2048, 768; FP16 = 61_064_245_248; epar = 3 * HID * FF
tokb = 7_400_000
def other(nk, attn=6, emb=5, head=4):
    ap = HID * 4096 + 2 * HID * 512 + 4096 * HID; bp = lambda b: b + 17 / 128 if b < 16 else 16
    return NL * ap * bp(attn) / 8 + 151936 * HID * bp(emb) / 8 + 151936 * HID * bp(head) / 8 + nk * HID * 2 + nk + (NL * (2 * HID + 256) + HID) * 2 + tokb
def keep_for(bpw, ratio=0.099):
    k = 1
    while True:
        nk = (k + 1) * NL
        if nk * epar * bpw / 8 + other(nk) > ratio * FP16 or k + 1 > NE: return k
        k += 1
def nominal(m, b, g=128):
    base = m.replace("_had", "")
    if base in ("gptq", "gptq_clip", "rtn_noclip", "rtnq", "nf", "awq", "awq_gptq", "omni", "admm", "owq", "hqq"): return b + (16 + b) / g if base not in ("nf",) else b + 16 / g
    if base == "ecsq": return b + 16 / g * 0 + 0.0  # target_bpw already includes scale bits (solve_c subtracts sbits/g)
    if base in ("vq", "rvq", "tcq", "gptvq"): return b + 0.01
    if base == "squeeze": return b + 16 * 2 ** b / 2048 + 0.0045 * 32
    if base == "spqr": return b + 0.5 + 0.32
    return b
common = dict(ds="skill7", quant="gptq", attn_b=6, emb_b=5, head_b=4, sets=["SEL"])
def cfg(name, m, b, g=128, **kw):
    bpw = nominal(m, b, g); k = kw.pop("keep", None) or keep_for(bpw)
    return dict(name=name, m=m, eb=b, eg=g, keep=k, nom_bpw=round(bpw, 3), **common, **kw)
if __name__ == "__main__":
    out = [dict(name="base", ds="skill7", sets=["SEL"], ncal=2000)]
    # A. method sweep at 3-bit-class and 2-bit-class, keep chosen to hit 9.9%
    for b in (3, 2):
        out += [cfg(f"gptq_had_b{b}", "gptq_had", b), cfg(f"gptq_b{b}", "gptq", b), cfg(f"awq_gptq_b{b}", "awq_gptq", b), cfg(f"nf_had_b{b}", "nf_had", b),
                cfg(f"ecsq_had_b{b}", "ecsq_had", b, g=256), cfg(f"ecsq_rdo_had_b{b}", "ecsq_had", b, g=256, lam=0.05),
                cfg(f"vq4_had_b{b}", "vq_had", b, D=4), cfg(f"gptvq_b{b}", "gptvq", b, D=2) if b==2 else cfg(f"gptvq_b{b}", "gptvq", b, D=2)]
    out += [cfg("tcq_had_b2", "tcq_had", 2), cfg("tcq_had_b3", "tcq_had", 3), cfg("e8_had_1.5", "vq_had", 1.5, D=8, cb="e8"), cfg("rvq8_had_b2", "rvq_had", 2, D=8),
            cfg("rtn_b3", "rtnq", 3), cfg("rtn_b2", "rtnq", 2)]
    # B. group size sweeps (gptq_had b3, b2)
    for g in (32, 64, 256): out += [cfg(f"gptq_had_b3_g{g}", "gptq_had", 3, g=g), cfg(f"gptq_had_b2_g{g}", "gptq_had", 2, g=g)]
    # C. mixed per-expert bits by routing mass at equal bytes (avg ~2.5): 
    for hf in (0.3, 0.5): out.append(dict(cfg(f"mix_had_h{int(hf*100)}_3_2", "gptq_had", 2.5, keep=keep_for(2.5 + 0.2)), mix=dict(hi_frac=hf, hi=3, lo=2)))
    out += [cfg("gptq_had_b2.5_ecsq", "ecsq_had", 2.5, g=256)]
    json.dump(out, open("cfgs_sweep.json", "w"), indent=0); print(len(out)); [print(c["name"], c["keep"], c["nom_bpw"]) for c in out]
