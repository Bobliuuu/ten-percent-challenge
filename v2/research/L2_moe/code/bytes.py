"""bytes(model, keep_frac, expert_bpw, attn_bpw, emb_bpw): payload + fp16 group scale (g=128: +0.125 bpw) ; router rows & norms bf16; excludes entropy-coding gain and tokenizer/config (~16 MB)."""
def sizes(L, d, E, mi, nh, nkv, hd, V, tie=False):
    ex = L * E * 3 * d * mi
    at = L * (d * nh * hd + 2 * d * nkv * hd + nh * hd * d)
    em = V * d * (1 if tie else 2)
    rt = L * E * d
    return dict(expert=ex, attn=at, emb=em, router=rt)
MODELS = {
 "Qwen3-30B-A3B": sizes(48, 2048, 128, 768, 32, 4, 128, 151936),
 "Qwen3-235B-A22B": sizes(94, 4096, 128, 1536, 64, 4, 128, 151936),
}
def total_bytes(m, keep, eb, ab=16, mb=16, gs=0.125):
    s = MODELS[m]
    q = lambda bits: bits + (gs if bits < 16 else 0)
    return (s["expert"] * keep * q(eb) + s["attn"] * q(ab) + s["emb"] * q(mb) + s["router"] * keep * 16) / 8
if __name__ == "__main__":
    for m in MODELS:
        s = MODELS[m]; fp16 = sum(s.values()) * 2
        print(m, "params(B)", {k: round(v/1e9, 2) for k, v in s.items()}, "fp16 GB", round(fp16/1e9, 1), "10% budget GB", round(fp16/1e10, 2))
        for keep in (1, .75, .5, .375, .25):
            row = []
            for eb in (16, 8, 4, 3, 2.5, 2):
                row.append(f"{eb}b:{total_bytes(m, keep, eb, 6, 6)/fp16*100:5.1f}%")
            print(f"  keep {keep:5}  ", " ".join(row))
