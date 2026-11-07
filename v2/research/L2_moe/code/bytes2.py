"""Correct byte table for Qwen3-235B-A22B. PRUNE fraction p (keep 1-p). Expert bits = entropy-coded b-bit symmetric RTN/GPTQ codes (measured entropy of RTN g128 codes on real experts: 3b 2.149, 4b 3.300, 5b 4.365; GPTQ assumed same) + 0.125 bpw fp16 scale per 128.
Everything counted: experts, attn (ab bits +scale), embed+lm_head (mb bits +scale), router rows of kept experts (mb bits), norms (bf16), tokenizer/config 16 MB, code 0.15 MB."""
from bytes import MODELS
ENT = {2: 0.775, 3: 2.149, 4: 3.300, 5: 4.365, 6: 5.4}
def total(p, eb, ab=6, mb=6, m="Qwen3-235B-A22B"):
    s = MODELS[m]; keep = 1 - p
    ex = s["expert"] * keep * (ENT[eb] + .125) / 8
    at = s["attn"] * (ab + .125) / 8; em = s["emb"] * (mb + .125) / 8; rt = s["router"] * keep * mb / 8
    norms = 94 * (2 * 4096 + 2 * 128 + 4096) * 2 + 4096 * 2
    return ex + at + em + rt + norms + 16e6 + 0.15e6
fp16 = sum(MODELS["Qwen3-235B-A22B"].values()) * 2
if __name__ == "__main__":
    print(f"fp16 = {fp16/1e9:.2f} GB ; 10% = {fp16/1e10:.2f} GB ; 9.6% = {fp16*.096/1e9:.2f} GB")
    for p in (0.0, .25, .375, .5, .625, .75):
        print(f"prune {p:5}: " + "  ".join(f"{eb}b {total(p,eb)/fp16*100:5.2f}% " for eb in (3, 4)) + f" | 3b with attn/emb 5b: {total(p,3,5,5)/fp16*100:5.2f}% | 3b attn/emb 4b: {total(p,3,4,4)/fp16*100:5.2f}%")
