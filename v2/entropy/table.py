import json
LIN=13.21e9; REST=94.6e6; BF16=29540067328
print("cfg | fixed | H0 | best-ctx(name) | mix | RC real | scale lossless bits/scale | total bpp(RC+scale) | bytes | ratio_bf16 (REST=94.6MB incl 4bit emb32k)")
for rot in (1,0):
  for L in (4,6,8,16):
    n=f"L{L}_rot{rot}"; rs=json.load(open(f"res_{n}.json")); tot=sum(r["N"] for r in rs); w=lambda g:sum(r["N"]*g(r) for r in rs)/tot
    ks=rs[0]["adapt"]; best=min(ks,key=lambda k:w(lambda r:r["adapt"][k]))
    sc=w(lambda r:r["scale_exp_ctx_left"])+10   # per-scale bits (nscale weight ~ N)
    rc=w(lambda r:r["rc"]["bits_per_param"]); fx=w(lambda r:r["fixed"]); h0=w(lambda r:r["H0"])
    mix=w(lambda r:r["mix"]); bb=w(lambda r:r["adapt"][best])
    ok=all(r["rc"]["decode_ok"] for r in rs)
    for lab,idx,s in (("fixed+fp16 scale",fx,16),("H0+fp16 scale",h0,16),("RC ctx+lossless scale",rc,sc)):
        bpp=idx+s/128; by=LIN*bpp/8+REST
        print(f"{n} {lab}: idx {idx:.4f} scale {s:.2f}b/128 -> {bpp:.4f} bpp, {by/1e9:.3f} GB, ratio {by/BF16:.4f}")
    print(f"   best ctx {best} {bb:.4f}  mix {mix:.4f}  RCok {ok}  gain RC vs H0 {h0-rc:.4f} bpp ({100*(h0-rc)/h0:.2f}%)")
