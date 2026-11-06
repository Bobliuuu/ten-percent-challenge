exec(open('/Users/jhome/ten-percent/trackA/ev2.py').read())
r3(fresh(),"vocabprune fp32")
r3(rtn_model(8,128,16 if False else 8),"RTN enc8 emb8")
r3(rtn_model(4,128,4),"RTN enc4 g128 emb4")
r3(rtn_model(3,128,4),"RTN enc3 g128 emb4")
for b,g,e in [(4,128,4),(3,128,4),(3,64,4)]:
    m,pk=gptq_quantize(fresh(),lambda l,n:b,g,embbits=e,clip=True); r3(m,f"GPTQclip enc{b} g{g} emb{e} bytes~{bytes_est(pk)/1e6:.1f}")
