import sys; sys.path.insert(0,'/Users/jhome/ten-percent/trackA')
from lib import *
keep=build_vocab(); T=remap_table(keep)
tx,y=dev_data(); y=np.array(y)
B=encode(tx,T)
base=base_model(); 
Bfull=encode(tx,torch.arange(tok.vocab_size))
BL=logits(base,Bfull); BP=BL.argmax(-1).numpy()
def fresh(): return prune_vocab(base_model(),keep,64)
def rep(m,name):
    p=logits(m,B).argmax(-1).numpy(); return f"{name}: acc={np.mean(p==y):.4f} agree={np.mean(p==BP):.4f}"
LIN=["attention.self.query","attention.self.key","attention.self.value","attention.output.dense","intermediate.dense","output.dense"]
def encoder_linears(m,l):
    L=m.bert.encoder.layer[l]; return {n:L.get_submodule(n) for n in LIN}
def rtn_model(bits_enc,g,bits_emb,gemb=128,bits_pool=8):
    m=fresh()
    for l in range(12):
        for n,mod in encoder_linears(m,l).items():
            q,s,z=quant_rtn(mod.weight.data,bits_enc,g); mod.weight.data=dequant(q,s,z,g)
    w=m.bert.embeddings.word_embeddings.weight.data
    q,s,z=quant_rtn(w,bits_emb,gemb); m.bert.embeddings.word_embeddings.weight.data=dequant(q,s,z,gemb)
    return m
if __name__=="__main__":
    print(rep(base.to("cpu") if False else fresh(),"pruned-vocab fp32"))
    for be,g in [(8,128),(4,128),(4,64),(3,128),(3,64),(2,64)]:
        print(rep(rtn_model(be,g,4 if be>=4 else 4),f"RTN enc{be} g{g} emb4"),flush=True)
    for be in [8,6,4,3,2]:
        print(rep(rtn_model(8,128,be),f"RTN enc8 emb{be}"),flush=True)
