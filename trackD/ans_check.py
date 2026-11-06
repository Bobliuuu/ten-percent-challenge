# real range-ANS roundtrip (constriction) on 3-bit GPTQ-like codes of one real weight matrix; compares actual bytes vs Huffman/entropy
import numpy as np, torch, constriction, sys
sys.path.insert(0,'.')
from transformers import AutoModelForSequenceClassification
import engine
m=AutoModelForSequenceClassification.from_pretrained(engine.M)
W=m.bert.encoder.layer[5].intermediate.dense.weight.data
for bits in (2,3,4):
    d,q,meta=engine.rtn(W,bits,64); codes=q.numpy().astype(np.int32).ravel()
    cnt=np.bincount(codes,minlength=2**bits).astype(np.float64); p=(cnt/cnt.sum()).astype(np.float32); p=np.maximum(p,1e-6); p/=p.sum()
    enc=constriction.stream.stack.AnsCoder(); model=constriction.stream.model.Categorical(p,perfect=False)
    enc.encode_reverse(codes,model); comp=enc.get_compressed(); nbytes=len(comp)*4
    dec=constriction.stream.stack.AnsCoder(comp); out=dec.decode(model,len(codes))
    assert (out==codes).all()
    print(bits,"bits: fixed %d B, huffman %d B, entropy %d B, ANS actual %d B (+table %d B) roundtrip OK"%(codes.size*bits//8,engine.huff_bits(q,2**bits)//8,engine.ent_bits(q,2**bits)//8,nbytes,(2**bits)*2))
