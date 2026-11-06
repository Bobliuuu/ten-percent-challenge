import struct,json,numpy as np,lzma,zlib,collections
raw=open('/Users/jhome/ten-percent/trackA/model.bin','rb').read();n=struct.unpack("<I",raw[:4])[0];meta=json.loads(raw[4:4+n]);data=raw[4+n:]
def H(a,k):
    c=np.bincount(a,minlength=k).astype(float);p=c[c>0]/c.sum();return -(p*np.log2(p)).sum()
tot=collections.Counter()
for k,d in meta['q'].items():
    O,I=d['shape'];b=d['bits'];g=d['g'];o1,o2,o3=d['o'];N=O*I
    bits=np.unpackbits(np.frombuffer(data[o1:o1+(N*b+7)//8],dtype=np.uint8))[:N*b].reshape(N,b);q=(bits<<np.arange(b)).sum(1)
    ng=O*(I//g);z=np.frombuffer(data[o3:o3+2*ng],dtype=np.float16).astype(int)
    s=data[o2:o2+2*ng]
    hq=H(q,2**b);hz=H(z,2**b)
    typ='emb' if 'word' in k else 'enc'
    tot[typ+'_raw']+=(N*b+7)//8; tot[typ+'_H']+=N*hq/8; tot[typ+'_z_raw16']+=2*ng; tot[typ+'_zH']+=ng*hz/8; tot[typ+'_zpack']+=ng*b/8; tot[typ+'_s']+=2*ng
    tot[typ+'_slzma']+=len(lzma.compress(s,preset=9))
    if typ=='enc' and ('layer.0.' in k or 'layer.6.' in k): print(k,b,g,f"H(q)={hq:.3f}",f"H(z)={hz:.2f}")
    if typ=='emb': print(k,b,f"H(q)={hq:.3f}",f"H(z)={hz:.2f}",O)
for k,v in sorted(tot.items()): print(k,f"{v/1e6:.3f}MB")
