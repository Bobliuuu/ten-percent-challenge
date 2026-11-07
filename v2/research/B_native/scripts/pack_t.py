"""Lossless pack of Bonsai-8B-unpacked (1-bit, g128): sign bits + fp16 scale per group; everything else raw fp16. Verifies bit-exact round trip."""
import sys,glob,os,json,lzma,torch,numpy as np
from safetensors import safe_open
src=glob.glob(sys.argv[1])[0]; out=sys.argv[2]; os.makedirs(out,exist_ok=True); G=128
bits=open(out+"/bits.bin","wb"); scl=open(out+"/scales.bin","wb"); small={}; index=[]
nbits=nscale=0
for f in sorted(glob.glob(src+"/*.safetensors")):
    with safe_open(f,"pt") as sf:
        for k in sf.keys():
            t=sf.get_tensor(k)
            if t.ndim==2 and t.numel()>1e6:
                g=t.reshape(-1,G); s=g.abs().amax(1)
                c=(torch.sign(g)+1).numpy().astype(np.uint8).reshape(-1,32,4)
                pb=(c[:,:,0]<<6)|(c[:,:,1]<<4)|(c[:,:,2]<<2)|c[:,:,3]
                bits.write(pb.tobytes()); scl.write(s.numpy().tobytes()); index.append((k,list(t.shape)))
                nbits+=pb.size; nscale+=s.numel()*2
            else: small[k]=t
bits.close(); scl.close()
torch.save(small,out+"/small.pt")
# round trip verify
b=np.fromfile(out+"/bits.bin",dtype=np.uint8); s=np.fromfile(out+"/scales.bin",dtype=np.float16)
ob=os_=0; ok=True; checked=0
for f in sorted(glob.glob(src+"/*.safetensors")):
    with safe_open(f,"pt") as sf:
        for k in sf.keys():
            t=sf.get_tensor(k)
            if k in small: ok&=torch.equal(small[k],t); continue
            n=t.numel(); ng=n//G
            pb=b[ob:ob+ng*32].reshape(ng,32); sc=torch.from_numpy(s[os_:os_+ng].copy())
            sg=np.stack([(pb>>6)&3,(pb>>4)&3,(pb>>2)&3,pb&3],-1).reshape(ng,128).astype(np.float32)-1
            r=(torch.from_numpy(sg).half()*sc[:,None]).reshape(t.shape)
            ok&=torch.equal(r,t); ob+=ng*32; os_+=ng; checked+=1
print("bit-exact roundtrip:",ok,"matrices",checked)
json.dump(index,open(out+"/index.json","w"))
print("bits.bin",os.path.getsize(out+"/bits.bin"),"scales.bin",os.path.getsize(out+"/scales.bin"),"small.pt",os.path.getsize(out+"/small.pt"))
