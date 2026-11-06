from build import *
BITS=dict(q=(2,128),k=(2,128),v=(3,128),a=(3,128),o=(3,256))
enc={}
for l in range(12):
    for n in LIN:
        s=SHORT[n]
        enc[f"L{l}.{n}"]=list((4 if l<6 else 3,128) if s=="i" else BITS[s])
cfg=dict(enc=enc,emb=[[2,128],[3,128],[3,128],[3,128],[3,128]],pool=[6,64],calib="dev",beta=0.0)
t=time.time(); r=build(cfg,HERE+"/A_rans.bin"); print(r,time.time()-t)
