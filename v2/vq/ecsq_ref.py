"""Reference: entropy-coded uniform scalar quantiser (centroid recon) on N(0,1): best SQNR at given entropy. Gap vs RD bound."""
import torch, math
x=torch.randn(4_000_000)
for D in [1.2,1.3,1.4,1.5,1.6,1.8,2.0]:
    best=None
    for dz in [0.0,0.1,0.2,0.3]:
        q=torch.sign(x)*torch.floor(x.abs()/D+0.5-dz); 
        vals=q.unique(); rec=torch.zeros_like(x)
        for v in vals: m=q==v; rec[m]=x[m].mean()
        p=torch.stack([(q==v).float().mean() for v in vals]); H=-(p*p.log2()).sum().item(); mse=((x-rec)**2).mean().item()
        if best is None or mse<best[0]: best=(mse,H,dz)
    print(f"step {D}: dz {best[2]} H {best[1]:.3f} bits SQNR {-10*math.log10(best[0]):.2f} dB  bound {6.0206*best[1]:.2f}")
