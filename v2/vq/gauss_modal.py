"""Gaussian-source TCQ benchmark on a GPU (Modal, detached). Results -> volume tenpct-vq:/gauss.json, printed too."""
import modal, json, os
app = modal.App("tenpct-vq-gauss"); vol = modal.Volume.from_name("tenpct-vq", create_if_missing=True)
HERE = os.path.dirname(os.path.abspath(__file__))
img = modal.Image.debian_slim(python_version="3.11").pip_install("torch==2.5.1", "numpy<2.3").add_local_file(HERE + "/tcq.py", "/code/tcq.py")
@app.function(image=img, gpu="A100-40GB", timeout=3 * 3600, volumes={"/vol": vol})
def run(cfgs):
    import sys, math, torch, time; sys.path.insert(0, "/code"); import tcq
    out = []
    for (L, V, kb, iters) in cfgs:
        t0 = time.time(); tr = tcq.Trellis(L, V, kb, "cuda"); tr.train_codebook(iters=iters, nseq=2048 if L <= 12 else 1024, T=128 // V * 2 // 2, verbose=False)
        X = torch.randn(2048, 128 // V * V // V, V).cuda() if False else torch.randn(2048, 256 // V, V).cuda()
        st, Xq = tr.quantize(X); mse = float(((X - Xq) ** 2).mean()); ok = bool((tr.unpack(tr.pack_bits(st)) == st).all())
        r = dict(L=L, V=V, kb=kb, bpw=kb / V, sqnr=-10 * math.log10(mse), bound=6.0206 * kb / V, decode_ok=ok, secs=time.time() - t0); print(r, flush=True); out.append(r)
        torch.save(tr.C.cpu(), f"/vol/cb_L{L}_V{V}_kb{kb}.pt")
    json.dump(out, open("/vol/gauss.json", "w")); vol.commit(); return out
@app.local_entrypoint()
def main():
    cfgs = [(8,2,3,8),(10,2,3,8),(12,2,3,10),(12,1,2,8),(12,4,6,10),(14,2,3,8),(16,2,3,6),(12,2,2,8),(12,2,4,8)]
    print(run.remote(cfgs))
