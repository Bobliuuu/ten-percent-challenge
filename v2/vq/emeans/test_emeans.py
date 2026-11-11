"""Unit tests for the opt-in e-means codebook fitter in v2/moe_quant/qlib.py (CPU):  python -m pytest v2/vq/emeans/test_emeans.py -q"""
import os, sys, math
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "moe_quant"))
import torch, qlib as Q

def test_recipe_matches_reference_laws():
    # FAISS contrib/e_means: quality law for d <= 16, throughput above; power-of-two batch; alpha0 from the step law, clamped to [1e-4, 1]
    B, a0, rows = Q.emeans_recipe(100000, 256, 8, 5.0)
    raw = 0.00439175 * 256 ** 0.134749 * 100000 ** 0.876473 * 5 ** 0.813344
    assert B == Q._em_pow2(raw) and B & (B - 1) == 0
    assert math.isclose(a0, min(max(0.8244 * 256 ** -0.4617 * B ** 0.9003 * 100000 ** -0.4498 * 5 ** -0.1057, 1e-4), 1.0))
    assert rows == [100000] * 5
    Bt, _, _ = Q.emeans_recipe(100000, 256, 64, 5.0); assert Bt > B                  # throughput law (d > 16) gives a larger batch
    assert Q.emeans_recipe(1000, 10, 8, 2.5)[2] == [1000, 500, 1000]                  # fractional passes: whole, prefix, closing pass

def test_lloyd_path_unchanged():
    X = torch.randn(5000, 4)
    assert torch.equal(Q.fit_codebook(X, 16, iters=5, seed=3), Q.kmeans_w(X, 16, iters=5, seed=3))

def test_emeans_separated_clusters_no_worse_than_lloyd25():
    # 32 well-separated blobs, random-row init: both fitters can merge blobs (local minima); e-means at 3 passes must not be worse than Lloyd at 25
    g = torch.Generator().manual_seed(0); K = 32; ctr = torch.randn(K, 8, generator=g) * 10
    X = ctr[torch.randint(0, K, (40000,), generator=g)] + 0.1 * torch.randn(40000, 8, generator=g)
    mse = lambda C: float((X - C[Q.nn_argmax(X, C, 0.5 * (C ** 2).sum(1))]).pow(2).mean())
    for seed in range(3):
        C = Q.emeans_w(X, K, epochs=3, seed=seed); assert C.shape == (K, 8) and torch.isfinite(C).all()
        assert mse(C) <= 1.01 * mse(Q.kmeans_w(X, K, iters=25, seed=seed))

def test_emeans_not_worse_than_lloyd_at_matched_passes_gaussian():
    X = torch.randn(60000, 8, generator=torch.Generator().manual_seed(1))
    mse = lambda C: float((X - C[Q.nn_argmax(X, C, 0.5 * (C ** 2).sum(1))]).pow(2).mean())
    assert mse(Q.fit_codebook(X, 256, iters=5, method="emeans")) <= mse(Q.fit_codebook(X, 256, iters=5, method="lloyd"))

def test_weights_and_vq_optin():
    X = torch.randn(20000, 8); w = torch.rand(20000)
    assert Q.emeans_w(X, 64, epochs=2, wts=w).shape == (64, 8)
    W = torch.randn(64, 512); vq = Q.VQ(D=8, K=64, fit=True, fitter="emeans"); vq.prepare(W); out = vq.quant(W[:, :8])
    assert out.shape == (64, 8) and torch.isfinite(out).all()
