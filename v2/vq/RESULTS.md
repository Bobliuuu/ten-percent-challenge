# v2/vq results (Qwen2.5-14B-Instruct, MMLU dev n=1987, base 0.8007)

Method: random sign + Kronecker(orthogonal, Hadamard) incoherence rotation on both sides of every linear, block-LDLQ (Hessian from calib.jsonl, sequential layers) with an 8-dim shared Gaussian k-means codebook (K=2^logk, bits/w = logk/8, codebook 2^logk*8*2 bytes shared). Decode verified: packed-bit roundtrip exact, decode == quantized rotated-frame weights (maxdiff 2e-7). Embeddings/head NOT quantized in these runs; byte total adds an estimate of 0.108 GB fixed (trimmed 32k-vocab 4-bit embedding + norms/biases + 4 head rows), trimming not implemented here.

| run | bits/w body | est total/bf16 | dev acc | agree w/ base |
|---|---|---|---|---|
| m_mlp13_attn14 | 1.654 | 0.0961 (in budget) | 0.6195 | 0.705 |
| m_k14 (uniform) | 1.75 | 0.1015 (just over) | 0.6578 | 0.745 |
| m_mlp12_attn16 | 1.614 | 0.0939 | 0.5742 | 0.652 |

(Modal base acc 0.8012 on that GPU; 0.8007 reproduced on A100 box.) Slope about +4 pts per +0.1 bit. Best point at <=0.10: ~0.62-0.64 dev acc (uniform ~1.70 bit would be ~0.64), i.e. ~16-18 pts below base. Matching base is not reachable at 10% with this training-free method; extrapolation says ~2.3+ bits needed.

Failed / not finished: Modal GPU pool was queued for hours; AWS p4d/p4de had no capacity; Lambda A100 box was shared (OOM from other jobs, lost one full 1.625-bit run at final eval; RAM starved). 0.5B smoke (chance-level at <=2 bits) only validated mechanics. Code written but NOT run (queued then stopped): per-matrix sensitivity mode (sens12) + allocate.py greedy bit allocation, tgt (fp-activation target, W*Hfq*Hqq^-1) option, smult/ncal/damp sweeps (cfg_r2.json, cfg_r3.json), streaming eval. All in vq.py.
