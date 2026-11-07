import torch
@torch.no_grad()
def gptq(W, H, bits, group=128, damp=0.01, block=128):
    """W [out,in] (any float), H [in,in] fp32 (sum x x^T). symmetric per-(row,group) absmax scale. returns fake-quant W (same dtype)."""
    dt = W.dtype; W = W.float().clone(); n = W.shape[1]; H = H.clone()
    dead = torch.diag(H) == 0; H[dead, dead] = 1; W[:, dead] = 0
    H += damp * torch.mean(torch.diag(H)) * torch.eye(n, device=H.device)
    try:
        Hinv = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(H)), upper=True)
    except Exception:
        H += 10 * damp * torch.mean(torch.diag(H)) * torch.eye(n, device=H.device)
        Hinv = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(H)), upper=True)
    qm = 2 ** (bits - 1) - 1; Q = torch.zeros_like(W)
    for i1 in range(0, n, block):
        i2 = min(i1 + block, n); W1 = W[:, i1:i2].clone(); Q1 = torch.zeros_like(W1); E1 = torch.zeros_like(W1); Hi = Hinv[i1:i2, i1:i2]
        # group size == block == 128: one scale per row for this block, from current (error-updated) weights
        s = (W1.abs().amax(1, keepdim=True).clamp(min=1e-8) / qm)
        for i in range(i2 - i1):
            w = W1[:, i:i+1]; d = Hi[i, i]
            q = (w / s).round().clamp(-qm - 1, qm) * s
            Q1[:, i:i+1] = q; e = (w - q) / d
            W1[:, i:] -= e @ Hi[i:i+1, i:]; E1[:, i:i+1] = e
        Q[:, i1:i2] = Q1; W[:, i2:] -= E1 @ Hinv[i1:i2, i2:]
    return Q.to(dt)

@torch.no_grad()
def gptq_batched(W, H, bits, group=128, damp=0.01):
    """W [B,out,in], H [B,in,in] fp32 (batched GPTQ, sym per-(row,group128), block=group). returns fake-quant W."""
    dt = W.dtype; W = W.float().clone(); B, out, n = W.shape; H = H.clone()
    idx = torch.arange(n, device=H.device)
    dead = H[:, idx, idx] == 0
    H[:, idx, idx] = torch.where(dead, torch.ones_like(H[:, idx, idx]), H[:, idx, idx]); W = W.masked_fill(dead[:, None, :], 0)
    H[:, idx, idx] += damp * H[:, idx, idx].mean(1, keepdim=True)
    L = torch.linalg.cholesky(H)
    Hinv = torch.linalg.cholesky(torch.cholesky_inverse(L), upper=True)
    qm = 2 ** (bits - 1) - 1; Q = torch.zeros_like(W)
    for i1 in range(0, n, group):
        i2 = i1 + group; W1 = W[:, :, i1:i2].clone(); Q1 = torch.zeros_like(W1); E1 = torch.zeros_like(W1); Hi = Hinv[:, i1:i2, i1:i2]
        s = W1.abs().amax(2, keepdim=True).clamp(min=1e-8) / qm
        for i in range(group):
            w = W1[:, :, i:i+1]; d = Hi[:, i, i][:, None, None]
            q = (w / s).round().clamp(-qm - 1, qm) * s
            Q1[:, :, i:i+1] = q; e = (w - q) / d
            W1[:, :, i:] -= e @ Hi[:, i:i+1, i:]; E1[:, :, i:i+1] = e
        Q[:, :, i1:i2] = Q1; W[:, :, i2:] -= E1 @ Hinv[:, i1:i2, i2:]
    return Q.to(dt)
