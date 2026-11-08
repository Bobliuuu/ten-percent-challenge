"""Training-free trellis-coded quantiser (QTIP-style bitshift trellis), pure PyTorch.
State s in [0,2^L); step emits kb new bits u; next state = ((s<<kb)|u) & (2^L-1); state s emits the V-dim code C[s].
Rate = kb/V bits per weight, fixed. Tail-biting (cyclic) so a length-T sequence costs exactly T*kb bits.
Codebook C (2^L x V) is Lloyd-trained on synthetic N(0,1) data through the trellis (no model weights involved).
Viterbi is reshape/min only (predecessors of ns are (ns>>kb) + j*2^(L-kb)).
"""
import math, torch


class Trellis:
    def __init__(s, L=12, V=2, kb=3, dev="cpu", C=None, seed=0):
        s.L, s.V, s.kb, s.dev = L, V, kb, dev
        s.S = 1 << L
        g = torch.Generator().manual_seed(seed)
        s.C = (torch.randn(s.S, V, generator=g) if C is None else C.clone()).to(dev).float()

    @property
    def bpw(s):
        return s.kb / s.V

    def _viterbi(s, X, start_hi=None):
        """X [B,T,V]. start_hi [B] or None: required high (L-kb) bits of the start state (also required low bits of end state).
        Returns states [B,T] (int64), cost [B]."""
        B, T, V = X.shape
        S, kb, L = s.S, s.kb, s.L
        nl = 1 << (L - kb)
        nj = 1 << kb
        C = s.C
        Cn = (C * C).sum(1)  # [S]
        INF = 1e30
        cost = torch.zeros(B, S, device=X.device)
        if start_hi is not None:
            # start state s0 = (hi << kb) | u0  => s0 >> kb == hi ... wait s0 has L bits: high (L-kb) bits = hi, low kb bits free
            mask = torch.full((B, S), INF, device=X.device)
            base = (start_hi.long() << kb)[:, None] + torch.arange(nj, device=X.device)[None]
            mask.scatter_(1, base, 0.0)
            cost = mask
        bp = torch.empty(T, B, nl, dtype=torch.uint8, device=X.device)
        d0 = (X[:, 0] @ C.T)  # placeholder to size
        for t in range(T):
            d = (X[:, t] * X[:, t]).sum(-1, keepdim=True) - 2 * (X[:, t] @ C.T) + Cn[None]
            if t == 0:
                cost = cost + d
                continue
            # predecessor index j*nl + low ; cost viewed [B, nj, nl]
            m, a = cost.view(B, nj, nl).min(1)
            bp[t] = a.to(torch.uint8)
            cost = m.repeat_interleave(nj, dim=1) + d  # ns = low*nj + u
        if start_hi is not None:
            # end state must have low (L-kb) bits == start_hi
            ns = torch.arange(S, device=X.device)
            ok = (ns & (nl - 1))[None] == start_hi.long()[:, None]
            cost = torch.where(ok, cost, torch.full_like(cost, INF))
        best, e = cost.min(1)
        st = torch.empty(B, T, dtype=torch.long, device=X.device)
        cur = e
        for t in range(T - 1, -1, -1):
            st[:, t] = cur
            if t > 0:
                j = bp[t].gather(1, (cur >> kb)[:, None]).squeeze(1).long()
                cur = (cur >> kb) + (j << (L - kb))
        return st, best

    @torch.no_grad()
    def quantize(s, X, tailbite=True):
        """X [B,T,V] -> (states [B,T], Xq [B,T,V]). Tail-biting via two-pass (free-start pass fixes the overlap bits)."""
        st, _ = s._viterbi(X)
        if tailbite:
            hi = st[:, -1] & ((1 << (s.L - s.kb)) - 1)
            st, _ = s._viterbi(X, start_hi=hi)
        return st, s.C[st]

    def pack_bits(s, st):
        """states [B,T] -> bit tensor of kb*T bits per sequence (the stored payload). Requires tail-biting."""
        return st & ((1 << s.kb) - 1)  # u_t per step (kb bits each)

    def unpack(s, u, ):
        """Rebuild states from u [B,T] (kb bits each) of a tail-biting stream. First state high bits = last (L-kb) bits of the cyclic stream."""
        B, T = u.shape
        kb, L = s.kb, s.L
        # last state's low L bits are the last L bits; s0 = (last (L-kb) bits) << kb | u0.  Build s_{T-1} from last ceil(L/kb) u's.
        n = -(-L // kb)
        sl = torch.zeros(B, dtype=torch.long, device=u.device)
        for i in range(n):
            sl = (sl << kb) | u[:, T - n + i]
        hi = sl & ((1 << (L - kb)) - 1)
        cur = (hi << kb) | u[:, 0]
        out = [cur]
        for t in range(1, T):
            cur = ((cur << kb) | u[:, t]) & (s.S - 1)
            out.append(cur)
        return torch.stack(out, 1)

    @torch.no_grad()
    def train_codebook(s, iters=10, nseq=2048, T=128, seed=1, verbose=True):
        g = torch.Generator(device="cpu").manual_seed(seed)
        for it in range(iters):
            X = torch.randn(nseq, T, s.V, generator=g).to(s.dev)
            st, Xq = s.quantize(X)
            mse = float(((X - Xq) ** 2).mean())
            sums = torch.zeros(s.S, s.V, device=s.dev)
            cnt = torch.zeros(s.S, device=s.dev)
            sums.index_add_(0, st.flatten(), X.reshape(-1, s.V))
            cnt.index_add_(0, st.flatten(), torch.ones(st.numel(), device=s.dev))
            upd = cnt > 0
            s.C[upd] = sums[upd] / cnt[upd, None]
            if verbose:
                print(f"  lloyd it{it} mse/dim {mse:.4f} ({-10*math.log10(mse):.2f} dB; RD bound {6.0206*s.bpw:.2f} dB) empty {int((~upd).sum())}", flush=True)
        # final unit-variance rescale of reconstruction is not applied; scale handled by per-matrix sigma
        return s


def gaussian_test(L, V, kb, dev, iters=6, nseq=1024, T=128):
    tr = Trellis(L, V, kb, dev)
    tr.train_codebook(iters=iters, nseq=nseq, T=T)
    X = torch.randn(nseq, T, V).to(dev)
    st, Xq = tr.quantize(X)
    mse = float(((X - Xq) ** 2).mean())
    u = tr.pack_bits(st)
    st2 = tr.unpack(u)
    ok = bool((st2 == st).all())
    return mse, -10 * math.log10(mse), 6.0206 * tr.bpw, ok


if __name__ == "__main__":
    import sys
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    for (L, V, kb) in [(int(a) for a in x.split(",")) for x in sys.argv[1:]]:
        r = gaussian_test(L, V, kb, dev)
        print(f"L={L} V={V} kb={kb} bpw={kb/V}: mse {r[0]:.4f} SQNR {r[1]:.2f} dB, RD bound {r[2]:.2f} dB, gap {r[2]-r[1]:.2f} dB, tailbite decode ok={r[3]}", flush=True)
