"""Power of the pre-registered NI test for a given exam size n and margin. Exact enumeration over (b,c) (as trackN/power.py,
but n is a parameter). Usage: power.py N [margin]  -> CSV of P(pass) for true diff delta and discordance d."""
import sys, functools
from scipy.stats import binom
import stats as S
n = int(sys.argv[1]); m = float(sys.argv[2]) if len(sys.argv) > 2 else 0.01
@functools.lru_cache(None)
def ni(b, c): return S.ni_pvalue(n, b, c, -m, grid=30) < S.ALPHA
def p_pass(delta, d):
    pc, pb = (d + delta) / 2, (d - delta) / 2
    if pc < 0 or pb < 0: return float("nan")
    tot = 0.0
    for N in range(0, int(n * d * 3) + 8):
        pN = binom.pmf(N, n, pb + pc)
        if pN < 1e-10: continue
        for c in range(N + 1):
            if ni(N - c, c): tot += pN * binom.pmf(c, N, pc / (pb + pc))
    return tot
print("n,margin,delta,d,P_pass")
for delta in (0.0, -0.005, -0.01):
    for d in (0.02, 0.05, 0.10):
        print(f"{n},{m},{delta},{d},{p_pass(delta, d):.3f}", flush=True)
