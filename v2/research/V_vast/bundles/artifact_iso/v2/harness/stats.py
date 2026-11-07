"""Statistics for the paired non-inferiority criterion (ported from trackC/common.py; no model code here)."""
import numpy as np
MARGIN = 0.01    # default pre-registered margin: 1 accuracy point (see CRITERION.md)
ALPHA = 0.05
def clopper_pearson(k, n, alpha=0.05):
    from scipy.stats import beta
    lo = 0.0 if k == 0 else beta.ppf(alpha / 2, k, n - k + 1)
    hi = 1.0 if k == n else beta.ppf(1 - alpha / 2, k + 1, n - k)
    return float(lo), float(hi)


def mcnemar_exact(b, c):
    """Two-sided exact McNemar. b = base right/cand wrong, c = cand right/base wrong."""
    from scipy.stats import binomtest
    return 1.0 if b + c == 0 else float(binomtest(c, b + c, 0.5).pvalue)


def _tail_p(n, d_obs, theta, s):
    """P(D >= d_obs) where D = c - b, (b,c,rest)~Multinomial(n; (s-theta)/2,(s+theta)/2,1-s)."""
    from scipy.stats import binom
    N = np.arange(n + 1)
    wN = binom.pmf(N, n, s)
    q = (s + theta) / 2 / s
    k = np.ceil((d_obs + N) / 2).astype(int)           # c >= k  <=>  2c-N >= d_obs
    return float(np.sum(wN * binom.sf(k - 1, N, min(max(q, 0), 1))))


GAMMA = 0.001   # Berger-Boos nuisance-parameter confidence level


def _s_interval(n, N_obs, gamma=GAMMA):
    from scipy.stats import beta
    lo = 0.0 if N_obs == 0 else beta.ppf(gamma / 2, N_obs, n - N_obs + 1)
    hi = 1.0 if N_obs == n else beta.ppf(1 - gamma / 2, N_obs + 1, n - N_obs)
    return float(lo), float(hi)


def ni_pvalue(n, b, c, theta0, grid=80, gamma=GAMMA):
    """Exact test (Berger-Boos) of H0: acc_cand - acc_base <= theta0 (theta0=-margin for non-inferiority).
    Multinomial paired model, D=c-b. The discordance rate s=P(b)+P(c) is a nuisance parameter: p-value is the
    sup of the exact tail probability over a (1-gamma) Clopper-Pearson interval for s, plus gamma (valid, exact)."""
    d_obs = c - b
    slo, shi = _s_interval(n, b + c, gamma)
    slo = max(slo, abs(theta0), 1e-9); shi = max(shi, slo)
    ss = np.linspace(slo, shi, grid)
    return min(1.0, max(_tail_p(n, d_obs, theta0, s) for s in ss) + gamma)


def exact_lower_bound(n, b, c, alpha=ALPHA, grid=40):
    """One-sided (1-alpha) exact lower confidence bound on theta = acc_cand - acc_base (test inversion)."""
    lo, hi = -1.0, 1.0
    for _ in range(20):
        mid = (lo + hi) / 2
        if ni_pvalue(n, b, c, mid, grid) > alpha:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2


def paired_report(y, base_pred, cand_pred, margin=MARGIN, alpha=ALPHA):
    y, bp, cp = map(np.asarray, (y, base_pred, cand_pred))
    n = len(y)
    bc, cc = bp == y, cp == y
    b = int(np.sum(bc & ~cc)); c = int(np.sum(~bc & cc))
    both = int(np.sum(bc & cc)); neither = int(np.sum(~bc & ~cc))
    d = c - b
    p_ni = ni_pvalue(n, b, c, -margin)
    lb = exact_lower_bound(n, b, c, alpha)
    diff = (c - b) / n
    se = np.sqrt(max((b + c) - (c - b) ** 2 / n, 0.0)) / n      # paired Wald SE of the accuracy difference
    z = 1.959964
    stmt = (f"candidate - base accuracy = {100*diff:+.2f} pt, paired 95% CI [{100*(diff-z*se):+.2f}, {100*(diff+z*se):+.2f}] pt "
            f"(n={n}, base-only-right b={b}, cand-only-right c={c})")
    return dict(
        diff_ci95_paired_wald=(float(diff - z * se), float(diff + z * se)), plain_statement=stmt,
        n=n, base_correct=int(bc.sum()), cand_correct=int(cc.sum()),
        base_acc=float(bc.mean()), cand_acc=float(cc.mean()),
        base_ci95=clopper_pearson(int(bc.sum()), n), cand_ci95=clopper_pearson(int(cc.sum()), n),
        diff_cand_minus_base=float(cc.mean() - bc.mean()),
        both_right=both, neither_right=neither, base_only_right_b=b, cand_only_right_c=c,
        mcnemar_exact_two_sided_p=mcnemar_exact(b, c),
        margin=margin, alpha=alpha,
        ni_exact_unconditional_p=float(p_ni),
        exact_one_sided_lower_bound_on_diff=float(lb),
        non_inferior=bool(p_ni < alpha),
    )
