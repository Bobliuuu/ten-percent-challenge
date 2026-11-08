"""E0 ambiguity-floor audit. CPU-only, DEV per-item rows only. Reads v2/task_search/results/*.json
and v2/research/D_scale/results/{ct_Qwen2.5-72B_*,real_*}.json. Writes tables.json + tables.md."""
import json, math, collections, itertools, os, sys
import numpy as np
H = os.path.dirname(os.path.abspath(__file__)); V2 = os.path.dirname(H)
TS = V2 + '/task_search/results/'; DS = V2 + '/research/D_scale/results/'
OUT = {}; MD = []
def p(s=''): MD.append(s)
def wil(k, n, z=1.96):
    if n == 0: return (float('nan'),) * 2
    ph = k / n; d = 1 + z*z/n; c = ph + z*z/(2*n); a = z*math.sqrt(ph*(1-ph)/n + z*z/(4*n*n))
    return ((c-a)/d, (c+a)/d)
def load(f): return json.load(open(f))

# ---- gather runs: (model, task) -> {cfg: {id: row}}
runs = collections.defaultdict(dict)   # (model, task) -> cfg -> id->row
def add(model, d, rename=None, strip=False):
    for cfg, v in d.items():
        if 'rows' not in v: continue
        if not strip and cfg[:2] in ('i_', 'n_'): continue
        c = cfg.split('_', 1)[-1] if strip else cfg
        for r in v['rows']:
            t = rename if (rename and r['task'] == 'NEWS') else r['task']
            runs[(model, t)].setdefault(c, {})[r['id']] = r
for m in ('7B', '14B', '32B'):
    mm = f'Qwen2.5-{m}-Instruct'
    add(m, load(TS + mm + '_r1.json'), rename='NEWS6')
    for suf in ('_i', '_n'):
        f = TS + mm + suf + '.json'
        if os.path.exists(f): add(m, load(f), rename='NEWS', strip=True)
CFGS = ['r100', 'r1125', 'r125']; BPW = {'r096': '~1.47', 'r100': '1.57', 'r1125': '1.79', 'r125': '1.96'}
RATIO = {'r096': 9.6, 'r100': 10.0, 'r1125': 11.3, 'r125': 12.5}

# ---- Table 1/2: retention, flip structure, floor
T1 = []
for (m, t), cf in sorted(runs.items()):
    if 'bf16' not in cf or not all(c in cf for c in CFGS): continue
    base = cf['bf16']; B = [i for i, r in base.items() if r['correct']]
    if len(B) < 30: continue
    n = len(B)
    ret = {c: sum(cf[c][i]['correct'] for i in B)/n for c in CFGS}
    fail = {c: np.array([not cf[c][i]['correct'] for i in B]) for c in CFGS}
    allfail = (fail['r100'] & fail['r1125'] & fail['r125']).mean()   # fails at every budget incl. 12.5%
    f2 = (fail['r100'] & fail['r1125']).mean()
    anyfail = (fail['r100'] | fail['r1125'] | fail['r125']).mean()
    exp_ind2 = fail['r100'].mean()*fail['r1125'].mean()
    exp_ind3 = fail['r100'].mean()*fail['r1125'].mean()*fail['r125'].mean()
    # cross-config agreement on ALL items (correct/incorrect labels of the three quantized configs)
    ids = list(base)
    agree = np.mean([len({cf[c][i]['correct'] for c in CFGS}) == 1 for i in ids])
    T1.append(dict(model=m, task=t, n_items=len(ids), base_acc=n/len(ids), n_bc=n,
                   ret100=ret['r100'], ret1125=ret['r1125'], ret125=ret['r125'],
                   floor_all3=allfail, fail_both_100_1125=f2, fail_any3=anyfail,
                   indep_both=exp_ind2, indep_all3=exp_ind3,
                   persistent_share_100=(allfail/max(fail['r100'].mean(), 1e-9)),
                   cfg_agree_all_items=agree))
OUT['T1'] = T1

# ---- margin-vs-flip curve (pooled, per model); m_min is min over answer tokens of base logit gap
BINS = [0, 2, 4, 6, 8, 10, 12, 16, 99]
def mcurve(models, tasks=None, cfgs=CFGS):
    acc = collections.defaultdict(lambda: [0, collections.Counter()])
    for (m, t), cf in runs.items():
        if m not in models or (tasks and t not in tasks) or 'bf16' not in cf or not all(c in cf for c in cfgs): continue
        for i, r in cf['bf16'].items():
            if not r['correct']: continue
            mm = r['m_min']; b = max(j for j in range(len(BINS)-1) if BINS[j] <= mm) if mm < 99 else len(BINS)-2
            a = acc[b]; a[0] += 1
            for c in cfgs: a[1][c] += (not cf[c][i]['correct'])
            # persistent flip: fail in all cfgs
            a[1]['all'] += all(not cf[c][i]['correct'] for c in cfgs)
    rows = []
    for b in sorted(acc):
        n, f = acc[b]; rows.append(dict(bin=f'[{BINS[b]},{BINS[b+1]})', n=n, **{('flip_' + c): f[c]/n for c in cfgs}, flip_all=f['all']/n))
    return rows
OUT['margin'] = {}
for key, ms, ts in [('32B pooled 12 tasks', ['32B'], None), ('14B pooled 12 tasks', ['14B'], None), ('32B INJR', ['32B'], ['INJR']), ('32B NEWS', ['32B'], ['NEWS']), ('32B TOOLSEL', ['32B'], ['TOOLSEL'])]:
    OUT['margin'][key] = mcurve(ms, ts)
# small-margin share and logistic-ish summary: P(flip all3 | m<4) vs overall
def small_share(model, task, thr=4):
    cf = runs[(model, task)]; B = [r for r in cf['bf16'].values() if r['correct']]
    return np.mean([r['m_min'] < thr for r in B]), np.mean([r['m_min'] < 2 for r in B])
OUT['small_margin'] = {f'{m}/{t}': small_share(m, t) for (m, t) in runs if 'bf16' in runs[(m, t)] and (m, t) in [(r['model'], r['task']) for r in T1]}

# ---- per-cell (src x label) retention, base-correct counts, for key tasks, per model
def cells(model, task, cfg):
    cf = runs[(model, task)]; out = collections.defaultdict(lambda: [0, 0, 0])
    for i, r in cf['bf16'].items():
        k = (r['role'], r['src'], r['kind']); out[k][2] += 1
        if r['correct']:
            out[k][0] += 1; out[k][1] += cf[cfg][i]['correct']
    return out
OUT['cells'] = {}
for (m, t) in [('32B', 'INJR'), ('32B', 'NEWS'), ('32B', 'TOOLSEL'), ('14B', 'INJR'), ('14B', 'TOOLSEL'), ('14B', 'NEWS')]:
    if (m, t) not in runs or 'r100' not in runs[(m, t)]: continue
    d = {}
    for c in CFGS:
        for k, (nb, kept, ntot) in cells(m, t, c).items():
            e = d.setdefault('|'.join(k), dict(n_items=ntot, n_bc=nb)); e['ret_' + c] = kept/nb if nb else None
            e['wilson_lo_' + c] = wil(kept, nb)[0] if nb else None
    OUT['cells'][f'{m}/{t}'] = d

# ---- label noise / confusable pairs (base errors): map first-token id -> label via correct rows
def nk(k): return k.split('_', 1)[1] if k[:2] in ('S1', 'S2', 'S3') and '_' in k else k
def confusions(model, task):
    cf = runs[(model, task)]; base = cf['bf16']
    tok = {}
    for r in base.values():
        if r['correct']: tok[r['pred_first']] = nk(r['kind'])
    cm = collections.Counter(); n_unk = 0
    for r in base.values():
        if r['correct']: continue
        if r['pred_first'] in tok and tok[r['pred_first']] != nk(r['kind']): cm[(nk(r['kind']), tok[r['pred_first']])] += 1
        else: n_unk += 1
    return cm, n_unk
OUT['confusions'] = {}
for (m, t) in [('32B', 'INJR'), ('32B', 'NEWS'), ('32B', 'TOOLSEL'), ('32B', 'FIN'), ('32B', 'INJ')]:
    if (m, t) in runs and 'bf16' in runs[(m, t)]:
        cm, nu = confusions(m, t); OUT['confusions'][f'{m}/{t}'] = dict(pairs={f'{a}->{b}': v for (a, b), v in cm.most_common(8)}, other=nu)
# base error by cell
OUT['base_cells'] = {}
for (m, t) in [('32B', 'INJR'), ('32B', 'NEWS'), ('32B', 'TOOLSEL')]:
    cf = runs[(m, t)]['bf16']; d = collections.defaultdict(lambda: [0, 0])
    for r in cf.values():
        k = f"{r['role']}|{r['src']}|{r['kind']}"; d[k][1] += 1; d[k][0] += r['correct']
    OUT['base_cells'][f'{m}/{t}'] = {k: dict(acc=a/n, n=n) for k, (a, n) in sorted(d.items())}

# ---- 72B teacher-forced label tasks: agreement across 7 quantized configs (9.6%) vs base
def r72(f):
    return {r['id']: r for r in load(DS + f)['rows']}
b72 = r72('ct_Qwen2.5-72B_bf16.json')
cfg72 = {k: r72(f'ct_Qwen2.5-72B_{k}.json') for k in ['t096', 'C72_gam1', 'V3_gam1_ncal256_emb8_lcm', 'V7_gam1_ncal256_emb8_lcm_self', 'V12_gam1_ncal256_emb8', 'V9_gam1_ncal256_emb8_lcm_injplus', 'V6_gam1_ncal192_emb8_lblonly']}
T72 = []
for t in sorted({r['task'] for r in b72.values()}):
    B = [i for i, r in b72.items() if r['task'] == t and r['correct']]
    allids = [i for i, r in b72.items() if r['task'] == t]
    fl = np.array([[not cfg72[c][i]['correct'] for i in B] for c in cfg72])
    T72.append(dict(task=t, n_bc=len(B), base_acc=len(B)/len(allids), mean_ret=1-fl.mean(), best_ret=1-fl.mean(1).min(),
                    worst_ret=1-fl.mean(1).max(), floor_allcfg=fl.all(0).mean(), fail_any=fl.any(0).mean(),
                    fail_ge_half=(fl.mean(0) >= .5).mean()))
OUT['T72'] = T72
# real generation 72B: base fp8 vs artifacts V3/V7
T72R = []
for task, arts in [('inj', ['V3', 'V7']), ('toolsel', ['V3', 'V7'])]:
    b = {r['id']: r for r in load(DS + f'real_base72_{task}.json')['rows']}
    A = [{r['id']: r for r in load(DS + f'real_art72_{a}_{task}.json')['rows']} for a in arts]
    B = [i for i, r in b.items() if r['correct']]
    fl = np.array([[not a[i]['correct'] for i in B] for a in A])
    cell = collections.defaultdict(lambda: [0, 0, 0, 0])
    for j, i in enumerate(B):
        k = '|'.join(b[i]['cell']); cell[k][0] += 1; cell[k][1] += fl[0, j]; cell[k][2] += fl[1, j]; cell[k][3] += fl.all(0)[j]
    T72R.append(dict(task=task, n_items=len(b), n_bc=len(B), base_acc=len(B)/len(b), ret_V3=1-fl[0].mean(), ret_V7=1-fl[1].mean(),
                     floor_both=fl.all(0).mean(), cells={k: dict(n_bc=v[0], ret_V3=1-v[1]/v[0], ret_V7=1-v[2]/v[0], both_fail=v[3]/v[0]) for k, v in sorted(cell.items())}))
OUT['T72R'] = T72R

# ---- Section 5: Wilson-based exam sizing
def n_needed(p, thr=0.95, z=1.96, nmax=200000):
    for n in list(range(20, 2000, 5)) + list(range(2000, nmax, 100)):
        k = round(p*n)
        if wil(k, n, z)[0] >= thr: return n
    return None
SZ = {}
for pt in [0.96, 0.97, 0.975, 0.98, 0.985, 0.99, 0.995, 1.0]:
    SZ[pt] = dict(n_bc_95=n_needed(pt, z=1.96), n_bc_90=n_needed(pt, z=1.645))
OUT['sizing'] = SZ
# pass probability (point-estimate rule: cell kept-fraction >=95%, overall >=98%), cells specified as (n_items, base_acc, true_ret)
rng = np.random.default_rng(0)
def pass_prob(cells_, draws=4000, shift=0.0, uncert=True):
    """cells_: list of (n_items, base_acc, ret_mean, n_bc_dev). Dev uncertainty on ret via Beta(k+1,n-k+1)."""
    ok = 0; ok_o = 0; ok_c = 0
    for _ in range(draws):
        tot_bc = tot_kept = 0; allc = True
        for (n, ba, ret, ndev) in cells_:
            p_ = min(max(ret + shift, 0), 1)
            if uncert and ndev: p_ = rng.beta(p_*ndev + 1, (1-p_)*ndev + 1)
            nb = rng.binomial(n, ba); kept = rng.binomial(nb, p_)
            tot_bc += nb; tot_kept += kept; allc &= (kept/nb >= 0.95)
        o = tot_kept/tot_bc >= 0.98; ok_o += o; ok_c += allc; ok += (o and allc)
    return ok/draws, ok_o/draws, ok_c/draws
def spec_from_cells(modeltask, cfg, cellmap, n_per_cell):
    out = []
    for k, e in OUT['cells'][modeltask].items():
        key = k.split('|', 1)[1] if False else k
        out.append(((n_per_cell[k] if isinstance(n_per_cell, dict) else n_per_cell), e['n_bc']/e['n_items'], e['ret_' + cfg], e['n_bc']))
    return out
EXAM = {}
# INJ lock: S1 inj 800, S1 benign 800, S2 inj 360, S2 benign 360 (v2/tournament/INJ/LOCK.json note). INJR dev cells -> map S1->promptshield, S3->deepset, S2audit->S2
def inj_spec(model, cfg, shift=0.0):
    c = OUT['cells'][f'{model}/INJR']; mp = {'S1': 'S1|promptshield', 'S2audit': 'S2audit|jailbreakhub_audit'}
    sp = []
    for lab, nlock in [('injection', 800), ('benign', 800)]:
        for kk, e in c.items():
            if kk.startswith(mp['S1']) and kk.endswith(lab): sp.append((nlock, e['n_bc']/e['n_items'], e['ret_' + cfg], e['n_bc']))
    for lab in ['injection', 'benign']:
        for kk, e in c.items():
            if kk.startswith(mp['S2audit']) and kk.endswith(lab): sp.append((360, e['n_bc']/e['n_items'], e['ret_' + cfg], e['n_bc']))
    return sp
def news_spec(model, cfg):
    c = OUT['cells'][f'{model}/NEWS']; sp = []
    for kk, e in c.items():
        if not (kk.startswith('S1|') or kk.startswith('S2probe|')): continue   # BBC is dev-only OOD proxy, not in lock; HuffPost stands in for the S2 block
        sp.append((400, e['n_bc']/e['n_items'], e['ret_' + cfg], e['n_bc']))
    return sp
PP = []
for model in ['32B', '14B']:
    for cfg in CFGS:
        if f'{model}/INJR' in OUT['cells'] and 'ret_' + cfg in next(iter(OUT['cells'][f'{model}/INJR'].values())):
            PP.append(dict(task='INJ', model=model, cfg=cfg, ratio=RATIO[cfg], cells=len(inj_spec(model, cfg)), **dict(zip(('p_pass', 'p_overall', 'p_cells'), pass_prob(inj_spec(model, cfg))))))
        if f'{model}/NEWS' in OUT['cells'] and 'ret_' + cfg in next(iter(OUT['cells'][f'{model}/NEWS'].values()) or {}) :
            PP.append(dict(task='NEWS', model=model, cfg=cfg, ratio=RATIO[cfg], cells=len(news_spec(model, cfg)), **dict(zip(('p_pass', 'p_overall', 'p_cells'), pass_prob(news_spec(model, cfg))))))
OUT['pass_prob'] = PP
# What true retention is needed? sweep uniform retention across all cells for each task's cell structure/base accs
SW = {}
for name, specf in [('INJ', lambda: inj_spec('32B', 'r100')), ('NEWS', lambda: news_spec('32B', 'r100'))]:
    sp = specf(); SW[name] = {}
    for ret in [0.96, 0.97, 0.98, 0.985, 0.99, 0.995]:
        s2 = [(n, ba, ret, 0) for (n, ba, _, _) in sp]
        SW[name][ret] = pass_prob(s2, draws=3000, uncert=False)[0]
OUT['sweep_uniform'] = SW

# ---- exact-binomial sizing: n_bc needed so that P(cell passes) >= target
from math import lgamma, log, exp
def binom_cdf_ge(n, p, kmin):
    if p >= 1: return 1.0
    ks = np.arange(kmin, n+1); lp = np.array([lgamma(n+1)-lgamma(k+1)-lgamma(n-k+1)+k*log(p)+(n-k)*log(1-p) for k in ks])
    return float(np.exp(lp).sum())
def n_for_prob(p, target=0.8, rule='point', thr=0.95):
    for n in range(20, 6000, 5):
        if rule == 'point': kmin = math.ceil(thr*n - 1e-9)
        else:  # wilson lower bound >= thr
            kmin = next((k for k in range(n+1) if wil(k, n)[0] >= thr), n+1)
            if kmin > n: continue
        if binom_cdf_ge(n, p, kmin) >= target: return n
    return None
SZ2 = {}
for pt in [0.96, 0.97, 0.98, 0.99, 0.995]:
    SZ2[pt] = dict(point_80=n_for_prob(pt, .8, 'point'), point_95=n_for_prob(pt, .95, 'point'), wilson_50=n_needed(pt), wilson_80=n_for_prob(pt, .8, 'wilson'))
OUT['sizing_exact'] = SZ2
# per-task cell lists with dev base accuracy (cells: label x source). items needed = n_bc / base_acc_cell
CELLTAB = {}
for name, mt, keep in [('INJ (lock: S1,S2 x inj/benign)', '32B/INJR', lambda k: k.startswith('S1|') or k.startswith('S2audit|')),
                       ('NEWS (plan: AG,HuffPost x 4 classes)', '32B/NEWS', lambda k: k.startswith('S1|') or k.startswith('S2probe|')),
                       ('TOOLSEL (S1,S3 x tool/none)', '32B/TOOLSEL', lambda k: True)]:
    rows = []
    for k, e in OUT['cells'][mt].items():
        if not keep(k): continue
        ba = e['n_bc']/e['n_items']
        r = dict(cell=k, base_acc=ba, dev_ret100=e['ret_r100'])
        for pt in (0.98, 0.99):
            nb = SZ2[pt]['point_80']; r[f'items_p{pt}_pass80'] = math.ceil(nb/ba)
            nb = SZ2[pt]['wilson_80']; r[f'items_p{pt}_wilson80'] = math.ceil(nb/ba)
        rows.append(r)
    CELLTAB[name] = rows
OUT['celltab'] = CELLTAB

json.dump(OUT, open(H + '/tables.json', 'w'), indent=1, default=float)
print('ok')
