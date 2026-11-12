import json, itertools
base = dict(target="down", nlayers=10, K=0.001, steps=100, lr=0.5, l0=0.0, klw=1.0, ndraw=2, conds=["real", "shuffled"])
vars_ = {
 "a_base": {},
 "b_K0002": dict(K=0.0002), "c_K0005": dict(K=0.0005), "d_K0003_lr2": dict(K=0.003, lr=2.0),
 "e_lr02": dict(lr=0.2), "f_lr2": dict(lr=2.0), "g_steps250": dict(steps=250),
 "h_kl3": dict(klw=3.0), "i_kl03": dict(klw=0.3), "j_l0_1e-5": dict(l0=1e-5), "k_l0_1e-4": dict(l0=1e-4),
 "l_all_K001": dict(target="all", K=0.001), "m_col_K005": dict(target="col", K=0.005), "n_L5": dict(nlayers=5),
}
names = list(vars_)
for i in range(0, len(names), 1):
    n = names[i]; c = dict(base, name=n, **vars_[n])
    json.dump(dict(stage="fb", tag="dev_" + n, art="s7_k102_e3", dev=True, sets=["SEL", "CAL", "FB_TEACH", "FB_ANNOT"], new_label="Recipe", **{"del": [c]}), open(f"job_dev_{n}.json", "w"), indent=1)
print(len(names))
