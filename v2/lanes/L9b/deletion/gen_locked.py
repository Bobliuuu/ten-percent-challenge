import json, sys
cfg = json.loads(sys.argv[1]); tag = sys.argv[2]
for d0 in (0, 2, 4, 6):
    c = dict(cfg, name=f"{cfg['name']}_d{d0}", ndraw=2, draw0=d0, conds=["real", "shuffled", "wrong"])
    json.dump(dict(stage="fb", tag=f"locked_{tag}_d{d0}", art="s7_k102_e3", dev=False, sets=["EXAM", "CAL", "FB_TEACH", "FB_ANNOT"], new_label="Recipe", exam_draws=(2 if d0 == 0 else 0), save_mask=(d0 == 0), **{"del": [c]}), open(f"job_locked_{tag}_d{d0}.json", "w"), indent=1)
