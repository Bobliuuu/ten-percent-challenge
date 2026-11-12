import json, glob, statistics as st, sys
tag = sys.argv[1]; GATE = float(sys.argv[2]) if len(sys.argv) > 2 else 0.9
runs = []
for f in sorted(glob.glob(f"results/del_s7_k102_e3_locked_{tag}_d*.json")): runs += json.load(open(f))
out = dict(tag=tag, gate=GATE, empty=runs[0]["empty"], conds={}, draws=[])
for r in runs: out["draws"] += r["draws"]
m = lambda xs: round(st.mean(xs), 4)
for c in ("real", "shuffled", "wrong"):
    xs = [d[c] for d in out["draws"] if c in d]
    if not xs: continue
    g = [(x["teach"], x["neigh"]) if x["fit_acc"] >= GATE else (0.0, out["empty"]["neigh"]) for x in xs]
    out["conds"][c] = dict(n=len(xs), teach_mean=m([x["teach"] for x in xs]), teach_min=min(x["teach"] for x in xs), teach_max=max(x["teach"] for x in xs), neigh_mean=m([x["neigh"] for x in xs]), neigh_min=round(min(x["neigh"] for x in xs), 4),
        min_cell_mean=m([x["min_cell"] for x in xs]), leak_max=max(x["leak"] for x in xs), n_deleted_mean=round(st.mean(x["n_deleted"] for x in xs)), frac_deleted_mean=st.mean(x["frac_deleted"] for x in xs), carrier_bytes_mean=round(st.mean(x["carrier_bytes"] for x in xs)),
        fit_acc=[x["fit_acc"] for x in xs], teach_each=[round(x["teach"], 3) for x in xs], gated_teach_mean=m([a for a, _ in g]), gated_accepted=sum(1 for x in xs if x["fit_acc"] >= GATE), gated_neigh_mean=m([b for _, b in g]))
out["exam"] = [d["real"]["exam"] for d in out["draws"] if "exam" in d.get("real", {})]
json.dump(out, open(f"results/locked_{tag}_summary.json", "w"), indent=1)
print(json.dumps({k: v for k, v in out.items() if k != "draws"}, indent=1))
