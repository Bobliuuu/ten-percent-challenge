import json, glob, sys, statistics as st
pat = sys.argv[1] if len(sys.argv) > 1 else "results/del_s7_k102_e3_dev_*.json"
for f in sorted(glob.glob(pat)):
    for r in json.load(open(f)):
        row = {}
        for c in ("real", "shuffled", "wrong"):
            xs = [d[c] for d in r["draws"] if c in d]
            if xs: row[c] = (round(st.mean(x["teach"] for x in xs), 2), round(max(x["teach"] for x in xs), 2), round(st.mean(x["neigh"] for x in xs), 3), round(st.mean(x["n_deleted"] for x in xs)), round(st.mean(x["carrier_bytes"] for x in xs)))
        print(r["name"], "empty_neigh %.3f" % r["empty"]["neigh"], {k: "teach mean/max %s/%s neigh %s del %s bytes %s" % v for k, v in row.items()})
