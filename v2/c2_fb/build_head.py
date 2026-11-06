"""Build the shipped head.npz (global, fixed) from C2 CALIB only. python build_head.py <out_dir> [params-json]"""
from ana import *
import sys
rows, L, FO, y = ld("calib"); h, keep = build_head(rows, FO, set())
pr = dict(PR0); pr.update(json.loads(sys.argv[2]) if len(sys.argv) > 2 else {}); pr["lam0"] = pr.get("lam0", 1.0)
meta = json.dumps(dict(p=pr, r=int(h["P"].shape[0]), note="C2 feedback head; fit on C2 calib only")).encode()
np.savez(sys.argv[1] + "/head.npz", meta=np.frombuffer(meta, np.uint8), mu0=h["mu0"].astype(np.float16), P=h["P"].astype(np.float16), Ci=h["Ci"].astype(np.float16), m=h["m"].astype(np.float32))
print(os.path.getsize(sys.argv[1] + "/head.npz"), pr)
