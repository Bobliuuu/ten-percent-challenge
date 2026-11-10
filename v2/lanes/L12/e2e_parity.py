"""(1) Packed generation vs fake-quant engine on dev items. Builds the research engine (fake-quant, mass k25 + RTN4) exactly as run_pack.py, generates on ~200 dev (sel) items; then loads the PACKED artifact
/r/art with the standalone runtime and generates on the same rows; compares. Also checks the router/kept-set parity of the rebuilt engine vs the packed files tensor-by-tensor (quick, experts of 3 layers)."""
import sys, os, json, argparse, collections
ap = argparse.ArgumentParser(); ap.add_argument("--cfg", default="best.json"); ap.add_argument("--out"); ap.add_argument("--root", default="/workspace/fin"); ap.add_argument("--model", default="Qwen/Qwen3.5-35B-A3B"); ap.add_argument("--art", default="/r/art"); ap.add_argument("--n", type=int, default=200)
a = ap.parse_args(); ROOT = os.path.abspath(a.root); sys.path.insert(0, ROOT + "/v2/exam"); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch, loader as L, l12_core as C, l12_runtime as RT
from l12_core import log
T = "SKILL7"; cfg = json.load(open(a.cfg)); os.makedirs(a.out + "/data", exist_ok=True)
w = lambda n, rows: open(f"{a.out}/data/{n}.jsonl", "w").write("".join(json.dumps(r) + "\n" for r in rows))
cal = L.load(T, "cal_big"); sel = L.load(T, "sel"); w("cal", cal); w("sel", sel); w("rep", sel[:50])
rows = sel[::9][:a.n]; assert len(rows) == a.n
eng = C.Engine(a.model, a.out + "/data", a.out); eng.max_new = 6; eng.run_stats(len(cal))
eng.free_live(); eng.set_full_all(); eng.restore_nonexpert(); kept, by, nfb = eng.build(cfg)
rows_e = [dict(r, ids=eng.tok(eng.chat(r["user"]), add_special_tokens=False).input_ids) for r in rows]
fq = eng.gen_eval(rows_e); log("fake-quant done", collections.Counter(fq).most_common(8))
lm, tok = RT.load(a.art, carrier=False); R = RT.Runner(lm, tok); log("packed loaded")
# tensor spot check: engine live experts vs packed-loaded experts, layers 0, 20, 39
spot = {}
for l in (0, 20, 39):
    b = lm.model.layers[l].mlp; lv = eng.blocks[l].live
    spot[l] = dict(G=bool(torch.equal(b.G, lv.G)), U=bool(torch.equal(b.U, lv.U)), D=bool(torch.equal(b.D, lv.D)), router=bool(torch.equal(b.gate.weight, eng.orig_gate[l][torch.tensor(kept[l], device=eng.dev)])))
pk = R.gen([dict(r) for r in rows]); same = [x == y for x, y in zip(fq, pk)]
ok_fq = [x == r["answer"] for x, r in zip(fq, rows)]; ok_pk = [x == r["answer"] for x, r in zip(pk, rows)]
res = dict(n=len(rows), identical=sum(same), identical_frac=sum(same) / len(rows), acc_fakequant=sum(ok_fq) / len(rows), acc_packed=sum(ok_pk) / len(rows), spot=spot,
           diffs=[dict(id=r["id"], fq=x, pk=y, gold=r["answer"]) for r, x, y, s in zip(rows, fq, pk, same) if not s][:30], fq=fq, pk=pk, ids=[r["id"] for r in rows])
json.dump(res, open(a.out + "/parity.json", "w"), indent=1); print("E2E PARITY", json.dumps({k: v for k, v in res.items() if k not in ("fq", "pk", "ids")})); print("E2E DONE")
