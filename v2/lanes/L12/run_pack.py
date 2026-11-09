"""L12 packer job: builds the mass k25 + RTN4 body (cfg best.json), writes real packed bytes (l12_pack.export), sha256 of every file, then reloads from disk with l12_loader and compares EVERY packed tensor against the live engine weights (bit-exact in bf16)."""
import sys, os, json, argparse, hashlib
ap = argparse.ArgumentParser(); ap.add_argument("--cfg", default="best.json"); ap.add_argument("--out", default="/workspace/out_pack"); ap.add_argument("--root", default="/workspace/fin"); ap.add_argument("--model", default="Qwen/Qwen3.5-35B-A3B")
a = ap.parse_args(); ROOT = os.path.abspath(a.root); sys.path.insert(0, ROOT + "/v2/exam"); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch, loader as L, l12_core as C, l12_pack as P, l12_loader as LD
from l12_core import log
T = "SKILL7"; cfg = json.load(open(a.cfg)); os.makedirs(a.out + "/data", exist_ok=True)
w = lambda n, rows: open(f"{a.out}/data/{n}.jsonl", "w").write("".join(json.dumps(r) + "\n" for r in rows))
cal = L.load(T, "cal_big"); sel = L.load(T, "sel"); w("cal", cal); w("sel", sel); w("rep", sel[:50])
eng = C.Engine(a.model, a.out + "/data", a.out); eng.max_new = 6; eng.run_stats(len(cal))
eng.free_live(); eng.set_full_all(); eng.restore_nonexpert(); kept, by, nfb = eng.build(cfg); log("BYTES", json.dumps(by))
pk = a.out + "/packed"; res = dict(cfg=cfg, bytes_analytic=by); res["export"] = P.export(eng, kept, cfg, pk)
files = {}
for f in sorted(os.listdir(pk)):
    h = hashlib.sha256(); 
    with open(f"{pk}/{f}", "rb") as fh:
        for ch in iter(lambda: fh.read(1 << 24), b""): h.update(ch)
    files[f] = dict(bytes=os.path.getsize(f"{pk}/{f}"), sha256=h.hexdigest())
res["files"] = files; res["total_file_bytes"] = sum(v["bytes"] for v in files.values()); log("PACK files", len(files), res["total_file_bytes"])
# parity: reload from disk, compare every tensor to the live engine
D = LD.load_dir(pk); P_ = dict(eng.lm.named_parameters()); bad = []; n = 0; dt = eng.dtype
def cmp(name, got, ref):
    global n; n += 1
    if not torch.equal(got.to(dt).cpu(), ref.to(dt).cpu()): bad.append(name)
d = D["attn_shared"]
for nm in LD.names(d): cmp(nm, LD.dequant(d, nm), P_[nm].data)
for l in range(eng.NL):
    d = D[f"layer{l}"]; lv = eng.blocks[l].live; ks = len(kept[l])
    for nm, ref in (("g", lv.G), ("u", lv.U), ("d", lv.D)): cmp(f"L{l}.{nm}", LD.dequant(d, f"L{l}.{nm}"), ref)
    if l % 10 == 0: log("parity layer", l, "bad so far", len(bad))
for nmx, W in (("embed", eng.lm.model.embed_tokens.weight.data), ("head", eng.lm.lm_head.weight.data)):
    d = D[nmx]; parts = [LD.dequant(d, f"{nmx}.{i}") for i in range(len(LD.names(d)))]; cmp(nmx, torch.cat(parts, 0), W)
res["parity"] = dict(tensors_compared=n, mismatches=bad[:20], n_mismatch=len(bad), exact=len(bad) == 0)
json.dump(res, open(a.out + "/pack_result.json", "w"), indent=1); log("PACK_DONE", json.dumps(res["parity"]), res["total_file_bytes"], res["export"]["ratio_vs_full_checkpoint"]); print("PACK_DONE")
