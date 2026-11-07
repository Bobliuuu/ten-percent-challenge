"""Verification of a TPQ1 artifact against the quantiser-side simulation and the harness checks.
  python verify.py --art ART --sim-dir SIMDIR [--sim-hashes ART.sim_hashes.json] [--report ART.quant_report.json] [--base-arch ART.base_arch.json]
 1. bit-exactness: for every tensor, loader output (decoded rANS -> dequant -> bf16) torch.equal quantiser-simulated bf16 tensor
 2. size: exact bytes of every file vs 10% of fp16 base bytes, and vs the quantiser's estimate (entropy*1.005 + 11/g bits)
 3. harness/v3 checks.size_check + no_new_parts (+ manifest decode-and-compare)"""
import argparse, glob, json, os, sys, time
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import loader as LD


def bitexact(art, sim_dir, dev, workers):
    from safetensors import safe_open
    c = LD.Container(os.path.join(art, LD.BLOB), workers=workers); sim = {}
    for f in sorted(glob.glob(os.path.join(sim_dir, "sim-*.safetensors"))):
        with safe_open(f, "pt") as h:
            for k in h.keys(): sim[k] = f
    bad, n, nel, t0, dt = [], 0, 0, time.time(), 0.0
    for name in c.names():
        t = c.T[name]
        if t["kind"] != "q": continue
        t1 = time.time(); w = c.tensor(name, dev); dt += time.time() - t1
        with safe_open(sim[name], "pt") as h: s = h.get_tensor(name).to(dev)
        ok = bool(torch.equal(w, s)); n += 1; nel += w.numel()
        if not ok: bad.append(dict(name=name, mismatch=int((w != s).sum()), of=w.numel()))
    c.close()
    return dict(tensors_compared=n, elements=nel, all_equal=not bad, mismatches=bad[:10], decode_dequant_seconds=dt, wall_seconds=time.time() - t0)


def bitexact_hashes(art, hashes, dev, workers):
    """no sim tensors on disk (e.g. 32B+): compare sha256 of the loader's bf16 bytes with the quantiser-side sha256 of its simulated tensor"""
    H = json.load(open(hashes)); c = LD.Container(os.path.join(art, LD.BLOB), workers=workers); bad, n, dt = [], 0, 0.0
    for name in c.names():
        if c.T[name]["kind"] != "q": continue
        t1 = time.time(); w = c.tensor(name, dev); dt += time.time() - t1; n += 1
        if LD.sha_tensor(w) != H[name]: bad.append(name)
    c.close(); return dict(mode="sha256", tensors_compared=n, all_equal=not bad, mismatches=bad[:10], decode_dequant_seconds=dt)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--art", required=True); ap.add_argument("--sim-dir"); ap.add_argument("--sim-hashes"); ap.add_argument("--report"); ap.add_argument("--base-arch"); ap.add_argument("--out")
    ap.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 1)); a = ap.parse_args(); dev = "cuda" if torch.cuda.is_available() else "cpu"
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "harness", "v3")); import checks
    res = {}
    if a.sim_dir and glob.glob(os.path.join(a.sim_dir, "sim-*.safetensors")): res["bitexact"] = bitexact(a.art, a.sim_dir, dev, a.workers)
    elif a.sim_hashes: res["bitexact"] = bitexact_hashes(a.art, a.sim_hashes, dev, a.workers)
    rep = json.load(open(a.report or a.art.rstrip("/") + ".quant_report.json")); base_bytes = rep["base_bytes"]
    sz = checks.size_check(a.art, base_bytes); files = {f["file"]: f["bytes"] for f in sz["files"]}
    fixed = sum(b for f, b in files.items() if f != LD.BLOB)
    est_total = rep["est_quant_bytes"] + rep["raw_bytes"] + fixed          # estimate formula (dense_stream): quant entropy est + raw tensors + non-weight files + code
    q_real = sum(r["real_bytes"] for r in rep["rows"]); q_est = sum(r["est_bytes"] for r in rep["rows"])
    res["size"] = dict(total_bytes=sz["candidate_bytes"], base_fp16_bytes=base_bytes, ratio=sz["byte_ratio"], size_ok=sz["ok"], files=files,
                       estimate_total_bytes=est_total, estimate_ratio=est_total / base_bytes, real_minus_estimate_pct_of_base=100 * (sz["candidate_bytes"] - est_total) / base_bytes,
                       quant_payload=dict(real_bytes=q_real, est_bytes=q_est, real_codes=sum(r["real_codes"] for r in rep["rows"]), real_scale_exp=sum(r["real_exp"] for r in rep["rows"]),
                                          real_scale_mant=sum(r["real_mant"] for r in rep["rows"]), est_code_bytes=sum(r["n"] * r["ent_bits"] * 1.005 / 8 for r in rep["rows"]),
                                          est_scale_bytes=sum(r["n"] * 11 / r["g"] / 8 for r in rep["rows"])),
                       blob_overhead_bytes=files.get(LD.BLOB, 0) - q_real - rep["raw_bytes"])
    if a.base_arch:
        ba = json.load(open(a.base_arch)); arch = ba["arch"]; bc = ba.get("config")
        nnp = checks.no_new_parts(a.art, arch, bc, allowed_extra=("model.tpq",))
        c = LD.Container(os.path.join(a.art, LD.BLOB)); hm = c.manifest(); c.close(); man = json.load(open(os.path.join(a.art, "tensor_manifest.json")))
        res["no_new_parts"] = {k: v for k, v in nnp.items() if k not in ("files",)}
        res["manifest_vs_container"] = dict(equal=all(man[k][1] == hm[k][1] for k in man) and set(man) == set(hm), n=len(man))
    print(json.dumps(res, indent=1, default=str))
    if a.out: json.dump(res, open(a.out, "w"), indent=1, default=str)


if __name__ == "__main__": main()
