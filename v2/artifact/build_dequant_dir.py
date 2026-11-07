"""Build a TPQ1 artifact on a box that holds the D_scale/L3-style quantised dir: L{i}.pt (packed 4-bit codes, fp16 scales, int8 sg) + emb.pt/head.pt (DEQUANTISED bf16 only).
  python build_dequant_dir.py --q QDIR --base HF_SNAPSHOT --out ART [--head-L 8 --emb-L 8 --g 128 --mbits 10 --verify 0,40,79]
Body: lossless re-encode of the saved integer codes + scales (bit-exact by construction; checked against an independent fp32 reference built from the packed codes).
emb/head: integer codes were NOT saved -> recovered by re-rotating the saved bf16 tensor and RTN-snapping to its (L-level, g=128) grid; the result is checked against the saved
tensor and the number of differing bf16 elements is reported (exactness NOT guaranteed for head, which was GPTQ-quantised)."""
import argparse, glob, json, math, os, sys, time, shutil
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import loader as LD, convert_l3 as CV, writer as WR
from safetensors import safe_open


def log(*a): print(time.strftime("%H:%M:%S"), *a, flush=True)


def requant(w, path, name, seed, tag, L, g, mbits, hid, nrows_chunk=2048, src=None):
    from quantize import signs
    W = torch.load(path, map_location="cpu"); SRC = src if src is not None else W; R, C = W.shape; sg = signs(seed, tag, C, "cpu"); w.add_signs(tag, sg)
    codes = torch.empty(R, C, dtype=torch.uint8); sc = torch.empty(R, C // g, dtype=torch.float16); diff = 0
    for r in range(0, R, nrows_chunk):
        Wr = LD.rot(SRC[r:r + nrows_chunk].float(), sg); cq, sq = LD.rtn_rows(Wr, L, g, mbits); codes[r:r + nrows_chunk] = cq; sc[r:r + nrows_chunk] = sq
        deq = LD.dequant(cq, sq, L, g, sg, "cpu"); diff += int((deq != W[r:r + nrows_chunk]).sum())
    st = w.add_q(name, codes.numpy(), sc.numpy(), L, g, mbits, tag); del codes, sc
    return dict(bf16_elements_differing_from_saved=diff, elements=R * C, bytes=sum(st.values()))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--q", required=True); ap.add_argument("--base", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--layers", type=int, default=80); ap.add_argument("--emb-L", type=int, default=8); ap.add_argument("--head-L", type=int, default=8); ap.add_argument("--g", type=int, default=128)
    ap.add_argument("--mbits", type=int, default=10); ap.add_argument("--verify", default="0,40,79"); ap.add_argument("--no-embhead", action="store_true"); a = ap.parse_args()
    t0 = time.time(); os.makedirs(a.out, exist_ok=True); w = LD.Writer(os.path.join(a.out, LD.BLOB)); cache = {}; lin = set(); rep = {}
    for i in range(a.layers):
        d = torch.load(os.path.join(a.q, f"L{i}.pt"), map_location="cpu"); st = CV.convert_layer(w, i, d, a.g, a.mbits, cache); lin |= set(st)
        if i % 8 == 0: log("layer", i, f"{os.path.getsize(w.path) / 1e9:.2f}GB")
    body = os.path.getsize(w.path); rep["body_blob_bytes"] = body; files = sorted(glob.glob(a.base + "/*.safetensors")); shapes = {}; dtypes = {}; raw = 0
    cfg = json.load(open(a.base + "/config.json")); hid = cfg["hidden_size"]
    for f in files:
        with safe_open(f, "pt") as h:
            for k in h.keys(): sl = h.get_slice(k); shapes[k] = list(sl.get_shape()); dtypes[k] = sl.get_dtype()
    for f in files:
        with safe_open(f, "pt") as h:
            for k in h.keys():
                if k in lin or k in ("model.embed_tokens.weight", "lm_head.weight"): continue
                x = h.get_tensor(k); w.add_raw(k, x, "BF16"); raw += x.numel() * 2
    rep["raw_bytes"] = raw
    if not a.no_embhead:
        src = None                                    # embedding quantiser is deterministic RTN: re-run it on the ORIGINAL embedding (exact), compare with the saved dequantised tensor
        for f in files:
            with safe_open(f, "pt") as h:
                if "model.embed_tokens.weight" in h.keys(): src = h.get_tensor("model.embed_tokens.weight")
        rep["emb"] = requant(w, os.path.join(a.q, "emb.pt"), "model.embed_tokens.weight", 88888, "emb", a.emb_L, a.g, a.mbits, hid, src=src); log("emb", rep["emb"])
        if "lm_head.weight" in shapes: rep["head"] = requant(w, os.path.join(a.q, "head.pt"), "lm_head.weight", 99999, "head", a.head_L, a.g, a.mbits, hid); log("head", rep["head"])
    for f in ("config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json"):
        if os.path.exists(os.path.join(a.base, f)): shutil.copy(os.path.join(a.base, f), os.path.join(a.out, f))
    json.dump({k: [dtypes[k], shapes[k]] for k in shapes}, open(os.path.join(a.out, "tensor_manifest.json"), "w"), separators=(",", ":"))
    rep["blob_bytes"] = w.finish(meta=dict(source=a.q, mbits=a.mbits)); WR.assemble(a.out)
    sizes = {f: os.path.getsize(os.path.join(a.out, f)) for f in os.listdir(a.out)}; base = sum(2 * math.prod(s) for s in shapes.values())
    rep.update(files=sizes, total_bytes=sum(sizes.values()), base_fp16_bytes=base, ratio=sum(sizes.values()) / base, secs=time.time() - t0)
    ver = []; c = LD.Container(os.path.join(a.out, LD.BLOB), workers=8)
    for i in [int(x) for x in a.verify.split(",") if x]:
        d = torch.load(os.path.join(a.q, f"L{i}.pt"), map_location="cpu")
        for mod, e in d.items():
            cc = e["codes"]; R_, C2 = cc.shape; q = torch.stack([cc & 15, cc >> 4], -1).view(R_, C2 * 2).float()
            ref = LD.unrot((q - (e["L"] - 1) / 2) * e["scales"].float().repeat_interleave(a.g, 1), e["sg"].float()).to(torch.bfloat16)
            ver.append([i, mod, bool(torch.equal(ref, c.tensor(f"model.layers.{i}.{mod}.weight", "cpu")))]); log("verify", ver[-1])
    c.close(); rep["bitexact_sample"] = ver; rep["bitexact_sample_all_equal"] = all(v[2] for v in ver)
    json.dump(rep, open(a.out.rstrip("/") + ".build_report.json", "w"), indent=1); log("BUILD_REPORT", json.dumps(rep)[:2500])


if __name__ == "__main__": main()
