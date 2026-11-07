"""Modal CPU job: Llama-3.1-405B L3 quantised layers (volume tenpct-l3-405b:/out/t096/q) -> TPQ1 artifact (volume tenpct-artifact:/l3_405b/art).
Deploy: modal deploy modal_405b.py ; spawn: python -c "import modal; modal.Function.from_name('tenpct-artifact-405b','convert').spawn(run='t096')" """
import modal, os, json
V2 = os.path.expanduser("~/ten-percent/v2")
app = modal.App("tenpct-artifact-405b")
img = (modal.Image.debian_slim(python_version="3.11").pip_install("torch==2.5.1", "transformers==4.56.2", "numpy", "safetensors", "constriction")
       .add_local_dir(V2 + "/artifact", "/app/v2/artifact", ignore=["__pycache__", "*.pyc", ".pytest_cache", "results", "tests"]))
l3 = modal.Volume.from_name("tenpct-l3-405b"); outv = modal.Volume.from_name("tenpct-artifact", create_if_missing=True)


@app.function(image=img, cpu=32, memory=196608, timeout=86400, volumes={"/w": l3, "/out": outv}, ephemeral_disk=800 * 1024)
def convert(run="t096", emb_codes="", head_codes="", layers=126, mbits=10, verify_layers="0,62,125"):
    import sys, time, glob, shutil, math
    import torch
    sys.path[:0] = ["/app/v2/artifact"]
    import loader as LD, convert_l3 as CV, writer as WR
    from safetensors import safe_open
    T0 = time.time(); art = "/scratch/art"; os.makedirs(art, exist_ok=True); O = "/out/l3_405b"; os.makedirs(O, exist_ok=True)
    def log(*a): print(time.strftime("%H:%M:%S"), *a, flush=True)
    MD = "/w/bf16"; Q = f"/w/out/{run}/q"; w = LD.Writer(f"{art}/model.tpq"); cache = {}; rows = []; lin = set()
    for i in range(layers):
        d = torch.load(f"{Q}/L{i}.pt", map_location="cpu"); st = CV.convert_layer(w, i, d, 128, mbits, cache)
        lin |= set(st); rows.append(sum(sum(v.values()) for v in st.values()))
        if i % 5 == 0 or i == layers - 1: log("layer", i, f"blob={os.path.getsize(art + '/model.tpq') / 1e9:.2f}GB {time.time() - T0:.0f}s")
    body = os.path.getsize(art + "/model.tpq")
    files = sorted(glob.glob(MD + "/*.safetensors")); shapes = {}; dtypes = {}; raw = 0
    for f in files:
        with safe_open(f, "pt") as h:
            for k in h.keys(): sl = h.get_slice(k); shapes[k] = list(sl.get_shape()); dtypes[k] = sl.get_dtype()
    for f in files:
        with safe_open(f, "pt") as h:
            for k in h.keys():
                if k in lin or k in ("model.embed_tokens.weight", "lm_head.weight"): continue
                x = h.get_tensor(k); w.add_raw(k, x, "BF16" if x.dtype == torch.bfloat16 else None); raw += x.numel() * 2
    for name, path, seed_tag in (("model.embed_tokens.weight", emb_codes, "emb"), ("lm_head.weight", head_codes, "head")):
        if path:
            e = torch.load(path, map_location="cpu"); cd = e["codes"]; cd = CV.unpack4(cd).numpy() if cd.shape[1] * 2 == shapes[name][1] else cd.numpy()
            sgk = f"s_{seed_tag}"; w.add_signs(sgk, e["sg"].float()); w.add_q(name, cd, e["scales"].half().numpy(), int(e["L"]), 128, mbits, sgk)
    full = bool(emb_codes and head_codes)
    for f in ("config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json"):
        if os.path.exists(f"{MD}/{f}"): shutil.copy(f"{MD}/{f}", f"{art}/{f}")
    json.dump({k: [dtypes[k], shapes[k]] for k in shapes}, open(f"{art}/tensor_manifest.json", "w"), separators=(",", ":"))
    total_blob = w.finish(meta=dict(source="L3 " + run, mbits=mbits)); WR.assemble(art)
    sizes = {f: os.path.getsize(f"{art}/{f}") for f in os.listdir(art)}; base_bytes = sum(2 * math.prod(s) for s in shapes.values())
    rep = dict(full=full, body_bytes_without_emb_head_raw=body, raw_bytes=raw, blob=total_blob, files=sizes, total=sum(sizes.values()), base_fp16_bytes_from_headers=base_bytes, ratio=sum(sizes.values()) / base_bytes,
               ratio_vs_810e9=sum(sizes.values()) / 810e9, ratio_vs_811p7e9=sum(sizes.values()) / 811.7e9, secs=time.time() - T0)
    # sampled bit-exactness vs an independent fp32 reference built straight from the L3 packed layers
    ver = []
    c = LD.Container(f"{art}/model.tpq", workers=16)
    for i in [int(x) for x in verify_layers.split(",") if x]:
        d = torch.load(f"{Q}/L{i}.pt", map_location="cpu")
        for mod, e in d.items():
            R_, C2 = e["codes"].shape; C_ = C2 * 2; cc = e["codes"]; q = torch.stack([cc & 15, cc >> 4], -1).view(R_, C_).float()
            Qm = (q - (e["L"] - 1) / 2) * e["scales"].float().repeat_interleave(128, 1); ref = LD.unrot(Qm, e["sg"].float()).to(torch.bfloat16)
            got = c.tensor(f"model.layers.{i}.{mod}.weight", "cpu"); ver.append((i, mod, bool(torch.equal(ref, got))))
    c.close(); rep["bitexact_sample"] = ver; rep["bitexact_all_equal"] = all(v[2] for v in ver)
    json.dump(rep, open(f"{O}/report_{run}_{'full' if full else 'partial'}.json", "w"), indent=1)
    log("REPORT", json.dumps(rep)[:3000])
    if full:
        shutil.copytree(art, f"{O}/art", dirs_exist_ok=True)
    outv.commit(); return rep
