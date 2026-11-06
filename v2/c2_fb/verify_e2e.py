"""End-to-end on Modal: overlay predict.py + head.npz onto the volume copy of the C2 artifact, then (a) load(art, None) on C2 dev, (b) ingest k annotations -> memory file, (c) load(art, memory) on dev. Never imports predict.py from inside the artifact dir (loaded from /root/code).
modal deploy verify_e2e.py && python verify_e2e.py   -> RESULT_JSON in app logs + Volume tenpct-c2fb:/e2e/result.json"""
import modal, json, os, sys
R = os.path.dirname(os.path.abspath(__file__)) + "/../"
app = modal.App("tenpct-c2fb-e2e")
img = (modal.Image.debian_slim().pip_install("torch", "transformers==4.51.3", "numpy", "safetensors", "constriction", "accelerate")
       .add_local_file(R + "commercial/C2/dev.jsonl", "/root/c_dev.jsonl").add_local_file(R + "commercial/C2/pool.jsonl", "/root/c_pool.jsonl")
       .add_local_dir(R + "c2_fb/artifact_code", "/root/code"))
vol = modal.Volume.from_name("tenpct-c2fb", create_if_missing=True); cres = modal.Volume.from_name("tenpct-C-res")

@app.function(image=img, gpu=["H100", "A100-80GB", "L40S", "A10G"], cpu=8, memory=65536, timeout=14400, volumes={"/d": vol, "/a": cres})
def run():
    import numpy as np, shutil, time, importlib.util, hashlib, collections
    t0 = time.time(); art = "/tmp/art"; shutil.copytree("/a/art_c120_dz15", art)
    shutil.rmtree(art + "/__pycache__", ignore_errors=True)
    shutil.copy("/root/code/predict.py", art + "/predict.py"); shutil.copy("/root/code/head.npz", art + "/head.npz")
    def tree():
        h = hashlib.sha256()
        for r, _, fs in sorted(os.walk(art)):
            for f in sorted(fs): h.update(os.path.relpath(os.path.join(r, f), art).encode()); h.update(hashlib.sha256(open(os.path.join(r, f), "rb").read()).digest())
        return h.hexdigest()
    th = tree()
    spec = importlib.util.spec_from_file_location("predict_mod", "/root/code/predict.py"); P = importlib.util.module_from_spec(spec); spec.loader.exec_module(P)
    ld = lambda s: [json.loads(l) for l in open(f"/root/c_{s}.jsonl")]
    dev, pool = ld("dev"), ld("pool")
    for r in dev + pool: r.pop("tools", None); r.pop("gold_calls", None)
    y = np.array([r["answer"] for r in dev]); out = dict(tree_sha256=th, size=sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(art) for f in fs))
    f0 = P.load(art, None); p0 = np.array(f0(dev)); out["dev_acc_head_on_k0"] = float((p0 == y).mean())
    os.environ["TENPCT_NOHEAD"] = "1"; fb = P.load(art, None, dev="cuda"); pb = np.array(fb(dev)); out["dev_acc_head_off"] = float((pb == y).mean()); del os.environ["TENPCT_NOHEAD"]
    # S2-style annotation: pick a pool tool with >=5 items and a dev tool support; ingest k=5 -> memory
    tn = P.tool_name; cnt = collections.Counter(tn(r["names"][r["answer"]]) for r in pool if r["answer"] != 4)
    devc = collections.Counter(tn(r["names"][r["answer"]]) for r in dev if r["answer"] != 4)
    tools = [t for t, c in cnt.most_common() if c >= 5 and devc[t] >= 5][1:4]; out["tools"] = tools; res = {}
    for t in tools:
        ann = [r for r in pool if r["answer"] != 4 and tn(r["names"][r["answer"]]) == t][:5]
        mem = f"/tmp/mem_{t}.json"; nb = P.ingest(art, ann, None, mem, B=f0.build)
        fm = P.load(art, mem); pm = np.array(fm(dev)); ie = [i for i, r in enumerate(dev) if r["answer"] != 4 and tn(r["names"][r["answer"]]) == t]
        res[t] = dict(mem_bytes=nb, new_before=float((p0[ie] == y[ie]).mean()), new_after=float((pm[ie] == y[ie]).mean()), dev_before=float((p0 == y).mean()), dev_after=float((pm == y).mean()), n_eval=len(ie))
        shutil.copy(mem, f"/d/e2e_mem_{t}.json")
    out["s2_e2e"] = res; out["tree_after"] = tree(); out["tree_unchanged"] = out["tree_after"] == th; out["secs"] = time.time() - t0
    os.makedirs("/d/e2e", exist_ok=True); json.dump(out, open("/d/e2e/result.json", "w")); vol.commit(); print("RESULT_JSON " + json.dumps(out)); return out

if __name__ == "__main__":
    print(modal.Function.from_name("tenpct-c2fb-e2e", "run").spawn().object_id)
