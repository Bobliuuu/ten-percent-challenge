"""Frozen-feature extraction from the DECODED BitNet-2B artifact (predict._build/_fwd), MDC-6 calib/dev/pool only (never locked).
Deploy+spawn:  modal deploy feat_fb.py && python feat_fb.py launch      Output: Volume lowbit-data:/fb/<name>.npz  (X fp16 [N,5120], L [N,14], y, task)"""
import modal, json, os, sys
R = os.path.dirname(os.path.abspath(__file__)) + "/../"
app = modal.App("tenpct-lowbit-fb-feat")
img = (modal.Image.debian_slim().pip_install("torch", "transformers==4.57.1", "numpy", "safetensors", "constriction", "accelerate")
       .add_local_file(R + "alt/dev.jsonl", "/root/a_dev.jsonl").add_local_file(R + "alt/pool.jsonl", "/root/a_pool.jsonl")
       .add_local_file(R + "alt/calib.jsonl", "/root/a_calib.jsonl").add_local_dir(R + "lowbit_fb/artifact_code", "/root/code"))
vol = modal.Volume.from_name("lowbit-data", create_if_missing=True)
HOLD = {"ag_news": 3, "dbpedia": 12, "emotion": 2}
OTHER = "Other / none of these"
ART = "/d/art/bitnet2b_v42k_e4"

@app.function(image=img, gpu=["A100-40GB", "A100-80GB", "L40S", "A10G"], cpu=8, memory=65536, timeout=14400, volumes={"/d": vol})
def run(group):
    import numpy as np, torch, time, importlib.util
    spec = importlib.util.spec_from_file_location("predict", "/root/code/predict.py"); P = importlib.util.module_from_spec(spec); spec.loader.exec_module(P)
    t0 = time.time(); B = P._build(ART); print("built", time.time() - t0, flush=True)
    ld = lambda sp: [json.loads(l) for l in open(f"/root/a_{sp}.jsonl")]
    os.makedirs("/d/fb", exist_ok=True)
    def extract(name, rows):
        N = len(rows); L = np.full((N, 14), -1e4, np.float32); X = None
        with torch.no_grad():
            for i, r in enumerate(rows):
                lg, f = P._fwd(B, r)
                if X is None: X = np.zeros((N, len(f)), np.float16)
                L[i, :len(lg)] = lg; X[i] = f
                if i % 500 == 0: print(name, i, N, time.time() - t0, flush=True)
        y = np.array([r["answer"] for r in rows]); k = np.array([len(r["names"]) for r in rows])
        acc = float((np.array([L[i, :k[i]].argmax() for i in range(N)]) == y).mean())
        np.savez(f"/d/fb/{name}.npz", X=X, L=L, y=y, task=np.array([r["task"] for r in rows]), ytrue=np.array([r.get("ytrue", r["answer"]) for r in rows]))
        vol.commit(); out = dict(name=name, n=N, acc=acc, secs=time.time() - t0); print("RESULT_JSON " + json.dumps(out), flush=True); return out
    res = []
    if group in ("calib", "dev", "pool"):
        res.append(extract("mdc_" + group, ld(group)))
    elif group == "hold":
        for sp in ("calib", "dev", "pool"):
            for tn, h in HOLD.items():
                rows = []
                for r in ld(sp):
                    if r["task"] != tn: continue
                    idx = [i for i in range(len(r["names"])) if i != h]
                    rr = dict(r); rr["names"] = [r["names"][i] for i in idx] + [OTHER]
                    rr["ytrue"] = r["answer"]; rr["answer"] = idx.index(r["answer"]) if r["answer"] != h else len(idx)
                    rows.append(rr)
                res.append(extract(f"hold_{tn}_{sp}", rows))
    return res

@app.local_entrypoint()
def main():
    for g in ("calib", "dev", "pool", "hold"): run.spawn(g)
