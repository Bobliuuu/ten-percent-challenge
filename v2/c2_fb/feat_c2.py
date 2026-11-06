"""Frozen-feature extraction from the DECODED C2 7B artifact (predict._build), C2 calib/dev/pool only (NEVER locked).
modal deploy feat_c2.py && python feat_c2.py   (spawns).  Output: Volume tenpct-c2fb:/f/<split>.npz (L [N,14], FQ fp16 [N,H], FO fp16 [N,5,H], y)"""
import modal, json, os
R = os.path.dirname(os.path.abspath(__file__)) + "/../"
app = modal.App("tenpct-c2fb-feat")
img = (modal.Image.debian_slim().pip_install("torch", "transformers==4.51.3", "numpy", "safetensors", "constriction", "accelerate")
       .add_local_file(R + "commercial/C2/dev.jsonl", "/root/c_dev.jsonl").add_local_file(R + "commercial/C2/pool.jsonl", "/root/c_pool.jsonl")
       .add_local_file(R + "commercial/C2/calib.jsonl", "/root/c_calib.jsonl").add_local_dir(R + "c2_fb/artifact_code", "/root/code"))
vol = modal.Volume.from_name("tenpct-c2fb", create_if_missing=True)
cres = modal.Volume.from_name("tenpct-C-res")

@app.function(image=img, gpu=["H100", "A100-80GB", "L40S", "A10G"], cpu=8, memory=65536, timeout=14400, volumes={"/d": vol, "/a": cres})
def run(split):
    import numpy as np, time, importlib.util
    spec = importlib.util.spec_from_file_location("predict", "/root/code/predict.py"); P = importlib.util.module_from_spec(spec); spec.loader.exec_module(P)
    t0 = time.time(); B = P._build("/a/art_c120_dz15"); print("built", time.time() - t0, flush=True)
    rows = [json.loads(l) for l in open(f"/root/c_{split}.jsonl")]; N = len(rows); BS = 32
    L, Q, O = [], [], []
    for b in range(0, N, BS):
        l, q, o = B["run"](rows[b:b + BS]); L.append(l); Q.append(q); O.append(o)
        if (b // BS) % 10 == 0: print(split, b, N, time.time() - t0, flush=True)
    L = np.concatenate(L); y = np.array([r["answer"] for r in rows])
    acc = float((L[:, :5].argmax(1) == y).mean())
    os.makedirs("/d/f", exist_ok=True)
    np.savez(f"/d/f/{split}.npz", L=L, FQ=np.concatenate(Q), FO=np.concatenate(O), y=y); vol.commit()
    out = dict(split=split, n=N, acc_raw=acc, secs=time.time() - t0); print("RESULT_JSON " + json.dumps(out), flush=True); return out

if __name__ == "__main__":
    f = modal.Function.from_name("tenpct-c2fb-feat", "run")
    for s in ("dev", "calib", "pool"): print(s, f.spawn(s).object_id)
