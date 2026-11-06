"""Final real-code-path verification on Modal (C2 DEV only). Overlays predict.py+head.npz on the volume copy of art_c120_dz15.
 (a) plain art_c120_dz15 with its ORIGINAL predict.py (raw, no bias) vs new predict.load(art, None): decisions must agree 100%.
 (b) new load(art, memory) for the default feedback sets: dev acc (retention), new-tool / pair-error items before/after.
 (c) GPU predict.ingest(art, ann) vs the memory file produced from stored features (make_feedback_memory.py).
modal deploy verify_final.py && python verify_final.py"""
import modal, json, os
R = os.path.dirname(os.path.abspath(__file__)) + "/../"
app = modal.App("tenpct-c2fb-final")
img = (modal.Image.debian_slim().pip_install("torch", "transformers==4.51.3", "numpy", "safetensors", "constriction", "accelerate")
       .add_local_file(R + "commercial/C2/dev.jsonl", "/root/c_dev.jsonl")
       .add_local_dir(R + "c2_fb/artifact_code", "/root/code").add_local_dir(R + "c2_fb/feedback_sets", "/root/fb")
       .add_local_file(R + "compress_C/artifact_src/predict.py", "/root/orig_predict.py"))
vol = modal.Volume.from_name("tenpct-c2fb", create_if_missing=True); cres = modal.Volume.from_name("tenpct-C-res")

@app.function(image=img, gpu=["H100", "A100-80GB", "L40S", "A10G"], cpu=8, memory=65536, timeout=14400, volumes={"/d": vol, "/a": cres})
def run():
    import numpy as np, shutil, time, importlib.util, hashlib, torch
    t0 = time.time(); import tempfile; art = tempfile.mkdtemp() + "/art"; shutil.copytree("/a/art_c120_dz15", art); shutil.rmtree(art + "/__pycache__", ignore_errors=True)
    def tree():
        h = hashlib.sha256()
        for r, _, fs in sorted(os.walk(art)):
            for f in sorted(fs): h.update(os.path.relpath(os.path.join(r, f), art).encode()); h.update(hashlib.sha256(open(os.path.join(r, f), "rb").read()).digest())
        return h.hexdigest()
    def imp(n, p): s = importlib.util.spec_from_file_location(n, p); m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
    dev = [json.loads(l) for l in open("/root/c_dev.jsonl")]
    for r in dev: r.pop("tools", None); r.pop("gold_calls", None)
    y = np.array([r["answer"] for r in dev]); out = {}
    O = imp("orig", "/root/orig_predict.py"); fo = O.load(art, None, use_bias=False); pplain = np.array(fo(dev)); out["plain_orig_dev_acc"] = float((pplain == y).mean()); del fo; torch.cuda.empty_cache()
    shutil.copy("/root/code/predict.py", art + "/predict.py"); shutil.copy("/root/code/head.npz", art + "/head.npz"); th = tree(); out["tree_sha256"] = th
    P = imp("newp", "/root/code/predict.py"); f0 = P.load(art, None); p1 = np.array(f0(dev))
    out["stage1_dev_acc"] = float((p1 == y).mean()); out["stage1_agree_with_plain"] = float((p1 == pplain).mean())
    LG, _, FO = f0.build["run"](dev); H = P.Head(art); plan = json.load(open("/root/fb/feedback_plan.json")); tn = P.tool_name
    ie = plan["new_tool_dev_idx"]; ip = plan["pair_dev_err_idx"]; res = {}
    for name in ("newtool_k5", "pair_k1", "combined_k6"):
        mp = f"/root/fb/memory_{name}.json"; M = P.Mem(mp); pm = np.array(P.decide(H, M, dev, LG, FO))
        res[name] = dict(mem_bytes=os.path.getsize(mp), dev_acc=float((pm == y).mean()), dev_acc_stage1=float((p1 == y).mean()), changed_vs_stage1=int((pm != p1).sum()),
                         newtool_items=len(ie), newtool_before=float((p1[ie] == y[ie]).mean()), newtool_after=float((pm[ie] == y[ie]).mean()),
                         pair_err_items=len(ip), pair_before=float((p1[ip] == y[ip]).mean()), pair_after=float((pm[ip] == y[ip]).mean()))
    out["real_load_memory_combined_dev_acc"] = float((np.array(P.load(art, "/root/fb/memory_combined_k6.json".replace("combined_k6", "combined_k6"))(dev)) == y).mean()) if False else None
    out["feedback"] = res
    # real path: load(art, memory) with the combined memory
    del f0; torch.cuda.empty_cache(); fm = P.load(art, "/root/fb/memory_combined_k6.json"); out["real_load_combined_dev_acc"] = float((np.array(fm(dev)) == y).mean())
    # GPU ingest vs stored-feature memory
    ann = json.load(open("/root/fb/ann_newtool_k5.json")); nb = P.ingest(art, ann, None, "/tmp/mem_gpu_%d.json" % os.getpid(), B=fm.build)
    a, b = P.Mem("/tmp/mem_gpu_%d.json" % os.getpid()), P.Mem("/root/fb/memory_newtool_k5.json")
    out["gpu_ingest_bytes"] = nb; out["gpu_ingest_maxdiff_mu"] = float(max(np.abs(P._dec(a.proto[t]["mu"]) - P._dec(b.proto[t]["mu"])).max() for t in b.proto)); out["gpu_ingest_tp_equal"] = a.tp == b.tp
    out["tree_unchanged"] = tree() == th; out["pycache_present"] = os.path.exists(art + "/__pycache__"); out["secs"] = time.time() - t0
    os.makedirs("/d/final", exist_ok=True); json.dump(out, open("/d/final/result.json", "w"), indent=1); vol.commit(); print("RESULT_JSON " + json.dumps(out)); return out

if __name__ == "__main__":
    print(modal.Function.from_name("tenpct-c2fb-final", "run").spawn().object_id)
