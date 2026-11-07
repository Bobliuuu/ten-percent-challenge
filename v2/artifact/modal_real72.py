"""Real-exam eval of a REAL packed TPQ1 72B artifact on Modal (H200:2): volume tenpct-artifact:/<ART> -> loader.export_bf16 -> vLLM TP=2 bf16 -> D's real TOOLSEL/INJ dev exams.
Deploy: modal deploy modal_real72.py ; spawn: python launch_real72.py art72_V7  (results -> volume tenpct-artifact:/real72/<ART>/ and returned)"""
import modal, os, json
V2 = os.path.expanduser("~/ten-percent/v2")
app = modal.App("tenpct-artifact-real72")
img = (modal.Image.debian_slim(python_version="3.11").pip_install("vllm==0.11.0", "transformers==4.57.1", "huggingface_hub<1.0", "safetensors", "numpy<2.3", "constriction")
       .add_local_dir(V2 + "/artifact", "/app/v2/artifact", ignore=["__pycache__", "*.pyc", ".pytest_cache", "results", "tests"])
       .add_local_dir(V2 + "/research/D_scale/real", "/app/real")
       .add_local_file(V2 + "/research/D_scale/vllm_real.py", "/app/vllm_real.py"))
outv = modal.Volume.from_name("tenpct-artifact", create_if_missing=True)


@app.function(image=img, gpu="H200:2", cpu=32, memory=262144, timeout=4 * 3600, volumes={"/out": outv}, ephemeral_disk=700 * 1024)
def run(art="art72_V7", tasks="toolsel,inj"):
    import subprocess, time, shutil, hashlib
    T0 = time.time(); O = f"/out/real72/{art}"; os.makedirs(O, exist_ok=True); lg = open(O + "/log.txt", "a")
    def log(*a):
        s = time.strftime("%H:%M:%S ") + f"[{time.time() - T0:.0f}s] " + " ".join(map(str, a)); print(s, flush=True); lg.write(s + "\n"); lg.flush()
    A = f"/out/{art}"; log("art files", os.listdir(A), os.path.getsize(A + "/model.tpq"))
    h = hashlib.sha256(); f = open(A + "/model.tpq", "rb")
    for b in iter(lambda: f.read(1 << 24), b""): h.update(b)
    log("model.tpq sha256", h.hexdigest())
    B = "/scratch/bf16"; os.makedirs("/scratch", exist_ok=True)
    r = subprocess.run(["python", "-c", f"import sys; sys.path.insert(0,'/app/v2/artifact'); import loader; loader.export_bf16('{A}','{B}',dev='cuda')"], capture_output=True, text=True)
    log("export rc", r.returncode, r.stdout[-500:], r.stderr[-1500:])
    if r.returncode: return dict(error="export", err=r.stderr[-2000:])
    log("bf16 bytes", sum(os.path.getsize(f"{B}/{x}") for x in os.listdir(B)))
    shutil.copy("/app/vllm_real.py", "/app/vllm_real_tp2.py"); s = open("/app/vllm_real_tp2.py").read()
    s = s.replace('llm = LLM(md, dtype="bfloat16", max_model_len=8192, gpu_memory_utilization=0.92, enable_prefix_caching=False)', 'llm = LLM(md, dtype="bfloat16", max_model_len=8192, gpu_memory_utilization=0.92, enable_prefix_caching=False, tensor_parallel_size=2)')
    assert "tensor_parallel_size=2" in s; s = s.replace('os.path.dirname(os.path.abspath(__file__)) + "/real"', '"/app/real"'); open("/app/vllm_real_tp2.py", "w").write(s)
    r = subprocess.run(["python", "/app/vllm_real_tp2.py", B, f"{O}/real_{art}", tasks], capture_output=True, text=True)
    log("vllm rc", r.returncode, "\n".join(l for l in r.stdout.splitlines() if "REAL_DONE" in l), r.stderr[-1500:])
    outv.commit(); res = {}
    for t in tasks.split(","):
        p = f"{O}/real_{art}_{t}.json"
        if os.path.exists(p): res[t] = json.load(open(p))
    log("done"); outv.commit(); return res
