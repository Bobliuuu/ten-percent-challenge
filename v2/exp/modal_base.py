"""X01 72B base references on Modal (H100:2, TP2). ALWAYS: modal run --detach v2/exp/modal_base.py::main --model Qwen/Qwen2.5-72B-Instruct --tag 72b
   (FP8 secondary: --model RedHatAI/Qwen2.5-72B-Instruct-FP8-dynamic --tag 72b_fp8).  Results land on volume tenpct-exp:/X01 ; fetch: modal volume get tenpct-exp X01 <dir>"""
import modal, os, subprocess, sys, shutil
app = modal.App("tenpct-x01-base")
R = os.path.dirname(os.path.abspath(__file__)); V2 = os.path.dirname(R)
vol = modal.Volume.from_name("tenpct-exp", create_if_missing=True)
img = (modal.Image.debian_slim(python_version="3.11").pip_install("vllm==0.11.0", "transformers==4.57.1", "huggingface_hub<1.0", "scipy", "datasets<4")
       .add_local_file(R + "/run_retention.py", "/code/v2/exp/run_retention.py").add_local_file(R + "/analyze_pair.py", "/code/v2/exp/analyze_pair.py")
       .add_local_file(R + "/SPLITS.json", "/code/v2/exp/SPLITS.json").add_local_dir(R + "/splits", "/code/v2/exp/splits")
       .add_local_file(V2 + "/entry32b/scorer.py", "/code/v2/entry32b/scorer.py").add_local_file(V2 + "/entry32b/dev_inj.jsonl", "/code/v2/entry32b/dev_inj.jsonl")
       .add_local_file(V2 + "/harness/stats.py", "/code/v2/harness/stats.py"))
@app.function(image=img, gpu="H100:2", timeout=6 * 3600, volumes={"/vol": vol})
def run(model: str, tag: str, sets: str = "SEL,REP,GEN,INJ"):
    env = dict(os.environ, HF_HOME="/vol/hf")
    os.makedirs("/vol/X01", exist_ok=True)
    for p in (1, 2):
        subprocess.check_call([sys.executable, "/code/v2/exp/run_retention.py", "--model", model, "--exp", "X01", "--arm", f"base_{tag}", "--seed", "0", "--sets", sets,
                               "--pass", str(p), "--tp", "2", "--backend", "vllm", "--util", "0.90"], env=env, cwd="/code")
        subprocess.check_call("cp -r /code/v2/exp/results/X01/. /vol/X01/", shell=True); vol.commit()
@app.local_entrypoint()
def main(model: str, tag: str, sets: str = "SEL,REP,GEN,INJ"): run.remote(model, tag, sets)
