"""Modal CPU launcher for the leftover stats (ternary shard 3 + embed/head lossless stats). modal run --detach v2/byte_tricks/modal_stats.py::main"""
import modal, os
HERE = os.path.dirname(os.path.abspath(__file__))
app = modal.App("tenpct-l13-stats"); vol = modal.Volume.from_name("tenpct-l13", create_if_missing=True)
img = (modal.Image.debian_slim(python_version="3.11").pip_install("torch==2.8.0", "transformers==4.55.0", "numpy", "huggingface_hub", "zstandard", "safetensors")
       .add_local_dir(HERE, "/code/byte_tricks", ignore=["results", "__pycache__"]))
@app.function(image=img, timeout=3 * 3600, volumes={"/res": vol}, cpu=4, memory=49152)
def run():
    import subprocess, os, glob, shutil
    os.chdir("/code/byte_tricks"); os.makedirs("results", exist_ok=True)
    cmds = [["python", "stats_weights.py", "prism-ml/Ternary-Bonsai-27B-unpacked", "--shards", "1", "--skip", "2", "--tensors", "12", "--discrete", "--dev", "cpu", "--out", "results/stats_ternary_skip2.json"]]
    for m in ["Qwen/Qwen2.5-14B-Instruct", "Qwen/Qwen2.5-32B-Instruct", "Qwen/Qwen3-30B-A3B", "prism-ml/Bonsai-8B-unpacked", "prism-ml/Ternary-Bonsai-27B-unpacked"]: cmds.append(["python", "stats_embed.py", m])
    for c in cmds:
        r = subprocess.run(c, capture_output=True, text=True); print(" ".join(c), r.returncode, r.stdout[-1500:], r.stderr[-400:], flush=True)
        for f in glob.glob("results/*.json"): print("RESULT_JSON", os.path.basename(f), open(f).read()[:6000].replace("\n", " "), flush=True); os.makedirs("/res/stats", exist_ok=True); shutil.copy(f, "/res/stats/")
        vol.commit()
@app.local_entrypoint()
def main(): run.spawn(); print("spawned")
