"""Box pipeline (no Modal): 8B quantise (levels all 3, GPTQ) -> assemble -> verify -> harness final_proof with REAL netns isolation -> vLLM parity.  python run_g18.py STAGES  (quant,verify,proof,vllm)"""
import json, math, os, subprocess, sys, shutil, time, glob
W = os.path.expanduser("~/work/tp"); V2 = W + "/v2"; PY = sys.executable; OUT = W + "/out8b"; os.makedirs(OUT, exist_ok=True)
sys.path[:0] = [V2 + "/artifact", V2 + "/harness/v3"]
os.environ["HF_HOME"] = "/opt/dlami/nvme/hf"
stages = sys.argv[1].split(","); art = OUT + "/art"; sim = OUT + "/sim"; MODEL = "Qwen/Qwen3-8B"
def log(*a): print(time.strftime("%H:%M:%S"), *a, flush=True)
from huggingface_hub import snapshot_download
snap = snapshot_download(MODEL, allow_patterns=["*.safetensors", "*.json", "merges.txt", "vocab.json"]); log("snap", snap)
import checks
arch = checks.arch_of_dir(snap); fp16 = sum(2 * math.prod(s) for _, s in arch.values())
json.dump(dict(base=MODEL, arch=arch, base_fp16_bytes=fp16, config=json.load(open(snap + "/config.json"))), open(OUT + "/art.base_arch.json", "w"))
if "quant" in stages:
    import quantize as QZ, writer as WR
    shutil.rmtree(art, ignore_errors=True); shutil.rmtree(sim, ignore_errors=True)
    rep = QZ.quantize_model(snap, art, dict(attn=3, o=3, mlp=3, down=3, emb=3, head=3), g=256, mbits=8, gptq=True, calib=V2 + "/tournament/T1/data/s1_calib.jsonl", ncal=128, sim_dir=sim, log=log)
    WR.assemble(art); json.dump(rep, open(OUT + "/quant.json", "w")); shutil.copy(OUT + "/art.quant_report.json", OUT + "/quant_report.json") if os.path.exists(OUT + "/art.quant_report.json") else None
if "verify" in stages:
    r = subprocess.run([PY, V2 + "/artifact/verify.py", "--art", art, "--sim-dir", sim, "--sim-hashes", OUT + "/art.sim_hashes.json", "--base-arch", OUT + "/art.base_arch.json", "--out", OUT + "/verify.json"], capture_output=True, text=True); log(r.stderr[-800:]); log("verify done")
if "proof" in stages:
    D = OUT + "/data"; env = dict(os.environ, PYTHONPATH=V2 + "/harness/v3")
    if not os.path.exists(D + "/exam.jsonl"):
        p = subprocess.run([PY, V2 + "/harness/v3/make_dev_smoke.py", "--base", MODEL, "--out", D, "--per-cell", "20", "--pool", "500", "--n-ann", "12", "--n-teach", "30", "--n-nb", "30"], capture_output=True, text=True, env=env, cwd=V2 + "/harness/v3"); log(p.stdout[-600:], p.stderr[-600:]); assert p.returncode == 0
    run = OUT + "/run"; shutil.rmtree(run, ignore_errors=True); os.makedirs(run)
    p = subprocess.run([PY, V2 + "/harness/v3/final_proof.py", "--exam", D + "/exam.jsonl", "--artifact", art, "--base-id", MODEL, "--base-arch", OUT + "/art.base_arch.json", "--annotations", D + "/ann.jsonl", "--teach", D + "/teach.jsonl",
                        "--neighbor", D + "/nb.jsonl", "--tag", "dev8b_netns", "--out-dir", run, "--source-map", "S3=S2", "--isolation", "netns", "--python", PY], capture_output=True, text=True, env=env, cwd=V2 + "/harness/v3")
    log("FINAL_PROOF rc", p.returncode); log(p.stdout[-5000:]); log(p.stderr[-2500:])
if "vllm" in stages:
    import loader as LD
    ex = OUT + "/bf16"; shutil.rmtree(ex, ignore_errors=True); t = time.time(); LD.export_bf16(art, ex); log("export_bf16 s", time.time() - t, "bytes", sum(os.path.getsize(f) for f in glob.glob(ex + "/*.safetensors")))
