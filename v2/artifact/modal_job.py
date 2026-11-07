"""Modal driver: quantise -> write TPQ1 artifact -> verify (bit-exact, size) -> real-vs-sim greedy equality -> harness/v3 final_proof on a DEV-ONLY mini exam.
  modal run --detach v2/artifact/modal_job.py::main --stages probe,quant,verify,devgen,proof     (1 GPU; results on volume tenpct-artifact:/<tag>/ and printed as RESULT_JSON)
NEVER touches locked/S2: only probe_items / s1_dev / s3_dev / s1_calib and LOCK.json (hash guard) are shipped in the image."""
import modal, os, json
V2 = os.path.expanduser("~/ten-percent/v2"); T1 = V2 + "/tournament/T1"
app = modal.App("tenpct-artifact")
img = (modal.Image.debian_slim(python_version="3.11").pip_install("torch==2.5.1", "transformers==4.56.2", "accelerate", "numpy", "safetensors", "constriction", "huggingface_hub", "pytest")
       .add_local_dir(V2 + "/artifact", "/app/v2/artifact", ignore=["__pycache__", "*.pyc", ".pytest_cache"])
       .add_local_dir(V2 + "/harness/v3", "/app/v2/harness/v3", ignore=["__pycache__", "*.pyc", ".pytest_cache", "standin", "tests"])
       .add_local_file(V2 + "/harness/scorecard.py", "/app/v2/harness/scorecard.py")
       .add_local_file(T1 + "/scorer.py", "/app/v2/tournament/T1/scorer.py").add_local_file(T1 + "/probe_items.jsonl", "/app/v2/tournament/T1/probe_items.jsonl")
       .add_local_file(T1 + "/data/LOCK.json", "/app/v2/tournament/T1/data/LOCK.json")
       .add_local_file(T1 + "/data/s1_calib.jsonl", "/app/v2/tournament/T1/data/s1_calib.jsonl").add_local_file(T1 + "/data/s1_dev.jsonl", "/app/v2/tournament/T1/data/s1_dev.jsonl")
       .add_local_file(T1 + "/data/s3_dev.jsonl", "/app/v2/tournament/T1/data/s3_dev.jsonl"))
hfv = modal.Volume.from_name("tenpct-v2-hf"); outv = modal.Volume.from_name("tenpct-artifact", create_if_missing=True)
GPU = os.environ.get("ART_GPU", "A100-80GB,H100,L40S,A100-40GB").split(",")   # fallback list: first available


@app.function(image=img, gpu=GPU, cpu=16, memory=65536, timeout=36000, volumes={"/vol": hfv, "/out": outv})
def pipeline(tag="q3_8b", model="Qwen/Qwen3-8B", stages="probe,quant,verify,devgen,proof", levels="", gptq=True, ncal=128, g=256, mbits=8, ndev=100, devnew=256):
    import glob, subprocess, sys, time, shutil, math, hashlib
    os.environ.update(HF_HOME="/vol/hf", HF_HUB_OFFLINE="0"); sys.path[:0] = ["/app/v2/artifact", "/app/v2/harness/v3"]
    import torch, loader as LD, quantize as QZ, writer as WR, checks
    W = f"/work/{tag}"; O = f"/out/{tag}"; os.makedirs(W, exist_ok=True); os.makedirs(O, exist_ok=True); R = {}
    T0 = time.time(); logf = open(f"{O}/log.txt", "a")
    def log(*a):
        s = time.strftime("%H:%M:%S ") + " ".join(map(str, a)); print(s, flush=True); logf.write(s + "\n"); logf.flush()
    def commit(): outv.commit()
    def save(k, v):
        R[k] = v; json.dump(R, open(f"{O}/result.json", "w"), indent=1, default=str); commit(); log("RESULT_JSON " + json.dumps({k: v}, default=str)[:4000])
    snap = sorted(glob.glob(f"/vol/hf/hub/models--{model.replace('/', '--')}/snapshots/*") + glob.glob(f"/vol/models--{model.replace('/', '--')}/snapshots/*"))[-1]; log("snapshot", snap, "GPU", torch.cuda.get_device_name(0))
    art, sim = f"{W}/art", f"{W}/sim"; ST = stages.split(",")
    arch = checks.arch_of_dir(snap); fp16 = sum(2 * math.prod(s) for _, s in arch.values())
    json.dump(dict(base=model, arch=arch, base_fp16_bytes=fp16, config=json.load(open(snap + "/config.json"))), open(f"{W}/art.base_arch.json", "w"))
    cand = [QZ.parse_levels(x) for x in ([levels] if levels else ["attn=3,o=3,mlp=3,down=3,emb=4,head=4", "attn=3,o=3,mlp=3,down=3,emb=3,head=3", "attn=2,o=3,mlp=3,down=2,emb=3,head=3"])]
    calib = "/app/v2/tournament/T1/data/s1_calib.jsonl"
    if "probe" in ST and not levels:
        for i, lv in enumerate(cand):                      # RTN size probes (no calibration, no sim): pick the first config under 9.6% of fp16
            t = time.time(); p = f"{W}/probe{i}"; shutil.rmtree(p, ignore_errors=True)
            rep = QZ.quantize_model(snap, p, lv, g=g, mbits=mbits, gptq=False, log=lambda *a: None); r = (rep["blob"] + 12_000_000) / fp16
            log("probe", i, lv, "blob_ratio~", round(r, 4), f"{time.time() - t:.0f}s"); save(f"probe{i}", dict(levels=lv, ratio_est=r)); shutil.rmtree(p, ignore_errors=True)
            if r <= 0.096: cand = [lv]; break
        else: cand = [cand[-1]]
    lv = cand[0] if len(cand) == 1 else cand[0]
    if "quant" in ST:
        shutil.rmtree(art, ignore_errors=True); shutil.rmtree(sim, ignore_errors=True); t = time.time()
        rep = QZ.quantize_model(snap, art, lv, g=g, mbits=mbits, gptq=gptq, calib=calib, ncal=ncal, sim_dir=sim, log=log); WR.assemble(art)
        save("quant", dict(levels=lv, gptq=gptq, seconds=time.time() - t, **rep)); shutil.copytree(art, f"{O}/art", dirs_exist_ok=True)
        for f in ("art.quant_report.json", "art.sim_hashes.json"): shutil.copy(f"{W}/{f}", f"{O}/{f}")
        commit()
    if "verify" in ST:
        r = subprocess.run([sys.executable, "/app/v2/artifact/verify.py", "--art", art, "--sim-dir", sim, "--sim-hashes", f"{W}/art.sim_hashes.json", "--base-arch", f"{W}/art.base_arch.json", "--out", f"{O}/verify.json", "--workers", "16"], capture_output=True, text=True)
        log(r.stdout[-3000:], r.stderr[-1500:]); save("verify", json.load(open(f"{O}/verify.json")))
    if "devgen" in ST:
        probe = [json.loads(l) for l in open("/app/v2/tournament/T1/probe_items.jsonl")]; per = {}; items = []
        for r_ in probe:
            k = (r_["source"], r_["kind"])
            if per.get(k, 0) < math.ceil(ndev / 6): items.append(r_); per[k] = per.get(k, 0) + 1
        items = items[:ndev]; view = [{k: r_[k] for k in ("id", "query", "tools")} for r_ in items]; json.dump(view, open(f"{W}/dev_view.json", "w"))
        t = time.time(); p = subprocess.run([sys.executable, f"{art}/answer.py", "--artifact", art, "--in", f"{W}/dev_view.json", "--out", f"{W}/dev_real.json", "--max-new-tokens", str(devnew), "--bs", "32", "--timings", f"{W}/dev_timings.json"], capture_output=True, text=True)
        log(p.stderr[-1500:]); assert p.returncode == 0; real_s = time.time() - t; real = json.load(open(f"{W}/dev_real.json"))
        from safetensors.torch import load_file; from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
        sd = {}
        for f in sorted(glob.glob(f"{sim}/sim-*.safetensors")): sd.update(load_file(f, device="cuda"))
        for f in glob.glob(snap + "/*.safetensors"):
            for k, v in load_file(f, device="cuda").items(): sd.setdefault(k, v)
        conf = AutoConfig.from_pretrained(art); m = AutoModelForCausalLM.from_config(conf, dtype=torch.bfloat16).cuda(); m.load_state_dict(sd, strict=False); m.eval(); del sd
        tok = AutoTokenizer.from_pretrained(art); tok.padding_side = "left"
        try: from transformers import GenerationConfig; m.generation_config = GenerationConfig.from_pretrained(art)
        except Exception: pass
        t = time.time(); simtxt, tm = LD.generate_with(m, tok, view, devnew, 32, dev="cuda"); sim_s = time.time() - t
        same = [a == b["text"] for a, b in zip(simtxt, real)]
        # harness scorer accuracy of both (dev items)
        sys.path.insert(0, "/app/v2/tournament/T1"); import scorer as S
        acc = lambda txts: sum(bool(S.score_item(t_, r_)["correct"]) for t_, r_ in zip(txts, items)) / len(items)
        save("devgen", dict(n=len(view), identical=sum(same), identical_all=all(same), real_wall_s=real_s, sim_wall_s=sim_s, real_timings=json.load(open(f"{W}/dev_timings.json")), acc_real=acc([x["text"] for x in real]), acc_sim=acc(simtxt)))
        json.dump(dict(real=real, sim=simtxt), open(f"{O}/devgen_texts.json", "w")); commit(); del m; torch.cuda.empty_cache()
    if "proof" in ST:
        D = f"{W}/data"; env = dict(os.environ, PYTHONPATH="/app/v2/harness/v3")
        if not os.path.exists(D + "/exam.jsonl"):
            p = subprocess.run([sys.executable, "/app/v2/harness/v3/make_dev_smoke.py", "--base", model, "--out", D, "--per-cell", "20", "--pool", "500", "--n-ann", "12", "--n-teach", "30", "--n-nb", "30"], capture_output=True, text=True, env=env, cwd="/app/v2/harness/v3")
            log(p.stdout[-1500:], p.stderr[-1500:]); assert p.returncode == 0
        t = time.time(); run = f"{W}/run"; shutil.rmtree(run, ignore_errors=True); os.makedirs(run)
        p = subprocess.run([sys.executable, "/app/v2/harness/v3/final_proof.py", "--exam", f"{D}/exam.jsonl", "--artifact", art, "--base-id", model, "--base-arch", f"{W}/art.base_arch.json", "--annotations", f"{D}/ann.jsonl",
                            "--teach", f"{D}/teach.jsonl", "--neighbor", f"{D}/nb.jsonl", "--tag", "dev8b", "--out-dir", run, "--source-map", "S3=S2", "--isolation", "auto"], capture_output=True, text=True, env=env, cwd="/app/v2/harness/v3")
        log("FINAL_PROOF rc", p.returncode); log(p.stdout[-6000:]); log(p.stderr[-3000:]); shutil.copytree(run, f"{O}/run", dirs_exist_ok=True); shutil.copytree(D, f"{O}/data", dirs_exist_ok=True)
        summ = {}
        for f in glob.glob(f"{run}/*.PROOF_SUMMARY.json"): summ = json.load(open(f))
        save("proof", dict(rc=p.returncode, seconds=time.time() - t, stdout_tail=p.stdout[-3000:], stderr_tail=p.stderr[-2000:], summary=summ))
    log("ALL_DONE", f"{time.time() - T0:.0f}s"); commit(); return R


@app.local_entrypoint()
def main(stages: str = "probe,quant,verify,devgen,proof", tag: str = "q3_8b", model: str = "Qwen/Qwen3-8B", levels: str = "", gptq: bool = True):
    print(pipeline.spawn(tag=tag, model=model, stages=stages, levels=levels, gptq=gptq).object_id)
