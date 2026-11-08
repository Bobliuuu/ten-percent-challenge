"""memit2: careful second attempt at a NON-HEAD closed-form feedback edit (MEMIT-style) on existing internal weights (MLP down_proj, several consecutive layers).
No lm_head / embedding change, 0 added bytes, no weight gradient training.
  keys   k = input of down_proj (act(gate)*up, d_ff dim) at the last prompt token and at each teacher-forced label position (fp32 stats)
  C      = E[k k^T] over >=100k tokens of unlabeled task-format prompts (calib_text; optional wikitext) at EACH edited layer, fp32, per-token mean
  target v* (tgt="adam"): ROME/MEMIT free delta vector on the MLP output of the LAST edited layer, 30 Adam steps on the delta ONLY (an activation, not a weight);
         (tgt="closed"): delta = alpha * gamma*(W_U[label]-mean W_U[top-5 base competitors])/rms, alpha line-searched with FORWARD passes only
  update layer l (ascending): R = (v* - current MLP out at last layer)/(L_last-l+1); keys recomputed with the already-updated model; dW = R K^T (lam*C + K K^T)^-1
Eval: teacher-forced exact match of ALL label tokens (+EOS), neighbours = base-correct neighbour rows, controls empty/wrong/mixed_wrong/shuffle (same draws as fbcmp harness).
Decisive checks: weights saved to disk and reloaded, REAL greedy generate, 4-bit RTN (g128) re-quantisation of edited matrices."""
import modal, json, os
app = modal.App("tenpct-memit2")
HERE = os.path.dirname(os.path.abspath(__file__))
img = (modal.Image.debian_slim().pip_install("torch", "transformers==4.51.3", "numpy", "accelerate", "huggingface_hub", "safetensors")
       .add_local_file(HERE + "/core.py", "/root/core.py").add_local_dir(HERE + "/../data_tox", "/root/data_tox").add_local_dir(HERE + "/../data_route", "/root/data_route"))
vol = modal.Volume.from_name("tenpct-memit2", create_if_missing=True); hf = modal.Volume.from_name("tenpct-v2-hf")


def _core():
    """hot-load core.py from the volume (/r/code/core.py) if present, else the image copy"""
    p = "/r/code/core.py" if os.path.exists("/r/code/core.py") else "/root/core.py"
    ns = {"__name__": "core"}; exec(compile(open(p).read(), p, "exec"), ns); return ns


@app.function(image=img, gpu="A100-80GB", cpu=8, memory=65536, timeout=43200, volumes={"/r": vol, "/hf": hf}, env={"HF_HOME": "/hf"})
def run_server(idle_min: int = 45):
    """one A100 job that stays alive and executes every cfg json dropped into /r/inbox (code hot-reloaded from /r/code/core.py) -> no re-queue between experiments"""
    import glob, time, traceback
    last = time.time()
    while time.time() - last < idle_min * 60:
        vol.reload(); fs = sorted(glob.glob("/r/inbox/*.json"))
        if not fs: time.sleep(10); continue
        f = fs[0]
        try: cfgs = json.load(open(f))
        except Exception: time.sleep(3); continue
        os.rename(f, f.replace("/inbox/", "/running_")); print("SERVER running", f, flush=True)
        for c in cfgs:
            try: _core()["_run"](c)
            except Exception as e: traceback.print_exc(); print("CFG FAILED", c.get("name"), repr(e), flush=True)
        vol.commit(); last = time.time()
    print("SERVER idle exit", flush=True)


@app.function(image=img, gpu="A100-80GB", cpu=8, memory=65536, timeout=43200, volumes={"/r": vol, "/hf": hf}, env={"HF_HOME": "/hf"})
def run_gpu(cfgs):
    for c in cfgs:
        try: _core()["_run"](c)
        except Exception as e:
            import traceback; traceback.print_exc(); print("CFG FAILED", c["name"], repr(e), flush=True)


@app.function(image=img, gpu="A10G", cpu=8, memory=32768, timeout=43200, volumes={"/r": vol, "/hf": hf}, env={"HF_HOME": "/hf"})
def run_a10g(cfgs):
    for c in cfgs:
        try: _core()["_run"](c)
        except Exception as e:
            import traceback; traceback.print_exc(); print("CFG FAILED", c["name"], repr(e), flush=True)


@app.function(image=img, cpu=8, memory=32768, timeout=14400, volumes={"/r": vol, "/hf": hf}, env={"HF_HOME": "/hf"})
def run_cpu(cfgs):
    for c in cfgs:
        c["device"] = "cpu"
        try: _core()["_run"](c)
        except Exception as e:
            import traceback; traceback.print_exc(); print("CFG FAILED", c["name"], repr(e), flush=True)


@app.local_entrypoint()
def main(cfgs: str = "", where: str = "gpu"):
    """modal run --detach memit2.py --cfgs cfgs_x.json [--where cpu]"""
    if where == "server": print("SERVER", run_server.remote(), flush=True); return
    f = {"gpu": run_gpu, "a10g": run_a10g, "cpu": run_cpu}[where]
    print("RUN", f.remote(json.load(open(cfgs))), flush=True)
