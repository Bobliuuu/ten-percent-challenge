"""L10 feedback + exam jobs on the PACKED body (decoded by ship/packlib on the GPU). All detached.
  python3 modal_fb2.py dev   CFG.json NAME      L11 run_exam.py on OUR dev data (own-data proxy + tuning)
  python3 modal_fb2.py lock  CFG.json NAME      L11 run_exam.py on the shared locked-exam splits (teach/neighbours)
  python3 modal_fb2.py exam  -        NAME      locked SKILL7 exam scored once with the packed body (+ one real/shuffled/empty feedback arm each, real greedy generation)
Results -> volume tenpct-tasksearch:/l10v2/NAME/ (res.json | exam.json)"""
import modal, json, sys, os
HERE = os.path.dirname(os.path.abspath(__file__)); FA = HERE + "/../../fb_ablation"
app = modal.App("tenpct-l10fb2")
vol = modal.Volume.from_name("tenpct-tasksearch", create_if_missing=True); priv = modal.Volume.from_name("tenpct-l10-priv", create_if_missing=True)
img = (modal.Image.debian_slim(python_version="3.11").pip_install("torch==2.4.1", "transformers==4.55.0", "numpy", "accelerate", "safetensors", "constriction")
       .add_local_file(HERE + "/ship/packlib.py", "/app/packlib.py").add_local_file(HERE + "/ship/common.py", "/app/common.py")
       .add_local_file(FA + "/fblib.py", "/app/fblib.py").add_local_file(FA + "/run_exam.py", "/app/run_exam.py").add_local_file(FA + "/exam_data/SKILL7.json", "/app/exam_data/SKILL7.json")
       .add_local_dir(HERE + "/data", "/app/data").add_local_file(HERE + "/fbrun/build_dev_dd.py", "/app/build_dev_dd.py").add_local_file(HERE + "/results/exam_skill7_omni7.json", "/app/prev_exam.json"))
def _go(job):
    import subprocess, os, sys, json, shutil, time
    sys.path.insert(0, "/app"); import packlib
    os.chdir("/app"); out = f"/vol/l10v2/{job['name']}"; os.makedirs(out, exist_ok=True)
    pack = job.get("pack", "/priv/omni7_body.pack"); body = "/tmp/body"; t = time.time(); meta = packlib.write_hf_dir(pack, body); print("decoded pack", round(time.time() - t), "s", flush=True)
    if job["mode"] in ("dev", "lock"):
        task = "SKILL7"
        if job["mode"] == "dev": subprocess.run(["python", "build_dev_dd.py", "/app/data", "/app/exam_data/L10DEV.json"], check=True); task = "L10DEV"
        cfg = dict(job["cfg"], model=body, task=task); json.dump(cfg, open("/tmp/cfg.json", "w"))
        r = subprocess.run(["python", "run_exam.py", "/tmp/cfg.json", out]); print("RUN_EXAM rc", r.returncode, flush=True)
    else:
        exam_job(job, body, out)
    vol.commit(); print("FB2_DONE", job["name"], flush=True)
def exam_job(job, body, out):
    import sys, json, random, hashlib, torch, numpy as np, tempfile
    import common, fblib
    from fblib import Env
    from transformers import AutoTokenizer
    rd = lambda p: [json.loads(l) for l in open(p) if l.strip()]
    ex = rd("/app/data/exam_skill7.jsonl"); teach = rd("/app/data/s7_teach.jsonl"); ann = rd("/app/data/s7_ann_pool.jsonl"); nbid = set(json.load(open("/app/data/s7_neighbours.json"))); nb = [r for r in ex if r["id"] in nbid]
    prev = json.load(open("/app/prev_exam.json")); out_d = {"n_exam": len(ex), "base": prev["base"], "note": "base = bf16 text path preds from the earlier single exam run; quant = PACKED body decoded by packlib"}
    DD = json.load(open("/app/exam_data/SKILL7.json")); cal = DD["cal"]; random.Random(0).shuffle(cal); NEW = "Recipe"
    tok = AutoTokenizer.from_pretrained(body); tok.pad_token = tok.pad_token or tok.eos_token
    m = common.load(body, patch=False)
    gen = lambda rows: dict(zip([r["id"] for r in rows], common.generate(m, tok, [r["user"] for r in rows], bs=32)))
    out_d["quant"] = gen(ex); out_d["quant_teach"] = out_d["empty_teach"] = gen(teach); del m; torch.cuda.empty_cache()
    json.dump(out_d, open(out + "/exam.json", "w")); print("packed-body exam generated", flush=True)
    td = tempfile.mkdtemp(); open(f"{td}/calib_old.jsonl", "w").write("".join(json.dumps(dict(text=r["user"], gold=r["kind"])) + "\n" for r in cal[:300]))
    env = Env(body, td, raw=True, ncal=300, log=print); mk = lambda rows: [dict(text=r["user"], pid=env.pid(r["user"]), id=r["id"]) for r in rows]
    P_ex, P_te, P_nb = mk(ex), mk(teach), mk(nb); A = mk(ann); SH = mk(cal[300:700])
    g = lambda P: dict(zip([p["id"] for p in P], env.gen_text([p["pid"] for p in P], 8)))
    ANN_BASE = env.gen_text([a["pid"] for a in A], 6); rr = random.Random(100 * 25 + 0); mt = job["method"]
    ix = [i for i in range(len(A)) if ANN_BASE[i] != NEW]; ix = rr.sample(ix, 25); sh = rr.sample(range(len(SH)), 25)
    for arm, items in (("real", [(A[i], NEW) for i in ix]), ("shuffled", [(SH[i], NEW) for i in sh])):
        env.restore(); info = env.apply(mt, items); out_d[arm + "_teach"] = g(P_te); out_d[arm + "_neigh"] = g(P_nb)
        if arm == "real": out_d["real_exam"] = g(P_ex); out_d["ann_ids"] = [A[i]["id"] for i in ix]; out_d["ann_pred"] = [ANN_BASE[i] for i in ix]; out_d["real_bytes"] = env.delta_stats(); out_d["cfg"] = mt
        env.restore()
    json.dump(out_d, open(out + "/exam.json", "w")); print("EXAM_ARMS_DONE", flush=True)
fn = app.function(image=img, timeout=10 * 3600, volumes={"/vol": vol, "/priv": priv}, gpu="A100-80GB", memory=96000, name="go_fb2")(_go)
if __name__ == "__main__":
    mode, cf, name = sys.argv[1:4]; job = dict(mode=mode, name=name)
    if cf != "-": job["cfg"] = json.load(open(cf))
    else: job["method"] = dict(tag="o_prot_L8_m32", carrier="o", solver="prot", m=32, layers=list(range(20, 28)))
    with modal.enable_output(), app.run(detach=True): print("CALL", name, fn.spawn(job).object_id, flush=True)
