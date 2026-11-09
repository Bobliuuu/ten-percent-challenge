"""L10 feedback + exam jobs on the PACKED body (decoded by ship/packlib on the GPU). All detached.
  python3 modal_fb2.py dev   CFG.json NAME      L11 run_exam.py on OUR dev data (own-data proxy + tuning)
  python3 modal_fb2.py lock  CFG.json NAME      L11 run_exam.py on the shared locked-exam splits (teach/neighbours)
  python3 modal_fb2.py exam  -        NAME      locked SKILL7 exam scored once with the packed body (+ one real/shuffled/empty feedback arm each, real greedy generation)
Results -> volume tenpct-tasksearch:/l10v2/NAME/ (res.json | exam.json)"""
import modal, json, sys, os
HERE = os.path.dirname(os.path.abspath(__file__)); FA = HERE + "/../../fb_ablation"
app = modal.App("tenpct-l10fb2")
vol = modal.Volume.from_name("tenpct-tasksearch", create_if_missing=True); priv = modal.Volume.from_name("tenpct-l10-priv", create_if_missing=True); hfvol = modal.Volume.from_name("tenpct-v2-hf", create_if_missing=True)
img = (modal.Image.debian_slim(python_version="3.11").pip_install("torch==2.4.1", "transformers==4.55.0", "numpy", "accelerate", "safetensors", "constriction")
       .add_local_file(HERE + "/ship/packlib.py", "/app/packlib.py").add_local_file(HERE + "/ship/common.py", "/app/common.py")
       .add_local_file(FA + "/fblib.py", "/app/fblib.py").add_local_file(FA + "/run_exam.py", "/app/run_exam.py").add_local_file(FA + "/exam_data/SKILL7.json", "/app/exam_data/SKILL7.json")
       .add_local_dir(HERE + "/data", "/app/data").add_local_dir(HERE + "/ship", "/app/ship").add_local_file(HERE + "/fbrun/build_dev_dd.py", "/app/build_dev_dd.py").add_local_file(HERE + "/results/exam_skill7_omni7.json", "/app/prev_exam.json"))
def _go(job):
    import subprocess, os, sys, json, shutil, time
    sys.path.insert(0, "/app"); import packlib
    os.chdir("/app"); out = f"/vol/l10v2/{job['name']}"; os.makedirs(out, exist_ok=True)
    pack = job.get("pack", "/priv/omni7_body.pack"); body = "/tmp/body"; t = time.time()
    if job["mode"] not in ("cli", "embdbg"): meta = packlib.write_hf_dir(pack, body); print("decoded pack", round(time.time() - t), "s", flush=True); import gc, torch; gc.collect(); torch.cuda.empty_cache()
    if job["mode"] == "embdbg":
        embdbg(pack, out); vol.commit(); return
    if job["mode"] == "cli":
        cli_job(pack, out); vol.commit(); print("FB2_DONE", job["name"], flush=True); return
    if job["mode"] in ("dev", "lock"):
        task = "SKILL7"
        if job["mode"] == "dev": subprocess.run(["python", "build_dev_dd.py", "/app/data", "/app/exam_data/L10DEV.json"], check=True); task = "L10DEV"
        cfg = dict(job["cfg"], model=body, task=task); json.dump(cfg, open("/tmp/cfg.json", "w"))
        r = subprocess.run(["python", "run_exam.py", "/tmp/cfg.json", out]); print("RUN_EXAM rc", r.returncode, flush=True)
    else:
        exam_job(job, body, out)
    vol.commit(); print("FB2_DONE", job["name"], flush=True)
def embdbg(pack, out):
    import torch, numpy as np, json, math, sys; sys.path.insert(0, "/app"); import packlib
    from safetensors import safe_open
    H = packlib.read_header(pack); e = next(x for x in H["entries"] if x["name"] == "model.embed_tokens.weight"); f = open(pack, "rb")
    R, C = e["shape"]; res = dict(shape=[R, C], g=e["g"], Lv=e["Lv"], seed=e["seed"])
    f.seek(e["sym"][0]); qi = packlib.rc_decode(f.read(e["sym"][1]), e["counts"], R * C).astype(np.uint8); res["decoded_hist"] = np.bincount(qi, minlength=e["Lv"]).tolist(); res["stored_counts"] = e["counts"]
    res["hist_equal"] = res["decoded_hist"] == e["counts"]
    with safe_open("/hf/l10_omni7b/model.safetensors", "pt") as g: W = g.get_tensor("model.embed_tokens.weight").cuda().float()
    Hb = packlib._hb("cuda"); sg = packlib.signs(e["seed"], C, "cuda"); We = ((W * sg).view(R, C // 128, 128) @ Hb).reshape(R, C); half = (e["Lv"] - 1) / 2
    # recompute qi/scales as the packer does
    PG = torch.tensor([1.0, .9, .8, .7, .62, .55, .5, .45, .4, .36, .32, .28, .25, .22, .19, .16, .13]).cuda(); Lv = e["Lv"]
    def quant(x, s): t = x / s; qi_ = torch.clamp(torch.round(t + half), 0, Lv - 1); return (qi_ - half) * s, qi_
    def qparams(x):
        am = x.abs().amax(1, keepdim=True).clamp(min=1e-8); best = None
        for p in PG:
            s = (am * p / max(half, 0.5)).half().float().clamp(min=1e-8); q = quant(x, s)[0]; er = ((q - x) ** 2).sum(1, keepdim=True)
            if best is None: best = (er, s)
            else: m = er < best[0]; best = (torch.where(m, er, best[0]), torch.where(m, s, best[1]))
        return best[1]
    mism = 0; first = None; QI = []
    for r in range(0, R, 8192):
        x = We[r:r + 8192].reshape(-1, 128); s = qparams(x); q, qi2 = quant(x, s); QI.append(qi2.to(torch.uint8).cpu())
    QI = torch.cat(QI).numpy().reshape(-1); d = np.nonzero(QI != qi)[0]; res["qi_mismatch"] = int(len(d)); res["first_mismatch"] = d[:5].tolist(); res["n"] = int(QI.size)
    # numerics: packer-style whole-matrix unrot vs loader dequant (same decoded levels/scales)
    f.seek(e["hi"][0]); hi = packlib.rc_decode(f.read(e["hi"][1]), e["hcounts"], e["nsc"]).astype(np.uint16); f.seek(e["lo"][0]); lo = np.frombuffer(f.read(e["lo"][1]), dtype=np.uint8).astype(np.uint16)
    sc = ((hi << 8) | lo).astype(np.uint16).view(np.float16)
    Qm = ((torch.from_numpy(qi).cuda().float().view(R, C // 128, 128) - half) * torch.from_numpy(sc.copy()).cuda().float().view(R, C // 128, 1)).view(R, C)
    A = packlib.unrot(Qm, sg, Hb).to(torch.bfloat16); B = packlib.dequant(e, f, "cuda", Hb)
    res["loader_vs_wholeunrot_diff_elems"] = int((A != B).sum()); res["maxabs"] = float((A.float() - B.float()).abs().max())
    A2 = torch.cat([packlib.unrot(Qm[r:r + 8192], sg, Hb) for r in range(0, R, 8192)]).to(torch.bfloat16); res["chunked_vs_whole_diff"] = int((A2 != A).sum()); res["chunked_vs_loader_diff"] = int((A2 != B).sum())
    print("EMBDBG", json.dumps(res), flush=True); json.dump(res, open(out + "/embdbg.json", "w"))
def cli_job(pack, out):
    """standalone commands from the packed body only: answer -> feedback (12 annotations) -> answer on held-out new-label items"""
    import subprocess, json, time
    rd = lambda p: [json.loads(l) for l in open(p) if l.strip()]; P = lambda u: u.split("Request: ", 1)[1]
    b = [P(r["user"]) for r in rd("/app/data/fb_skill/skill_new_b.jsonl")[:30]]; old = [P(r["user"]) for r in rd("/app/data/fb_skill/skill_old.jsonl")[900:930]]; log = {}
    ann = [json.loads(l) for l in open("/app/ship/example_ann.jsonl")]; open("/tmp/ann.jsonl", "w").write("".join(json.dumps(dict(text=r["text"], label="Recipe")) + "\n" for r in ann))
    run = lambda c: (lambda t0, r: (r.stdout.strip().split("\n"), round(time.time() - t0), r.stderr[-400:]))(time.time(), subprocess.run(c, capture_output=True, text=True, cwd="/app/ship"))
    log["answer_before"] = run(["python", "answer.py", "--pack", pack] + b[:10] + old[:10]); print(log, flush=True)
    log["feedback"] = run(["python", "feedback.py", "--pack", pack, "--annotations", "/tmp/ann.jsonl", "--calib", "/app/ship/calib_old.jsonl", "--out", "/tmp/body_fb.pack"]); print(log["feedback"], flush=True)
    log["answer_after"] = run(["python", "answer.py", "--pack", "/tmp/body_fb.pack"] + b + old); log["sizes"] = [os.path.getsize(pack), os.path.getsize("/tmp/body_fb.pack")]
    o = log["answer_after"][0]; log["teach_after"] = sum(x == "Recipe" for x in o[:30]) / 30; log["old_labels_after"] = o[30:]
    json.dump(log, open(out + "/cli.json", "w"), indent=1); print("CLI", log["teach_after"], log["sizes"], flush=True)
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
fn = app.function(image=img, timeout=10 * 3600, volumes={"/vol": vol, "/priv": priv, "/hf": hfvol}, gpu=["A100-40GB", "A100-80GB"], memory=96000, name="go_fb2")(_go)
if __name__ == "__main__":
    mode, cf, name = sys.argv[1:4]; job = dict(mode=mode, name=name)
    if cf != "-": job["cfg"] = json.load(open(cf))
    else: job["method"] = dict(tag="o_prot_L8_m24", carrier="o", solver="prot", m=24, layers=list(range(20, 28)))
    with modal.enable_output(), app.run(detach=True): print("CALL", name, fn.spawn(job).object_id, flush=True)
