"""Modal jobs for the L5 shipped packed loader (artifact in PRIVATE volume tenpct-l5:/pk, dequantised verification checkpoint tenpct-l5:/q32_s7). All on one A100-80GB (the claimed minimum).
  modal run --detach modal_ship.py::verify      decode vs checkpoint (tensors + 200 dev items), results /r/res/verify.json
  modal run --detach modal_ship.py::evalcols --cols 1536"""
import modal, os
HERE = os.path.dirname(os.path.abspath(__file__)); SHIP = HERE + "/../ship"; V2 = HERE + "/../../.."
app = modal.App("tenpct-l5-ship")
img = (modal.Image.debian_slim(python_version="3.11").pip_install("torch==2.8.0", "transformers==4.57.1", "accelerate", "numpy", "constriction", "safetensors")
       .add_local_dir(SHIP, "/root/ship").add_local_file(V2 + "/fb_ablation/exam_data/SKILL7.json", "/root/exam_data/SKILL7.json").add_local_file(HERE + "/rep/data/ct_items.jsonl", "/root/dev200.jsonl").add_local_file(HERE + "/exam_base.json", "/root/exam_base.json"))
vol = modal.Volume.from_name("tenpct-l5")
KW = dict(image=img, gpu="A100-80GB", cpu=16, memory=200000, timeout=43200, volumes={"/r": vol})
def sh(cmd, cwd="/root"):
    import subprocess
    p = subprocess.Popen(cmd, cwd=cwd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for l in p.stdout: print(l, end="", flush=True)
    p.wait(); vol.commit(); return p.returncode
@app.function(**KW)
def verify():
    import sys, json, glob, time, torch; sys.path.insert(0, "/root/ship"); os.makedirs("/r/res", exist_ok=True)
    import common, answer
    from packed_loader import load_packed
    from safetensors import safe_open
    t0 = time.time(); tok, m = load_packed("/r/pk"); tload = time.time() - t0; P = dict(m.named_parameters()); res = dict(load_secs=tload, gpu=torch.cuda.get_device_name(0), peak_gb=torch.cuda.max_memory_allocated() / 2**30)
    n = eq = 0; maxd = 0.0; worst = []; seen = set()
    for f in sorted(glob.glob("/r/q32_s7/*.safetensors")):
        with safe_open(f, "pt", device="cuda") as sf:
            for k in sf.keys():
                w = sf.get_tensor(k); n += 1; seen.add(k)
                if k not in P: worst.append((k, "missing")); continue
                d = (P[k].data.float() - w.float()).abs().max().item(); eq += int(d == 0); maxd = max(maxd, d)
                if d > 0: worst.append((k, d))
    res.update(tensors_compared=n, tensors_bit_equal=eq, max_abs_diff=maxd, worst=worst[:10], params_not_in_ckpt=[k for k in P if k not in seen][:5])
    print(res, flush=True)
    items = [json.loads(l) for l in open("/root/dev200.jsonl")]; pf = common.prompt_fn(common.task()); P0 = common.task()["prompt"].split("{t}")[0]
    reqs = [i["user"][len(P0):] for i in items]; o1 = answer.answer_texts(tok, m, reqs, pf); del m, P; torch.cuda.empty_cache()
    from transformers import AutoModelForCausalLM
    m2 = AutoModelForCausalLM.from_pretrained("/r/q32_s7", torch_dtype=torch.bfloat16, device_map="cuda").eval(); tok2 = tok
    o2 = answer.answer_texts(tok2, m2, reqs, pf)
    gold = [i["answer"] for i in items]
    res.update(dev_n=len(items), dev_agree=sum(x == y for x, y in zip(o1, o2)) / len(items), acc_packed=sum(x == g for x, g in zip(o1, gold)) / len(items), acc_ckpt=sum(x == g for x, g in zip(o2, gold)) / len(items), disagreements=[(r, x, y) for r, x, y in zip(reqs, o1, o2) if x != y][:10])
    json.dump(res, open("/r/res/verify.json", "w"), indent=1); vol.commit(); print("VERIFY", res, flush=True)
@app.function(**KW)
def evalcols(cols: int, draws: int = 3, nosub: bool = False):
    os.makedirs("/r/res", exist_ok=True)
    sh(f"cd /root && python ship/eval_locked.py /r/pk /r/res/eval_c{cols}.json --cols {cols} --draws {draws} {'--no-subprocess' if nosub else ''}")
@app.function(**KW)
def exam():
    """Score the packed artifact ONCE on the sealed exam (bytes changed vs the first build) + 500 dev rows, real greedy generation, retention vs base-correct (base outputs from results/final)."""
    import sys, json, collections, time, torch; sys.path.insert(0, "/root/ship"); os.makedirs("/r/res", exist_ok=True)
    import common, answer; from packed_loader import load_packed
    tok, m = load_packed("/r/pk"); rows = json.load(open("/root/exam_base.json")); P0 = common.task()["prompt"].split("{t}")[0]; pf = common.prompt_fn(common.task())
    o = answer.answer_texts(tok, m, [r["user"][len(P0):] for r in rows], pf); res = {}
    for role in ("exam", "dev"):
        R = [(r, x) for r, x in zip(rows, o) if r["role"] == role]; ok = [(r, x == r["gold"]) for r, x in R]; bk = [(r, c) for r, c in ok if r["base_ok"]]
        k = collections.defaultdict(list); s_ = collections.defaultdict(list)
        for r, c in bk: k[r["kind"]].append(c); s_[r["src"]].append(c)
        res[role] = dict(n=len(R), base_acc=sum(r["base_ok"] for r, _ in R) / len(R), cand_acc=sum(c for _, c in ok) / len(R), retention=sum(c for _, c in bk) / len(bk), n_base_ok=len(bk), by_kind={a: sum(b) / len(b) for a, b in k.items()}, by_src={a: sum(b) / len(b) for a, b in s_.items()})
        if role == "exam":
            cl = [(r, c) for r, c in bk if r.get("tier") == "clean"]; res[role]["clean_only_retention"] = sum(c for _, c in cl) / len(cl)
    res["outputs"] = {r["id"]: x for r, x in zip(rows, o)}; json.dump(res, open("/r/res/exam_new_body.json", "w")); vol.commit(); print("EXAM", {k: v for k, v in res.items() if k != "outputs"}, flush=True)
@app.local_entrypoint()
def main(): pass
