"""L10 final scoring ONCE on the shared locked SKILL7 exam (v2/exam). Stages, all real greedy generation via ship/common.generate:
 base (bf16 Omni-7B thinker text path) | quant (dequantised 9.88% artifact) | quant+feedback(real: 25 annotated Recipe mistakes from skill7_ann_pool, column-sparse down_proj closed form)
 | feedback shuffled (25 old-skill items labelled Recipe) | empty feedback (= quant). Feedback arms scored on skill7_teach (164) + neighbour exam ids; real arm also on the full exam.
Result -> Volume tenpct-tasksearch:/l10exam/exam_skill7_omni7.json"""
import modal, json, os
HERE = os.path.dirname(os.path.abspath(__file__)); app = modal.App("tenpct-l10exam")
vol = modal.Volume.from_name("tenpct-tasksearch", create_if_missing=True); hfvol = modal.Volume.from_name("tenpct-v2-hf", create_if_missing=True)
img = (modal.Image.debian_slim(python_version="3.11").pip_install("torch==2.4.1", "transformers==4.55.0", "numpy", "accelerate")
       .add_local_dir(HERE + "/ship", "/app/ship").add_local_dir(HERE + "/data", "/app/data"))
def _go(job):
    import sys, torch, json, hashlib, random; sys.path.insert(0, "/app/ship"); import common, feedback as FB
    from transformers import AutoTokenizer
    rd = lambda p: [json.loads(l) for l in open(p) if l.strip()]
    ex = rd("/app/data/exam_skill7.jsonl"); teach = rd("/app/data/s7_teach.jsonl"); ann = rd("/app/data/s7_ann_pool.jsonl"); nbid = set(json.load(open("/app/data/s7_neighbours.json")))
    old = rd("/app/data/fb_skill/skill_old.jsonl"); calib = [json.loads(l)["query"] for l in open("/app/data/fb_skill/skill_calib.jsonl")]
    nb = [r for r in ex if r["id"] in nbid]; NEW = "Recipe"; out = {"n_exam": len(ex)}; H = lambda s: int(hashlib.sha256(s.encode()).hexdigest()[:8], 16)
    tok = AutoTokenizer.from_pretrained(job["quant"]); tok.pad_token = tok.pad_token or tok.eos_token
    gen = lambda m, rows: dict(zip([r["id"] for r in rows], common.generate(m, tok, [r["user"] for r in rows], bs=32)))
    m = common.load(job["base"], patch=False); out["base"] = gen(m, ex); del m; torch.cuda.empty_cache()
    m = common.load(job["quant"], patch=False); out["quant"] = gen(m, ex); out["quant_teach"] = gen(m, teach)
    json.dump(out, open("/tmp/partial.json", "w"))
    cal = FB.calib_seqs(m, tok, calib)
    anns = sorted(ann, key=lambda r: H("l10ann|" + r["id"]))   # real mistakes only: the quantised model does not already say Recipe
    pred = gen(m, anns[:80]); mist = [r for r in anns[:80] if pred[r["id"]] != NEW][:25]; out["ann_ids"] = [r["id"] for r in mist]; out["ann_pred"] = [pred[r["id"]] for r in mist]
    oldp = sorted(old, key=lambda r: H("l10shuf|" + r["id"]))[:25]
    cfg = dict(layers=6, cols=256); out["cfg"] = cfg
    for arm, rows in (("real", mist), ("shuffled", oldp)):
        A = [dict(query=r["user"], label=NEW) for r in rows]; P = FB.compute_patch(m, tok, A, None, calib, cal=cal, **cfg)
        out[arm + "_bytes"] = sum(c.numel() * 2 + v.numel() * 2 for c, v in zip(P["cols"], P["vals"])) if P else 0
        out[arm + "_teach"] = gen(m, teach); out[arm + "_neigh"] = gen(m, nb)
        if arm == "real": out["real_exam"] = gen(m, ex); torch.save({k: P[k] for k in ("layers", "cols", "vals")}, "/tmp/patch_real.pt")
        if P: FB.undo(m, P)
    out["empty_teach"] = out["quant_teach"]
    os.makedirs("/vol/l10exam", exist_ok=True); json.dump(out, open("/vol/l10exam/exam_skill7_omni7.json", "w")); vol.commit(); print("EXAM_DONE", flush=True)
fn = app.function(image=img, timeout=6 * 3600, volumes={"/vol": vol, "/hf": hfvol}, gpu=["A100-40GB", "L40S", "A100-80GB"], memory=64000, name="go_l10exam")(_go)
if __name__ == "__main__":
    with modal.enable_output(), app.run(detach=True): print(fn.spawn(dict(base="/hf/l10_omni7b", quant="/hf/l10_deq_omni7")).object_id, flush=True)
