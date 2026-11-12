"""L9b job runner (pure python; runs on Modal via modal_l9b.py or on any GPU box with R_ROOT=/r)."""
import json, os
COMMIT = lambda: None
def sets_of(job):
    import core
    S = {}
    for n in job.get("sets", ["SEL", "REP"]):
        pre = job.get("prefix", "skill7"); p = f"/root/l9b/data/{pre}_{n.lower()}.jsonl" if n in ("SEL", "EXAM", "CAL", "FB_TEACH", "FB_ANNOT") else f"/r/l9b/{n.lower()}.jsonl"
        S[n] = core.rows_of(p)
    return S

def main(job):
    import sys, time, torch; sys.path.insert(0, "/root/l9b"); import core
    from transformers import AutoTokenizer, AutoModelForCausalLM
    T0 = time.time(); log = lambda *a: print(f"[{time.time()-T0:6.0f}s]", *a, flush=True)
    os.makedirs("/r/l9b/res", exist_ok=True); stage = job["stage"]; tag = job.get("tag", stage)
    def commit(): COMMIT()
    if stage == "pack":
        tok = AutoTokenizer.from_pretrained(core.MID); tok.padding_side = "left"
        ng = torch.cuda.device_count()
        model = AutoModelForCausalLM.from_pretrained(core.MID, dtype=torch.bfloat16, device_map="auto", max_memory={i: "74GiB" for i in range(ng)}).eval()
        for p in model.parameters(): p.requires_grad_(False)
        R = core.Runner(model, tok); S = sets_of(job); log("loaded")
        if job.get("base", True):
            for n, rows in S.items():
                f = f"/r/l9b/res/base_{job.get('prefix','skill7')}_{n}.json"
                if os.path.exists(f) and not job.get("redo_base"): continue
                outs = R.gen(rows); json.dump(dict(outs=outs, cor=core.correct(outs, rows), ids=[r["id"] for r in rows]), open(f, "w")); log("base", n, sum(core.correct(outs, rows)) / len(rows)); commit()
        mf = f"/r/l9b/res/mass_{job.get('prefix','skill7')}.pt"
        if not os.path.exists(mf):
            blocks, mass, STAT = core.install_router_hooks(model)
            M = core.mass_chunks(R, core.rows_of(f"/root/l9b/data/{job.get('prefix','skill7')}_cal.jsonl"), blocks, mass, STAT); torch.save(M, mf); commit(); log("mass", tuple(M.shape))
        M = torch.load(mf)
        for v in job["variants"]:
            name = v["name"]; rs = np_rng(v.get("draw", 0)); nch = M.shape[0]; sel = sorted(rs.choice(nch, nch // 2, replace=False).tolist()) if v.get("draw", 0) else list(range(nch))
            mv = M[sel].sum(0); kept = torch.topk(mv, v["k"], dim=1).indices.sort(1).values
            adir = f"/r/l9b/art_{name}"; t1 = time.time()
            tot = core.pack_model(model, tok, kept, adir, eb=v.get("eb", 3), attn_b=v.get("attn_b", 6), emb_b=v.get("emb_b", 4), head_b=v.get("head_b", 4), extra=dict(name=name, draw=v.get("draw", 0), chunks=sel))
            import hashlib
            def sha(p):
                h = hashlib.sha256()
                with open(p, "rb") as fh:
                    for b in iter(lambda: fh.read(1 << 24), b""): h.update(b)
                return h.hexdigest()
            files = {f: dict(bytes=os.path.getsize(os.path.join(adir, f)), sha256=sha(os.path.join(adir, f))) for f in sorted(os.listdir(adir))}
            r = dict(name=name, bytes=tot, secs=round(time.time() - t1), files=files); print("PACK_JSON", json.dumps(r), flush=True); commit()
        return
    if stage == "loadtest":
        os.environ["PL_DEBUG"] = "1"
        import gc, packed_loader, hashlib
        adir = f"/r/l9b/art_{job['art']}"; rows = core.rows_of("/root/l9b/data/skill7_sel.jsonl")[:job.get("n", 200)]; res = {}
        def bufinfo(model): return {n: str(b.dtype) for n, b in model.named_buffers() if "inv_freq" in n}
        def chk(model):
            h = hashlib.sha256()
            for n, p in sorted(model.named_parameters()): h.update(n.encode()); h.update(p.data.contiguous().view(torch.int16).cpu().numpy().tobytes())
            return h.hexdigest()
        for tag, fn in (("packed_loader.load", lambda: packed_loader.load(adir)), ("core.load_packed", lambda: core.load_packed(adir))):
            torch.cuda.reset_peak_memory_stats(); t = time.time(); model, tok_, man = fn(); lt = time.time() - t; lp = torch.cuda.max_memory_allocated() / 2**30; R = core.Runner(model, tok_)
            torch.cuda.reset_peak_memory_stats(); R.gen(rows[:1]); sp = torch.cuda.max_memory_allocated() / 2**30; torch.cuda.reset_peak_memory_stats(); outs = R.gen(rows); res[tag] = dict(load_s=round(lt, 1), load_peak_gb=round(lp, 1), single_request_peak_gb=round(sp, 1), peak_gb_with_200_item_batched_gen=round(torch.cuda.max_memory_allocated() / 2**30, 1), param_sha256=chk(model), inv_freq_dtypes=sorted(set(bufinfo(model).values())), outs=outs, acc=sum(core.correct(outs, rows)) / len(rows))
            log(tag, {k: v for k, v in res[tag].items() if k != "outs"}); del model, R; gc.collect(); torch.cuda.empty_cache()
        a, b = res["core.load_packed"], res["packed_loader.load"]
        res["identical_params"] = a["param_sha256"] == b["param_sha256"]; res["identical_outputs"] = a["outs"] == b["outs"]; res["n"] = len(rows)
        json.dump(res, open(f"/r/l9b/res/loadtest_{job['art']}.json", "w")); commit(); print("RESULT_JSON", json.dumps({k: v for k, v in res.items()}), flush=True); return
    if stage in ("eval", "fb"):
        sys.path.insert(0, "/root/l9b")
        adir = f"/r/l9b/art_{job['art']}"; model, tok, man = core.load_packed(adir); R = core.Runner(model, tok); S = sets_of(job); log("loaded packed", job["art"])
        if stage == "eval":
            out = dict(art=job["art"], man_cfg=man["cfg"], extra=man["extra"], sets={})
            for n, rows in S.items():
                b = json.load(open(f"/r/l9b/res/base_{job.get('prefix','skill7')}_{n}.json")); assert b["ids"] == [r["id"] for r in rows]
                outs = R.gen(rows); cor = core.correct(outs, rows); sc = core.score(cor, b["cor"], rows); sc["sample"] = outs[:5]; out["sets"][n] = sc
                json.dump(dict(outs=outs, cor=cor), open(f"/r/l9b/res/outs_{job['art']}_{n}.json", "w")); log(n, {k: v for k, v in sc.items() if k != "cells"})
            json.dump(out, open(f"/r/l9b/res/eval_{job['art']}.json", "w")); commit(); print("RESULT_JSON", json.dumps(out), flush=True); return
        if job.get('del'):
            sys.path.insert(0, '/root/l9b/deletion'); import del_run; del_run.run(job, model, tok, R, S, man, log, commit); return
        import fb_run; fb_run.run(job, model, tok, R, S, man, log, commit)
def np_rng(seed):
    import numpy as np; return np.random.RandomState(1000 + seed)

