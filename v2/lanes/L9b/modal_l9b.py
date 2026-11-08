"""L9b Modal app (Qwen3.5-35B-A3B real packed artifact). Deploy: modal deploy modal_l9b.py ; launch: python3 launch_l9b.py job.json  (spawn; results on volume tenpct-l9b:/res + RESULT_JSON in logs).
Stages: pack (2 GPUs: base outputs, routing-mass chunks, write packed artifacts), eval (1 GPU: load packed artifact, generate on sets), fb (feedback on packed artifact)."""
import modal, json, os
app = modal.App("tenpct-l9b")
HERE = os.path.dirname(os.path.abspath(__file__))
img = (modal.Image.debian_slim(python_version="3.11").pip_install("torch==2.8.0", "transformers==5.22.2", "numpy", "accelerate", "huggingface_hub", "sentencepiece", "protobuf", "tiktoken", "safetensors", "kernels")
       .add_local_dir(HERE, "/root/l9b", ignore=["__pycache__", "*.pyc"]))
vol = modal.Volume.from_name("tenpct-l9b", create_if_missing=True)
hfvol = modal.Volume.from_name("tenpct-v2-hf", create_if_missing=True)

def sets_of(job):
    import core
    S = {}
    for n in job.get("sets", ["SEL", "REP"]):
        p = f"/root/l9b/data/skill7_{n.lower()}.jsonl" if n in ("SEL", "REP", "CAL", "FB_TEACH", "FB_ANNOT") else f"/r/l9b/{n.lower()}.jsonl"
        S[n] = core.rows_of(p)
    return S

def main(job):
    import sys, time, torch; sys.path.insert(0, "/root/l9b"); import core
    from transformers import AutoTokenizer, AutoModelForCausalLM
    T0 = time.time(); log = lambda *a: print(f"[{time.time()-T0:6.0f}s]", *a, flush=True)
    os.makedirs("/r/l9b/res", exist_ok=True); stage = job["stage"]; tag = job.get("tag", stage)
    def commit():
        try: vol.commit()
        except Exception: pass
    if stage == "pack":
        tok = AutoTokenizer.from_pretrained(core.MID); tok.padding_side = "left"
        ng = torch.cuda.device_count()
        model = AutoModelForCausalLM.from_pretrained(core.MID, dtype=torch.bfloat16, device_map="auto", max_memory={i: "74GiB" for i in range(ng)}).eval()
        for p in model.parameters(): p.requires_grad_(False)
        R = core.Runner(model, tok); S = sets_of(job); log("loaded")
        if job.get("base", True):
            for n, rows in S.items():
                f = f"/r/l9b/res/base_{n}.json"
                if os.path.exists(f) and not job.get("redo_base"): continue
                outs = R.gen(rows); json.dump(dict(outs=outs, cor=core.correct(outs, rows), ids=[r["id"] for r in rows]), open(f, "w")); log("base", n, sum(core.correct(outs, rows)) / len(rows)); commit()
        mf = f"/r/l9b/res/mass_{job.get('cal','CAL')}.pt"
        if not os.path.exists(mf):
            blocks, mass, STAT = core.install_router_hooks(model)
            M = core.mass_chunks(R, core.rows_of(f"/root/l9b/data/skill7_{job.get('cal','CAL').lower()}.jsonl"), blocks, mass, STAT); torch.save(M, mf); commit(); log("mass", tuple(M.shape))
        M = torch.load(mf)
        for v in job["variants"]:
            name = v["name"]; rs = np_rng(v.get("draw", 0)); nch = M.shape[0]; sel = sorted(rs.choice(nch, nch // 2, replace=False).tolist()) if v.get("draw", 0) else list(range(nch))
            mv = M[sel].sum(0); kept = torch.topk(mv, v["k"], dim=1).indices.sort(1).values
            adir = f"/r/l9b/art_{name}"; t1 = time.time()
            tot = core.pack_model(model, tok, kept, adir, eb=v.get("eb", 3), attn_b=v.get("attn_b", 6), emb_b=v.get("emb_b", 4), head_b=v.get("head_b", 4), extra=dict(name=name, draw=v.get("draw", 0), chunks=sel))
            r = dict(name=name, bytes=tot, secs=round(time.time() - t1)); print("PACK_JSON", json.dumps(r), flush=True); commit()
        return
    if stage in ("eval", "fb"):
        sys.path.insert(0, "/root/l9b")
        adir = f"/r/l9b/art_{job['art']}"; model, tok, man = core.load_packed(adir); R = core.Runner(model, tok); S = sets_of(job); log("loaded packed", job["art"])
        if stage == "eval":
            out = dict(art=job["art"], man_cfg=man["cfg"], extra=man["extra"], sets={})
            for n, rows in S.items():
                b = json.load(open(f"/r/l9b/res/base_{n}.json")); assert b["ids"] == [r["id"] for r in rows]
                outs = R.gen(rows); cor = core.correct(outs, rows); sc = core.score(cor, b["cor"], rows); sc["sample"] = outs[:5]; out["sets"][n] = sc
                json.dump(dict(outs=outs, cor=cor), open(f"/r/l9b/res/outs_{job['art']}_{n}.json", "w")); log(n, {k: v for k, v in sc.items() if k != "cells"})
            json.dump(out, open(f"/r/l9b/res/eval_{job['art']}.json", "w")); commit(); print("RESULT_JSON", json.dumps(out), flush=True); return
        import fb_run; fb_run.run(job, model, tok, R, S, man, log, commit)
def np_rng(seed):
    import numpy as np; return np.random.RandomState(1000 + seed)

SZ = dict(s=dict(gpu="A100-80GB", memory=96000), m2=dict(gpu="A100-80GB:2", memory=256000))
FN = {k: app.function(image=img, timeout=24 * 3600, volumes={"/r": vol, "/hf": hfvol}, env={"HF_HOME": "/hf", "HF_HUB_ENABLE_HF_TRANSFER": "0"}, name="run_" + k, **v)(main) for k, v in SZ.items()}
