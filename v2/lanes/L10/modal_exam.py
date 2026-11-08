"""L10 final scoring ONCE on the shared locked SKILL7 exam: base (bf16 Omni-7B thinker text path) vs shipped-simulation (dequantised 9.88% artifact), real greedy generation via ship/common.generate.
python3 modal_exam.py  -> Volume tenpct-tasksearch:/l10exam/exam_skill7_omni7.json (per-item outputs)"""
import modal, json, os
HERE = os.path.dirname(os.path.abspath(__file__)); app = modal.App("tenpct-l10exam")
vol = modal.Volume.from_name("tenpct-tasksearch", create_if_missing=True); hfvol = modal.Volume.from_name("tenpct-v2-hf", create_if_missing=True)
img = (modal.Image.debian_slim(python_version="3.11").pip_install("torch==2.4.1", "transformers==4.55.0", "numpy", "accelerate")
       .add_local_dir(HERE + "/ship", "/app/ship").add_local_file(HERE + "/data/exam_skill7.jsonl", "/app/exam.jsonl"))
def _go(job):
    import sys, torch, json; sys.path.insert(0, "/app/ship"); import common
    from transformers import AutoTokenizer
    rows = [json.loads(l) for l in open("/app/exam.jsonl")]; out = {}
    for name, d in (("base", job["base"]), ("quant", job["quant"])):
        tok = AutoTokenizer.from_pretrained(d); tok.pad_token = tok.pad_token or tok.eos_token; m = common.load(d, patch=False)
        o = common.generate(m, tok, [r["user"] for r in rows], bs=32); out[name] = {r["id"]: x for r, x in zip(rows, o)}; del m; torch.cuda.empty_cache()
    os.makedirs("/vol/l10exam", exist_ok=True); json.dump(out, open("/vol/l10exam/exam_skill7_omni7.json", "w")); vol.commit(); print("EXAM_DONE", flush=True)
fn = app.function(image=img, timeout=6 * 3600, volumes={"/vol": vol, "/hf": hfvol}, gpu=["A100-40GB", "L40S", "A100-80GB"], memory=64000, name="go_l10exam")(_go)
if __name__ == "__main__":
    with modal.enable_output(), app.run(detach=True): print(fn.spawn(dict(base="/hf/l10_omni7b", quant="/hf/l10_deq_omni7")).object_id, flush=True)
