"""L10 REAL PACK job: requantise Omni-7B text path (same recipe as jobs4 + last-8 o_proj kept bf16 as the feedback carrier), write the entropy-coded body to the private volume,
verify per-tensor sha256 of the decoded pack vs in-process weights, run dev eval. python3 modal_pack.py  (spawns detached). Volume tenpct-l10-priv:/omni7_body.pack(+.json)"""
import modal, json, sys, os
HERE = os.path.dirname(os.path.abspath(__file__))
app = modal.App("tenpct-l10pack")
vol = modal.Volume.from_name("tenpct-tasksearch", create_if_missing=True); hfvol = modal.Volume.from_name("tenpct-v2-hf", create_if_missing=True); priv = modal.Volume.from_name("tenpct-l10-priv", create_if_missing=True)
img = (modal.Image.debian_slim(python_version="3.11").pip_install("torch==2.4.1", "transformers==4.55.0", "numpy", "accelerate", "huggingface_hub", "safetensors", "constriction")
       .add_local_file(HERE + "/dense_core_l10.py", "/code/dense_core.py").add_local_file(HERE + "/prep_text.py", "/code/prep_text.py").add_local_file(HERE + "/ship/packlib.py", "/code/packlib.py")
       .add_local_file(HERE + "/data/items_dev.jsonl", "/code/easy_items.jsonl").add_local_file(HERE + "/data/calib.jsonl", "/code/easy_calib.jsonl")
       .add_local_file(HERE + "/../../compress/dense/dense_data.json", "/code/dense_data.json"))
def _go(job):
    import os, sys
    os.environ.update(HF_HOME="/hf", DENSE_HF="/vol", DENSE_DATA="/code/dense_data.json", CT_ITEMS="/code/easy_items.jsonl", CT_CALIB="/code/easy_calib.jsonl", HF_HUB_ENABLE_HF_TRANSFER="0", DENSE_GPU="1", L10_SHIP="/code")
    sys.path.insert(0, "/code"); import prep_text; M = prep_text.ensure("Qwen/Qwen2.5-Omni-7B", "/hf/l10_omni7b", False, "omni")
    import dense_core
    cfg = dict(model=M, ctasks=1, calib_ct=1, gen=True, base_bytes_override=21464450880, **job["cfg"])
    res = dense_core._run(dict(multi=[cfg])); tag = os.path.basename(M)
    f = f"/vol/dense/ct_{tag}_{cfg['name']}.json"; print("RES", res, flush=True)
    priv.commit(); vol.commit(); print("PACK_JOB_DONE", flush=True)
fn = app.function(image=img, timeout=10 * 3600, volumes={"/vol": vol, "/hf": hfvol, "/priv": priv}, gpu="A100-80GB", memory=128000, name="go_pack")(_go)
if __name__ == "__main__":
    job = dict(cfg=dict(name=sys.argv[1] if len(sys.argv) > 1 else "pk_r985", target_ratio=0.0985, g=256, scale_bits=14, emb_L=4, head_L=4, gen_bs=48, fp16_keep_o_last=8, pack_path="/priv/omni7_body.pack"))
    with modal.enable_output(), app.run(detach=True): print("CALL", fn.spawn(job).object_id, flush=True)
