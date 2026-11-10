"""L3 own-data dry run (same protocol as L9b sec 11 / L1 / L5 owndata). Base Qwen3-30B-A3B-Instruct-2507 vs packed L3 SKILL7 artifact (volume tenpct-l3-art:/skill7x).
Arms: (1) no feedback (packed vs base): Banking77-200 77-label menu, same queries under the 7-skill router prompt (agreement), 100 generic prompts; (2) final carrier (shipped feedback.py defaults:
prot, layers 40-47, m32, 24 annotations = 12 new + 12 real old-skill errors of the packed model): Recipe (dev ann pool) and Timer real draw 0 -> same own-data sets; (3) unseen Timer, 3 draws x real/shuffled/wrong,
28 held-out Timer requests, 60 Alarm + 60 other SKILL7 SEL neighbours; base Timer emission also measured.
Deploy + spawn (crash-safe):  modal deploy v2/lanes/L3/owndata/modal_owndata.py ; python3 v2/lanes/L3/owndata/launch.py
Results: volume tenpct-l3-art:/results/owndata_l3.json, printed as RESULT_JSON."""
import modal, os
HERE = os.path.dirname(os.path.abspath(__file__)); L3 = os.path.dirname(HERE); V2 = os.path.dirname(os.path.dirname(L3))
app = modal.App("tenpct-l3-owndata")
img = (modal.Image.debian_slim(python_version="3.11").pip_install("torch==2.6.0", "transformers==4.57.1", "numpy", "accelerate", "huggingface_hub<1.0", "scipy")
       .add_local_dir(L3, "/root/l3", ignore=["results", "__pycache__", "*.pt", "proxy", "jobs"]).add_local_dir(V2 + "/moe_entry/data", "/root/data"))
vol = modal.Volume.from_name("tenpct-moe"); art = modal.Volume.from_name("tenpct-l3-art")
@app.function(image=img, gpu="H200", cpu=8, memory=190000, timeout=6 * 3600, volumes={"/r": vol, "/art": art}, env={"HF_HOME": "/r/hf"}, ephemeral_disk=600000)
def run():
    import sys; sys.path[:0] = ["/root/l3", "/root/l3/owndata"]; import job; r = job.main(); art.commit(); return r
