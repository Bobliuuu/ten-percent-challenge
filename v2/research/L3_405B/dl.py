"""L3 download: SillyTilly/Meta-Llama-3.1-405B-Instruct (bf16, ungated) -> Volume tenpct-l3-405b:/bf16. CPU only.
deploy: modal deploy dl.py ; launch: python dl.py launch"""
import modal, sys, json, os
app = modal.App("l3-405b-dl")
vol = modal.Volume.from_name("tenpct-l3-405b", version=2)
img = modal.Image.debian_slim().pip_install("huggingface_hub", "hf_transfer", "requests").env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
REPO = "SillyTilly/Meta-Llama-3.1-405B-Instruct"

@app.function(image=img, volumes={"/w": vol}, timeout=6 * 3600, cpu=4, memory=16384, retries=3)
def dl(files):
    from huggingface_hub import hf_hub_download
    import time
    for f in files:
        t = time.time()
        p = f"/w/bf16/{f}"
        if os.path.exists(p + ".done"): continue
        hf_hub_download(REPO, f, local_dir="/w/bf16")
        open(p + ".done", "w").write("1"); vol.commit()
        print("DL", f, round(time.time() - t), flush=True)
    return len(files)

@app.function(image=img, timeout=600)
def listing():
    from huggingface_hub import list_repo_files
    return [f for f in list_repo_files(REPO) if not f.startswith(".")]

if __name__ == "__main__":
    f = modal.Function.from_name("l3-405b-dl", "dl"); L = modal.Function.from_name("l3-405b-dl", "listing")
    files = L.remote(); print(len(files), files[:5])
    n = 24
    for i in range(n):
        c = f.spawn(files[i::n]); print("spawned", i, c.object_id)
