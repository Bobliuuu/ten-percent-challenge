"""Reference build + sha256 of the L1 weight files: packs prism-ml/Bonsai-8B-unpacked with pack1.py on a Modal CPU container and prints/writes sha256 of every output.
  modal run --detach v2/lanes/L1/build/modal_hash_pack.py     (writes SHA256SUMS lines to the container log and returns them)"""
import modal, os
app = modal.App("l1-hash-pack")
img = (modal.Image.debian_slim(python_version="3.11").pip_install("torch==2.5.1", "safetensors", "numpy", "huggingface_hub")
       .add_local_file(os.path.join(os.path.dirname(os.path.abspath(__file__)), "pack1.py"), "/pack1.py"))
@app.function(image=img, cpu=8, memory=65536, ephemeral_disk=524288, timeout=7200)
def run():
    import subprocess, hashlib, glob, os
    from huggingface_hub import snapshot_download
    snap = snapshot_download("prism-ml/Bonsai-8B-unpacked", local_dir="/tmp/snap", allow_patterns=["*.safetensors", "*.json", "*.txt", "*.jinja"])
    print("snapshot sha256 + sizes:", flush=True)
    sums = []
    for f in sorted(glob.glob("/tmp/snap/*")):
        if os.path.isfile(f): sums.append(("SNAP", os.path.basename(f), os.path.getsize(f)))
    subprocess.check_call(["python", "/pack1.py", "/tmp/snap", "/tmp/art"])
    out = []
    for f in sorted(glob.glob("/tmp/art/*")) + [g for g in sorted(glob.glob("/tmp/snap/*")) if g.split("/")[-1] in ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "config.json", "generation_config.json")]:
        h = hashlib.sha256()
        with open(f, "rb") as fh:
            for b in iter(lambda: fh.read(1 << 24), b""): h.update(b)
        out.append(f"{h.hexdigest()}  {os.path.basename(f)}  {os.path.getsize(f)}"); print(out[-1], flush=True)
    return "\n".join(out)
@app.local_entrypoint()
def main():
    r = run.remote(); open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "SHA256SUMS.build"), "w").write(r + "\n"); print(r)
