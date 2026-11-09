"""Build the L1 Bonsai-8B artifact weights from the public snapshot and publish them PRIVATELY into the Modal Volume `tenpct-l1-art` (dir /art) with a sha256 manifest.
  modal run --detach v2/lanes/L1/build/modal_publish.py
Checks every hash against v2/lanes/L1/SHA256SUMS (bits/scales/index/config/tokenizer/vocab/merges; small.pt is a torch.save pickle: size+hash reported)."""
import modal, os
HERE = os.path.dirname(os.path.abspath(__file__))
app = modal.App("tenpct-l1-publish")
vol = modal.Volume.from_name("tenpct-l1-art", create_if_missing=True)
img = (modal.Image.debian_slim(python_version="3.11").pip_install("torch==2.5.1", "safetensors", "numpy", "huggingface_hub")
       .add_local_file(HERE + "/pack1.py", "/pack1.py").add_local_file(HERE + "/../SHA256SUMS", "/SHA256SUMS"))
@app.function(image=img, cpu=8, memory=65536, ephemeral_disk=524288, timeout=7200, volumes={"/vol": vol})
def run():
    import subprocess, hashlib, glob, os, shutil, json
    from huggingface_hub import snapshot_download
    snap = snapshot_download("prism-ml/Bonsai-8B-unpacked", local_dir="/tmp/snap")
    subprocess.check_call(["python", "/pack1.py", "/tmp/snap", "/tmp/art"])
    for f in ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "config.json", "generation_config.json", "chat_template.jinja", "LICENSE", "NOTICE.txt"):
        if os.path.exists("/tmp/snap/" + f): shutil.copy("/tmp/snap/" + f, "/tmp/art/" + f)
    def sha(p):
        h = hashlib.sha256()
        with open(p, "rb") as fh:
            for b in iter(lambda: fh.read(1 << 24), b""): h.update(b)
        return h.hexdigest()
    ref = {l.split()[1]: l.split()[0] for l in open("/SHA256SUMS") if l.strip()}
    lines = []; ok = True
    for f in sorted(os.listdir("/tmp/art")):
        p = "/tmp/art/" + f; h = sha(p); lines.append(f"{h}  {f}  {os.path.getsize(p)}")
        if f in ref and ref[f] != h: ok = False; lines.append(f"MISMATCH vs SHA256SUMS: {f}")
    print("\n".join(lines), "\nALL_MATCH_SHA256SUMS" if ok else "\nMISMATCH", flush=True)
    if os.path.exists("/vol/art"): shutil.rmtree("/vol/art")
    shutil.copytree("/tmp/art", "/vol/art")
    open("/vol/art/SHA256SUMS.artifact", "w").write("\n".join(lines) + "\n"); vol.commit()
    return "\n".join(lines) + ("\nALL_MATCH_SHA256SUMS" if ok else "\nMISMATCH")
@app.local_entrypoint()
def main():
    r = run.remote(); open(HERE + "/SHA256SUMS.artifact", "w").write(r + "\n"); print(r)
