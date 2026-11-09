"""One-off diagnostic: GPU peak while loading + rotary buffer dtype/equality between core.load_packed and packed_loader.load. modal run modal_memtest.py"""
import modal, os, json
app = modal.App("tenpct-l9b-memtest"); HERE = os.path.dirname(os.path.abspath(__file__))
img = (modal.Image.debian_slim(python_version="3.11").pip_install("torch==2.8.0", "transformers==5.22.2", "numpy", "accelerate", "huggingface_hub", "sentencepiece", "protobuf", "tiktoken", "safetensors")
       .add_local_dir(HERE, "/root/l9b", ignore=["__pycache__", "*.pyc", "bank6q", "results"]))
vol = modal.Volume.from_name("tenpct-l9b")
@app.function(image=img, gpu="A100-80GB", memory=96000, timeout=1800, volumes={"/r": vol})
def run():
    import sys, torch, gc, hashlib; sys.path.insert(0, "/root/l9b"); os.environ["PL_DEBUG"] = "1"
    import core, packed_loader
    adir = "/r/l9b/art_s7_k102_e3"; out = {}
    for tag, fn in (("packed_loader", packed_loader.load), ("core", core.load_packed)):
        torch.cuda.reset_peak_memory_stats(); m, tok, man = fn(adir)
        bufs = {n: (str(b.dtype), hashlib.sha256(b.cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()[:12]) for n, b in m.named_buffers()}
        out[tag] = dict(peak=torch.cuda.max_memory_allocated() / 2**30, bufs={k: v for k, v in list(bufs.items())[:4]}, nbuf=len(bufs), bufdig=hashlib.sha256(json.dumps(bufs, sort_keys=True).encode()).hexdigest()[:12])
        print(tag, out[tag], flush=True); del m; gc.collect(); torch.cuda.empty_cache()
    return out
@app.local_entrypoint()
def main(): print(json.dumps(run.remote()))
