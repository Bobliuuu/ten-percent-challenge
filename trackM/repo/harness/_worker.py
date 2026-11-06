"""Prediction worker: reads texts JSON, writes predictions JSON. Never sees labels.
  --mode base                       : the base model from the HF cache
  --mode artifact --artifact D [--memory M] : D/predict.py:load(artifact_dir, memory_path) -> f(list[str]) -> list[int]"""
import argparse, importlib.util, json, os, sys, time
ap = argparse.ArgumentParser()
ap.add_argument("--mode", required=True); ap.add_argument("--texts", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--artifact"); ap.add_argument("--memory")
a = ap.parse_args()
texts = json.load(open(a.texts)); t0 = time.time()
if a.mode == "base":
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    M = "textattack/bert-base-uncased-SST-2"
    tok = AutoTokenizer.from_pretrained(M); m = AutoModelForSequenceClassification.from_pretrained(M).eval()
    pred = []
    with torch.no_grad():
        for i in range(0, len(texts), 64):
            b = tok(texts[i:i + 64], padding=True, truncation=True, max_length=64, return_tensors="pt")
            pred += m(**b).logits.argmax(-1).tolist()
else:
    sys.path.insert(0, a.artifact)
    spec = importlib.util.spec_from_file_location("artifact_predict", os.path.join(a.artifact, "predict.py"))
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    f = mod.load(a.artifact, a.memory)
    pred = [int(x) for x in f(texts)]
assert len(pred) == len(texts) and all(p in (0, 1) for p in pred)
json.dump(dict(pred=pred, seconds=time.time() - t0), open(a.out, "w"))
