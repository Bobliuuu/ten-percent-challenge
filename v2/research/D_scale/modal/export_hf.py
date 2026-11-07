"""Export a dense_stream quantised dir (q_<name>) as a bf16 HF checkpoint of dequantised weights (for fast vLLM evaluation).
usage: python export_hf.py MODEL_DIR QDIR OUT_DIR G"""
import sys, os, json, glob, shutil, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dense_stream import ST, QLinear, Quantiser, linn
from safetensors.torch import save_file
md, qdir, out, g = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]); FP8 = len(sys.argv) > 5 and sys.argv[5] == "fp8"; QCFG = sys.argv[6] if FP8 else None
os.makedirs(out, exist_ok=True); st = ST(md); Qz = Quantiser(g); conf = json.load(open(md + "/config.json")); nl = conf["num_hidden_layers"]
wm = {}
def put(fn, d):
    save_file({k: v.contiguous().cpu() for k, v in d.items()}, out + "/" + fn); [wm.__setitem__(k, fn) for k in d]
for i in range(nl):
    d = torch.load(f"{qdir}/L{i}.pt"); t = {}
    for n in linn:
        e = d[n]; q = QLinear(e["scales"].shape[0], e["codes"].shape[1] * 2, g, e["L"], Qz.Hb)
        q.codes = e["codes"].cuda(); q.scales = e["scales"].cuda(); q.sg = e["sg"].cuda().float(); W = q.weight()
        if FP8:
            sc = (W.float().abs().amax(1, keepdim=True) / 448.0).clamp(min=1e-12); t[f"model.layers.{i}.{n}.weight"] = (W.float() / sc).to(torch.float8_e4m3fn); t[f"model.layers.{i}.{n}.weight_scale"] = sc.to(torch.bfloat16)
        else: t[f"model.layers.{i}.{n}.weight"] = W
    for k in st.wm:
        if k.startswith(f"model.layers.{i}.") and k not in t: t[k] = st.get(k, "cuda")
    put(f"layer{i}.safetensors", t)
rest = {"model.embed_tokens.weight": torch.load(qdir + "/emb.pt"), "model.norm.weight": st.get("model.norm.weight", "cpu")}
if not conf.get("tie_word_embeddings"): rest["lm_head.weight"] = torch.load(qdir + "/head.pt")
put("rest.safetensors", rest)
json.dump(dict(metadata=dict(total_size=0), weight_map=wm), open(out + "/model.safetensors.index.json", "w"))
for f in glob.glob(md + "/*"):
    b = os.path.basename(f)
    if b.endswith((".json", ".txt", ".jinja")) and b != "model.safetensors.index.json": shutil.copy(f, out + "/" + b)
if FP8:
    c = json.load(open(out + "/config.json")); c["quantization_config"] = json.load(open(QCFG)); json.dump(c, open(out + "/config.json", "w"))
print("EXPORT_DONE", out)
