"""Build the real compressed artifact for Ternary-Bonsai-2-27B (lossy only in: group-scale log-quantisation (K steps/octave), vocab prefix truncation).
usage: art_build.py OUTDIR K NV"""
import os, sys, json, lzma, shutil, numpy as np, torch
os.environ["HF_HOME"] = "/home/ubuntu/work/hf"
from huggingface_hub import snapshot_download
from safetensors import safe_open
out, K, NV = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]); os.makedirs(out, exist_ok=True)
SPEC0, SPEC1 = 248044, 248077
keep_rows = np.array(list(range(NV)) + list(range(SPEC0, SPEC1))) if NV else None
src = snapshot_download("prism-ml/Ternary-Bonsai-2-27B-mlx-2bit", allow_patterns=["model.safetensors"])
f = safe_open(src + "/model.safetensors", "pt"); keys = list(f.keys())
names = sorted(set(k.rsplit(".", 1)[0] for k in keys if k.endswith(".scales")))
trit_f = open(out + "/trits.bin", "wb"); rowbase = []; qbytes = []; index = []; n_trit = 0
sh = torch.arange(0, 32, 2, dtype=torch.int32)
for base in names:
    w = f.get_tensor(base + ".weight"); s = f.get_tensor(base + ".scales").float()
    w = w.view(torch.int32) if w.dtype != torch.int32 else w
    if keep_rows is not None and (base.endswith("embed_tokens") or base.endswith("lm_head")):
        kr = torch.from_numpy(keep_rows); w = w[kr]; s = s[kr]
    rows = w.shape[0]; assert (s > 0).all(), base
    codes = ((w[:, :, None] >> sh) & 3).reshape(-1).to(torch.uint8); assert int(codes.max()) <= 2
    pad = (-codes.numel()) % 5
    if pad: codes = torch.cat([codes, torch.zeros(pad, dtype=torch.uint8)])
    c5 = codes.reshape(-1, 5).to(torch.int32); packed = (c5[:, 0] + 3 * c5[:, 1] + 9 * c5[:, 2] + 27 * c5[:, 3] + 81 * c5[:, 4]).to(torch.uint8)
    trit_f.write(packed.numpy().tobytes()); n_trit += rows * w.shape[1] * 16
    L = torch.log2(s); med = L.median(dim=1).values.half(); q = torch.round((L - med.float()[:, None]) * K).clamp(-127, 127).to(torch.int8)
    rowbase.append(med.numpy().tobytes()); qbytes.append(q.numpy().tobytes()); index.append((base, rows, int(w.shape[1] * 16), s.shape[1]))
trit_f.close()
open(out + "/rowbase.bin", "wb").write(b"".join(rowbase))
open(out + "/scales.q.xz", "wb").write(lzma.compress(b"".join(qbytes), format=lzma.FORMAT_XZ, preset=9 | lzma.PRESET_EXTREME))
small = {}; sdt = {}
for k in keys:
    if k.endswith((".scales", ".biases", ".signs")) or k.startswith("vision_tower"): continue
    if k.rsplit(".", 1)[0] in names and k.endswith(".weight"): continue
    t = f.get_tensor(k)
    if t.dtype in (torch.uint32, torch.int32): continue
    sdt[k] = str(t.dtype)
    if t.dtype == torch.float32 and torch.equal(t.half().float(), t): t = t.half()
    small[k] = t
torch.save(small, out + "/small.pt")
json.dump({"K": K, "NV": NV, "index": index, "n_trit": n_trit, "small_dtype": sdt}, open(out + "/index.json", "w"))
# tokenizer (trimmed)
psrc = snapshot_download("Qwen/Qwen3.8-27B", allow_patterns=["tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "config.json", "generation_config.json", "vocab.json", "merges.txt"])
if NV:
    tj = json.load(open(psrc + "/tokenizer.json")); m = tj["model"]
    vocab = {t: i for t, i in m["vocab"].items() if i < NV}; merges = [mg for mg in m["merges"] if (lambda a, b: a in vocab and b in vocab and (a + b) in vocab)(*(mg if isinstance(mg, list) else mg.split(" ")))]
    sp = {}
    for a in tj["added_tokens"]: sp[a["id"]] = a
    new_added = []; mp = {}
    for j, oid in enumerate(range(SPEC0, SPEC1)):
        a = dict(sp[oid]); a["id"] = NV + j; mp[oid] = NV + j; new_added.append(a)
    tj["added_tokens"] = new_added; m["vocab"] = vocab; m["merges"] = merges
    json.dump(tj, open(out + "/tokenizer.json", "w"), ensure_ascii=False)
    json.dump({"vocab_map": {str(k): v for k, v in mp.items()}, "NV": NV}, open(out + "/vocabmap.json", "w"))
    for fn in ["tokenizer_config.json", "chat_template.jinja", "config.json", "generation_config.json"]: shutil.copyfile(psrc + "/" + fn, out + "/" + fn)
else:
    for fn in ["tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "config.json", "generation_config.json"]: shutil.copyfile(psrc + "/" + fn, out + "/" + fn)
tot = {fn: os.path.getsize(out + "/" + fn) for fn in sorted(os.listdir(out))}
FP16 = 2 * (26869760000 + 26.2e6)   # fp16 bytes of the LM (ternary params + small), see survey
print("BYTES", json.dumps(tot)); print("TOTAL", sum(tot.values()), "pct_of_fp16", 100 * sum(tot.values()) / 53808408928)
