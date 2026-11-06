"""Pack captured quantization (push.py cfg.pack dir) into a real artifact dir. usage: pack_artifact.py PACKDIR OUTDIR
Files: body.bin (ANS streams: weight indices + scale codes), index.json.xz, signs.bin, emb.bin, misc.bin.xz, tok.tar.xz, config.json, predict.py"""
import sys, os, json, lzma, io, glob, tarfile, shutil, numpy as np, torch, constriction
pk, out = sys.argv[1], sys.argv[2]; os.makedirs(out, exist_ok=True)
def ans_enc(sym, nsym):
    cnt = np.bincount(sym, minlength=nsym).astype(np.int64); cnt = np.maximum(cnt, 1)
    m = constriction.stream.model.Categorical((cnt / cnt.sum()).astype(np.float64), perfect=False)
    c = constriction.stream.stack.AnsCoder(); c.encode_reverse(sym.astype(np.int32), m)
    return cnt, c.get_compressed()
files = glob.glob(f"{pk}/*.*.pt")
def key(f):
    b = os.path.basename(f)[:-3]; o, n = b.split(".", 1); return int(o), n
files.sort(key=key)
body = open(f"{out}/body.bin", "wb"); off = 0; idx = []; signs = io.BytesIO(); sgmap = {}
GRP = {"self_attn.q_proj": "qkv", "self_attn.k_proj": "qkv", "self_attn.v_proj": "qkv", "self_attn.o_proj": "o", "mlp.gate_proj": "gu", "mlp.up_proj": "gu", "mlp.down_proj": "d"}
ent_bits = 0; sc_bits = 0; nw = 0
for f in files:
    o, n = key(f); d = torch.load(f); R, C = d["shape"]; K = d["K"]; g = d["g"]
    wi = d["idx"].numpy().reshape(-1); si = d["sc"].numpy().reshape(-1)
    cw, zw = ans_enc(wi, 2 * K + 1); cs, zs = ans_enc(si, 256)
    gk = (o, GRP[n])
    if gk not in sgmap:
        sgmap[gk] = signs.tell(); signs.write(np.packbits(d["sg"].numpy().astype(np.uint8)).tobytes())
    e = dict(name=f"{o}.{n}", R=R, C=C, g=g, K=K, cw=cw.tolist(), cs=cs.tolist(), ow=off, lw=len(zw), os=off + len(zw) * 4, ls=len(zs), sgoff=sgmap[gk])
    body.write(zw.tobytes()); body.write(zs.tobytes()); off += (len(zw) + len(zs)) * 4
    idx.append(e); ent_bits += len(zw) * 32; sc_bits += len(zs) * 32; nw += R * C
    print(e["name"], len(zw) * 32 / (R * C), flush=True)
body.close()
open(f"{out}/signs.bin", "wb").write(signs.getvalue())
mc = torch.load(f"{pk}/misc.pt",weights_only=False)
keep = mc["keep"].numpy().astype(np.int64)
ec = mc["ecode"][keep].numpy().astype(np.uint8); es = mc["escale"][keep].numpy().astype(np.float16)
H = ec.shape[1]; pk4 = (ec[:, 0::2] | (ec[:, 1::2] << 4)).astype(np.uint8)
open(f"{out}/emb.bin", "wb").write(pk4.tobytes() + es.tobytes())
miscmeta = {}; blob = io.BytesIO()
for k, v in mc["misc"].items():
    a = v.to(torch.bfloat16).view(torch.int16).numpy(); miscmeta[k] = [list(v.shape), blob.tell()]; blob.write(a.tobytes())
Wl = mc["Wl"].to(torch.bfloat16).view(torch.int16).numpy(); miscmeta["__Wl"] = [list(Wl.shape), blob.tell()]; blob.write(Wl.tobytes())
open(f"{out}/misc.bin.xz", "wb").write(lzma.compress(blob.getvalue(), preset=9))
meta = dict(model=mc["model"], n_keep=len(keep), H=H, keep=np.diff(keep, prepend=0).tolist(), misc=miscmeta, bias=mc["bias"], orig_idx=mc["orig_idx"], letter_ids=mc["letter_ids"],
            nwords=len(idx), mods=idx, ecfg=mc["cfg"].get("ec"), g=mc["cfg"]["g"], vocab_cfg=mc["cfg"].get("vocab"))
open(f"{out}/index.json.xz", "wb").write(lzma.compress(json.dumps(meta, separators=(",", ":")).encode(), preset=9))
# tokenizer + config
from transformers import AutoTokenizer, AutoConfig
tok = AutoTokenizer.from_pretrained(mc["model"]); td = f"{pk}/_tok"; tok.save_pretrained(td)
keepf = [f for f in os.listdir(td) if f in ("tokenizer.json", "tokenizer_config.json")]
with tarfile.open(f"{out}/tok.tar.xz", "w:xz") as t:
    for f in keepf: t.add(f"{td}/{f}", arcname=f)
cfgj = AutoConfig.from_pretrained(mc["model"]).to_dict(); cfgj["num_hidden_layers"] = len(mc["orig_idx"])
json.dump(cfgj, open(f"{out}/config.json", "w"), separators=(",", ":"))
shutil.copy(os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifact_src", "predict.py"), f"{out}/predict.py")
tot = 0
for f in sorted(os.listdir(out)):
    s = os.path.getsize(f"{out}/{f}"); tot += s; print(f"{s:>14,d} {f}")
print("TOTAL", tot, "body_bpw_actual(w only)", ent_bits / nw, "scale bits/w", sc_bits / nw)
