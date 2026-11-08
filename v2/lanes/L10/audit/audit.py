"""L10 denominator audit: real safetensors headers (range requests), split text-path vs rest. No token needed (ungated mirrors)."""
import json, struct, sys, re, urllib.request, collections
def get(url, rng=None):
    h = {"User-Agent": "x"}
    if rng: h["Range"] = rng
    return urllib.request.urlopen(urllib.request.Request(url, headers=h), timeout=60).read()
def api(rid): return json.loads(get(f"https://huggingface.co/api/models/{rid}?blobs=true"))
DT = {"BF16":2,"F16":2,"F32":4,"F8_E4M3":1,"I8":1,"U8":1,"I32":4,"I64":8,"F64":8,"BOOL":1}
def tensors(rid):
    info = api(rid); out = {}
    files = [s["rfilename"] for s in info["siblings"] if s["rfilename"].endswith(".safetensors")]
    for f in files:
        u = f"https://huggingface.co/{rid}/resolve/main/{f}"
        n = struct.unpack("<Q", get(u, "bytes=0-7"))[0]
        hd = json.loads(get(u, f"bytes=8-{8+n-1}"))
        for k, v in hd.items():
            if k == "__metadata__": continue
            c = 1
            for d in v["shape"]: c *= d
            out[k] = (c, v["dtype"])
    return out, info
def classify(name, rid):
    # returns 'text' / 'cross' / 'other'
    n = name
    if re.search(r"vision|visual|audio|talker|token2wav|code2wav|image|siglip|speech|mm_projector|multi_modal|projector|patch_embed|vit|thinker\.visual", n) and "language_model" not in n.replace("vision_model","") : return "other"
    return "text"
CANDS = json.load(open(sys.argv[1]))
res = []
for c in CANDS:
    rid = c["id"]
    try: T, info = tensors(rid)
    except Exception as e: res.append(dict(id=rid, error=repr(e)[:120])); print(rid, "ERR", e); continue
    tot = sum(n * 2 for n, d in T.values())          # fp16-equivalent bytes of every tensor (2 B/param)
    disk = sum(s.get("size") or 0 for s in info["siblings"] if s["rfilename"].endswith(".safetensors"))
    pre = c["text_re"]; ex = c.get("exclude_re")
    txt = 0; vis = 0; xat = 0
    for k, (n, d) in T.items():
        if re.search(pre, k) and not (ex and re.search(ex, k)): txt += n
        else: vis += n
    xl = set(re.search(r"layers\.(\d+)\.", k).group(1) for k in T if "cross_attn" in k and "layers." in k)
    xa = sum(n for k, (n, d) in T.items() if re.search(pre, k) and (m := re.search(r"layers\.(\d+)\.", k)) and m.group(1) in xl)
    r = dict(id=rid, total_params=sum(n for n, d in T.values()), disk_bytes=disk, fp16_bytes=tot, text_params_incl_xattn=txt, xattn_params=xa, text_params_used=txt - xa, nontext_params=vis,
             budget_bytes=tot * 0.10)
    r["bpw_used"] = r["budget_bytes"] * 8 / r["text_params_used"]
    r["bpw_incl_xattn"] = r["budget_bytes"] * 8 / txt
    prefixes = collections.Counter(".".join(k.split(".")[:2]) for k in T)
    r["top_prefixes"] = {p: sum(n for k, (n, d) in T.items() if ".".join(k.split(".")[:2]) == p) for p in prefixes}
    res.append(r); print(rid, {k: (round(v/1e9, 3) if isinstance(v, (int, float)) and k != "bpw_used" and k != "bpw_incl_xattn" else v) for k, v in r.items() if k != "top_prefixes"}, flush=True)
json.dump(res, open(sys.argv[2], "w"), indent=1)
