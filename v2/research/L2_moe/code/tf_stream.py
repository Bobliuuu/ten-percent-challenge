"""Teacher-forced layer-streaming predictor for retention. usage:
 tf_stream.py SNAP SAL.pt ITEMS.jsonl OUT.json GPU  cfg1,cfg2,...     cfg = base | METHOD:PRUNEFRAC[:BITS]  (e.g. reap:0.5:4 ; freq:0.5 ; reap:0.5:3)
Per layer loads weights once, then advances every config's hidden stream (mask-pruned + RTN fake-quant of expert weights if BITS). Final: per item/token argmax agreement
with the gold native-format tool-call tokens (teacher forced). Item 'ok' = all response tokens argmax==gold (call items) / first token != <tool_call> (nocall items).
Retention = P(ok under cfg | ok under base). Dev items only."""
import sys, json, torch, time, os
sys.path.insert(0, "/opt/dlami/nvme/work/l2/code"); sys.path.insert(0, "/opt/dlami/nvme/work/l2/data")
from moe_lib import *
from pruned_loader import rtn_fake
from gptq import gptq, gptq_batched
import transformers.models.qwen3_moe.modeling_qwen3_moe as M
from transformers import AutoTokenizer, AutoConfig
from safetensors import safe_open
snap, salf, itemsf, outf, gpu, cfgs = sys.argv[1:7]
cfgs = cfgs.split(",")
dev = torch.device(f"cuda:{gpu}"); torch.cuda.set_device(dev)
cfg = AutoConfig.from_pretrained(snap); cfg._attn_implementation = "sdpa"; tok = AutoTokenizer.from_pretrained(snap)
items = [json.loads(l) for l in open(itemsf)]
sal = torch.load(salf) if salf != "none" else None
L, E = cfg.num_hidden_layers, cfg.num_experts
TC = tok.convert_tokens_to_ids("<tool_call>")
seqs = []
LABEL = os.environ.get("MODE") == "label"
if LABEL:
    items = [dict(it, source=it["role"], gold=[1]) for it in items if it["task"] in os.environ.get("TASKS", "INJ,SENT,TOOLSEL,FIN,ROUTE").split(",")]
for it in items:
    if LABEL:
        pr = tok.apply_chat_template([{"role": "user", "content": it["user"]}], add_generation_prompt=True, tokenize=False)
        seqs.append((tok(pr, add_special_tokens=False).input_ids, tok(it["answer"], add_special_tokens=False).input_ids + [tok.convert_tokens_to_ids("<|im_end|>")])); continue
    p = tok(chat_prompt(tok, it), add_special_tokens=False).input_ids; r = tok(gold_qwen(it) + "<|im_end|>", add_special_tokens=False).input_ids
    seqs.append((p, r))
order = sorted(range(len(items)), key=lambda i: len(seqs[i][0]) + len(seqs[i][1]))
BS = int(os.environ.get("BS", 12))
batches = []; cur = []
for i in order:
    T = len(seqs[i][0]) + len(seqs[i][1])
    if cur and (len(cur) >= BS or (len(cur) + 1) * T > 8000): batches.append(cur); cur = []
    cur.append(i)
if cur: batches.append(cur)
def mk_mask(spec):
    if spec == "base": return None
    a = spec.split(":"); method, frac = a[0], float(a[1]); m = {}
    if frac == 0: return None
    g = torch.Generator().manual_seed(0)
    if method in ("greap", "gfreq"):   # global (layerwise-variable) allocation: prune experts with lowest per-layer-normalised mass share anywhere
        mass = (sal["cnt"] * sal["reap"]) if method == "greap" else sal["cnt"].clone()
        share = mass / mass.sum(1, keepdim=True)
        thr = share.flatten().sort().values[int(round(frac * L * E))]
        pr = share < thr
        for l in range(L): m[l] = pr[l].to(dev)
        print("global alloc kept/layer min,max:", int((~pr).sum(1).min()), int((~pr).sum(1).max()), flush=True)
        return m
    for l in range(L):
        k = int(round(E * frac))
        sc = {"freq": sal["cnt"][l], "reap": sal["reap"][l], "ean": sal["ean"][l], "gate": sal["gatesum"][l], "rand": torch.rand(E, generator=g)}[method].clone()
        if method in ("reap", "ean"): sc[sal["cnt"][l] == 0] = -1.0
        pr = torch.zeros(E, dtype=torch.bool); pr[sc.argsort()[:k]] = True; m[l] = pr.to(dev)
    return m
masks = {c: mk_mask(c) for c in cfgs}
bits = {c: (int(c.split(":")[2].lstrip("g")) if c.count(":") >= 2 else 0) for c in cfgs}
isg = {c: (c.count(":") >= 2 and c.split(":")[2].startswith("g")) for c in cfgs}
GL = os.environ.get("GPTQ_LAYERS", "all"); GL = None if GL == "all" else set(int(x) for x in GL.split(","))
NCAL = int(os.environ.get("NCAL", 96))
wm = json.load(open(f"{snap}/model.safetensors.index.json"))["weight_map"]; files = {}
def get(n):
    f = wm[n]
    if f not in files: files[f] = safe_open(f"{snap}/{f}", "pt", device="cpu")
    return files[f].get_tensor(n)
emb = get("model.embed_tokens.weight").to(dev)
H = {}   # cfg -> list of hidden per batch
for c in cfgs:
    H[c] = []
for b in batches:
    T = max(len(seqs[i][0]) + len(seqs[i][1]) for i in b)
    ids = torch.full((len(b), T), tok.pad_token_id or 0, dtype=torch.long)
    for j, i in enumerate(b):
        s = seqs[i][0] + seqs[i][1]; ids[j, :len(s)] = torch.tensor(s)
    h = torch.nn.functional.embedding(ids.to(dev), emb).bfloat16()
    for c in cfgs: H[c].append(h.clone())
HC = {}
if any(isg.values()):
    cal = [json.loads(l) for l in open("/opt/dlami/nvme/work/l2/data/s1_calib.jsonl")]
    import random; random.Random(1).shuffle(cal); cal = cal[:NCAL]
    ctexts = sorted([chat_prompt(tok, it) + gold_qwen(it) + "<|im_end|>" for it in cal], key=len)
    gen_t = [json.loads(l)["text"] for l in open("/opt/dlami/nvme/work/l2/data/general.jsonl")][-24:]
    ctexts = sorted(ctexts + gen_t, key=len); cb = []
    for i in range(0, len(ctexts), 8):
        e = tok(ctexts[i:i+8], return_tensors="pt", padding=True, truncation=True, max_length=2048)
        cb.append((torch.nn.functional.embedding(e.input_ids.to(dev), emb).bfloat16(), e.attention_mask.bool().to(dev)))
    for c in cfgs:
        if isg[c]: HC[c] = [(h.clone(), v) for h, v in cb]
rot = M.Qwen3MoeRotaryEmbedding(cfg, device=dev)
M.Qwen3MoeSparseMoeBlock.forward = fwd
STATE["stats"] = None; STATE["valid"] = None
t0 = time.time()
for l in range(L):
    with torch.device("meta"): lay = M.Qwen3MoeDecoderLayer(cfg, l)
    lay.mlp._li = l; pre = f"model.layers.{l}."
    sd0 = {k[len(pre):]: get(k).to(dev, torch.bfloat16) for k in wm if k.startswith(pre)}
    lay = lay.to_empty(device=dev); lay.eval()
    for c in cfgs:
        sd = sd0; gdone = False
        if isg[c] and (GL is None or l in GL):
            Hs = {}; hooks = []
            def mk(name):
                def hk(m, a):
                    x = a[0].reshape(-1, a[0].shape[-1]).float()
                    if x.shape[0] == 0: return
                    Hs[name] = Hs.get(name, 0) + x.T @ x
                return hk
            lay.load_state_dict(sd0, strict=True); STATE["mask"] = masks[c]
            for e in range(E):
                if masks[c] is not None and bool(masks[c][l][e]): continue
                hooks.append(lay.mlp.experts[e].gate_proj.register_forward_pre_hook(mk(f"{e}.in")))
                hooks.append(lay.mlp.experts[e].down_proj.register_forward_pre_hook(mk(f"{e}.act")))
            with torch.no_grad():
                for (h, valid) in HC[c]:
                    STATE["valid"] = None; pos = torch.arange(h.shape[1], device=dev)[None].expand(h.shape[0], -1)
                    lay(h, attention_mask=None, position_ids=pos, position_embeddings=rot(h, pos))
            for hk in hooks: hk.remove()
            sd = dict(sd0)
            have = [e for e in range(E) if f"{e}.in" in Hs]
            if have:
                Hin = torch.stack([Hs[f"{e}.in"] for e in have]); Hact = torch.stack([Hs[f"{e}.act"] for e in have])
                for pn, Hh in (("gate_proj", Hin), ("up_proj", Hin), ("down_proj", Hact)):
                    Wst = torch.stack([sd0[f"mlp.experts.{e}.{pn}.weight"] for e in have])
                    Qs = torch.cat([gptq_batched(Wst[i:i+16], Hh[i:i+16], bits[c]) for i in range(0, len(have), 16)])
                    for j, e in enumerate(have): sd[f"mlp.experts.{e}.{pn}.weight"] = Qs[j]
                    del Wst, Qs
                del Hin, Hact
            for e in range(E):
                if f"{e}.in" not in Hs:
                    for pn in ("gate_proj", "up_proj", "down_proj"):
                        k = f"mlp.experts.{e}.{pn}.weight"; sd[k] = rtn_fake(sd0[k], bits[c])
            del Hs
            lay.load_state_dict(sd, strict=True)
            with torch.no_grad():
                for bi, (h, valid) in enumerate(HC[c]):
                    pos = torch.arange(h.shape[1], device=dev)[None].expand(h.shape[0], -1)
                    o = lay(h, attention_mask=None, position_ids=pos, position_embeddings=rot(h, pos)); HC[c][bi] = ((o[0] if isinstance(o, tuple) else o), valid)
            gdone = True
        elif bits[c]:
            sd = {k: (rtn_fake(v, bits[c]) if ("experts" in k and v.shape[-1] % 128 == 0) else v) for k, v in sd0.items()}
            gdone = False
        lay.load_state_dict(sd, strict=True)
        if isg[c] and not gdone:
            STATE["mask"] = masks[c]
            with torch.no_grad():
                for bi, (h, valid) in enumerate(HC[c]):
                    pos = torch.arange(h.shape[1], device=dev)[None].expand(h.shape[0], -1)
                    o = lay(h, attention_mask=None, position_ids=pos, position_embeddings=rot(h, pos)); HC[c][bi] = ((o[0] if isinstance(o, tuple) else o), valid)
        STATE["mask"] = masks[c]; STATE["valid"] = None
        with torch.no_grad():
            for bi, b in enumerate(batches):
                h = H[c][bi]; pos = torch.arange(h.shape[1], device=dev)[None].expand(h.shape[0], -1)
                o = lay(h, attention_mask=None, position_ids=pos, position_embeddings=rot(h, pos))
                H[c][bi] = o[0] if isinstance(o, tuple) else o
        del sd
    del lay, sd0; torch.cuda.empty_cache()
    print("layer", l, round(time.time() - t0), flush=True)
normw = get("model.norm.weight").to(dev, torch.bfloat16); head = get("lm_head.weight").to(dev, torch.bfloat16)
res = {c: {} for c in cfgs}
with torch.no_grad():
    for c in cfgs:
        for bi, b in enumerate(batches):
            h = H[c][bi].float(); h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + cfg.rms_norm_eps); h = (h * normw.float()).bfloat16()
            for j, i in enumerate(b):
                p, r = seqs[i]; lg = (h[j, len(p) - 1:len(p) + len(r) - 1] @ head.T).float()
                am = lg.argmax(-1).cpu().tolist(); lp = torch.log_softmax(lg, -1)
                gold = torch.tensor(r, device=dev); nll = (-lp[torch.arange(len(r)), gold]).cpu().tolist()
                it = items[i]
                ok = (am[0] != TC) if (not it["gold"]) else all(a == g for a, g in zip(am, r))
                res[c][it["id"]] = dict(ok=bool(ok), tokacc=sum(a == g for a, g in zip(am, r)) / len(r), nll=sum(nll) / len(r), first_tc=am[0] == TC, kind=it["kind"], source=it["source"], task=it.get("task"), src=it.get("src"), am=am if c == "base" else None)
json.dump(res, open(outf, "w"))
print("done", time.time() - t0)
