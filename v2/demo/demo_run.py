"""REAL demo data for the explainer video. DEV rows only (never locked). Launch: cd v2/demo && modal run --detach demo_run.py
Writes demo_outputs.json to Volume tenpct-demo:/demo_outputs.json and prints RESULT_JSON:<json>.
Base = bf16 Qwen2.5-7B-Instruct, frozen zero-shot letter prompt, argmax over 5 letter logits (same as harness task_base.py).
Ours = Volume tenpct-c2fb:/art_c2fb loaded through its own predict.py (stage 1 = no memory); feedback = memory_newtool_k5.json."""
import json, os, sys, time, random
import modal
HERE = os.path.dirname(os.path.abspath(__file__)); V2 = os.path.dirname(HERE)
app = modal.App("tenpct-demo-run")
img = (modal.Image.debian_slim().pip_install("torch", "transformers==4.51.3", "accelerate", "numpy", "safetensors", "scipy", "constriction", "huggingface_hub")
       .add_local_file(f"{V2}/harness/common.py", "/h/common.py")
       .add_local_file(f"{V2}/commercial/C2/dev.jsonl", "/h/dev.jsonl")
       .add_local_file(f"{V2}/c2_fb/feedback_sets/ann_newtool_k5.json", "/h/ann_newtool_k5.json")
       .add_local_file(f"{V2}/c2_fb/feedback_sets/memory_newtool_k5.json", "/h/memory_newtool_k5.json")
       .add_local_file(f"{V2}/c2_fb/feedback_sets/feedback_plan.json", "/h/feedback_plan.json"))
art_vol = modal.Volume.from_name("tenpct-c2fb")
out_vol = modal.Volume.from_name("tenpct-demo", create_if_missing=True)
hf = modal.Volume.from_name("tenpct-v2-hf", create_if_missing=True)
BASE = "Qwen/Qwen2.5-7B-Instruct"; TREE = "6290414d96f3119799516442cab5106243dede6a5b29256f24669f0581ee9661"
LET = "ABCDEFGHIJKLMN"
NONE_TXT = "None of these tools can do this"

def softmax(x):
    import numpy as np
    x = np.asarray(x, np.float64); x = x - x.max(); e = np.exp(x); return (e / e.sum()).tolist()
def tname(o): return o if o.startswith("None of") else o.split(":")[0].strip()

@app.function(image=img, gpu=["A100-40GB", "L40S", "A10G"], timeout=7200, volumes={"/art": art_vol, "/out": out_vol, "/hf": hf}, env={"HF_HOME": "/hf"})
def run():
    sys.dont_write_bytecode = True; sys.path.insert(0, "/h")
    import numpy as np, torch, importlib.util, shutil
    import common as C
    art = "/art/art_c2fb"
    h = C.tree_sha256(art); print("tree hash", h, flush=True)
    assert h == TREE, "artifact tree hash mismatch"
    body = C.sha256_file(f"{art}/body.bin"); print("body sha", body); assert body.startswith("0304a803")
    abytes = sum(os.path.getsize(p) for _, p in C.tree_files(art))
    rows = [json.loads(l) for l in open("/h/dev.jsonl")]
    for r in rows: r.pop("tools", None); r.pop("gold_calls", None)
    y = np.array([r["answer"] for r in rows]); n = len(rows)
    # ---- base ----
    from transformers import AutoTokenizer, AutoModelForCausalLM
    def fmt(r):
        s = r["inst"] + "\n\nText: " + r["text"] + "\n\nOptions:\n"
        for l, c in zip(LET, r["names"]): s += f"{l}. {c}\n"
        return s + "Answer:"
    tok = AutoTokenizer.from_pretrained(BASE); m = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16).cuda().eval()
    ids = [tok.encode(" " + l)[0] for l in LET]; BL = np.zeros((n, 5), np.float32)
    with torch.no_grad():
        for i, r in enumerate(rows):
            x = tok.encode(fmt(r))[-1024:]; BL[i] = m(torch.tensor([x]).cuda()).logits[0, -1, ids[:5]].float().cpu().numpy()
    del m; torch.cuda.empty_cache()
    bacc = float((BL.argmax(1) == y).mean()); print("base dev acc", bacc, flush=True)
    # ---- ours ----
    shutil.copy(f"{art}/predict.py", "/tmp/predict_art.py")
    spec = importlib.util.spec_from_file_location("predict_art", "/tmp/predict_art.py"); P = importlib.util.module_from_spec(spec); spec.loader.exec_module(P)
    t0 = time.time(); f = P.load(art, None); load_secs = time.time() - t0; print("load secs", load_secs, flush=True)
    LG, _, FO = f.build["run"](rows); OL = LG[:, :5]
    oacc = float((OL.argmax(1) == y).mean()); print("ours dev acc", oacc, flush=True)
    bp, op = BL.argmax(1), OL.argmax(1)
    # ---- examples ----
    plan = json.load(open("/h/feedback_plan.json"))
    def ok_row(i):
        r = rows[i]; return len(r["text"]) <= 140 and all(len(tname(o)) <= 26 for o in r["names"][:4])
    elig = [i for i in range(n) if ok_row(i)]
    def pick(cond, k, taken, seed):
        c = [i for i in elig if cond(i) and i not in taken]; random.Random(seed).shuffle(c); return c[:k]
    taken = set(); sel = []
    for cond, k, sd in [(lambda i: bp[i] == y[i] and op[i] == y[i] and y[i] != 4, 3, 1),
                        (lambda i: op[i] == y[i] and bp[i] != y[i], 2, 2),
                        (lambda i: bp[i] == y[i] and op[i] != y[i], 2, 3),
                        (lambda i: y[i] == 4 and bp[i] == 4 and op[i] == 4, 1, 4)]:
        c = pick(cond, k, taken, sd); sel += c; taken |= set(c)
    print("pool sizes ok; selected", sel, flush=True)
    def ex(i):
        r = rows[i]; return dict(id=r.get("id", f"C2-dev-{i}"), query=r["text"], options=[tname(o) for o in r["names"]], gold_idx=int(y[i]),
                                 base=dict(pred=int(bp[i]), probs=softmax(BL[i])), ours=dict(pred=int(op[i]), probs=softmax(OL[i])))
    examples = [ex(i) for i in sel]
    # ---- feedback ----
    ann = json.load(open("/h/ann_newtool_k5.json")); tool = plan["new_tool"]
    H = P.Head(art); Mm = P.Mem("/h/memory_newtool_k5.json"); mem_bytes = os.path.getsize("/h/memory_newtool_k5.json")
    U = H.proj(FO); G = H.gscore(U)
    AS = np.stack([P._scores(H, Mm, LG[i], U[i], G[i], rows[i]["names"]) for i in range(n)])
    apred = AS.argmax(1); aacc = float((apred == y).mean())
    # also verify through real load(art, memory) on the new-tool items
    didx = plan["new_tool_dev_idx"]; f2 = P.load(art, "/h/memory_newtool_k5.json", dev="cuda")
    real_after = f2([rows[i] for i in didx]); assert list(real_after) == [int(apred[i]) for i in didx], "load(memory) disagrees with manual scoring"
    print("dev acc with newtool_k5 memory", aacc, "new-tool items before/after", [(int(op[i]), int(apred[i]), int(y[i])) for i in didx], flush=True)
    short = [i for i in didx if len(rows[i]["text"]) <= 200]
    short.sort(key=lambda i: (not (op[i] != y[i] and apred[i] == y[i]), len(rows[i]["text"])))   # prefer items the feedback fixes, then short
    test_i = (short + [i for i in didx if i not in short])[:3]
    test = [dict(query=rows[i]["text"], options=[tname(o) for o in rows[i]["names"]], gold_idx=int(y[i]), before_pred=int(op[i]), after_pred=int(apred[i]),
                 before_probs=softmax(OL[i]), after_probs=softmax(AS[i])) for i in test_i]
    out = dict(meta=dict(base_model=BASE, artifact_bytes=int(abytes), base_bytes=15231233024, artifact_load_secs=load_secs, dev_acc_base=bacc, dev_acc_ours=oacc, n_dev=n),
               examples=examples,
               feedback=dict(tool=tool, annotations=[dict(query=a["text"], correct_option_text=tname(a["names"][a["answer"]])) for a in ann],
                             memory_bytes=int(mem_bytes), test=test))
    out["_extra_not_in_schema"] = dict(dev_acc_ours_with_memory=aacc, newtool_dev_items=len(didx), newtool_before_correct=int(sum(op[i] == y[i] for i in didx)),
                                       newtool_after_correct=int(sum(apred[i] == y[i] for i in didx)), tree_sha256=h, body_sha256=body, gpu=torch.cuda.get_device_name(0))
    json.dump(out, open("/out/demo_outputs.json", "w"), indent=1); out_vol.commit()
    print("RESULT_JSON:" + json.dumps(out)); return out

@app.local_entrypoint()
def main():
    c = run.spawn(); print("spawned", c.object_id)
