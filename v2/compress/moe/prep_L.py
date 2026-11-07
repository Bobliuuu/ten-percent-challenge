"""Bigger labelled item-level eval set evalL: ag_news (>=300), sst2 (>=300), hermes tool-call with gold (>=200); disjoint from calibA/evalA/evalB by prompt hash. -> volume moe_data/evalL.jsonl"""
import modal, json, random, hashlib
app = modal.App("tenpct-moe-prepL")
img = modal.Image.debian_slim().pip_install("datasets==3.6.0", "huggingface_hub", "pyarrow")
vol = modal.Volume.from_name("tenpct-v2-hf", create_if_missing=True)
@app.function(image=img, volumes={"/hf": vol}, timeout=1800)
def prep():
    import os; os.environ["HF_HOME"] = "/hf"
    from datasets import load_dataset
    used = set()
    for n in ("evalA", "calibA", "evalB", "calibB"):
        for l in open(f"/hf/moe_data/{n}.jsonl"): used.add(hashlib.md5(json.dumps(json.loads(l)["msgs"]).encode()).hexdigest())
    R = random.Random(99); rows = []
    def add(msgs, gold, cat):
        h = hashlib.md5(json.dumps(msgs).encode()).hexdigest()
        if h in used: return False
        rows.append(dict(cat=cat, msgs=msgs, gold=gold)); return True
    lab = ["World", "Sports", "Business", "Science/Technology"]
    d = load_dataset("fancyzhx/ag_news", split="test"); idx = list(range(len(d))); R.shuffle(idx); n = 0
    for i in idx:
        r = d[i]
        if add([{"role": "user", "content": "Classify the news article into one of: World, Sports, Business, Science/Technology. Answer with the label only.\n\n" + r["text"]}], lab[r["label"]], "sumcls"): n += 1
        if n >= 400: break
    d = load_dataset("stanfordnlp/sst2", split="validation"); idx = list(range(len(d))); R.shuffle(idx); n = 0
    for i in idx:
        r = d[i]
        if add([{"role": "user", "content": "What is the sentiment of this review (positive or negative)? Answer with one word.\n\n" + r["sentence"]}], "positive" if r["label"] else "negative", "sumcls"): n += 1
        if n >= 400: break
    d = load_dataset("NousResearch/hermes-function-calling-v1", "func_calling_singleturn", split="train"); idx = list(range(len(d))); R.shuffle(idx); n = 0
    for i in idx:
        c = d[i]["conversations"]
        try:
            sysm = [x for x in c if x["from"] == "system"][0]["value"]; hum = [x for x in c if x["from"] == "human"][0]["value"]; gpt = [x for x in c if x["from"] == "gpt"][0]["value"]
        except Exception: continue
        if len(sysm.split()) > 700 or "<tool_call>" not in gpt: continue
        if add([{"role": "system", "content": sysm}, {"role": "user", "content": hum}], gpt, "json"): n += 1
        if n >= 300: break
    with open("/hf/moe_data/evalL.jsonl", "w") as f:
        for j, r in enumerate(rows): r["id"] = f"evalL{j}"; f.write(json.dumps(r) + "\n")
    print({c: sum(1 for r in rows if r["cat"] == c) for c in ("sumcls", "json")}, len(rows)); vol.commit()
@app.local_entrypoint()
def main(): prep.remote()
