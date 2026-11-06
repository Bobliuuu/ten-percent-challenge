"""Build eval / calibration prompt sets from public HF datasets (runs on Modal CPU). Writes to volume tenpct-v2-hf:/moe_data/*.jsonl
Sets: evalA (400, mix chat/json/sumcls/pii), calibA (same mix, disjoint), calibB (domain-shifted: math/code/multilingual), evalB (120 shifted held-out).
Each row: {id, cat, msgs:[{role,content}], gold}   gold = dataset reference response (used for calib text + teacher-forced text)."""
import modal, json, random, hashlib
app = modal.App("tenpct-moe-prep")
img = modal.Image.debian_slim().pip_install("datasets==3.6.0", "huggingface_hub", "pyarrow")
vol = modal.Volume.from_name("tenpct-v2-hf", create_if_missing=True)

def trunc(s, n): w = s.split(); return " ".join(w[:n])

@app.function(image=img, volumes={"/hf": vol}, timeout=3600, cpu=4)
def prep():
    import os
    os.environ["HF_HOME"] = "/hf"
    from datasets import load_dataset
    R = random.Random(1234); pools = {}; log = []
    def load(name, *a, **k):
        try:
            d = load_dataset(name, *a, **k); log.append(f"OK {name} {a} n={len(d)}"); return d
        except Exception as e:
            log.append(f"FAIL {name} {a}: {repr(e)[:200]}"); return None
    def add(cat, rows): R.shuffle(rows); pools.setdefault(cat, []).extend(rows)
    # --- chat Q&A
    d = load("databricks/databricks-dolly-15k", split="train")
    if d:
        rows = []
        for r in d:
            if r["category"] in ("open_qa", "general_qa", "brainstorming", "classification", "information_extraction", "closed_qa") and len(r["response"].split()) > 3:
                u = r["instruction"] + (("\n\n" + trunc(r["context"], 250)) if r["context"] else "")
                rows.append(dict(cat="chat", msgs=[{"role": "user", "content": u}], gold=trunc(r["response"], 200)))
        add("chat", rows)
    # --- JSON / function-call formatting
    d = load("NousResearch/hermes-function-calling-v1", "func_calling_singleturn", split="train")
    rows = []
    if d:
        for r in d:
            c = r["conversations"]
            try:
                sysm = [x for x in c if x["from"] == "system"][0]["value"]; hum = [x for x in c if x["from"] == "human"][0]["value"]
                gpt = [x for x in c if x["from"] == "gpt"][0]["value"]
            except Exception: continue
            if len(sysm.split()) > 700: continue
            rows.append(dict(cat="json", msgs=[{"role": "system", "content": sysm}, {"role": "user", "content": hum}], gold=gpt))
    if len(rows) < 300:
        d2 = load("glaiveai/glaive-function-calling-v2", split="train")
        if d2:
            import re
            for r in d2.select(range(0, min(len(d2), 6000))):
                m = re.search(r"USER: (.*?)\n\n\nASSISTANT: (.*?)<\|endoftext\|>", r["chat"], re.S)
                if m and "functioncall" in m.group(2):
                    rows.append(dict(cat="json", msgs=[{"role": "system", "content": r["system"]}, {"role": "user", "content": m.group(1)}], gold=m.group(2)))
    # also JSON-formatting from dolly extraction: ask for JSON output
    add("json", rows)
    # --- summarisation / classification
    rows = []
    d = load("abisee/cnn_dailymail", "3.0.0", split="test")
    if d:
        for r in d.select(range(0, 3000)):
            rows.append(dict(cat="sumcls", msgs=[{"role": "user", "content": "Summarize the following article in two sentences.\n\n" + trunc(r["article"], 330)}], gold=r["highlights"]))
    d = load("fancyzhx/ag_news", split="test")
    if d:
        lab = ["World", "Sports", "Business", "Science/Technology"]
        for r in d.select(range(0, 3000)):
            rows.append(dict(cat="sumcls", msgs=[{"role": "user", "content": "Classify the news article into one of: World, Sports, Business, Science/Technology. Answer with the label only.\n\n" + r["text"]}], gold=lab[r["label"]]))
    d = load("stanfordnlp/sst2", split="validation")
    if d:
        for r in d:
            rows.append(dict(cat="sumcls", msgs=[{"role": "user", "content": "What is the sentiment of this review (positive or negative)? Answer with one word.\n\n" + r["sentence"]}], gold="positive" if r["label"] else "negative"))
    add("sumcls", rows)
    # --- PII rewriting
    rows = []
    for nm in ("ai4privacy/pii-masking-200k", "ai4privacy/pii-masking-300k"):
        d = load(nm, split="train")
        if d:
            for r in d.select(range(0, min(len(d), 8000))):
                if r.get("language", "English") not in ("English", "en"): continue
                if not (r.get("masked_text") or r.get("target_text")): continue
                if len(r["source_text"].split()) > 200: continue
                rows.append(dict(cat="pii", msgs=[{"role": "user", "content": "Rewrite the following text, replacing every piece of personally identifiable information (names, emails, phone numbers, addresses, ids) with a bracketed placeholder such as [NAME] or [EMAIL].\n\n" + r["source_text"]}], gold=r.get("masked_text") or r.get("target_text") or ""))
            break
    add("pii", rows)
    # --- shifted domain B: math / code / multilingual
    rows = []
    d = load("openai/gsm8k", "main", split="train")
    if d:
        for r in d: rows.append(dict(cat="math", msgs=[{"role": "user", "content": r["question"]}], gold=r["answer"]))
    add("math", rows); rows = []
    d = load("iamtarun/python_code_instructions_18k_alpaca", split="train")
    if d:
        for r in d:
            if len(r["output"].split()) < 220:
                rows.append(dict(cat="code", msgs=[{"role": "user", "content": r["instruction"] + (("\n\n" + r["input"]) if r["input"] else "")}], gold=r["output"]))
    add("code", rows); rows = []
    d = load("papluca/language-identification", split="train")
    if d:
        for r in d:
            if r["labels"] != "en" and 8 < len(r["text"].split()) < 150:
                rows.append(dict(cat="multi", msgs=[{"role": "user", "content": "Translate the following text to English.\n\n" + r["text"]}], gold=""))
    add("multi", rows)
    print("\n".join(log)); print({k: len(v) for k, v in pools.items()})
    # --- dedupe + disjoint splits
    seen = set(); 
    for k in pools:
        out = []
        for r in pools[k]:
            h = hashlib.md5(json.dumps(r["msgs"]).encode()).hexdigest()
            if h in seen: continue
            seen.add(h); out.append(r)
        pools[k] = out
    def take(cat, n):
        p = pools.get(cat, []); out, pools[cat] = p[:n], p[n:]; return out
    evalA = take("chat", 110) + take("json", 90) + take("sumcls", 100) + take("pii", 100)
    calA = take("chat", 130) + take("json", 110) + take("sumcls", 120) + take("pii", 120)
    evalB = take("math", 40) + take("code", 40) + take("multi", 40)
    calB = take("math", 170) + take("code", 160) + take("multi", 150)
    os.makedirs("/hf/moe_data", exist_ok=True)
    for nm, rows in (("evalA", evalA), ("calibA", calA), ("evalB", evalB), ("calibB", calB)):
        for i, r in enumerate(rows): r["id"] = f"{nm}{i}"
        with open(f"/hf/moe_data/{nm}.jsonl", "w") as f:
            for r in rows: f.write(json.dumps(r) + "\n")
        print(nm, len(rows), {c: sum(1 for r in rows if r["cat"] == c) for c in set(r["cat"] for r in rows)})
    open("/hf/moe_data/prep.log", "w").write("\n".join(log))
    vol.commit()

@app.local_entrypoint()
def main(): prep.remote()
