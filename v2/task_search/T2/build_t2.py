"""TASK_SEARCH2 (copy/derive tasks) DEV-ONLY builder. Public train/dev rows only, sha256-ordered, no locked rows touched.
Tasks: PIIRED (redact), PIIDET (detect->JSON), NER (->JSON list), LOGTPL (log line -> event template), SQL (schema+question -> SQL), TOOLARG (function call w/ args).
Output items_dev.jsonl (id,task,role,src,kind,user,answer) + calib.jsonl. Licences in LICENCES.json."""
import json, hashlib, collections, ast, csv, io, re, urllib.request, sys
from datasets import load_dataset
H = lambda s: hashlib.sha256(s.encode()).hexdigest()
N = int(sys.argv[1]) if len(sys.argv) > 1 else 300; NC = 40
items, calib = [], []
def add(task, role, src, rows, n, kindcap=None):
    """rows: (rid, kind, user, answer); dedupe on user; take n by sha order stratified loosely: first n of hash order, last NC to calib"""
    seen = set(); rr = []
    for rid, kind, user, ans in rows:
        k = H(user)
        if k in seen: continue
        seen.add(k); rr.append((H(task + src + str(rid)), rid, kind, user, ans))
    rr.sort()
    for _, rid, kind, user, ans in rr[:n]: items.append(dict(id=f"{task}|{src}|{rid}", task=task, role=role, src=src, kind=kind, user=user, answer=ans))
    for _, rid, kind, user, ans in rr[-NC:]: calib.append(dict(id=f"{task}|{src}|cal|{rid}", task=task, role=role, src=src, kind=kind, user=user, answer=ans))
    print(task, src, "pool", len(rr), "kept", min(n, len(rr)), collections.Counter(r[2] for r in rr[:n]).most_common(8), flush=True)

PRI = ["PASSWORD", "SSN", "IBAN", "CARD", "IP", "PHONE", "EMAIL"]
TAGS = "EMAIL, PHONE, SSN, CARD, IBAN, IP, PASSWORD"
M1 = dict(EMAIL="EMAIL", PHONENUMBER="PHONE", SSN="SSN", CREDITCARDNUMBER="CARD", IBAN="IBAN", IPV4="IP", IPV6="IP", PASSWORD="PASSWORD")
M2 = dict(email="EMAIL", phone_number="PHONE", ssn="SSN", credit_card_number="CARD", iban="IBAN", ipv4="IP", ipv6="IP", password="PASSWORD")
IR = ("You are a privacy redaction service. Rewrite the text exactly as given, but replace every span containing one of these PII types with its tag in square brackets: " + TAGS + ".\n"
      "EMAIL=email address, PHONE=phone number, SSN=social security number, CARD=payment card number, IBAN=bank account IBAN, IP=IP address (v4 or v6), PASSWORD=password. Leave everything else (names, dates, addresses, amounts) unchanged.\n"
      "Output only the redacted text.\n\nText: {X}")
ID = ("You are a PII detection service. List every span in the text that contains one of these PII types: " + TAGS + ".\n"
      "EMAIL=email address, PHONE=phone number, SSN=social security number, CARD=payment card number, IBAN=bank account IBAN, IP=IP address (v4 or v6), PASSWORD=password. Ignore names, dates, addresses and any other data.\n"
      "Output only a JSON list of [type, value] pairs in order of appearance, value copied exactly from the text, e.g. [[\"EMAIL\",\"a@b.com\"],[\"IP\",\"10.0.0.1\"]]. Output [] if none.\n\nText: {X}")  # leak-ok
def piirows(texts):  # texts: (rid, text, [(s,e,tag)])
    R, D = [], []
    for rid, t, sp in texts:
        sp = sorted(sp); ok = all(sp[i][1] <= sp[i + 1][0] for i in range(len(sp) - 1))
        if not sp or not ok: continue
        if any(t[s:e] == "" for s, e, _ in sp): continue
        o, p = "", 0
        for s, e, g in sp: o += t[p:s] + f"[{g}]"; p = e
        o += t[p:]
        kind = next(g for g in PRI if any(x[2] == g for x in sp))
        R.append((rid, kind, IR.replace("{X}", t), o)); D.append((rid, kind, ID.replace("{X}", t), json.dumps([[g, t[s:e]] for s, e, g in sp], ensure_ascii=False)))
    return R, D
# ---- PII S1 ai4privacy (licence: ai4privacy custom/open, see LICENCES.json)
d = load_dataset("ai4privacy/pii-masking-200k", split="train", streaming=True); tx = []
for i, r in enumerate(d):
    if i >= 60000: break
    if r["language"] != "en": continue
    pm = ast.literal_eval(r["privacy_mask"]) if isinstance(r["privacy_mask"], str) else r["privacy_mask"]
    if any(m["label"] == "IP" for m in pm): continue
    sp = [(m["start"], m["end"], M1[m["label"]]) for m in pm if m["label"] in M1]
    if sp and len(r["source_text"]) <= 700: tx.append((r["id"], r["source_text"], sp))
R, D = piirows(tx); add("PIIRED", "S1", "ai4privacy", R, N); add("PIIDET", "S1", "ai4privacy", D, N)
# ---- PII S2 gretel synthetic finance (Apache-2.0)
d = load_dataset("gretelai/synthetic_pii_finance_multilingual", split="train", streaming=True); tx = []
for i, r in enumerate(d):
    if i >= 20000: break
    if r["language"] != "English": continue
    t = r["generated_text"]; spans = json.loads(r["pii_spans"]); pos = 0
    for pi, para in enumerate(t.split("\n\n")):
        a = t.index(para, pos); pos = a + len(para)
        if not (80 <= len(para) <= 800): continue
        sp = [(s["start"] - a, s["end"] - a, M2[s["label"]]) for s in spans if s["label"] in M2 and s["start"] >= a and s["end"] <= a + len(para)]
        # all spans (any label) fully inside/outside paragraph; drop paragraphs where a K-span is cut by boundary
        cut = any(s["label"] in M2 and (s["start"] < a + len(para) and s["end"] > a + len(para) or s["start"] < a < s["end"]) for s in spans)
        if sp and not cut: tx.append((f"{r['index']}p{pi}", para, sp))
R, D = piirows(tx); add("PIIRED", "S2", "gretel_pii_fin", R, N); add("PIIDET", "S2", "gretel_pii_fin", D, N)
# ---- NER (CoNLL-2003 train: Reuters, research-only; WNUT17 train: CC-BY-4.0)
NI = ("Extract all named entities from the text. Entity types: {M}.\nOutput only a JSON list of [entity text, type] pairs in order of appearance, entity text copied exactly from the text, e.g. [[\"Paris\",\"LOC\"]]. Output [] if none.\n\nText: {X}")
def ner(ds, names, prio, menu):
    out = []
    for r in ds:
        tk, tg = r["tokens"], [names[x] for x in r["ner_tags"]]; ents = []; cur = None
        for w, g in zip(tk, tg):
            if g.startswith("B-") or (g.startswith("I-") and (cur is None or cur[1] != g[2:])):
                cur = [[w], g[2:]]; ents.append(cur)
            elif g.startswith("I-"): cur[0].append(w)
            else: cur = None
        if not ents or len(tk) > 70: continue
        kind = next(p for p in prio if any(e[1] == p for e in ents))
        out.append((r["id"], kind, NI.replace("{M}", menu).replace("{X}", " ".join(tk)), json.dumps([[" ".join(e[0]), e[1]] for e in ents], ensure_ascii=False)))
    return out
cn = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-MISC", "I-MISC"]
d = load_dataset("eriktks/conll2003", split="train", streaming=True, revision="refs/convert/parquet")
add("NER", "S1", "conll2003", ner(d, cn, ["MISC", "ORG", "LOC", "PER"], "PER (person), ORG (organization), LOC (location), MISC (other: nationalities, events, etc.)"), N)
wn = ["O", "B-corporation", "I-corporation", "B-creative-work", "I-creative-work", "B-group", "I-group", "B-location", "I-location", "B-person", "I-person", "B-product", "I-product"]
d = load_dataset("leondz/wnut_17", split="train", streaming=True, revision="refs/convert/parquet")
add("NER", "S2", "wnut17", ner(d, wn, ["creative-work", "product", "group", "corporation", "location", "person"], "person, location, corporation, group, creative-work, product"), N)
# ---- LOGTPL (loghub 2k, free for research)
LI = "You are a log parsing service. Convert the log message to its event template: keep the constant words exactly and replace every variable value (numbers, ids, IP addresses, paths, hostnames, names, sizes) with <*>.\nOutput only the template.\n\nLog message: {X}"
SY1 = ["Apache", "HDFS", "Linux", "OpenSSH", "Hadoop", "Zookeeper", "Spark", "HealthApp"]; SY2 = ["Android", "BGL", "HPC", "Mac", "OpenStack", "Proxifier", "Thunderbird", "Windows"]
for role, sysl, srcn in (("S1", SY1, "loghub_a"), ("S2", SY2, "loghub_b")):
    rows = []
    for s in sysl:
        try: txt = urllib.request.urlopen(f"https://raw.githubusercontent.com/logpai/loghub/master/{s}/{s}_2k.log_structured.csv", timeout=60).read().decode("utf8", "replace")
        except Exception as e: print("skip", s, e); continue
        for j, r in enumerate(csv.DictReader(io.StringIO(txt))):
            if 15 <= len(r["Content"]) <= 400 and "<*>" in r["EventTemplate"]: rows.append((f"{s}{j}", s, LI.replace("{X}", r["Content"]), r["EventTemplate"]))
    add("LOGTPL", role, srcn, rows, N)
# ---- SQL (sql-create-context CC-BY-4.0 from WikiSQL/Spider ; gretel synthetic_text_to_sql Apache-2.0)
SI = "Write a SQL query that answers the question for the given database schema.\nOutput only the SQL query on one line, no explanation.\n\nSchema:\n{S}\n\nQuestion: {Q}"
def shape(q):
    u = q.upper(); return "join" if " JOIN " in u else "group" if "GROUP BY" in u else "agg" if re.search(r"\b(COUNT|SUM|AVG|MIN|MAX)\(", u) else "where" if " WHERE " in u else "plain"
d = load_dataset("b-mc2/sql-create-context", split="train", streaming=True); rows = []
for i, r in enumerate(d):
    if i >= 30000: break
    rows.append((i, shape(r["answer"]), SI.replace("{S}", r["context"]).replace("{Q}", r["question"]), r["answer"]))
add("SQL", "S1", "sql_create_context", rows, N)
d = load_dataset("gretelai/synthetic_text_to_sql", split="train", streaming=True); rows = []
for i, r in enumerate(d):
    if i >= 30000: break
    ctx = " ".join(x.strip() + ";" for x in r["sql_context"].split(";") if x.strip().upper().startswith("CREATE"))
    if not ctx or "\n" in r["sql"] or len(ctx) > 1500: continue
    rows.append((r["id"], shape(r["sql"]), SI.replace("{S}", ctx).replace("{Q}", r["sql_prompt"]), r["sql"]))
add("SQL", "S2", "gretel_text2sql", rows, N)
# ---- TOOLARG (glaive-function-calling-v2, Apache-2.0); S1/S2 = function-name hash halves
TI = "You can call these functions:\n{F}\n\nConversation so far:\n{C}\n\nIf the next assistant step is a function call, output only the call as JSON: {\"name\": <function name>, \"arguments\": {<arguments>}} with argument values taken from the conversation. Output only the JSON."
d = load_dataset("glaiveai/glaive-function-calling-v2", split="train", streaming=True); rows = {"S1": [], "S2": []}
for i, r in enumerate(d):
    if i >= 40000: break
    m = re.search(r"ASSISTANT: <functioncall> (\{.*?\}) <\|endoftext\|>", r["chat"], re.S)
    if not m: continue
    pre = r["chat"][:m.start()]
    if "FUNCTION RESPONSE" in pre: continue
    cm = re.match(r"(\{\"name\": \"[^\"]+\"), \"arguments\": '(.*)'\}$", m.group(1), re.S)
    if not cm: continue
    try: args = json.loads(cm.group(2)); name = json.loads(cm.group(1) + "}")["name"]
    except Exception: continue
    if not isinstance(args, dict) or not args: continue
    fn = r["system"].split("-\n", 1)[-1].strip()
    if len(fn) > 2500 or len(pre) > 1500: continue
    conv = pre.replace("USER:", "User:").replace("ASSISTANT:", "Assistant:").replace("<|endoftext|>", "").strip()
    gold = json.dumps(dict(name=name, arguments=args), ensure_ascii=False)
    kind = {1: "1arg", 2: "2args"}.get(len(args), "3+args")
    rows["S1" if int(H(name)[:8], 16) % 2 == 0 else "S2"].append((i, kind, TI.replace("{F}", fn).replace("{C}", conv), gold))
for k in rows: add("TOOLARG", k, "glaive_" + k, rows[k], N)
json.dump(dict(ai4privacy="ai4privacy/pii-masking-200k: custom open licence (see dataset card; synthetic)", gretel_pii_fin="gretelai/synthetic_pii_finance_multilingual: Apache-2.0", conll2003="eriktks/conll2003: Reuters corpus, research-only licence", wnut17="leondz/wnut_17: CC-BY-4.0", loghub="logpai/loghub: free for research/academic", sql_create_context="b-mc2/sql-create-context: CC-BY-4.0", gretel_text2sql="gretelai/synthetic_text_to_sql: Apache-2.0", glaive="glaiveai/glaive-function-calling-v2: Apache-2.0"), open("LICENCES.json", "w"), indent=1)
with open("items_dev.jsonl", "w") as f:
    for x in items: f.write(json.dumps(x, ensure_ascii=False) + "\n")
with open("calib.jsonl", "w") as f:
    for x in calib: f.write(json.dumps(x, ensure_ascii=False) + "\n")
print("TOTAL", len(items), len(calib))
