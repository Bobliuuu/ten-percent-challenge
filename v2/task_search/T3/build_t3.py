"""TASK_CATALOGUE probe builder (T3). DEV DATA ONLY: public train/dev splits, sha256-ordered, class-balanced, dedup on input text.
Never builds or reads a locked exam. Output: items_dev.jsonl (id,task,role,src,kind,user,answer), calib.jsonl, PLAN.json (pool counts per class).
Output of every task = ONE real label word (exact string match). Each task has >=2 distinct sources (S1 = tuning role, S2 = other-source role).
Usage: python build_t3.py [N_per_src=160] [task ...]"""
import json, hashlib, collections, sys, os, random, re, ast
from datasets import load_dataset
H = lambda s: hashlib.sha256(s.encode()).hexdigest()
HERE = os.path.dirname(os.path.abspath(__file__))
N = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 160
ONLY = [a for a in sys.argv[1:] if not a.isdigit()]
NC = int(os.environ.get("T3_NC", "30"))
items, calib, plan = [], [], {}
def trunc(t, n=1200): t = re.sub(r"\s+", " ", str(t)).strip(); return t if len(t) <= n else t[:n]
def stream(name, cfg=None, split="train", limit=60000):
    d = load_dataset(name, cfg, split=split, streaming=True)
    for i, r in enumerate(d):
        if i >= limit: break
        yield r
def add(task, role, src, rows, labels, prompt, n=None):
    """rows: (rid, label, text). class-balanced: n/len(labels) per class by sha order; last NC (by hash) to calib."""
    n = n or N; per = max(1, n // len(labels)); seen = set(); by = collections.defaultdict(list)
    for rid, lab, txt in rows:
        if lab not in labels or not txt: continue
        k = H(txt)
        if k in seen: continue
        seen.add(k); by[lab].append((H(task + src + str(rid)), rid, lab, txt))
    pool = {l: len(by[l]) for l in labels}; kept = {}
    for l in labels:
        v = sorted(by[l]); kept[l] = min(per, len(v))
        for _, rid, lab, txt in v[:per]: items.append(dict(id=f"{task}|{src}|{rid}", task=task, role=role, src=src, kind=lab, user=prompt.replace("{X}", txt), answer=lab))
        for _, rid, lab, txt in v[-(NC // len(labels) + 1):]: calib.append(dict(id=f"{task}|{src}|cal|{rid}", task=task, role=role, src=src, kind=lab, user=prompt.replace("{X}", txt), answer=lab))
    plan.setdefault(task, {"labels": labels, "prompt": prompt, "src": {}})["src"][src] = dict(pool_scanned=pool, dev_kept=kept)
    print(task, src, "pool", pool, flush=True)
def P(desc, labels, field="Text"): return f"{desc}\nAnswer with exactly one word from: {', '.join(labels)}.\n\n{field}: {{X}}"
def run(name):
    return (not ONLY) or name in ONLY
def safe(name, fn):
    if not run(name): return
    try: fn()
    except Exception as e: print("BUILD-FAIL", name, type(e).__name__, str(e)[:200], flush=True)

# ---------------- PHISH2: email Phishing / Legit
def t_phish():
    L = ["Phishing", "Legitimate"]; pr = P("You are an email security filter. Decide whether this email is a phishing attempt or a legitimate email.", L, "Email")
    add("PHISH2", "S1", "zefang_phish", ((i, "Phishing" if r["Email Type"] == "Phishing Email" else "Legitimate", trunc(r["Email Text"])) for i, r in enumerate(stream("zefang-liu/phishing-email-dataset"))), L, pr)
    def s2():
        for i, r in enumerate(stream("puyang2025/seven-phishing-email-datasets", limit=40000)):
            t = (r.get("subject") or "") + "\n" + (r.get("text") or ""); yield i, ("Phishing" if int(r["label"]) == 1 else "Legitimate"), trunc(t)
    add("PHISH2", "S2", "sevenphish", s2(), L, pr)
safe("PHISH2", t_phish)

# ---------------- TOX2: toxic comment moderation (Toxic / Safe)
def t_tox():
    L = ["Toxic", "Safe"]; pr = P("You are a content moderation filter. Is this user message toxic (insulting, hateful, threatening or profane)?", L, "Message")
    def s1():
        for i, r in enumerate(stream("google/civil_comments", limit=120000)):
            if r["toxicity"] >= 0.5: yield i, "Toxic", trunc(r["text"])
            elif r["toxicity"] < 0.05: yield i, "Safe", trunc(r["text"])
    add("TOX2", "S1", "civil_comments", s1(), L, pr)
    def s2():
        for i, r in enumerate(stream("lmsys/toxic-chat", "toxicchat0124", limit=20000)): yield i, ("Toxic" if r["toxicity"] == 1 else "Safe"), trunc(r["user_input"])
    add("TOX2", "S2", "toxic_chat", s2(), L, pr)
safe("TOX2", t_tox)

# ---------------- HATE2: hate speech (Hate / Fine)
def t_hate():
    L = ["Hate", "Fine"]; pr = P("You are a trust-and-safety classifier. Does this post contain hate speech targeting a group of people?", L, "Post")
    add("HATE2", "S1", "tweet_eval_hate", ((i, "Hate" if r["label"] == 1 else "Fine", trunc(r["text"])) for i, r in enumerate(stream("cardiffnlp/tweet_eval", "hate"))), L, pr)
    add("HATE2", "S2", "davidson", ((i, "Hate" if r["class"] == 0 else "Fine", trunc(r["tweet"])) for i, r in enumerate(stream("tdavidson/hate_speech_offensive")) if r["class"] in (0, 2)), L, pr)
safe("HATE2", t_hate)

# ---------------- OFFEN2: offensive language (Offensive / Clean)
def t_off():
    L = ["Offensive", "Clean"]; pr = P("Is this social media post offensive (profanity, insults, slurs)?", L, "Post")
    add("OFFEN2", "S1", "tweet_eval_off", ((i, "Offensive" if r["label"] == 1 else "Clean", trunc(r["text"])) for i, r in enumerate(stream("cardiffnlp/tweet_eval", "offensive"))), L, pr)
    add("OFFEN2", "S2", "davidson", ((i, "Offensive" if r["class"] in (0, 1) else "Clean", trunc(r["tweet"])) for i, r in enumerate(stream("tdavidson/hate_speech_offensive"))), L, pr)
safe("OFFEN2", t_off)

# ---------------- NLI3: entailment / neutral / contradiction
def t_nli():
    L = ["Entailment", "Neutral", "Contradiction"]; m = {0: "Entailment", 1: "Neutral", 2: "Contradiction"}
    pr = "Natural language inference. Decide whether the hypothesis follows from the premise.\nAnswer with exactly one word from: Entailment, Neutral, Contradiction.\n\nPremise: {P}\nHypothesis: {H}"
    def fmt(p, h): return pr.replace("{P}", trunc(p, 600)).replace("{H}", trunc(h, 300))
    # items need pre-formatted prompt: pass text as P||H and replace
    def mk(rows):
        for i, lab, p, h in rows: yield i, lab, (p, h)
    # custom add (two-field)
    for role, src, rows in [("S1", "snli", ((i, m[r["label"]], (r["premise"], r["hypothesis"])) for i, r in enumerate(stream("stanfordnlp/snli", limit=40000)) if r["label"] in m)),
                            ("S2", "anli", ((i, m[r["label"]], (r["premise"], r["hypothesis"])) for i, r in enumerate(stream("facebook/anli", split="train_r1", limit=20000)) if r["label"] in m)),
                            ("S3", "mnli", ((i, m[r["label"]], (r["premise"], r["hypothesis"])) for i, r in enumerate(stream("nyu-mll/multi_nli", limit=40000)) if r["label"] in m))]:
        rr = ((i, l, fmt(p, h)) for i, l, (p, h) in rows)
        # use fmt'd text as the whole user prompt
        add("NLI3", role, src, rr, L, "{X}")
safe("NLI3", t_nli)

# ---------------- ENTAIL2: yes/no entailment (RTE, SciTail)
def t_ent():
    L = ["Yes", "No"]; pr = "Does the hypothesis follow from the premise?\nAnswer with exactly one word from: Yes, No.\n\nPremise: {P}\nHypothesis: {H}"
    f = lambda p, h: pr.replace("{P}", trunc(p, 800)).replace("{H}", trunc(h, 300))
    add("ENTAIL2", "S1", "rte", ((i, "Yes" if r["label"] == 0 else "No", f(r["sentence1"], r["sentence2"])) for i, r in enumerate(stream("nyu-mll/glue", "rte"))), L, "{X}")
    add("ENTAIL2", "S2", "scitail", ((i, "Yes" if r["label"] == "entails" else "No", f(r["premise"], r["hypothesis"])) for i, r in enumerate(stream("allenai/scitail", "tsv_format", limit=20000))), L, "{X}")
safe("ENTAIL2", t_ent)

# ---------------- PARA2: duplicate / paraphrase
def t_para():
    L = ["Yes", "No"]; pr = "Do these two texts ask or say the same thing (are they duplicates / paraphrases)?\nAnswer with exactly one word from: Yes, No.\n\nA: {A}\nB: {B}"
    f = lambda a, b: pr.replace("{A}", trunc(a, 500)).replace("{B}", trunc(b, 500))
    add("PARA2", "S1", "qqp", ((i, "Yes" if r["label"] == 1 else "No", f(r["question1"], r["question2"])) for i, r in enumerate(stream("nyu-mll/glue", "qqp", limit=40000))), L, "{X}")
    add("PARA2", "S2", "mrpc", ((i, "Yes" if r["label"] == 1 else "No", f(r["sentence1"], r["sentence2"])) for i, r in enumerate(stream("nyu-mll/glue", "mrpc"))), L, "{X}")
safe("PARA2", t_para)

# ---------------- BOOLQ2 / ANSWERABLE
def t_bool():
    L = ["Yes", "No"]; pr = "Answer the question using the passage.\nAnswer with exactly one word from: Yes, No.\n\nPassage: {P}\nQuestion: {Q}"
    f = lambda p, q: pr.replace("{P}", trunc(p, 1500)).replace("{Q}", q)
    add("BOOLQ", "S1", "boolq", ((i, "Yes" if r["answer"] else "No", f(r["passage"], r["question"] + "?")) for i, r in enumerate(stream("google/boolq"))), L, "{X}")
safe("BOOLQ", t_bool)
def t_ans():
    L = ["Answerable", "Unanswerable"]; pr = "Can the question be answered from the passage?\nAnswer with exactly one word from: Answerable, Unanswerable.\n\nPassage: {P}\nQuestion: {Q}"
    f = lambda p, q: pr.replace("{P}", trunc(p, 1200)).replace("{Q}", q)
    add("ANSWERABLE", "S1", "squad2", ((r["id"], "Answerable" if r["answers"]["text"] else "Unanswerable", f(r["context"], r["question"])) for r in stream("rajpurkar/squad_v2", limit=40000)), L, "{X}")
safe("ANSWERABLE", t_ans)

# ---------------- PROMPT_SAFETY: harmful request?
def t_psafe():
    L = ["Unsafe", "Safe"]; pr = P("You are a safety guardrail for an AI assistant. Is this user request unsafe (asks for violence, crime, hate, self-harm, sexual content involving minors, privacy violation, or other harm)?", L, "Request")
    def s1():
        for i, r in enumerate(stream("nvidia/Aegis-AI-Content-Safety-Dataset-2.0", limit=40000)):
            p = r.get("prompt"); l = r.get("prompt_label")
            if p and p != "REDACTED" and l in ("safe", "unsafe"): yield i, ("Unsafe" if l == "unsafe" else "Safe"), trunc(p)
    add("PSAFE", "S1", "aegis2", s1(), L, pr)
    def s2():
        for i, r in enumerate(stream("lmsys/toxic-chat", "toxicchat0124", limit=20000)): yield i, ("Unsafe" if r["toxicity"] == 1 else "Safe"), trunc(r["user_input"])
    add("PSAFE", "S2", "toxic_chat", s2(), L, pr)
    def s3():
        seen = set()
        for i, r in enumerate(stream("PKU-Alignment/BeaverTails", split="330k_train", limit=60000)):
            if r["prompt"] in seen: continue
            seen.add(r["prompt"]); yield i, ("Safe" if r["is_safe"] else "Unsafe"), trunc(r["prompt"])
    add("PSAFE", "S3", "beavertails_prompt", s3(), L, pr)
safe("PSAFE", t_psafe)

# ---------------- SARC2: sarcasm / irony
def t_sarc():
    L = ["Sarcastic", "Sincere"]; pr = P("Is this text sarcastic or ironic?", L, "Text")
    add("SARC2", "S1", "headlines", ((i, "Sarcastic" if r["is_sarcastic"] == 1 else "Sincere", trunc(r["headline"])) for i, r in enumerate(stream("raquiba/Sarcasm_News_Headline"))), L, pr)
    add("SARC2", "S2", "tweet_irony", ((i, "Sarcastic" if r["label"] == 1 else "Sincere", trunc(r["text"])) for i, r in enumerate(stream("cardiffnlp/tweet_eval", "irony"))), L, pr)
safe("SARC2", t_sarc)

# ---------------- CODELANG: programming language ID
def t_code():
    LL = ["Python", "Java", "JavaScript", "Go", "PHP", "Ruby"]; pr = P("Identify the programming language of this code snippet.", LL, "Code")
    mp = {"python": "Python", "java": "Java", "javascript": "JavaScript", "go": "Go", "php": "PHP", "ruby": "Ruby"}
    def s1():
        for lg, nm in mp.items():
            for i, r in enumerate(stream("claudios/code_search_net", lg, limit=3000)): yield f"{lg}{i}", nm, trunc(r["whole_func_string"], 900)
    add("CODELANG", "S1", "codesearchnet", s1(), LL, pr)
    def s2():
        for lg, nm in mp.items():
            try:
                for i, r in enumerate(stream("google/code_x_glue_ct_code_to_text", lg, split="validation", limit=2000)): yield f"{lg}{i}", nm, trunc(r["original_string"] if "original_string" in r else r["code"], 900)
            except Exception as e: print("codexglue", lg, type(e).__name__)
    add("CODELANG", "S2", "codexglue_c2t", s2(), LL, pr)
safe("CODELANG", t_code)

# ---------------- DEFECT: C vulnerability yes/no (control, hard)
def t_defect():
    L = ["Vulnerable", "Safe"]; pr = P("Does this C function contain a security vulnerability?", L, "Code")
    add("DEFECT", "S1", "devign", ((i, "Vulnerable" if r["target"] else "Safe", trunc(r["func"], 1500)) for i, r in enumerate(stream("google/code_x_glue_cc_defect_detection", limit=20000))), L, pr)
safe("DEFECT", t_defect)

# ---------------- RCT5: sentence role in medical abstract
def t_rct():
    L = ["Background", "Objective", "Methods", "Results", "Conclusions"]; mp = {"BACKGROUND": "Background", "OBJECTIVE": "Objective", "METHODS": "Methods", "RESULTS": "Results", "CONCLUSIONS": "Conclusions"}
    pr = P("This sentence comes from the abstract of a medical research paper. Which section does it belong to?", L, "Sentence")
    add("RCT5", "S1", "pubmed_rct", ((i, mp[r["label"].upper()], trunc(r["text"], 600)) for i, r in enumerate(stream("armanc/pubmed-rct20k", limit=60000)) if r["label"].upper() in mp), L, pr)
safe("RCT5", t_rct)

# ---------------- FINSENT3
def t_fin():
    L = ["Bullish", "Bearish", "Neutral"]; pr = P("Classify the sentiment of this financial headline for an investor.", L, "Headline")
    add("FINSENT", "S1", "twitterfin", ((i, {0: "Bearish", 1: "Bullish", 2: "Neutral"}[r["label"]], trunc(r["text"])) for i, r in enumerate(stream("zeroshot/twitter-financial-news-sentiment"))), L, pr)
safe("FINSENT", t_fin)

# ---------------- PIIFLAG: does the text contain PII?
def t_pii():
    L = ["Yes", "No"]; pr = P("Does this text contain personally identifiable information such as an email address, phone number, ID number, card number, IP address or password?", L, "Text")
    def s1():
        for i, r in enumerate(stream("ai4privacy/pii-masking-200k", limit=30000)):
            if r["language"] != "en": continue
            pm = ast.literal_eval(r["privacy_mask"]) if isinstance(r["privacy_mask"], str) else r["privacy_mask"]
            has = any(m["label"] in ("EMAIL", "PHONENUMBER", "SSN", "CREDITCARDNUMBER", "IBAN", "IPV4", "IPV6", "PASSWORD") for m in pm)
            yield i, ("Yes" if has else "No"), trunc(r["source_text"], 700)
    add("PIIFLAG", "S1", "ai4privacy", s1(), L, pr)
    def s2():
        PL = ("email", "phone_number", "ssn", "credit_card_number", "iban", "ipv4", "ipv6", "password")
        for i, r in enumerate(stream("gretelai/synthetic_pii_finance_multilingual", limit=12000)):
            if r["language"] != "English": continue
            t = r["generated_text"]; sp = json.loads(r["pii_spans"]); pos = 0
            for pi, para in enumerate(t.split("\n\n")):
                a = t.index(para, pos); pos = a + len(para)
                has = any(s["label"] in PL and s["start"] >= a and s["end"] <= a + len(para) for s in sp)
                yield f"{i}.{pi}", ("Yes" if has else "No"), trunc(para, 700)
    add("PIIFLAG", "S2", "gretel_fin", s2(), L, pr)
safe("PIIFLAG", t_pii)

# ---------------- ADE: adverse drug event mention
def t_ade():
    L = ["Yes", "No"]; pr = P("Does this medical text report an adverse drug event (a side effect caused by a drug)?", L, "Text")
    add("ADE", "S1", "ade_corpus", ((i, "Yes" if r["label"] == 1 else "No", trunc(r["text"], 600)) for i, r in enumerate(stream("ade-benchmark-corpus/ade_corpus_v2", "Ade_corpus_v2_classification"))), L, pr)
safe("ADE", t_ade)

# ---------------- FAKENEWS
def t_fake():
    L = ["Fake", "Real"]; pr = P("Is this news article fake news or real news?", L, "Article")
    add("FAKENEWS", "S1", "gonzaloa", ((i, "Fake" if r["label"] == 0 else "Real", trunc(r["title"] + ". " + r["text"], 900)) for i, r in enumerate(stream("GonzaloA/fake_news", limit=20000))), L, pr)
safe("FAKENEWS", t_fake)

# ---------------- ARXIV / DBPEDIA / HUFFPOST fine topics (control: higher class counts)
def t_dbp():
    names = ["Company", "School", "Artist", "Athlete", "Politician", "Transport", "Building", "Nature", "Village", "Animal", "Plant", "Album", "Film", "Book"]
    L = ["Company", "Artist", "Athlete", "Politician", "Film", "Book"]; pr = P("What kind of entity does this Wikipedia abstract describe?", L, "Abstract")
    add("DBPEDIA6", "S1", "dbpedia14", ((i, names[r["label"]], trunc(r["content"], 700)) for i, r in enumerate(stream("fancyzhx/dbpedia_14", limit=40000))), L, pr)
safe("DBPEDIA6", t_dbp)

# ---------------- TICKETTYPE / PRIORITY (Tobi-Bueck helpdesk): hard control
def t_ticket():
    L = ["Incident", "Request", "Problem", "Change"]; pr = P("Classify this customer support ticket by type.", L, "Ticket")
    add("TICKETTYPE", "S1", "tobi_tickets", ((i, r["type"], trunc((r["subject"] or "") + ". " + (r["body"] or ""), 900)) for i, r in enumerate(stream("Tobi-Bueck/customer-support-tickets", limit=30000)) if r.get("language") == "en"), L, pr)
    L2 = ["High", "Medium", "Low"]; pr2 = P("Classify the priority of this customer support ticket.", L2, "Ticket")
    add("TICKETPRIO", "S1", "tobi_tickets", ((i, str(r["priority"]).capitalize(), trunc((r["subject"] or "") + ". " + (r["body"] or ""), 900)) for i, r in enumerate(stream("Tobi-Bueck/customer-support-tickets", limit=30000)) if r.get("language") == "en"), L2, pr2)
safe("TICKETTYPE", t_ticket)

# ---------------- UNFAIR: ToS clause unfair? (LexGLUE unfair_tos + legalbench)
def t_unfair():
    L = ["Unfair", "Fair"]; pr = P("You are a consumer-law reviewer. Is this clause from a terms-of-service agreement potentially unfair to the consumer?", L, "Clause")
    add("UNFAIR", "S1", "lexglue_unfairtos", ((i, "Unfair" if len(r["labels"]) > 0 else "Fair", trunc(r["text"], 800)) for i, r in enumerate(stream("coastalcph/lex_glue", "unfair_tos", limit=6000))), L, pr)
safe("UNFAIR", t_unfair)


# ---------------- TOPIC5: coarse topic routing, 2 corpora (Sports/Business/Health/Tech/Politics)
def t_topic():
    L = ["Sports", "Business", "Health", "Technology", "Politics"]; pr = P("Route this text to the department that covers its topic.", L, "Text")
    m1 = {"SPORTS": "Sports", "BUSINESS": "Business", "WELLNESS": "Health", "HEALTHY LIVING": "Health", "TECH": "Technology", "POLITICS": "Politics"}
    add("TOPIC5", "S1", "huffpost", ((i, m1[r["category"]], trunc(r["headline"] + ". " + r["short_description"], 600)) for i, r in enumerate(stream("heegyu/news-category-dataset", limit=210000)) if r["category"] in m1), L, pr)
    m2 = {"Sports": "Sports", "Business & Finance": "Business", "Health": "Health", "Computers & Internet": "Technology", "Politics & Government": "Politics"}
    names = ["Society & Culture", "Science & Mathematics", "Health", "Education & Reference", "Computers & Internet", "Sports", "Business & Finance", "Entertainment & Music", "Family & Relationships", "Politics & Government"]
    add("TOPIC5", "S2", "yahoo_answers", ((i, m2[names[r["topic"]]], trunc(r["question_title"] + " " + (r["question_content"] or ""), 600)) for i, r in enumerate(stream("community-datasets/yahoo_answers_topics", limit=120000)) if names[r["topic"]] in m2), L, pr)
    m3 = {"Business": "Business", "Sports": "Sports", "Politics": "Politics", "Tech": "Technology"}
safe("TOPIC5", t_topic)

# ---------------- INJ2: prompt injection / jailbreak (easy sources)
def t_inj():
    L = ["Injection", "Benign"]; pr = P("You are a security filter in front of an LLM application. Does this input try to hijack or jailbreak the model (override instructions, reveal the system prompt, role-play to evade rules)?", L, "Input")
    add("INJ2", "S1", "deepset", ((i, "Injection" if r["label"] == 1 else "Benign", trunc(r["text"], 900)) for i, r in enumerate(stream("deepset/prompt-injections"))), L, pr)
    add("INJ2", "S2", "jackhhao", ((i, "Injection" if r["type"] == "jailbreak" else "Benign", trunc(r["prompt"], 900)) for i, r in enumerate(stream("jackhhao/jailbreak-classification"))), L, pr)
safe("INJ2", t_inj)

# ---------------- LANG16: language ID with many languages
def t_lang():
    names = dict(ar="Arabic", bg="Bulgarian", de="German", el="Greek", en="English", es="Spanish", fr="French", hi="Hindi", it="Italian", ja="Japanese", nl="Dutch", pl="Polish", pt="Portuguese", ru="Russian", sw="Swahili", th="Thai", tr="Turkish", ur="Urdu", vi="Vietnamese", zh="Chinese")
    L = list(names.values()); pr = P("Identify the language of this text.", L, "Text")
    add("LANG20", "S1", "papluca", ((i, names[r["labels"]], trunc(r["text"], 500)) for i, r in enumerate(stream("papluca/language-identification", limit=70000))), L, pr, n=400)
safe("LANG20", t_lang)

# ---------------- CODEPROSE: code vs natural language
def t_cp():
    L = ["Code", "Prose"]; pr = P("Is this text source code or natural-language prose?", L, "Text")
    def s1():
        for lg in ["python", "java", "javascript", "go", "php", "ruby"]:
            for i, r in enumerate(stream("claudios/code_search_net", lg, limit=400)): yield f"{lg}{i}", "Code", trunc(r["whole_func_string"], 700)
        for i, r in enumerate(stream("fancyzhx/ag_news", limit=3000)): yield f"ag{i}", "Prose", trunc(r["text"], 700)
    add("CODEPROSE", "S1", "csn_vs_agnews", s1(), L, pr)
    def s2():
        for i, r in enumerate(stream("google/code_x_glue_cc_defect_detection", limit=1500)): yield f"d{i}", "Code", trunc(r["func"], 700)
        for i, r in enumerate(stream("stanfordnlp/imdb", limit=3000)): yield f"im{i}", "Prose", trunc(r["text"], 700)
    add("CODEPROSE", "S2", "devign_c_vs_imdb", s2(), L, pr)
safe("CODEPROSE", t_cp)

# ---------------- DEPT3: support / sales / billing style email routing (Bitext categories + Tobi queues)
def t_dept():
    L = ["Billing", "Technical", "Sales"]; pr = P("Route this customer email to the right team.", L, "Email")
    mq = {"Billing and Payments": "Billing", "Technical Support": "Technical", "Sales and Pre-Sales": "Sales"}
    add("DEPT3", "S1", "tobi_queue", ((i, mq[r["queue"]], trunc((r["subject"] or "") + ". " + (r["body"] or ""), 800)) for i, r in enumerate(stream("Tobi-Bueck/customer-support-tickets", limit=30000)) if r.get("language") == "en" and r["queue"] in mq), L, pr)
    mc = {"INVOICE": "Billing", "PAYMENT": "Billing", "REFUND": "Billing", "ORDER": "Sales", "SHIPPING": "Sales", "ACCOUNT": "Technical", "CONTACT": "Technical", "FEEDBACK": "Technical"}
    add("DEPT3", "S2", "bitext", ((i, mc[r["category"]], trunc(r["instruction"], 500)) for i, r in enumerate(stream("bitext/Bitext-customer-support-llm-chatbot-training-dataset", limit=30000)) if r["category"] in mc), L, pr)
safe("DEPT3", t_dept)


# ======================= ROUND 3 (cleaner label sets, second sources) =======================
def t_lang12b():
    names = dict(ar="Arabic", de="German", el="Greek", en="English", es="Spanish", fr="French", hi="Hindi", it="Italian", ja="Japanese", ru="Russian", th="Thai", tr="Turkish")
    L = list(names.values()); pr = P("Identify the language of this text.", L, "Text")
    add("LANG12B", "S1", "papluca", ((i, names[r["labels"]], trunc(r["text"], 500)) for i, r in enumerate(stream("papluca/language-identification", limit=70000)) if r["labels"] in names), L, pr, n=360)
    def s2():
        for lg in names:
            for i, r in enumerate(stream("mteb/amazon_massive_scenario", lg, limit=1500)): yield f"{lg}{i}", names[lg], trunc(r["text"], 300)
    add("LANG12B", "S2", "massive_utterances", s2(), L, pr, n=360)
safe("LANG12B", t_lang12b)

def t_news4():
    L = ["Business", "Entertainment", "Politics", "Sports"]; pr = P("Which newsroom desk should this news story go to?", L, "Story")
    m1 = {"BUSINESS": "Business", "ENTERTAINMENT": "Entertainment", "POLITICS": "Politics", "SPORTS": "Sports"}
    add("NEWS4C", "S1", "huffpost", ((i, m1[r["category"]], trunc(r["headline"] + ". " + r["short_description"], 500)) for i, r in enumerate(stream("heegyu/news-category-dataset", limit=210000)) if r["category"] in m1), L, pr)
    m2 = {"business": "Business", "entertainment": "Entertainment", "politics": "Politics", "sport": "Sports"}
    add("NEWS4C", "S2", "bbc_2004", ((i, m2[r["label_text"]], trunc(r["text"], 500)) for i, r in enumerate(stream("SetFit/bbc-news")) if r["label_text"] in m2), L, pr)
safe("NEWS4C", t_news4)

def t_sent2c():
    L = ["Positive", "Negative"]; pr = P("What is the sentiment of this text?", L, "Text")
    add("SENT2C", "S1", "yelp_restaurants", ((i, "Positive" if r["label"] == 1 else "Negative", trunc(r["text"], 600)) for i, r in enumerate(stream("fancyzhx/yelp_polarity", limit=30000))), L, pr)
    add("SENT2C", "S2", "imdb_movies", ((i, "Positive" if r["label"] == 1 else "Negative", trunc(r["text"], 700)) for i, r in enumerate(stream("stanfordnlp/imdb", limit=30000))), L, pr)
    add("SENT2C", "S3", "tweets", ((i, "Positive" if r["label"] == 2 else "Negative", trunc(r["text"], 300)) for i, r in enumerate(stream("cardiffnlp/tweet_eval", "sentiment")) if r["label"] in (0, 2)), L, pr)
safe("SENT2C", t_sent2c)

def t_tox2c():
    L = ["Toxic", "Fine"]; pr = ("You moderate an online community.\nToxic = the message clearly insults, demeans, threatens or harasses someone, or is hateful or very vulgar.\nFine = everything else.\nAnswer with exactly one word from: Toxic, Fine.\n\nMessage: {X}")
    def s1():
        for i, r in enumerate(stream("google/civil_comments", limit=160000)):
            if r["toxicity"] >= 0.8: yield i, "Toxic", trunc(r["text"], 600)
            elif r["toxicity"] <= 0.0: yield i, "Fine", trunc(r["text"], 600)
    add("TOX2C", "S1", "civil_comments", s1(), L, pr)
    add("TOX2C", "S2", "davidson_offensive", ((i, "Toxic" if (r["class"] in (0, 1) and r["hate_speech_count"] + r["offensive_language_count"] >= 3) else ("Fine" if r["class"] == 2 and r["neither_count"] >= 3 else ""), trunc(r["tweet"], 400)) for i, r in enumerate(stream("tdavidson/hate_speech_offensive"))), L, pr)
safe("TOX2C", t_tox2c)

def t_phish2c():
    L = ["Scam", "Genuine"]; pr = ("Is this email a scam or phishing attempt (it tries to trick the reader into paying money, giving credentials or personal data, or clicking a malicious link)?\nScam = yes. Genuine = an ordinary email.\nAnswer with exactly one word from: Scam, Genuine.\n\nEmail: {X}")
    add("PHISH2C", "S1", "zefang_phish", ((i, "Scam" if r["Email Type"] == "Phishing Email" else "Genuine", trunc(r["Email Text"], 900)) for i, r in enumerate(stream("zefang-liu/phishing-email-dataset"))), L, pr)
    def s2():
        for i, r in enumerate(stream("puyang2025/seven-phishing-email-datasets", limit=40000)):
            t = (r.get("subject") or "") + "\n" + (r.get("text") or ""); yield i, ("Scam" if int(r["label"]) == 1 else "Genuine"), trunc(t, 900)
    add("PHISH2C", "S2", "sevenphish", s2(), L, pr)
safe("PHISH2C", t_phish2c)

def t_pii2c():
    L = ["Sensitive", "Harmless"]; pr = ("Does the text contain a sensitive identifier: an email address, phone number, national ID / social security number, credit card number, IBAN, IP address or password?\nSensitive = yes, at least one. Harmless = none (names, dates, places, companies and amounts alone are harmless).\nAnswer with exactly one word from: Sensitive, Harmless.\n\nText: {X}")
    def s1():
        for i, r in enumerate(stream("ai4privacy/pii-masking-200k", limit=30000)):
            if r["language"] != "en": continue
            pm = ast.literal_eval(r["privacy_mask"]) if isinstance(r["privacy_mask"], str) else r["privacy_mask"]
            has = any(m["label"] in ("EMAIL", "PHONENUMBER", "SSN", "CREDITCARDNUMBER", "IBAN", "IPV4", "IPV6", "PASSWORD") for m in pm)
            yield i, ("Sensitive" if has else "Harmless"), trunc(r["source_text"], 700)
    add("PII2C", "S1", "ai4privacy", s1(), L, pr)
    def s2():
        PL = ("email", "phone_number", "ssn", "credit_card_number", "iban", "ipv4", "ipv6", "password")
        for i, r in enumerate(stream("gretelai/synthetic_pii_finance_multilingual", limit=12000)):
            if r["language"] != "English": continue
            t = r["generated_text"]; sp = json.loads(r["pii_spans"]); pos = 0
            for pi, para in enumerate(t.split("\n\n")):
                a = t.index(para, pos); pos = a + len(para)
                has = any(s["label"] in PL and s["start"] >= a and s["end"] <= a + len(para) for s in sp)
                yield f"{i}.{pi}", ("Sensitive" if has else "Harmless"), trunc(para, 700)
    add("PII2C", "S2", "gretel_fin", s2(), L, pr)
safe("PII2C", t_pii2c)


if not ONLY or True:
    tag = "_".join(ONLY) if ONLY else ""
    json.dump(plan, open(os.path.join(HERE, f"PLAN{tag}.json"), "w"), indent=1)
    with open(os.path.join(HERE, f"items_dev{tag}.jsonl"), "w") as f:
        for it in items: f.write(json.dumps(it, ensure_ascii=False) + "\n")
    with open(os.path.join(HERE, f"calib{tag}.jsonl"), "w") as f:
        for it in calib: f.write(json.dumps(it, ensure_ascii=False) + "\n")
    print("TOTAL items", len(items), "calib", len(calib), collections.Counter(i["task"] for i in items))
