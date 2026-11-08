"""EASY tasks (narrow label routing, 2-8 classes) DEV/CALIB builder. Public datasets, sha256 buckets: calib [0,40), dev [40,70), [70,100) untouched (future locked). No locked exam is built or read.
python build_easy.py -> items_dev.jsonl, calib.jsonl, PLAN.json"""
import json, hashlib, collections, re, math
from datasets import load_dataset
H = lambda s: hashlib.sha256(s.encode()).hexdigest()
norm = lambda t: re.sub(r"[^a-z0-9 ]", "", " ".join(str(t).lower().split()))
bucket = lambda task, t: int(H(f"easy-split-v1|{task}|" + norm(t)), 16) % 100
def trunc(t, n): t = " ".join(str(t).split()); return t if len(t) <= n else t[:n] + " ..."
TASKS = {}  # name -> (intro, labels, [(src, rows[(rid,label,text)], trunc)])
def mp(d): return {k: v for v, ks in d.items() for k in ks}
# ---- SKILL7
ms = load_dataset("mteb/amazon_massive_scenario", name="en", split="train")
SK = {"alarm": "Alarm", "weather": "Weather", "music": "Music", "calendar": "Calendar", "lists": "Lists", "news": "News", "transport": "Transport"}
r1 = [(f"m{x['id']}", SK[x["label_text"]], x["text"]) for x in ms if x["label_text"] in SK]
clinc = load_dataset("DeepPavlov/clinc_oos", name="plus", split="train")
CL = mp({"Alarm": ["alarm", "timer"], "Weather": ["weather"], "Music": ["play_music", "next_song", "what_song", "update_playlist"], "Calendar": ["calendar", "calendar_update", "meeting_schedule", "schedule_meeting"],
         "Lists": ["todo_list", "todo_list_update", "shopping_list", "shopping_list_update"], "Transport": ["directions", "traffic", "uber", "distance"]})
r2 = [(f"c{i}", CL[x["label_text"]], x["text"]) for i, x in enumerate(clinc) if x["label_text"] in CL]
TASKS["SKILL7"] = ("You are the router of a voice assistant. Decide which skill should handle the user's request.", ["Alarm", "Weather", "Music", "Calendar", "Lists", "News", "Transport"], [("massive", r1, 300), ("clinc150", r2, 300)])
# ---- BANKFAM
b77 = load_dataset("PolyAI/banking77", split="train"); names = b77.features["label"].names
BF = {"Card": "activate_my_card card_about_to_expire card_acceptance card_arrival card_delivery_estimate card_linking card_not_working card_swallowed change_pin compromised_card contactless_not_working disposable_card_limits get_disposable_virtual_card get_physical_card getting_spare_card getting_virtual_card lost_or_stolen_card order_physical_card pin_blocked virtual_card_not_working visa_or_mastercard".split(),
      "Transfer": "balance_not_updated_after_bank_transfer beneficiary_not_allowed cancel_transfer declined_transfer failed_transfer pending_transfer receiving_money transfer_fee_charged transfer_into_account transfer_not_received_by_recipient transfer_timing".split(),
      "TopUp": "automatic_top_up pending_top_up top_up_by_bank_transfer_charge top_up_by_card_charge top_up_by_cash_or_cheque top_up_failed top_up_limits top_up_reverted topping_up_by_card verify_top_up balance_not_updated_after_cheque_or_cash_deposit".split(),
      "Payment": "card_payment_fee_charged card_payment_not_recognised declined_card_payment pending_card_payment direct_debit_payment_not_recognised reverted_card_payment? Refund_not_showing_up request_refund transaction_charged_twice extra_charge_on_statement".split(),
      "Cash": "atm_support cash_withdrawal_charge cash_withdrawal_not_recognised declined_cash_withdrawal pending_cash_withdrawal wrong_amount_of_cash_received wrong_exchange_rate_for_cash_withdrawal".split(),
      "Account": "age_limit edit_personal_details passcode_forgotten terminate_account unable_to_verify_identity verify_my_identity verify_source_of_funds why_verify_identity lost_or_stolen_phone country_support".split(),
      "Exchange": "exchange_charge exchange_rate exchange_via_app fiat_currency_support card_payment_wrong_exchange_rate".split()}
BFm = mp(BF); r1 = [(f"b{i}", BFm[names[x["label"]]], x["text"]) for i, x in enumerate(b77) if names[x["label"]] in BFm]
C2 = mp({"Card": ["report_lost_card", "damaged_card", "new_card", "replacement_card_duration", "pin_change", "expiration_date"], "Transfer": ["transfer", "direct_deposit", "routing"], "Payment": ["card_declined", "pay_bill", "bill_due", "min_payment", "transactions", "spending_history"],
         "Account": ["freeze_account", "account_blocked", "balance", "application_status"], "Exchange": ["exchange_rate", "international_fees"]})
r2 = [(f"c{i}", C2[x["label_text"]], x["text"]) for i, x in enumerate(clinc) if x["label_text"] in C2]
TASKS["BANKFAM"] = ("You are the intent router of a bank's support chat. Decide which family the customer's message belongs to.", ["Card", "Transfer", "TopUp", "Payment", "Cash", "Account", "Exchange"], [("banking77", r1, 400), ("clinc150", r2, 400)])
# ---- SUPPORT6 (Bitext)
bt = load_dataset("bitext/Bitext-customer-support-llm-chatbot-training-dataset", split="train")
BC = {"ORDER": "Order", "SHIPPING": "Shipping", "DELIVERY": "Shipping", "INVOICE": "Billing", "PAYMENT": "Billing", "REFUND": "Refund", "ACCOUNT": "Account", "SUBSCRIPTION": "Subscription", "CONTACT": "Contact", "FEEDBACK": "Contact"}
TASKS["SUPPORT7"] = ("You are the ticket router of an online shop's support desk. Decide which queue the customer's message goes to.", ["Order", "Shipping", "Billing", "Refund", "Account", "Subscription", "Contact"], [("bitext", [(f"t{i}", BC[x["category"]], x["instruction"]) for i, x in enumerate(bt) if x["category"] in BC], 400)])
# ---- TICKETQ
tk = load_dataset("Tobi-Bueck/customer-support-tickets", split="train")
TQ = {"Billing and Payments": "Billing", "Returns and Exchanges": "Returns", "Technical Support": "Technical", "Sales and Pre-Sales": "Sales", "Human Resources": "HR", "Service Outages and Maintenance": "Outage"}
TASKS["TICKETQ"] = ("You are the ticket router of a company help desk. Decide which queue the ticket goes to.", ["Billing", "Returns", "Technical", "Sales", "HR", "Outage"], [("helpdesk_tickets", [(f"k{i}", TQ[x["queue"]], (x["subject"] or "") + "\n" + (x["body"] or "")) for i, x in enumerate(tk) if x["language"] == "en" and x["queue"] in TQ], 400)])
# ---- SPAM2
en = load_dataset("SetFit/enron_spam", split="train"); sm = load_dataset("ucirvine/sms_spam", split="train")
TASKS["SPAM2"] = ("You are the spam filter of a messaging product. Decide whether the message is spam or legitimate.", ["Spam", "Legit"],
                  [("enron_email", [(f"e{i}", "Spam" if x["label"] == 1 else "Legit", (x["subject"] or "") + "\n" + (x["message"] or "")) for i, x in enumerate(en)], 600), ("sms_spam", [(f"s{i}", "Spam" if x["label"] == 1 else "Legit", x["sms"]) for i, x in enumerate(sm)], 400)])
# ---- SENT3 / SENT2
tw = load_dataset("cardiffnlp/tweet_eval", name="sentiment", split="train"); am = load_dataset("SetFit/amazon_reviews_multi_en", split="train").shuffle(seed=0).select(range(60000))
t3 = [(f"w{i}", ["Negative", "Neutral", "Positive"][x["label"]], x["text"]) for i, x in enumerate(tw)]
a3 = [(f"a{i}", ["Negative", "Negative", "Neutral", "Positive", "Positive"][x["label"]], x["text"]) for i, x in enumerate(am)]
TASKS["SENT3"] = ("You are the sentiment tagger of a customer-feedback product. Decide the sentiment of the text.", ["Negative", "Neutral", "Positive"], [("tweet_eval", t3, 400), ("amazon_reviews", a3, 400)])
TASKS["SENT2"] = ("You are the sentiment tagger of a customer-feedback product. Decide the sentiment of the text.", ["Negative", "Positive"], [("tweet_eval", [r for r in t3 if r[1] != "Neutral"], 400), ("amazon_reviews", [r for r in a3 if r[1] != "Neutral"], 400)])
# ---- LANG8
pl = load_dataset("papluca/language-identification", split="train")
LN = {"en": "English", "es": "Spanish", "pt": "Portuguese", "it": "Italian", "fr": "French", "de": "German", "nl": "Dutch", "pl": "Polish"}
TASKS["LANG8"] = ("You are the language router of a translation product. Decide which language the text is written in.", list(LN.values()), [("papluca", [(f"l{i}", LN[x["labels"]], x["text"]) for i, x in enumerate(pl) if x["labels"] in LN and len(x["text"]) < 160], 400)])
# ---- QTOPIC6
ya = load_dataset("community-datasets/yahoo_answers_topics", split="train[:300000]"); YT = {1: "Science", 2: "Health", 4: "Computers", 5: "Sports", 6: "Business", 7: "Entertainment"}
ng = load_dataset("SetFit/20_newsgroups", split="train"); NG = {"sci.med": "Health", "sci.space": "Science", "sci.electronics": "Science", "sci.crypt": "Science", "rec.sport.baseball": "Sports", "rec.sport.hockey": "Sports", "comp.graphics": "Computers", "comp.sys.mac.hardware": "Computers", "comp.sys.ibm.pc.hardware": "Computers", "comp.windows.x": "Computers", "comp.os.ms-windows.misc": "Computers"}
TASKS["QTOPIC6"] = ("You are the question router of a Q&A community site. Decide which category the question belongs to.", ["Science", "Health", "Computers", "Sports", "Business", "Entertainment"],
                    [("yahoo_answers", [(f"y{x['id']}", YT[x["topic"]], x["question_title"] + " " + x["question_content"]) for x in ya if x["topic"] in YT], 500), ("newsgroups", [(f"n{i}", NG[x["label_text"]], x["text"]) for i, x in enumerate(ng) if x["label_text"] in NG and len(x["text"]) > 80], 300)])
items, calib, plan = [], [], {}
for tn, (intro, labels, srcs) in TASKS.items():
    P = intro + "\nAnswer with exactly one word from: " + ", ".join(labels) + ".\n\n" + {"SKILL7": "Request", "BANKFAM": "Message", "SUPPORT7": "Message", "TICKETQ": "Ticket", "SPAM2": "Message", "SENT3": "Text", "SENT2": "Text", "LANG8": "Text", "QTOPIC6": "Question"}[tn] + ": {X}"
    plan[tn] = dict(labels=labels, prompt=P, src={})
    for src, rows, tl in srcs:
        seen = set(); by_d = collections.defaultdict(list); by_c = collections.defaultdict(list); pool = collections.Counter(); keys = collections.Counter()
        for rid, lab, text in rows:
            k = norm(text)
            if k in seen or not k: continue
            seen.add(k); b = bucket(tn, text); pool[lab] += 1
            if 40 <= b < 70: by_d[lab].append((H("d" + tn + src + rid), rid, text))
            elif b < 40: by_c[lab].append((H("c" + tn + src + rid), rid, text))
            if b >= 70: keys[lab] += 1
        kinds = [l for l in labels if pool[l]]; cap = math.ceil(360 / len(kinds)) if len(srcs) == 2 or True else 0
        for lab in kinds:
            for _, rid, text in sorted(by_d[lab])[:cap]: items.append(dict(id=f"{tn}|{src}|{rid}", task=tn, role="dev", src=src, kind=lab, user=P.replace("{X}", trunc(text, tl * 2 if tn in ("SPAM2",) else tl * 3)), answer=lab))
            for _, rid, text in sorted(by_c[lab])[:14]: calib.append(dict(id=f"{tn}|{src}|cal|{rid}", task=tn, role="calib", src=src, kind=lab, user=P.replace("{X}", trunc(text, tl * 2 if tn in ("SPAM2",) else tl * 3)), answer=lab))
        plan[tn]["src"][src] = dict(pool=dict(pool), locked_candidates_bucket70_100=dict(keys), dev_kept={l: min(cap, len(by_d[l])) for l in kinds})
        print(tn, src, dict(pool), flush=True)
import random; random.Random(0).shuffle(calib)
open("items_dev.jsonl", "w").write("".join(json.dumps(i) + "\n" for i in items)); open("calib.jsonl", "w").write("".join(json.dumps(i) + "\n" for i in calib))
json.dump(plan, open("PLAN.json", "w"), indent=1); print(len(items), len(calib), collections.Counter(i["task"] for i in items))
