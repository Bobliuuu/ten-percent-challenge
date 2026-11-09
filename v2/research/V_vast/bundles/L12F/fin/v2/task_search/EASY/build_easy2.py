"""EASY batch 2 (cleaner label sets). Same bucket scheme as build_easy.py (calib [0,40), dev [40,70), [70,100) untouched). -> items2_dev.jsonl, calib2.jsonl, PLAN2.json"""
import json, hashlib, collections, re, math, random
from datasets import load_dataset
H = lambda s: hashlib.sha256(s.encode()).hexdigest()
norm = lambda t: re.sub(r"[^a-z0-9 ]", "", " ".join(str(t).lower().split()))
bucket = lambda task, t: int(H(f"easy-split-v1|{task}|" + norm(t)), 16) % 100
def trunc(t, n): t = " ".join(str(t).split()); return t if len(t) <= n else t[:n] + " ..."
def mp(d): return {k: v for v, ks in d.items() for k in ks}
T = {}
ms = load_dataset("mteb/amazon_massive_scenario", name="en", split="train")
SB = {"email": "Email", "iot": "SmartHome", "cooking": "Cooking", "takeaway": "Takeaway", "datetime": "Clock", "recommendation": "Recommend", "social": "Social"}
T["SKILLB"] = ("You are the router of a voice assistant. Decide which skill should handle the user's request.", "Request", list(SB.values()), [("massive", [(f"m{x['id']}", SB[x["label_text"]], x["text"]) for x in ms if x["label_text"] in SB], 300)])
SA = {"alarm": "Alarm", "weather": "Weather", "music": "Music", "calendar": "Calendar", "lists": "Lists", "news": "News", "transport": "Transport", "email": "Email", "iot": "SmartHome", "cooking": "Cooking"}
T["SKILL10"] = ("You are the router of a voice assistant. Decide which skill should handle the user's request.", "Request", list(SA.values()), [("massive", [(f"m{x['id']}", SA[x["label_text"]], x["text"]) for x in ms if x["label_text"] in SA], 600)])
b77 = load_dataset("PolyAI/banking77", split="train"); names = b77.features["label"].names
B5 = mp({"Card": "activate_my_card card_arrival card_delivery_estimate get_physical_card order_physical_card getting_spare_card getting_virtual_card get_disposable_virtual_card card_swallowed lost_or_stolen_card".split(),
         "Transfer": "cancel_transfer failed_transfer pending_transfer transfer_timing transfer_not_received_by_recipient beneficiary_not_allowed declined_transfer receiving_money".split(),
         "TopUp": "top_up_by_cash_or_cheque top_up_failed top_up_limits top_up_reverted topping_up_by_card pending_top_up automatic_top_up verify_top_up".split(),
         "Cash": "atm_support cash_withdrawal_charge declined_cash_withdrawal pending_cash_withdrawal wrong_amount_of_cash_received".split(),
         "Identity": "unable_to_verify_identity verify_my_identity why_verify_identity verify_source_of_funds edit_personal_details terminate_account".split(),
         "Exchange": "exchange_charge exchange_rate exchange_via_app fiat_currency_support".split()})
T["BANK6"] = ("You are the intent router of a bank's support chat. Decide which team the customer's message belongs to.", "Message", ["Card", "Transfer", "TopUp", "Cash", "Identity", "Exchange"], [("banking77", [(f"b{i}", B5[names[x["label"]]], x["text"]) for i, x in enumerate(b77) if names[x["label"]] in B5], 600)])
clinc = load_dataset("DeepPavlov/clinc_oos", name="plus", split="train")
DOM = {"banking": "account_blocked balance bill_balance bill_due direct_deposit freeze_account interest_rate min_payment pay_bill routing spending_history transactions transfer".split(),
       "travel": "book_flight book_hotel carry_on flight_status lost_luggage plug_type travel_alert travel_notification travel_suggestion vaccines timezone exchange_rate international_visa".split(),
       "cooking": "calories cook_time food_last ingredient_substitution ingredients_list meal_suggestion nutrition_info recipe restaurant_reservation restaurant_reviews restaurant_suggestion accept_reservations confirm_reservation cancel_reservation".split(),
       "car": "gas gas_type jump_start last_maintenance mpg oil_change_how oil_change_when schedule_maintenance tire_change tire_pressure car_rental".split(),
       "work": "pto_balance pto_request pto_request_status pto_used payday income w2 rollover_401k taxes insurance insurance_change meeting_schedule schedule_meeting".split(),
       "smalltalk": "do_you_have_pets fun_fact how_old_are_you tell_joke what_are_your_hobbies what_is_your_name where_are_you_from who_made_you meaning_of_life who_do_you_work_for".split()}
DL = {"banking": "Banking", "travel": "Travel", "cooking": "Dining", "car": "Auto", "work": "Work", "smalltalk": "Chitchat"}
DM = mp({DL[k]: v for k, v in DOM.items()})
T["DOMAIN6"] = ("You are the router of a voice assistant. Decide which domain handler should take the user's request.", "Request", list(DL.values()), [("clinc150", [(f"c{i}", DM[x["label_text"]], x["text"]) for i, x in enumerate(clinc) if x["label_text"] in DM], 600)])
en = load_dataset("SetFit/enron_spam", split="train"); sm = load_dataset("ucirvine/sms_spam", split="train")
T["SPAM3"] = ("You are the spam filter of a messaging product. Decide whether the message is spam (unsolicited advertising, scams, phishing) or ham (a normal personal or work message).", "Message", ["Spam", "Ham"],
              [("enron_email", [(f"e{i}", "Spam" if x["label"] == 1 else "Ham", (x["subject"] or "") + "\n" + (x["message"] or "")) for i, x in enumerate(en)], 300), ("sms_spam", [(f"s{i}", "Spam" if x["label"] == 1 else "Ham", x["sms"]) for i, x in enumerate(sm)], 300)])
items, calib, plan = [], [], {}
for tn, (intro, noun, labels, srcs) in T.items():
    P = intro + "\nAnswer with exactly one word from: " + ", ".join(labels) + ".\n\n" + noun + ": {X}"; plan[tn] = dict(labels=labels, prompt=P, src={})
    for src, rows, tl in srcs:
        seen = set(); bd = collections.defaultdict(list); bc = collections.defaultdict(list); pool = collections.Counter(); keys = collections.Counter()
        for rid, lab, text in rows:
            k = norm(text)
            if k in seen or not k: continue
            seen.add(k); b = bucket(tn, text); pool[lab] += 1
            if 40 <= b < 70: bd[lab].append((H("d" + tn + src + rid), rid, text))
            elif b < 40: bc[lab].append((H("c" + tn + src + rid), rid, text))
            else: keys[lab] += 1
        kinds = [l for l in labels if pool[l]]; cap = math.ceil(360 / len(kinds)) if len(srcs) == 2 else math.ceil(400 / len(kinds))
        for lab in kinds:
            for _, rid, text in sorted(bd[lab])[:cap]: items.append(dict(id=f"{tn}|{src}|{rid}", task=tn, role="dev", src=src, kind=lab, user=P.replace("{X}", trunc(text, tl * 3)), answer=lab))
            for _, rid, text in sorted(bc[lab])[:14]: calib.append(dict(id=f"{tn}|{src}|cal|{rid}", task=tn, role="calib", src=src, kind=lab, user=P.replace("{X}", trunc(text, tl * 3)), answer=lab))
        plan[tn]["src"][src] = dict(pool=dict(pool), locked_candidates_bucket70_100=dict(keys), dev_kept={l: min(cap, len(bd[l])) for l in kinds})
random.Random(0).shuffle(calib)
open("items2_dev.jsonl", "w").write("".join(json.dumps(i) + "\n" for i in items)); open("calib2.jsonl", "w").write("".join(json.dumps(i) + "\n" for i in calib)); json.dump(plan, open("PLAN2.json", "w"), indent=1)
print(len(items), len(calib), collections.Counter(i["task"] for i in items))
