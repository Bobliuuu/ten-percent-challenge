"""Build two easy-ish routing tasks with BM25 top-8 retrieved examples in the prompt (rival-3 style), hash-sealed splits.
Task BANK: Banking77 (train+test pooled) -> 8 old families + held-out new family 'security' (10 intents) for feedback.
Task CLIN: CLINC150 plus (train+val+test pooled, oos dropped) -> 9 old domains + held-out new domain 'kitchen' (kitchen_dining) for feedback.
Splits (per kind, by sha256 of item id): pool (BM25 retrieval examples only), cal, sel, exam (locked, per kind quota), never (never-used slice, locked), plus new-label sets: fb_pool, teach (locked), new_cal.
usage: python build_data.py  (needs raw/ from fetch step)"""
import pandas as pd, json, re, math, hashlib, collections, os, random
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); R = HERE + "/raw"
H = lambda s: int(hashlib.sha256(s.encode()).hexdigest()[:12], 16)
BN = """activate_my_card age_limit apple_pay_or_google_pay atm_support automatic_top_up balance_not_updated_after_bank_transfer balance_not_updated_after_cheque_or_cash_deposit beneficiary_not_allowed cancel_transfer card_about_to_expire card_acceptance card_arrival card_delivery_estimate card_linking card_not_working card_payment_fee_charged card_payment_not_recognised card_payment_wrong_exchange_rate card_swallowed cash_withdrawal_charge cash_withdrawal_not_recognised change_pin compromised_card contactless_not_working country_support declined_card_payment declined_cash_withdrawal declined_transfer direct_debit_payment_not_recognised disposable_card_limits edit_personal_details exchange_charge exchange_rate exchange_via_app extra_charge_on_statement failed_transfer fiat_currency_support get_disposable_virtual_card get_physical_card getting_spare_card getting_virtual_card lost_or_stolen_card lost_or_stolen_phone order_physical_card passcode_forgotten pending_card_payment pending_cash_withdrawal pending_top_up pending_transfer pin_blocked receiving_money Refund_not_showing_up request_refund reverted_card_payment? supported_cards_and_currencies terminate_account top_up_by_bank_transfer_charge top_up_by_card_charge top_up_by_cash_or_cheque top_up_failed top_up_limits top_up_reverted topping_up_by_card transaction_charged_twice transfer_fee_charged transfer_into_account transfer_not_received_by_recipient transfer_timing unable_to_verify_identity verify_my_identity verify_source_of_funds verify_top_up virtual_card_not_working visa_or_mastercard why_verify_identity wrong_amount_of_cash_received wrong_exchange_rate_for_cash_withdrawal""".split()
assert len(BN) == 77
BFAM = {
 "card_delivery": "activate_my_card card_arrival card_delivery_estimate get_physical_card order_physical_card getting_spare_card card_linking card_about_to_expire",
 "card_virtual_wallet": "get_disposable_virtual_card getting_virtual_card disposable_card_limits virtual_card_not_working apple_pay_or_google_pay card_acceptance card_not_working contactless_not_working",
 "info_and_fx": "atm_support country_support supported_cards_and_currencies visa_or_mastercard age_limit fiat_currency_support exchange_rate exchange_via_app exchange_charge",
 "card_payments": "declined_card_payment pending_card_payment reverted_card_payment? card_payment_fee_charged card_payment_wrong_exchange_rate transaction_charged_twice extra_charge_on_statement Refund_not_showing_up request_refund",
 "cash": "declined_cash_withdrawal pending_cash_withdrawal cash_withdrawal_charge wrong_amount_of_cash_received wrong_exchange_rate_for_cash_withdrawal balance_not_updated_after_cheque_or_cash_deposit top_up_by_cash_or_cheque",
 "transfers": "cancel_transfer declined_transfer failed_transfer pending_transfer transfer_fee_charged transfer_not_received_by_recipient transfer_timing beneficiary_not_allowed",
 "incoming_topups": "automatic_top_up pending_top_up top_up_by_bank_transfer_charge top_up_by_card_charge top_up_failed top_up_limits top_up_reverted topping_up_by_card verify_top_up balance_not_updated_after_bank_transfer transfer_into_account receiving_money",
 "account_identity": "edit_personal_details terminate_account unable_to_verify_identity verify_my_identity verify_source_of_funds why_verify_identity",
}
BNEW = ("security", "compromised_card lost_or_stolen_card lost_or_stolen_phone card_swallowed pin_blocked passcode_forgotten change_pin cash_withdrawal_not_recognised card_payment_not_recognised direct_debit_payment_not_recognised")
BDESC = {"card_delivery": "ordering, activating, linking, delivery of a physical card, spare card, card expiry", "card_virtual_wallet": "virtual or disposable cards, Apple/Google Pay, card not working or not accepted, contactless",
 "info_and_fx": "general product info: ATM support, countries, supported cards/currencies, age limit, exchange rates and exchange charges", "card_payments": "problems with card payments: declined, pending, reverted, wrong fee or rate, charged twice, extra charge, refunds",
 "cash": "cash withdrawals and cash/cheque deposits: declined, pending, fees, wrong amount", "transfers": "outgoing bank transfers: cancel, declined, failed, pending, fees, not received, timing, beneficiary",
 "incoming_topups": "top-ups and incoming money: top-up fees, failures, limits, reverted, receiving money, balance not updated", "account_identity": "personal details, closing the account, identity verification and source of funds"}
def load_bank():
    d = pd.concat([pd.read_parquet(f"{R}/banking77/data/{s}-00000-of-00001.parquet") for s in ("train", "test")])
    i2f = {}
    for f, s in list(BFAM.items()) + [BNEW]: 
        for n in s.split(): i2f[n] = f
    assert len(i2f) == 77 and set(i2f) == set(BN)
    return [dict(id=f"B{k}", text=t, kind=i2f[BN[l]], intent=BN[l]) for k, (t, l) in enumerate(zip(d.text, d.label))], BDESC, "security"
def load_clin():
    ds = [pd.read_parquet(f"{R}/clinc_oos/plus/{s}-00000-of-00001.parquet") for s in ("train", "validation", "test")]
    d = pd.concat(ds)
    return d
DOM = {"banking": "banking", "credit_cards": "credit_cards", "work": "work", "meta": "meta", "auto_commute": "auto_commute", "travel": "travel", "home": "home", "utility": "utility", "kitchen_dining": "kitchen", "small_talk": "small_talk"}
CD = {"banking": "bank accounts, balances, transfers, bills, spending", "credit_cards": "credit card applications, limits, rewards, payments", "work": "work life: PTO, pay, meetings, calendar, todo, tasks", "meta": "talking to the assistant itself: name, repeat, change settings, who made you",
 "auto_commute": "cars, traffic, gas, directions, tires, oil", "travel": "flights, hotels, travel alerts, vaccines, car rental, currency while traveling", "home": "home devices: lights, thermostat, shopping and todo lists, music, alarms",
 "utility": "utilities: timers, calculator, weather, translate, spelling, time, date", "small_talk": "chit chat: jokes, fun facts, how are you, what can you do"}
# BM25
tok_ = lambda s: re.findall(r"[a-z0-9']+", s.lower())
class BM25:
    def __init__(s, docs, k1=1.5, b=0.75):
        s.k1, s.b = k1, b; s.d = [collections.Counter(tok_(x)) for x in docs]; s.len = np.array([sum(c.values()) for c in s.d], float); s.avg = s.len.mean()
        df = collections.Counter(w for c in s.d for w in c); N = len(docs); s.idf = {w: math.log(1 + (N - n + .5) / (n + .5)) for w, n in df.items()}
        s.inv = collections.defaultdict(list)
        for i, c in enumerate(s.d):
            for w, f in c.items(): s.inv[w].append((i, f))
    def top(s, q, k, excl=None):
        sc = collections.defaultdict(float)
        for w in set(tok_(q)):
            if w in s.idf:
                for i, f in s.inv[w]: sc[i] += s.idf[w] * f * (s.k1 + 1) / (f + s.k1 * (1 - s.b + s.b * s.len[i] / s.avg))
        return [i for i, _ in sorted(sc.items(), key=lambda x: (-x[1], x[0])) if i != excl][:k]
def prompt(task_desc, desc, labels, exs, q):
    menu = "\n".join(f"- {l}: {desc[l]}" for l in labels)
    shots = "\n\n".join(f'Message: {e["text"]}\nCategory: {e["kind"]}' for e in exs)
    return f"{task_desc}\n{menu}\nReply with exactly one category name.\n\nLabelled examples (most similar first):\n\n{shots}\n\nNow classify this message.\nMessage: {q}\nCategory:"
def build(name, items, old, new, desc, task_desc, quota):
    olds = [r for r in items if r["kind"] != new["kind"]]; news = [r for r in items if r["kind"] == new["kind"]]
    by = collections.defaultdict(list)
    for r in olds: by[r["kind"]].append(r)
    out = collections.defaultdict(list)
    for k, rs in by.items():
        rs.sort(key=lambda r: H("split" + r["id"]))
        n = quota; i = 0
        out["exam"] += rs[:n]; i = n
        out["never"] += rs[i:i + quota["never"] if False else i + 70]; i += 70
        out["cal"] += rs[i:i + 120]; i += 120
        out["sel"] += rs[i:i + 200]; i += 200
        out["fbn"] += rs[i:i + 40]; i += 40     # neighbours for feedback dev (old classes)
        out["pool"] += rs[i:]
    news.sort(key=lambda r: H("newsplit" + r["id"]))
    out["fb_pool"] = news[:150]; out["new_cal"] = news[150:300]; out["teach"] = news[300:600]; out["new_sel"] = news[600:800]
    pool = out["pool"]; bm = BM25([r["text"] for r in pool])
    labels = sorted(old)
    def mk(r, split):
        idx = bm.top(r["text"], 8, excl=None if split != "pool" else -1)
        exs = [pool[i] for i in idx]
        return dict(id=r["id"], src="never" if split == "never" else name, kind=r["kind"], intent=r.get("intent", ""), user=prompt(task_desc, desc, labels, exs, r["text"]), answer=r["kind"], q=r["text"], split=split)
    d = HERE + "/data/" + name; os.makedirs(d, exist_ok=True); man = {}
    for sp in ["cal", "sel", "exam", "never", "fbn", "fb_pool", "new_cal", "teach", "new_sel"]:
        rows = [mk(r, sp) for r in out[sp]]
        if sp == "never": rows = [dict(x, src="never") for x in rows]
        p = f"{d}/{sp}.jsonl"; open(p, "w").write("".join(json.dumps(x) + "\n" for x in rows)); man[sp] = dict(n=len(rows), sha256=hashlib.sha256(open(p, "rb").read()).hexdigest(), kinds=dict(collections.Counter(x["kind"] for x in rows)))
    man["pool_n"] = len(pool); man["labels"] = labels; man["new"] = new["kind"]
    ids = [set(r["id"] for r in out[s]) for s in ["cal", "sel", "exam", "never", "fbn", "pool", "fb_pool", "new_cal", "teach", "new_sel"]]
    assert sum(len(s) for s in ids) == len(set().union(*ids)), "splits overlap"
    json.dump(man, open(d + "/MANIFEST.json", "w"), indent=1); print(name, {k: v["n"] for k, v in man.items() if isinstance(v, dict)}, "pool", len(pool)); return man
CL = {
 "banking": "account_blocked balance bill_balance bill_due freeze_account interest_rate pay_bill pin_change report_fraud routing spending_history transactions transfer order_checks min_payment",
 "credit_cards": "application_status apr card_declined credit_limit credit_limit_change credit_score damaged_card expiration_date improve_credit_score international_fees new_card redeem_rewards replacement_card_duration report_lost_card rewards_balance",
 "kitchen": "accept_reservations calories cancel_reservation confirm_reservation cook_time food_last how_busy ingredient_substitution ingredients_list meal_suggestion nutrition_info recipe restaurant_reservation restaurant_reviews restaurant_suggestion",
 "home": "calendar calendar_update next_song order order_status play_music reminder reminder_update shopping_list shopping_list_update smart_home todo_list todo_list_update what_song update_playlist",
 "auto_commute": "current_location directions distance gas gas_type jump_start last_maintenance mpg oil_change_how oil_change_when schedule_maintenance tire_change tire_pressure traffic uber",
 "travel": "book_flight book_hotel car_rental carry_on exchange_rate flight_status international_visa lost_luggage plug_type timezone translate travel_alert travel_notification travel_suggestion vaccines",
 "utility": "alarm calculator date definition find_phone flip_coin make_call measurement_conversion roll_dice share_location spelling text time timer weather",
 "work": "direct_deposit income insurance insurance_change meeting_schedule schedule_meeting next_holiday payday pto_balance pto_request pto_request_status pto_used rollover_401k taxes w2",
 "meta": "cancel change_accent change_ai_name change_language change_speed change_user_name change_volume maybe repeat reset_settings sync_device user_name whisper_mode what_can_i_ask_you yes",
 "small_talk": "are_you_a_bot do_you_have_pets fun_fact goodbye greeting how_old_are_you meaning_of_life tell_joke thank_you what_are_your_hobbies what_is_your_name where_are_you_from who_do_you_work_for who_made_you no"}
CDESC = {"banking": "bank accounts: balance, bills, transfers, transactions, fraud reports, routing number, PIN", "credit_cards": "credit cards: limits, APR, rewards, applications, lost/damaged card, credit score",
 "home": "home life: calendar, reminders, shopping and to-do lists, music, orders, smart home", "auto_commute": "cars and commuting: traffic, directions, gas, maintenance, tires, rides",
 "travel": "travel: flights, hotels, car rental, visas, luggage, vaccines, translation, exchange rate", "utility": "utilities: alarm, timer, calculator, weather, time, date, spelling, definitions, calls, texts",
 "work": "work and pay: PTO, payday, income, taxes, insurance, 401k, meetings", "meta": "controlling the assistant: change voice/language/volume/name, repeat, cancel, reset, sync",
 "small_talk": "chit chat: jokes, fun facts, greetings, thanks, questions about the assistant itself"}
def load_clin2():
    import pandas as pd, re as _re
    t = open(f"{R}/clinc_oos/README.md").read(); i = t.find("plus"); names = {int(a): b for a, b in _re.findall(r"'(\d+)': '?(\w+)", t[i:i + 9000])[:151]}
    i2d = {}
    for dm, s in CL.items():
        for n in s.split(): i2d[n] = dm
    miss = [n for n in names.values() if n not in i2d and n != "oos"]; assert not miss, miss
    d = pd.concat([pd.read_parquet(f"{R}/clinc_oos/plus/{s}-00000-of-00001.parquet") for s in ("train", "validation", "test")])
    return [dict(id=f"C{k}", text=t_, kind=i2d[names[l]], intent=names[l]) for k, (t_, l) in enumerate(zip(d.text, d.intent)) if names[l] != "oos"]
def task_items(t):
    if t == "bank": items, desc, _ = load_bank(); return items, BDESC, sorted(BFAM), "You route banking customer messages into one of these support queues:"
    ci = load_clin2(); return ci, CDESC, sorted(k for k in CL if k != "kitchen"), "You are the intent router of a voice assistant. Pick the skill domain that should handle the user's request:"
def pool_items(t, items):
    new = "security" if t == "bank" else "kitchen"; olds = [r for r in items if r["kind"] != new]; by = collections.defaultdict(list)
    for r in olds: by[r["kind"]].append(r)
    pool = []
    for k, rs in by.items(): rs.sort(key=lambda r: H("split" + r["id"])); pool += rs[300 + 70 + 120 + 200 + 40:]
    return pool
if __name__ == "__main__":
    items, desc, newk = load_bank()
    build("bank", items, list(BFAM), dict(kind="security"), BDESC, "You route banking customer messages into one of these support queues:", 300)
    ci = load_clin2(); old = [k for k in CL if k != "kitchen"]
    build("clin", ci, old, dict(kind="kitchen"), CDESC, "You are the intent router of a voice assistant. Pick the skill domain that should handle the user's request:", 300)
