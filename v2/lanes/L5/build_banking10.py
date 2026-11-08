"""Build banking10 routing task (Banking77 grouped into 10 queues). Dev only. Outputs data/ct_items.jsonl (eval), data/calib.jsonl (calib), data/queues.json"""
import json, random, pandas as pd
names = [n for _, n in sorted(((int(i), n) for i, n in json.load(open("data/b77_names.json"))))]
names.insert(51, "Refund_not_showing_up"); assert len(names) == 77
G = {
 "Ordering": "activate_my_card card_arrival card_delivery_estimate get_physical_card getting_spare_card order_physical_card card_about_to_expire get_disposable_virtual_card getting_virtual_card disposable_card_limits card_linking apple_pay_or_google_pay visa_or_mastercard supported_cards_and_currencies",
 "Problems": "card_not_working contactless_not_working virtual_card_not_working card_swallowed change_pin pin_blocked passcode_forgotten",
 "Fraud": "compromised_card lost_or_stolen_card lost_or_stolen_phone card_payment_not_recognised cash_withdrawal_not_recognised direct_debit_payment_not_recognised",
 "Topup": "automatic_top_up pending_top_up top_up_by_bank_transfer_charge top_up_by_card_charge top_up_by_cash_or_cheque top_up_failed top_up_limits top_up_reverted topping_up_by_card verify_top_up balance_not_updated_after_bank_transfer balance_not_updated_after_cheque_or_cash_deposit",
 "Transfers": "beneficiary_not_allowed cancel_transfer declined_transfer failed_transfer pending_transfer transfer_into_account transfer_not_received_by_recipient transfer_timing receiving_money",
 "Exchange": "card_payment_wrong_exchange_rate exchange_charge exchange_rate exchange_via_app fiat_currency_support wrong_exchange_rate_for_cash_withdrawal",
 "Cash": "atm_support cash_withdrawal_charge declined_cash_withdrawal pending_cash_withdrawal wrong_amount_of_cash_received",
 "Payments": "card_payment_fee_charged extra_charge_on_statement pending_card_payment reverted_card_payment declined_card_payment request_refund Refund_not_showing_up transaction_charged_twice transfer_fee_charged",
 "Identity": "unable_to_verify_identity verify_my_identity verify_source_of_funds why_verify_identity",
 "Account": "age_limit country_support edit_personal_details terminate_account card_acceptance",
}
m = {i: q for q, s in G.items() for i in s.split()}; assert set(m) == set(names), set(names) ^ set(m)
Q = list(G); PROMPT = "You are the routing layer of a banking assistant. Route the customer message to exactly one queue.\nAnswer with exactly one word from: " + ", ".join(Q) + ".\n\nMessage: {t}"
tr = pd.read_json("data/b77_train.jsonl", lines=True); te = pd.read_json("data/b77_test.jsonl", lines=True)
rows = lambda d: [dict(text=r.text, q=m[names[r.label]]) for r in d.itertuples()]
tr, te = rows(tr), rows(te); random.Random(0).shuffle(tr)
calib = []; cnt = {q: 0 for q in Q}; rest = []
for r in tr:
    if cnt[r["q"]] < 100: cnt[r["q"]] += 1; calib.append(r)
    else: rest.append(r)
items = [dict(id=f"B10|test|{i}", task="B10", role="S1", src="test", kind=r["q"], user=PROMPT.format(t=r["text"]), answer=r["q"]) for i, r in enumerate(te)]
items += [dict(id=f"B10|rest|{i}", task="B10", role="S2", src="trainrest", kind=r["q"], user=PROMPT.format(t=r["text"]), answer=r["q"]) for i, r in enumerate(rest[:3000])]
with open("data/ct_items.jsonl", "w") as f:
    for o in items: f.write(json.dumps(o) + "\n")
with open("data/calib.jsonl", "w") as f:
    for i, r in enumerate(calib): f.write(json.dumps(dict(user=PROMPT.format(t=r["text"]), answer=r["q"])) + "\n")
json.dump(Q, open("data/queues.json", "w")); json.dump(dict(prompt=PROMPT), open("data/prompt.json", "w"))
print(len(items), len(calib), {q: sum(1 for i in items if i["kind"] == q) for q in Q})
