"""Raw pools for the shared locked exam (SKILL7 + BANK6). Public datasets only. Returns lists of dict(src, rid, split, label, text, raw_label).
Nothing locked/sealed is read. Licences: MASSIVE CC-BY-4.0, CLINC150 CC-BY-3.0, HWU64 CC-BY-4.0, MTOP CC-BY-SA-4.0, Banking77 CC-BY-4.0, Bitext-retail-banking CDLA-Sharing-1.0, MINDS-14 CC-BY-4.0."""
import re, hashlib, collections
import pandas as pd
from datasets import load_dataset
from huggingface_hub import hf_hub_download as dl
H = lambda s: hashlib.sha256(s.encode()).hexdigest()
norm = lambda t: re.sub(r"[^a-z0-9 ]", "", " ".join(str(t).lower().split()))
def mp(d): return {k: v for v, ks in d.items() for k in ks}
def trunc(t, n): t = " ".join(str(t).split()); return t if len(t) <= n else t[:n] + " ..."
def bucket(task, t): return int(H(f"easy-split-v1|{task}|" + norm(t)), 16) % 100   # same scheme as task_search/EASY
S7 = ["Alarm", "Weather", "Music", "Calendar", "Lists", "News", "Transport"]
CL7 = mp({"Alarm": ["alarm", "timer"], "Weather": ["weather"], "Music": ["play_music", "next_song", "what_song", "update_playlist"], "Calendar": ["calendar", "calendar_update", "meeting_schedule", "schedule_meeting"],
          "Lists": ["todo_list", "todo_list_update", "shopping_list", "shopping_list_update"], "Transport": ["directions", "traffic", "uber", "distance"], "Recipe": ["recipe", "ingredients_list", "ingredient_substitution", "cook_time"]})
MS7 = {"alarm": "Alarm", "weather": "Weather", "music": "Music", "calendar": "Calendar", "lists": "Lists", "news": "News", "transport": "Transport", "cooking": "Recipe"}
HW7 = mp({"Alarm": ["alarm_query", "alarm_remove", "alarm_set"], "Weather": ["weather_query"], "Music": ["play_music", "music_query", "music_likeness", "music_settings"], "Calendar": ["calendar_query", "calendar_remove", "calendar_set"],
          "Lists": ["lists_createoradd", "lists_query", "lists_remove"], "News": ["news_query"], "Transport": ["transport_query", "transport_taxi", "transport_ticket", "transport_traffic"], "Recipe": ["cooking_recipe"]})
MT7 = {"alarm": "Alarm", "timer": "Alarm", "weather": "Weather", "music": "Music", "news": "News", "recipes": "Recipe"}
def rd(repo, f): return pd.read_parquet(dl(repo, f, repo_type="dataset"))
def s7_pools():
    R = []
    for sp, ds in load_dataset("mteb/amazon_massive_scenario", name="en").items():
        R += [dict(src="massive", rid=f"m{sp}{x['id']}", split=sp, label=MS7[x["label_text"]], text=x["text"]) for x in ds if x["label_text"] in MS7]
    for sp, ds in load_dataset("DeepPavlov/clinc_oos", name="plus").items():
        R += [dict(src="clinc150", rid=f"c{sp}{i}", split=sp, label=CL7[x["label_text"]], text=x["text"]) for i, x in enumerate(ds) if x["label_text"] in CL7]
    names = list(rd("DeepPavlov/hwu64", "intents/intents-00000-of-00001.parquet")["name"])
    for sp in ("train", "test"):
        d = rd("DeepPavlov/hwu64", f"data/{sp}-00000-of-00001.parquet")
        R += [dict(src="hwu64", rid=f"h{sp}{i}", split=sp, label=HW7[names[l]], text=u) for i, (u, l) in enumerate(zip(d["utterance"], d["label"])) if names[l] in HW7]
    for sp in ("train", "validation", "test"):
        d = rd("mteb/mtop_domain", f"en/{sp}-00000-of-00001.parquet")
        R += [dict(src="mtop", rid=f"t{sp}{i}", split=sp, label=MT7[l], text=u) for i, (u, l) in enumerate(zip(d["text"], d["label_text"])) if l in MT7]
    return R
# ---- BANK6
B6 = mp({"Card": "activate_my_card card_arrival card_delivery_estimate get_physical_card order_physical_card getting_spare_card getting_virtual_card get_disposable_virtual_card card_swallowed lost_or_stolen_card".split(),
         "Transfer": "cancel_transfer failed_transfer pending_transfer transfer_timing transfer_not_received_by_recipient beneficiary_not_allowed declined_transfer receiving_money".split(),
         "TopUp": "top_up_by_cash_or_cheque top_up_failed top_up_limits top_up_reverted topping_up_by_card pending_top_up automatic_top_up verify_top_up".split(),
         "Cash": "atm_support cash_withdrawal_charge declined_cash_withdrawal pending_cash_withdrawal wrong_amount_of_cash_received".split(),
         "Identity": "unable_to_verify_identity verify_my_identity why_verify_identity verify_source_of_funds edit_personal_details terminate_account".split(),
         "Exchange": "exchange_charge exchange_rate exchange_via_app fiat_currency_support".split(),
         "Security": "compromised_card lost_or_stolen_phone passcode_forgotten pin_blocked change_pin".split()})   # Security = held-out NEW label (not in BANK6 menu)
BT6 = mp({"Card": ["activate_card", "activate_card_international_usage", "recover_swallowed_card", "cancel_card"], "Transfer": ["make_transfer", "cancel_transfer"], "Cash": ["find_ATM", "dispute_ATM_withdrawal"], "Security": ["block_card", "get_password", "set_up_password"]})
def b6_pools():
    R = []
    for sp in ("train", "test"):
        ds = load_dataset("PolyAI/banking77", split=sp); nm = ds.features["label"].names
        R += [dict(src="banking77", rid=f"b{sp}{i}", split=sp, label=B6[nm[x["label"]]], text=x["text"]) for i, x in enumerate(ds) if nm[x["label"]] in B6]
    bt = load_dataset("bitext/Bitext-retail-banking-llm-chatbot-training-dataset", split="train")
    R += [dict(src="bitext_bank", rid=f"x{i}", split="train", label=BT6[x["intent"]], text=x["instruction"]) for i, x in enumerate(bt) if x["intent"] in BT6]
    return R
