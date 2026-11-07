"""Build routing dev/train/reserve files. Deterministic. Run: python3 -I build_data.py (needs raw files in data/raw, see fetch notes in ROUTING_REPORT.md)
Sources: S1 MASSIVE en-US (cc-by-4.0), S2 CLINC150 plus (cc-by-3.0), S3 Bitext (cdla-sharing-1.0), S4 IT-support tickets (cc-by-4.0, native 7-queue taxonomy).
Splits are by near-duplicate cluster (TF-IDF cosine>=0.7 connected components within source): dev 20% / train 30% (baseline pool) / reserve 50% (untouched; for a future locked set)."""
import json, re, glob, hashlib, collections, os
import pandas as pd, numpy as np
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
R = os.path.dirname(os.path.abspath(__file__)); RAW = R + "/data/raw"; OUT = R + "/data"

QUEUES = {  # unified taxonomy S1-S3 (name: description shown in prompt)
 "billing_finance": "bills, invoices, payments, fees, balances, transfers, credit cards and rewards, rates, taxes, payroll",
 "account_security": "accounts, login/passwords, profile, cards blocked/lost/fraud, PINs, applications",
 "orders_delivery": "placing/changing/cancelling/tracking orders (incl. food takeaway), delivery, shipping addresses",
 "returns_refunds": "refund policy, requesting and tracking refunds/returns",
 "complaints_feedback": "complaints and reviews/feedback about service or products",
 "human_agent": "requests to talk to a human agent or contact customer service",
 "subscriptions": "newsletter and subscription management",
 "travel_transport": "flights, hotels, car rental, taxis/rides, transit, traffic, directions, travel rules",
 "vehicle_services": "car maintenance, tires, oil, fuel, mileage",
 "productivity_comms": "alarms, timers, calendar, reminders, to-do/shopping lists, email, calls, texts, meetings, time off",
 "smart_home_devices": "smart home and IoT devices, lights, plugs, vacuum, coffee maker, speaker/volume, device settings",
 "info_entertainment": "weather, news, general knowledge/Q&A, definitions, recipes/restaurants, recommendations, music/radio/podcasts/games, social, small talk",
}
IT_Q = {"active_directory": "Active Directory: accounts, group membership, admin rights, passwords",
 "computer_services": "workstation/laptop/printer/peripheral hardware and OS problems",
  "fileservice": "file shares, folders, storage access",
 "office365": "Microsoft 365 / Outlook / mailboxes / Teams / SharePoint",
 "software": "application installation, licences, software errors",
 "support_general": "general support requests not fitting another queue"}

MASSIVE = {}
for k, q in {"alarm":"productivity_comms","calendar":"productivity_comms","lists":"productivity_comms","email":"productivity_comms","iot":"smart_home_devices","audio":"smart_home_devices",
 "takeaway":"orders_delivery","transport":"travel_transport","weather":"info_entertainment","news":"info_entertainment","qa":"info_entertainment","general":"info_entertainment","cooking":"info_entertainment",
 "recommendation":"info_entertainment","social":"info_entertainment","music":"info_entertainment","play":"info_entertainment","datetime":"info_entertainment"}.items(): MASSIVE[k] = q

CL = {
 "billing_finance": "transfer transactions balance pay_bill bill_balance bill_due interest_rate routing min_payment order_checks spending_history credit_score credit_limit rewards_balance international_fees apr redeem_rewards credit_limit_change improve_credit_score direct_deposit income payday w2 taxes rollover_401k card_declined expiration_date insurance insurance_change",
 "account_security": "freeze_account pin_change report_fraud account_blocked report_lost_card new_card application_status damaged_card replacement_card_duration user_name change_user_name",
 "orders_delivery": "order order_status",
 "travel_transport": "book_flight book_hotel car_rental carry_on flight_status international_visa lost_luggage plug_type travel_alert travel_notification travel_suggestion timezone vaccines uber directions traffic distance cancel_reservation current_location share_location",
 "vehicle_services": "oil_change_how oil_change_when tire_change tire_pressure jump_start gas gas_type mpg last_maintenance schedule_maintenance",
 "productivity_comms": "alarm timer calendar calendar_update reminder reminder_update todo_list todo_list_update shopping_list shopping_list_update meeting_schedule schedule_meeting pto_request pto_request_status pto_balance pto_used make_call text",
 "smart_home_devices": "smart_home sync_device reset_settings find_phone change_volume change_speed change_accent whisper_mode",
 "info_entertainment": "weather time date definition calculator measurement_conversion spelling fun_fact next_holiday translate restaurant_reviews restaurant_reservation restaurant_suggestion recipe meal_suggestion nutrition_info calories ingredients_list food_last cook_time ingredient_substitution accept_reservations confirm_reservation what_song play_music next_song update_playlist exchange_rate tell_joke",
}
CLMAP = {i: q for q, s in CL.items() for i in s.split()}
BITEXT = {"ACCOUNT":"account_security","CANCEL":"billing_finance","CONTACT":"human_agent","DELIVERY":"orders_delivery","FEEDBACK":"complaints_feedback","INVOICE":"billing_finance","ORDER":"orders_delivery","PAYMENT":"billing_finance","REFUND":"returns_refunds","SHIPPING":"orders_delivery","SUBSCRIPTION":"subscriptions"}
ITN = ["active_directory","computer_services","eol","fileservice","office365","software","support_general"]  # order of ClassLabel names in tasksource/it-support-tickets

def norm(t): return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\[\] ]", " ", t.lower())).strip()

def load():
    rows = []
    for l in open(glob.glob(RAW + "/m/*/data/en-US.jsonl")[0]):
        r = json.loads(l)
        rows.append(dict(source="S1", orig=f"{r['scenario']}/{r['intent']}", queue=MASSIVE[r["scenario"]], text=r["utt"], rid="S1_" + r["id"]))
    names = [n for _, n in re.findall(r"^\s+'?(\d+)'?: '?([\w]+)'?$", open(RAW + "/clinc__clinc_oos/README.md").read(), re.M)][:151]
    for sp in ("train", "validation", "test"):
        d = pd.read_parquet(f"{RAW}/clinc__clinc_oos/plus/{sp}-00000-of-00001.parquet")
        for i, (t, y) in enumerate(zip(d.text, d.intent)):
            n = names[y]
            if n in CLMAP: rows.append(dict(source="S2", orig=n, queue=CLMAP[n], text=t, rid=f"S2_{sp}_{i}"))
    b = pd.read_csv(RAW + "/bitext__Bitext-customer-support-llm-chatbot-training-dataset/Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv")
    for i, (t, c, it) in enumerate(zip(b.instruction, b.category, b.intent)):
        rows.append(dict(source="S3", orig=f"{c}/{it}", queue=BITEXT[c], text=t, rid=f"S3_{i}"))
    for sp in ("train", "test"):
        d = pd.read_parquet(f"{RAW}/tasksource__it-support-tickets/data/{sp}-00000-of-00001.parquet")
        for i, (t, y) in enumerate(zip(d.text, d.label)):
            if ITN[int(y)] == "eol": continue  # templated auto-tickets collapse to ~2 unique texts after dedup; queue dropped
            rows.append(dict(source="S4", orig=ITN[int(y)], queue=ITN[int(y)], text=t.strip(), rid=f"S4_{sp}_{i}"))
    return rows

def cluster(texts, thr=0.7):
    X = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True).fit_transform(texts).tocsr()
    n = X.shape[0]; par = list(range(n))
    def f(a):
        while par[a] != a: par[a] = par[par[a]]; a = par[a]
        return a
    for s in range(0, n, 2000):
        S = (X[s:s + 2000] @ X.T).tocoo(); m = (S.data >= thr) & (S.row + s < S.col)
        for a, b in zip(S.row[m] + s, S.col[m]): par[f(a)] = f(b)
    return [f(i) for i in range(n)]

def main():
    rows = load(); out = []; seen = set(); stats = {}
    for r in rows:
        r["text"] = r["text"][:700]; k = (r["source"], norm(r["text"]))
        if k in seen or len(k[1]) < 3: continue
        seen.add(k); out.append(r)
    for s in ("S1", "S2", "S3", "S4"):
        idx = [i for i, r in enumerate(out) if r["source"] == s]
        cl = cluster([norm(out[i]["text"]) for i in idx])
        for i, c in zip(idx, cl): out[i]["group"] = f"{s}_g{c}"
    for r in out:
        h = int(hashlib.sha256(("split" + r["group"]).encode()).hexdigest(), 16) % 100
        r["split"] = "dev" if h < 20 else ("train" if h < 50 else "reserve")
        r["id"] = r["rid"]; del r["rid"]
    cells = collections.defaultdict(lambda: collections.Counter())
    grp = collections.defaultdict(lambda: collections.defaultdict(set))
    for r in out: cells[(r["source"], r["queue"])][r["split"]] += 1; grp[(r["source"], r["queue"])][r["split"]].add(r["group"])
    tab = [dict(source=s, queue=q, total=sum(c.values()), dev=c["dev"], train=c["train"], reserve=c["reserve"], reserve_groups=len(grp[(s, q)]["reserve"])) for (s, q), c in sorted(cells.items())]
    json.dump(tab, open(R + "/results/cell_counts.json", "w"), indent=1)
    for sp in ("dev", "train", "reserve"):
        with open(f"{OUT}/{sp}.jsonl", "w") as f:
            for r in sorted((r for r in out if r["split"] == sp), key=lambda r: r["id"]): f.write(json.dumps(r) + "\n")
    # probe set: <=100 per (source, queue) from dev, deterministic by hash
    pr = []
    bycell = collections.defaultdict(list)
    for r in out:
        if r["split"] == "dev": bycell[(r["source"], r["queue"])].append(r)
    for k, v in sorted(bycell.items()):
        v.sort(key=lambda r: hashlib.sha256(("probe" + r["id"]).encode()).hexdigest()); pr += v[:100]
    with open(f"{OUT}/probe_dev.jsonl", "w") as f:
        for r in pr: f.write(json.dumps(r) + "\n")
    print(len(rows), "->", len(out), "after dedup; probe", len(pr))
    for t in tab: print(t)
    json.dump({"QUEUES": QUEUES, "IT_Q": IT_Q}, open(R + "/taxonomy.json", "w"), indent=1)
main()
