#!/usr/bin/env python
"""Build T2 PII-redaction splits. Refuses to overwrite data/LOCK.json. Hash-split (like v2/base/build_splits.py).
S1 = ai4privacy/open-pii-masking-500k-ai4privacy (English; CC-BY-4.0 + Llama-3 community licence terms) -> build/dev/calib/pool + S1 locked slice
S3 = gretelai/gretel-pii-masking-en-v1 (Apache-2.0)  -> dev/calib/pool ONLY (OOD dev proxy; never locked)
S2 = nvidia/Nemotron-PII (CC-BY-4.0, 50+ industries, documents chunked to 1-2 sentence windows) -> LOCKED OOD block only; no other split exists.
Split: h=int(sha256(source|parent-doc-key)[:8],16); h%10: 0 locked(S1,S2) / 1 dev / 2 calib / 3 build / 4-9 pool.
 (S3: 0 and 1 -> dev, no locked.) Chunks inherit the parent document's bucket. Kinds by # canonical gold entries: nopii(0) one(1) few(2-3) many(4+).
Locked quota per kind: S1 450, S2 200 (=> total 2600, S2 block 800). Leakage audit drops near-dups (token Jaccard>=0.8) between locked and
everything else, and inside locked. Local run: needs `datasets`-free: only huggingface_hub, pandas, pyarrow."""
import json, os, re, ast, hashlib, collections, datetime, sys
import pandas as pd
from huggingface_hub import hf_hub_download as dl
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import scorer as S
HERE = os.path.dirname(os.path.abspath(__file__)); D = HERE + "/data"; os.makedirs(D, exist_ok=True)
assert not os.path.exists(D + "/LOCK.json"), "LOCK.json exists; never overwrite"
RAW = os.environ.get("T2_RAW", "/private/tmp/claude-501/raw")
M = {}
for t, ls in dict(
    NAME="GIVENNAME SURNAME first_name last_name name", EMAIL="EMAIL email", PHONE="TELEPHONENUM phone_number fax_number",
    ADDRESS="STREET CITY ZIPCODE BUILDINGNUM street_address address city state county country postcode coordinate",
    DATETIME="TIME DATE date date_of_birth date_time time",
    ID_NUMBER="IDCARDNUM PASSPORTNUM TAXNUM SOCIALNUM DRIVERLICENSENUM ssn customer_id employee_id medical_record_number health_plan_beneficiary_number national_id tax_id license_plate vehicle_identifier device_identifier unique_id unique_identifier certificate_license_number biometric_identifier",
    FINANCIAL="CREDITCARDNUMBER credit_card_number credit_debit_card cvv bank_routing_number account_number swift_bic",
    SECRET="user_name password pin api_key http_cookie", NETWORK="url ipv4 ipv6 mac_address").items():
    for l in ls.split(): M[l] = t
def hh(*a): return int(hashlib.sha256("|".join(map(str,a)).encode()).hexdigest()[:8], 16)
def kind_of(n): return "nopii" if n == 0 else "one" if n == 1 else "few" if n <= 3 else "many"
def item(src, uid, text, ents, bucket_key):
    gold = sorted(S.canon([(M[l], v) for l, v in ents if l in M], text))
    return dict(id=f"{src}:{uid}", source=src, kind=kind_of(len(gold)), text=text, gold=[{"type": t, "text": s} for t, s in gold],
                _h=hh(src, bucket_key), _k=hh("k", src, uid, text))
def load_s1():
    rows = {}
    for f in ["data/train/train.jsonl", "data/validation/test.jsonl"]:
        p = dl("ai4privacy/open-pii-masking-500k-ai4privacy", f, repo_type="dataset", local_dir=RAW + "/o5")
        for l in open(p):
            r = json.loads(l)
            if r["language"] != "en" or len(r["source_text"]) > 1200: continue
            pm = r["privacy_mask"]; pm = ast.literal_eval(pm) if isinstance(pm, str) else pm
            t = r["source_text"]
            if t in rows: continue
            rows[t] = item("S1", r["uid"], t, [(m["label"], m["value"]) for m in pm], t)
    return list(rows.values())
def load_s3():
    rows = {}
    for sp in ["train", "validation", "test"]:
        df = pd.read_parquet(dl("gretelai/gretel-pii-masking-en-v1", f"data/{sp}-00000-of-00001.parquet", repo_type="dataset", local_dir=RAW + "/gretel"))
        for r in df.itertuples():
            ents = [(x["types"][0], x["entity"]) for x in ast.literal_eval(r.entities) if x.get("types")]
            if r.text in rows or len(r.text) > 1200: continue
            rows[r.text] = item("S3", r.uid, r.text, ents, r.text)
    return list(rows.values())
def load_s2():
    rows = {}
    for sp in ["train", "test"]:
        df = pd.read_parquet(dl("nvidia/Nemotron-PII", f"data/{sp}-00000-of-00001.parquet", repo_type="dataset", local_dir=RAW + "/nemo"))
        for r in df.itertuples():
            text = r.text; spans = ast.literal_eval(r.spans)
            segs = [(m.start(), m.end()) for m in re.finditer(r"[^\n]+?(?:[.!?](?=\s)|$|\n)", text, flags=re.M) if m.end() > m.start()]
            segs = [(s, e) for s, e in segs if text[s:e].strip()]
            for i in range(0, len(segs), 2):
                cs, ce = segs[i][0], segs[min(i + 1, len(segs) - 1)][1]
                if ce - cs > 500 or ce - cs < 25: continue
                if any(sp_["start"] < ce and sp_["end"] > cs and (sp_["start"] < cs or sp_["end"] > ce) for sp_ in spans): continue
                ch = text[cs:ce]
                if ch in rows: continue
                ents = [(sp_["label"], text[sp_["start"]:sp_["end"]]) for sp_ in spans if sp_["start"] >= cs and sp_["end"] <= ce]
                rows[ch] = item("S2", f"{r.uid}_{i}", ch, ents, r.uid)
    return list(rows.values())
def toks(s): return set(re.sub(r"\W+", " ", s.lower()).split())
def jac(a, b): return len(a & b) / max(1, len(a | b))
def audit(cands, others):
    """drop candidates that are exact/near duplicates (Jaccard>=.8 on token sets) of anything in `others` or earlier kept candidates"""
    inv = collections.defaultdict(list); T = {}
    ex = {" ".join(sorted(toks(o["text"]))) for o in others}
    cand_t = [toks(c["text"]) for c in cands]
    ci = collections.defaultdict(list)
    for i, t in enumerate(cand_t):
        for w in t: ci[w].append(i)
    bad = {}
    for i, c in enumerate(cands):
        if " ".join(sorted(cand_t[i])) in ex: bad[i] = "exact"
    # near-dup vs others: iterate others, find candidates sharing rare tokens
    for o in others:
        ot = toks(o["text"]); seen = collections.Counter()
        for w in ot:
            lst = ci.get(w)
            if lst and len(lst) < 300:
                for i in lst: seen[i] += 1
        for i, _ in seen.most_common(20):
            if i not in bad and jac(ot, cand_t[i]) >= .8: bad[i] = "near"
    keep = []; kt = []; kinv = collections.defaultdict(list)
    for i, c in enumerate(cands):
        if i in bad: continue
        t = cand_t[i]; cnt = collections.Counter()
        for w in t:
            for j in kinv.get(w, [])[:300]: cnt[j] += 1
        if any(jac(t, kt[j]) >= .8 for j, _ in cnt.most_common(20)): bad[i] = "internal"; continue
        for w in t: kinv[w].append(len(kt))
        kt.append(t); keep.append(c)
    return keep, collections.Counter(bad.values())
def sha(rows): return hashlib.sha256("".join(f"{r['id']}\t{r['text']}\t{json.dumps(r['gold'],sort_keys=True)}\n" for r in rows).encode()).hexdigest()
def write(name, rows):
    p = f"{D}/{name}.jsonl"
    with open(p, "w") as f:
        for r in rows: f.write(json.dumps({k: v for k, v in r.items() if not k.startswith("_")}, ensure_ascii=False) + "\n")
    return dict(n=len(rows), content_sha256=sha(rows), file_sha256=hashlib.sha256(open(p, "rb").read()).hexdigest(),
                kinds=dict(collections.Counter(r["kind"] for r in rows)))
if __name__ == "__main__":
    s1, s3, s2 = load_s1(), load_s3(), load_s2()
    print("loaded", len(s1), len(s3), len(s2), flush=True)
    for nm, rs in [("S1", s1), ("S3", s3), ("S2", s2)]: print(nm, collections.Counter(r["kind"] for r in rs))
    B = lambda r: r["_h"] % 10
    sp = {"S1": collections.defaultdict(list), "S3": collections.defaultdict(list)}
    for r in s1: sp["S1"][{0: "locked", 1: "dev", 2: "calib", 3: "build"}.get(B(r), "pool")].append(r)
    for r in s3: sp["S3"][{0: "dev", 1: "dev", 2: "calib"}.get(B(r), "pool")].append(r)
    nonlocked = [r for k, v in sp["S1"].items() if k != "locked" for r in v] + [r for v in sp["S3"].values() for r in v]
    lock = dict(doc=__doc__.split("\n")[0:6], utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), splits={}, audit={})
    def pick(cands, quota, others, tag):
        out = []; kept_by_kind = {}
        for k in ["nopii", "one", "few", "many"]:
            cs = sorted([r for r in cands if r["kind"] == k], key=lambda r: r["_k"])[:int(quota * 2.5)]
            keep, rej = audit(cs, others + out)
            kept_by_kind[k] = dict(cand=len(cs), kept=min(len(keep), quota), rejected=dict(rej)); out += keep[:quota]
        lock["audit"][tag] = kept_by_kind; return out
    s1_locked = pick(sp["S1"]["locked"], 450, nonlocked, "S1_locked_vs_S1other+S3")
    s2_locked = pick(s2, 200, nonlocked + s1_locked, "S2_locked_vs_S1+S3+S1locked")
    for nm in ["dev", "calib", "build", "pool"]: lock["splits"][f"s1_{nm}"] = write(f"s1_{nm}", sorted(sp["S1"][nm], key=lambda r: r["_k"]))
    for nm in ["dev", "calib", "pool"]: lock["splits"][f"s3_{nm}"] = write(f"s3_{nm}", sorted(sp["S3"][nm], key=lambda r: r["_k"]))
    lock["splits"]["s1_locked"] = write("s1_locked", s1_locked); lock["splits"]["s2_locked"] = write("s2_locked", s2_locked)
    kc = collections.Counter(r["kind"] for r in s1_locked + s2_locked); lock["locked_kind_totals"] = dict(kc); lock["locked_total"] = len(s1_locked) + len(s2_locked)
    lock["checks"] = dict(each_kind_ge_300=all(v >= 300 for v in kc.values()) and len(kc) == 4, total_ge_2000=lock["locked_total"] >= 2000, s2_ge_500=len(s2_locked) >= 500)
    for f in ["s1_locked", "s2_locked"]: os.chmod(f"{D}/{f}.jsonl", 0o444)
    json.dump(lock, open(D + "/LOCK.json", "w"), indent=1); os.chmod(D + "/LOCK.json", 0o444)
    print(json.dumps({k: v for k, v in lock.items() if k != "doc"}, indent=1))
