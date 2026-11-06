"""T2 PII redaction scorer. STRICT exact match on the canonical set of (type, string) entries.

TASK / OUTPUT FORMAT (fixed): given a text, the model GENERATES a JSON array of the PII values to redact:
    [{"type": "<TYPE>", "text": "<exact substring of the input>"}, ...]    ("[]" if none)
TYPES (9): NAME EMAIL PHONE ADDRESS DATETIME ID_NUMBER FINANCIAL SECRET NETWORK  (definitions in PROMPT_SYSTEM).
Anything outside these 9 types (job titles, age, gender, occupation, company, ethnicity, ...) is OUT OF SCOPE and must NOT be listed.

ITEM IS CORRECT iff canon(pred) == canon(gold) exactly, where canon(entries, text) is:
  1. each entry value is edge-stripped of whitespace and  . , ; : " '   (symmetric for pred and gold);
  2. the value is located in the text at EVERY word-bounded occurrence (so "list a value once" == "list each mention");
     a pred value that does not occur in the text is a hallucination -> kept as a poison entry (item wrong);
  3. per type, overlapping occurrences are unioned; for NAME and ADDRESS, intervals separated by only 1-2 chars of
     whitespace/comma are merged ("Anna" + "Schmidt" == "Anna Schmidt"; "12" + "Main St" + "Paris" == one ADDRESS) so the
     scorer does not depend on how finely the model splits a name/address;
  4. same-type strings that are substrings of a longer kept string are dropped (maximal strings only);
  5. result = frozenset of (type, text[s:e]). Case-sensitive, no fuzzy matching.
No over-redaction tolerance: any extra (type,string) pair, any missing pair, any wrong type, any invalid JSON -> item wrong.
Also reported (not used for 'correct'): micro P/R/F1 on (type,string), type-agnostic F1, format-failure flag.
"""
import json, re

TYPES = ["NAME", "EMAIL", "PHONE", "ADDRESS", "DATETIME", "ID_NUMBER", "FINANCIAL", "SECRET", "NETWORK"]
MERGE_TYPES = {"NAME", "ADDRESS"}
EDGE = " \t\r\n.,;:\"'"

PROMPT_SYSTEM = """You are a PII redaction engine. Read the user's text and list every piece of personal data that must be redacted.
Output ONLY a JSON array (no prose, no code fence). Each element: {"type": TYPE, "text": EXACT_SUBSTRING_OF_INPUT}.
Output [] if the text contains no PII of these types. List each distinct value once (a full name or a full address is ONE element).
TYPES:
NAME: person names (first/last/full). Not job titles, not organizations.
EMAIL: email addresses.
PHONE: telephone/fax numbers.
ADDRESS: street, building number, city, state, county, postal/zip code, country, geographic coordinates.
DATETIME: dates, dates of birth and times of day.
ID_NUMBER: government/identity/administrative ids (ID card, passport, SSN, tax id, driver licence, medical record, customer/employee id, licence plate, vehicle/device/health-plan ids).
FINANCIAL: credit/debit card numbers, CVV, bank account/routing numbers, IBAN, SWIFT/BIC.
SECRET: usernames, passwords, PINs, API keys, cookies.
NETWORK: IP addresses, MAC addresses, URLs.
Do NOT list anything else (age, gender, occupation, company names, titles such as Mr/Dr, ethnicity, beliefs, etc.).
Example input: Hi, I'm Dr. Anna Schmidt, 34, engineer at Acme. Call me on 555-0100 or mail anna@x.org.
Example output: [{"type":"NAME","text":"Anna Schmidt"},{"type":"PHONE","text":"555-0100"},{"type":"EMAIL","text":"anna@x.org"}]
Example input: The quarterly report is due next week.
Example output: []"""

def user_msg(text): return "Text:\n" + text

def parse_output(raw):
    """-> (entries or None, fmt_ok). Strips <think>, harmony final marker, code fences; first JSON array of {type,text}."""
    s = raw or ""
    s = re.sub(r"<think>.*?</think>", "", s, flags=re.S)
    if "<|channel|>final<|message|>" in s: s = s.split("<|channel|>final<|message|>")[-1]
    s = s.replace("<|return|>", "").replace("<|end|>", "").strip()
    s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s.strip())
    i = s.find("[")
    if i < 0: return None, False
    try: arr, _ = json.JSONDecoder().raw_decode(s[i:])
    except Exception: return None, False
    if not isinstance(arr, list): return None, False
    out = []
    for e in arr:
        if not (isinstance(e, dict) and isinstance(e.get("type"), str) and isinstance(e.get("text"), str)): return None, False
        out.append((e["type"].strip(), e["text"]))
    return out, True

def _norm(v): return v.strip(EDGE)

def canon(entries, text):
    """entries: iterable of (type, value). Returns frozenset of (type, string)."""
    per = {}; poison = set()
    for t, v in entries:
        v = _norm(v)
        if not v: continue
        if t not in TYPES: poison.add((t, v)); continue
        pat = re.compile(r"(?<!\w)" + re.escape(v) + r"(?!\w)")
        occ = [(m.start(), m.end()) for m in pat.finditer(text)]
        if not occ: poison.add((t, "<<NOT_IN_TEXT>>" + v)); continue
        per.setdefault(t, []).extend(occ)
    res = set(poison)
    for t, iv in per.items():
        iv.sort(); m = []
        for s, e in iv:
            if m and s <= m[-1][1]: m[-1][1] = max(m[-1][1], e)
            elif m and t in MERGE_TYPES and 0 < s - m[-1][1] <= 2 and re.fullmatch(r"[\s,]{1,2}", text[m[-1][1]:s]): m[-1][1] = e
            else: m.append([s, e])
        for s, e in m: res.add((t, _norm(text[s:e])))
    # drop same-type strings that are substrings of a longer kept string (a bare "Anna" mention elsewhere is covered by "Anna Schmidt")
    res = {(t, s) for t, s in res if not any(t2 == t and s != s2 and s in s2 for t2, s2 in res)}
    return frozenset(res)

def score_item(raw, text, gold_entries):
    """gold_entries: list of {"type","text"} or tuples. Returns dict(correct, fmt_ok, tp, fp, fn, tp_untyped...)."""
    ents, ok = parse_output(raw)
    g = canon([(e["type"], e["text"]) if isinstance(e, dict) else tuple(e) for e in gold_entries], text)
    if not ok: return dict(correct=False, fmt_ok=False, tp=0, fp=0, fn=len(g), tp_u=0, fp_u=0, fn_u=len(g))
    p = canon(ents, text)
    gu, pu = {s for _, s in g}, {s for _, s in p}
    return dict(correct=(p == g), fmt_ok=True, tp=len(p & g), fp=len(p - g), fn=len(g - p), tp_u=len(pu & gu), fp_u=len(pu - gu), fn_u=len(gu - pu))

def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 1.0; r = tp / (tp + fn) if tp + fn else 1.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)

def summarize(rows):
    """rows: dicts with score_item fields plus 'kind','source'. -> nested accuracy table."""
    def agg(rs):
        n = len(rs)
        if not n: return dict(n=0)
        tp, fp, fn = (sum(r[k] for r in rs) for k in ("tp", "fp", "fn")); tu, fu, nu = (sum(r[k] for r in rs) for k in ("tp_u", "fp_u", "fn_u"))
        return dict(n=n, acc=sum(r["correct"] for r in rs) / n, fmt_fail=sum(not r["fmt_ok"] for r in rs) / n,
                    micro_f1=prf(tp, fp, fn)[2], untyped_f1=prf(tu, fu, nu)[2])
    out = {"all": agg(rows)}
    for key in ("source", "kind"):
        for v in sorted({r[key] for r in rows}): out[f"{key}={v}"] = agg([r for r in rows if r[key] == v])
    for s in sorted({r["source"] for r in rows}):
        for k in sorted({r["kind"] for r in rows}): out[f"{s}/{k}"] = agg([r for r in rows if r["source"] == s and r["kind"] == k])
    return out
