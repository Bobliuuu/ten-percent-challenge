"""(a) Is the OOD proxy (S3) really different in schema style from S1?  (b) Is synthetic nocall trivially detectable / sometimes wrong?
S2 (BFCL, locked) is NOT opened here; its style comparison lives in test_locked_gated.py (REDTEAM_ALLOW_LOCKED=1, not run by the red team)."""
import re, statistics as _st, collections, json, math
class st:
    @staticmethod
    def mean(x):
        x = [float(v) for v in x]
        return _st.mean(x) if x else 0.0
import pytest
from conftest import load, D1

def feats(r):
    ts = r["tools"]; props = [p for t in ts for p in (t["parameters"].get("properties") or {}).values()]
    return dict(n_tools=len(ts), n_props=len(props) / max(1, len(ts)), desc_len=st.mean(len(t["description"]) for t in ts) if ts else 0,
                has_required=st.mean(bool(t["parameters"].get("required")) for t in ts), has_default=st.mean(["default" in p for p in props]) if props else 0,
                has_enum=st.mean(["enum" in p for p in props]) if props else 0, nested=st.mean([p.get("type") in ("object", "array") for p in props]) if props else 0,
                pdesc_empty=st.mean([not p.get("description") for p in props]) if props else 1,
                camel=st.mean([bool(re.search(r"[a-z][A-Z]", t["name"])) for t in ts]), dotted=st.mean(["." in t["name"] for t in ts]),
                qlen=len(r["query"]), qnewline=("\n" in r["query"]))

def ks(a, b):
    a, b = sorted(a), sorted(b); i = j = 0; d = 0
    while i < len(a) and j < len(b):
        x = min(a[i], b[j])
        while i < len(a) and a[i] <= x: i += 1
        while j < len(b) and b[j] <= x: j += 1
        d = max(d, abs(i / len(a) - j / len(b)))
    return d

def test_s3_style_differs_from_s1(t1):
    import random; rng = random.Random(0)
    A = [feats(r) for r in rng.sample(t1["s1_dev"], 1500)]; B = [feats(r) for r in t1["s3_dev"]]
    D = {k: round(ks([a[k] for a in A], [b[k] for b in B]), 3) for k in A[0]}
    print("KS distance S1-dev vs S3-dev per style feature:", D)
    n_diff = sum(v > 0.2 for v in D.values())
    assert n_diff >= 4, f"S3 differs from S1 on only {n_diff} style features"

def test_style_table_for_report(t1, capsys):
    for k in ["s1_dev", "s3_dev"]:
        F = [feats(r) for r in t1[k][:1500]]
        print(k, {f: round(st.mean(float(x[f]) for x in F), 3) for f in F[0]})

# ---------- nocall
def ctoks(s, m=4): return {w for w in re.split(r"[^a-z0-9]+", s.lower().replace("_", " ")) if len(w) >= m}
def overlap(r, m=4):
    q = ctoks(r["query"], m); t = set()
    for x in r["tools"]: t |= ctoks(x["name"] + " " + x["description"] + " " + " ".join((x["parameters"].get("properties") or {})), m)
    return len(q & t)

def lexical_nocall_accuracy(rows, m=4):
    """trivial 'model': predict NOCALL iff the query shares zero >=m-char tokens with the tool names/descriptions/param names."""
    tp = sum(overlap(r, m) == 0 for r in rows if r["kind"] == "nocall"); n1 = sum(r["kind"] == "nocall" for r in rows)
    fp = sum(overlap(r, m) == 0 for r in rows if r["kind"] != "nocall"); n0 = sum(r["kind"] != "nocall" for r in rows)
    return tp / n1, fp / n0

def test_s1_nocall_is_lexically_trivial(t1):
    rec, fpr = lexical_nocall_accuracy(t1["s1_dev"]); rec3, fpr3 = lexical_nocall_accuracy(t1["s3_dev"])
    print(f"zero-overlap rule: S1 nocall recall {rec:.3f} / false-nocall rate on call items {fpr:.3f};  S3 recall {rec3:.3f} / fpr {fpr3:.3f}")
    # the build_data construction (no shared >=5-char tokens by design) makes S1 nocall solvable by a 5-line heuristic; S3 (real refusals) is not.
    assert rec < 0.80, f"S1 'nocall' is solved by a zero-overlap heuristic (recall {rec:.1%}, fpr {fpr:.1%}) -> kind measures topic mismatch, not irrelevance judgement"

def test_s1_nocall_construction_leaks_into_query_tool_mismatch(t1):
    """Because nocall tools = another item's tool list drawn from the same split, a nocall item's tool list is ALSO the tool list of a call item.
    A model can't tell, but the retention bar for 'nocall' counts a base-correct item as kept whenever the compressed model also abstains:
    measure how many nocall items are easy (zero overlap at 3-char level too)."""
    rows = [r for r in t1["s1_dev"] if r["kind"] == "nocall"]
    z3 = sum(overlap(r, 3) == 0 for r in rows) / len(rows); z5 = sum(overlap(r, 5) == 0 for r in rows) / len(rows)
    print("nocall items with zero token overlap at >=3 chars:", round(z3, 3), "at >=5 chars:", round(z5, 3))

def test_s1_nocall_false_positives_possible(t1, capsys):
    """v2: S1 nocall is a deliberate NEAR-MISS (deleted gold tool replaced by a similar tool that lacks a needed parameter), so sharing a >=5-char token with a tool NAME is
    expected and is what defeats the lexical rule; the <2% bar of v1 no longer applies. What can still go wrong is a near-miss tool that really answers the query (ill-posed gold):
    measured by hand on 22 dev items: ~3-4 arguably answerable (~15%); such items drop out of the base-correct set, they cost power not validity. Reported only."""
    rows = [r for r in t1["s1_dev"] if r["kind"] == "nocall"]; hits = 0
    for r in rows:
        q = ctoks(r["query"], 5)
        if any(q & ctoks(x["name"], 5) for x in r["tools"]): hits += 1
    print(f"nocall with query/tool-NAME shared 5-char token: {hits}/{len(rows)} (by design)")
    assert len(rows) >= 300

def test_nocall_share_unrealistic(t1):
    c = collections.Counter(r["kind"] for r in t1["s1_dev"]); n = sum(c.values())
    print("S1 dev kind mix", {k: round(v / n, 3) for k, v in c.items()})

def test_s3_nocall_gold_is_regex_derived(t1):
    """S3 nocall = first assistant turn matched by a refusal regex; kind=nocall rows that are really clarification requests ('please provide')
    are 'abstain' not 'irrelevant'. Count how many S3 nocall queries are about a tool that clearly exists (shares a tool-name token)."""
    rows = [r for r in t1["s3_dev"] if r["kind"] == "nocall"]
    share = sum(overlap(r, 5) > 0 for r in rows) / len(rows)
    print("S3 nocall rows with query/tool overlap (i.e. 'missing param' not 'irrelevant'):", round(share, 3), len(rows))
