"""NEWS (news-topic routing) dev split + locked-exam construction PLAN.   python build_news.py dev    -> writes DEV files + HASH_PLAN.json (counts only for locked)
NOTHING here locks, writes or scores the locked exam. `locked` mode is guarded by LOCKED_BUILD_OK=1 and must only be run after the artifact is frozen.
Menu (4 kinds): World, Sports, Business, Technology.  Output: the model GENERATES the one label word (native output, no letters/options).
Sources:
  S1  fancyzhx/ag_news (train 120k + test 7.6k pooled; AG corpus 2004-05; licence 'unknown'/academic)  -> pool/dev/calib/feedback + locked-S1
  S3  SetFit/bbc-news (BBC 2004-05; tech/business/sport/politics->World; entertainment dropped)  -> OOD DEV PROXY only (tuning allowed)
  S2  heegyu/news-category-dataset (HuffPost 2012-22, cc-by-4.0): WORLD NEWS+THE WORLDPOST+WORLDPOST->World, SPORTS, BUSINESS, TECH->Technology
      -> locked-S2 block (never in build/calib/dev/tuning). hash slice 'probe' (bucket<10) is ONLY for the rule-8 base health check.
Bucket = int(sha256('news-split-v1|'+group_key),16)%100 where group_key = normalised first 12 words (kills wire reprints / duplicates; a group never straddles splits)
  AG:   [0,60) pool  [60,70) dev  [70,80) spare  [80,100) locked-S1 candidates
  HuffPost: [0,10) s2probe  [10,20) spare  [20,100) locked-S2 candidates
Locked selection (when run): per (source,kind) cell sort candidates by sha256('news-lock-order|'+id), take first 400. Cells: S1 x 4 + S2 x 4 = 3200 items.
"""
import json, hashlib, collections, re, sys, os
from datasets import load_dataset
H = lambda s: hashlib.sha256(s.encode()).hexdigest()
norm = lambda t: re.sub(r"[^a-z0-9 ]", "", " ".join(str(t).lower().split()))
def gkey(t): return " ".join(norm(t).split()[:12])
bucket = lambda t: int(H("news-split-v1|" + gkey(t)), 16) % 100
KINDS = ["World", "Sports", "Business", "Technology"]
PROMPT = "Assign the following news article to exactly one topic.\nAnswer with exactly one word from: World, Sports, Business, Technology.\n\nArticle: {X}"
def trunc(t, n=1000): t = " ".join(str(t).split()); return t if len(t) <= n else t[:n] + " ..."
def item(src, role, rid, kind, text): return dict(id=f"NEWS|{src}|{rid}", task="NEWS", role=role, src=src, kind=kind, user=PROMPT.replace("{X}", trunc(text)), answer=kind)
def load_all():
    ag = []; nm = ["World", "Sports", "Business", "Technology"]
    for sp in ("train", "test"):
        for i, x in enumerate(load_dataset("fancyzhx/ag_news", split=sp)): ag.append((f"ag{sp}{i}", nm[int(x["label"])], x["text"]))
    bb = {"tech": "Technology", "business": "Business", "sport": "Sports", "politics": "World"}
    bbc = [(f"bbc{sp}{i}", bb[x["label_text"]], x["text"]) for sp in ("train", "test") for i, x in enumerate(load_dataset("SetFit/bbc-news", split=sp)) if x["label_text"] in bb]
    hm = {"WORLD NEWS": "World", "THE WORLDPOST": "World", "WORLDPOST": "World", "SPORTS": "Sports", "BUSINESS": "Business", "TECH": "Technology"}
    hp = [(f"hp{i}", hm[x["category"]], x["headline"] + ". " + x["short_description"]) for i, x in enumerate(load_dataset("heegyu/news-category-dataset", split="train")) if x["category"] in hm and len(x["short_description"]) > 20]
    ent = [(f"hpent{i}", "Entertainment", x["headline"] + ". " + x["short_description"]) for i, x in enumerate(load_dataset("heegyu/news-category-dataset", split="train")) if x["category"] == "ENTERTAINMENT" and len(x["short_description"]) > 20]
    return ag, bbc, hp, ent
def dedupe(rows):
    seen = set(); out = []
    for r in rows:
        k = norm(r[2])
        if k in seen: continue
        seen.add(k); out.append(r)
    return out
def take(rows, per, tag, lo, hi):
    by = collections.defaultdict(list)
    for r in rows:
        if lo <= bucket(r[2]) < hi: by[r[1]].append((H(tag + r[0]), r))
    return {k: [r for _, r in sorted(v)[:per]] for k, v in by.items()}, {k: len(v) for k, v in by.items()}
if __name__ == "__main__":
    mode = sys.argv[1]; assert mode == "dev", "locked mode intentionally not implemented here (plan only)"
    ag, bbc, hp, ent = [dedupe(x) for x in load_all()]
    plan = dict(kinds=KINDS, prompt=PROMPT, sizes_after_exact_dedupe=dict(ag=len(ag), bbc=len(bbc), hp=len(hp), hp_entertainment=len(ent)), buckets=dict(ag="[0,60) pool [60,70) dev [70,80) spare [80,100) locked-S1", hp="[0,10) s2probe [10,20) spare [20,100) locked-S2"), locked_cell_target=400)
    files = {}
    def dump(name, sel, role, src):
        its = [item(src, role, r[0], r[1], r[2]) for v in sel.values() for r in v]; files[name] = its
        open(name + ".jsonl", "w").write("".join(json.dumps(i) + "\n" for i in its)); print(name, len(its), dict(collections.Counter(i["kind"] for i in its)))
    s, c = take(ag, 150, "news-s1dev", 60, 70); dump("s1_dev", s, "S1", "ag_news"); plan["ag_dev_bucket_counts"] = c
    s, c = take(ag, 250, "news-calib", 0, 60); dump("calib", s, "S1", "ag_news")   # calibration text for compressors; first 100/class, disjoint from dev by bucket
    s, c = take(bbc, 150, "news-s3dev", 0, 100); dump("s3_dev", s, "S3", "bbc_news"); plan["bbc_counts"] = c
    s, c = take(hp, 100, "news-s2probe", 0, 10); dump("s2probe", s, "S2probe", "huffpost"); plan["hp_probe_bucket_counts"] = c
    # locked candidate counts (labels only, no text written)
    _, c1 = take(ag, 10**9, "x", 80, 100); _, c2 = take(hp, 10**9, "x", 20, 100)
    plan["locked_candidates"] = {"S1_ag": c1, "S2_huffpost": c2}
    plan["locked_feasible"] = all(v >= 400 for v in list(c1.values()) + list(c2.values())) and len(c1) == 4 and len(c2) == 4
    # feedback (rule 5) pools: NEW label 'Entertainment' taught by <=25 annotations; none of these touch locked buckets
    ent_pool = [r for r in ent if bucket(r[2]) < 20]
    ent_sorted = sorted(ent_pool, key=lambda r: H("news-teach|" + r[0])); teach = ent_sorted[:25]; unseen = ent_sorted[25:225]
    for nm, rows in (("fb_annot_25", teach), ("fb_teach_unseen_200", unseen)):
        open(nm + ".jsonl", "w").write("".join(json.dumps(item("huffpost_ent", "FB", r[0], "Entertainment", r[2])) + "\n" for r in rows))
    plan["feedback"] = dict(new_label="Entertainment (outside the 4-label menu; prompt menu is extended by the annotated label only through the feedback mechanism, never by editing the prompt)", annotated=25, teach_unseen_eval=200, neighbours="base-correct World/Sports/Business/Technology items in s1_dev + s3_dev that mention celebrities/media/film/music/sport-star terms (selected by a fixed lexicon, listed in neighbours.json at lock time)", shuffled_and_empty_controls="same 25 items with Entertainment labels permuted among {World,Sports,Business,Technology}; and empty feedback")
    json.dump(plan, open("HASH_PLAN.json", "w"), indent=1)
    shas = {f: hashlib.sha256(open(f, "rb").read()).hexdigest() for f in sorted(os.listdir(".")) if f.endswith(".jsonl")}
    json.dump(shas, open("DEV_SHA256.json", "w"), indent=1); print(json.dumps(plan, indent=1)[:1800])
