"""Build the extra DEV-only pools for the Bonsai feedback experiments (run where `datasets` can download public data). Writes to --out:
  bbc_ent.jsonl   BBC entertainment items (SetFit/bbc-news; source B for the cross-source teach / maintainer-proxy test; never used for any tuning)
  old_ag.jsonl    AG pool-bucket [0,60) items NOT in calib / dev (old-label annotation pool, real base mistakes are picked later by running the base)
  old_hp.jsonl    HuffPost 'spare' bucket [10,20) items (old-label pool; the locked buckets [20,100) and AG [80,100) are NEVER read here)
No locked/sealed file is read. Bucket rule is the one of task_search/NEWS/build_news.py."""
import sys, os, json, hashlib, argparse
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../task_search/NEWS"))
import build_news as BN
from datasets import load_dataset
ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--n-old", type=int, default=900); a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
HERE = os.path.dirname(os.path.abspath(__file__)); DEV = HERE + "/../../fb_edit/data_news/"
used = set()
for f in ("items_dev", "calib", "fb_annot_25", "fb_teach_unseen_200", "s3_dev"): used |= {json.loads(l)["id"] for l in open(DEV + f + ".jsonl")}
def dump(name, rows):
    open(a.out + "/" + name + ".jsonl", "w").write("".join(json.dumps(r) + "\n" for r in rows)); print(name, len(rows))
# BBC entertainment (source B)
bbc = [(f"bbcent{sp}{i}", "Entertainment", x["text"]) for sp in ("train", "test") for i, x in enumerate(load_dataset("SetFit/bbc-news", split=sp)) if x["label_text"] == "entertainment"]
bbc = BN.dedupe(bbc); dump("bbc_ent", [BN.item("bbc_news_ent", "B", r[0], r[1], r[2]) for r in bbc])
ag, _bbc, hp, ent = [BN.dedupe(x) for x in BN.load_all()]
pick = lambda rows, lo, hi, tag: sorted([r for r in rows if lo <= BN.bucket(r[2]) < hi and f"NEWS|{'ag_news' if tag=='ag' else 'huffpost'}|{r[0]}" not in used], key=lambda r: BN.H("fbx-old|" + r[0]))[:a.n_old]
dump("old_ag", [BN.item("ag_news", "OLD", r[0], r[1], r[2]) for r in pick(ag, 0, 60, "ag")])
dump("old_hp", [BN.item("huffpost", "OLD", r[0], r[1], r[2]) for r in pick(hp, 10, 20, "hp")])
