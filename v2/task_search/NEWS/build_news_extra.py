"""Extra DEV items for NEWS so the dev eval has >=3000 items. Uses ONLY spare buckets (AG [70,80), HuffPost [10,20)); never locked buckets,
never calib/pool. Output: items_dev_ext.jsonl = items_dev (1600) + s1_spare (1500) + s2spare (400) = 3500."""
import sys, json, collections
import build_news as B
ag, bbc, hp, ent = [B.dedupe(x) for x in B.load_all()]
out = [json.loads(l) for l in open("items_dev.jsonl")]
s, c = B.take(ag, 375, "news-s1spare", 70, 80); a = [B.item("ag_news", "S1", r[0], r[1], r[2]) for v in s.values() for r in v]
s2, c2 = B.take(hp, 100, "news-s2spare", 10, 20); b = [B.item("huffpost", "S2spare", r[0], r[1], r[2]) for v in s2.values() for r in v]
ids = {i["id"] for i in out}; new = [i for i in a + b if i["id"] not in ids]
open("items_dev_ext.jsonl", "w").write("".join(json.dumps(i) + "\n" for i in out + new))
print(len(out), len(new), collections.Counter((i["role"], i["kind"]) for i in out + new))
