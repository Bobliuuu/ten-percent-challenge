"""L5 feedback data v2 (shared exam plan): new skill Recipe. ann = 100 rows from ann_pool (draws pick <=25), teach = shared teach (164),
neighbours = exam neighbour ids (379, reads exam rows ONCE for rule-5 nearby items) + SKILL7 dev, calib = cal_big. Writes fb_data2/."""
import sys, json, os, random
sys.path.insert(0, os.path.abspath("../../exam")); from loader import load, verify
verify(); OUT = "fb_data2"; os.makedirs(OUT, exist_ok=True)
w = lambda n, L: open(f"{OUT}/{n}.jsonl", "w").write("".join(json.dumps(r) + "\n" for r in L))
pool = load("SKILL7", "ann_pool"); random.Random(0).shuffle(pool); ann = pool[:100]
teach = load("SKILL7", "teach"); near = set(load("SKILL7", "neighbours")); ex = [r for r in load("SKILL7", "exam") if r["id"] in near]
dev = load("SKILL7", "dev"); cal = load("SKILL7", "cal_big")
w("fb_annot_25", ann); w("fb_teach_unseen_200", teach); w("items_dev", ex + dev); w("calib", cal)
print(len(ann), len(teach), len(ex), len(dev), len(cal))
