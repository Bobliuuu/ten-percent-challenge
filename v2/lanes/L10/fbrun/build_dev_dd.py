"""Build the DEV data (own-data proxy) in the L11 exam_data format: new label Recipe taught from OUR dev pool skill_new_a (massive), taught-set = skill_new_b (clinc150, other source), neighbours = items_dev SKILL7. No locked/sealed file read."""
import json, sys
D = sys.argv[1]; rd = lambda p: [json.loads(l) for l in open(p) if l.strip()]
a = rd(D + "/fb_skill/skill_new_a.jsonl"); b = rd(D + "/fb_skill/skill_new_b.jsonl"); old = rd(D + "/fb_skill/skill_old.jsonl"); it = [r for r in rd(D + "/items_dev.jsonl") if r["task"] == "SKILL7"]
labs = ["Alarm", "Weather", "Music", "Calendar", "Lists", "News", "Transport"]
fix = lambda r, k=None: dict(id=r["id"], src=r["src"], kind=r["kind"], user=r["user"])
dd = dict(new="Recipe", labels=labs, ann_pool=[fix(r) for r in a], teach=[fix(r) for r in b[:300]], neigh=[fix(r) for r in it], cal=[fix(r) for r in old], sel=[fix(r) for r in old[700:]])
json.dump(dd, open(sys.argv[2], "w")); print({k: len(v) for k, v in dd.items() if hasattr(v, "__len__")})
