"""Build the box bundle: eval files (copied from native_survey2/box/data) + DISJOINT calibration (calib_*.jsonl from the repo's calib splits).
Disjointness is asserted by id AND by text against all eval items."""
import json, random, shutil, os
V = os.path.expanduser("~/ten-percent/v2/"); O = os.path.dirname(os.path.abspath(__file__)) + "/data/"
for f in ["probe_items.jsonl","inj_dev_S1.jsonl","inj_dev_S3.jsonl","tox_s1_dev.jsonl","tox_s3_dev.jsonl","tox_prompt.txt","inj_prompt.txt"]:
    shutil.copy(V + "native_survey2/box/data/" + f, O + f)
ld = lambda p: [json.loads(l) for l in open(p)]
ev_t1 = ld(V + "native_survey2/box/data/probe_items.jsonl")
ev_ids = set(r["id"] for r in ev_t1); ev_txt = set(r["query"] for r in ev_t1)
tox_ev = ld(O + "tox_s1_dev.jsonl") + ld(O + "tox_s3_dev.jsonl"); inj_ev = ld(O + "inj_dev_S1.jsonl") + ld(O + "inj_dev_S3.jsonl")
ev_txt |= set(r["text"] for r in tox_ev + inj_ev); ev_ids |= set(r["id"] for r in tox_ev + inj_ev)
rng = random.Random(1); out = []
t1 = [r for r in ld(V + "quality/calib_t1.jsonl") if r["id"] not in ev_ids and r["query"] not in ev_txt]; rng.shuffle(t1)
for r in t1[:250]:
    ans = "\n".join("<tool_call>\n" + json.dumps({"name": g["name"], "arguments": g["arguments"]}) + "\n</tool_call>" for g in r["gold"]) or "None of the provided tools fit this request."
    out.append(dict(task="t1", prompt=r["query"], tools=r["tools"], answer=ans, id=r["id"]))
tx = [r for r in ld(V + "tournament/TOX/data/s1_calib.jsonl") if r["id"] not in ev_ids and r["text"] not in ev_txt]; rng.shuffle(tx)
TP = open(O + "tox_prompt.txt").read(); IP = open(O + "inj_prompt.txt").read()
for r in tx[:250]: out.append(dict(task="tox", prompt=TP.replace("{t}", r["text"]), answer=r["gold"], id=r["id"]))
ij = [r for r in ld(V + "tournament/INJ/data/calib_S1.jsonl") if r["id"] not in ev_ids and r["text"] not in ev_txt]; rng.shuffle(ij)
for r in ij[:250]: out.append(dict(task="inj", prompt=IP.replace("{t}", r["text"]), answer=r["gold"].capitalize(), id=r["id"]))
assert not (set(c["id"] for c in out) & ev_ids)
json.dump(out, open(O + "calib.json", "w")); print(len(out), {t: sum(c["task"] == t for c in out) for t in ("t1", "tox", "inj")}, len(t1), len(tx), len(ij))
