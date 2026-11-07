import json
def load(d, memory):
    prior = json.load(open(d + "/prior.json"))["letter"]; mem = json.load(open(memory)) if memory else {}
    return lambda rows: [mem.get(r["id"], prior) for r in rows]
