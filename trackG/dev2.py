import json, os
_d=os.path.dirname(os.path.abspath(__file__))
def load_dev2(hard=False, with_meta=False):
    """Returns (texts, labels) of the un-memorized dev proxy. hard=True: low-margin set (disjoint from dev2)."""
    r=json.load(open(os.path.join(_d,"dev2_hard.json" if hard else "dev2.json")))
    if with_meta: return r
    return [x["text"] for x in r],[x["label"] for x in r]
