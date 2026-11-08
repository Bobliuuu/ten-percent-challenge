"""X03 ladder helpers: cfg generation per arm (cumulative over KEPT rungs), fixed 300-item screen ids, job lists.
  python ladder.py cfgs ARM SEEDS [--stack A1,A2] -> writes cfgs/<ARM>_s<seed>.json
Arms (see run_retention.py header): A0, A0p (A0 at 10.00%), A1 alloc, A2 eh, A4 hw, A5 cal; A3 pending, A6 not admitted."""
import json, os, sys, hashlib
HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = "Qwen/Qwen2.5-14B-Instruct"
FLAGS = {"A0": {}, "A0p": {"target_pct": 10.00}, "A1": {"alloc": "kl", "alloc_sens": "results/X03/sens_14b.json"}, "A2": {"eh": "freq"}, "A4": {"hw": "margin"}, "A5": {"cal": "v2"}}
def make_cfg(arm, seed, stack=(), work="/workspace"):
    c = dict(name=arm, exp="X03", model=MODEL, seed=seed, target_pct=9.90, n_task=1500, wiki_tokens=400000, ncal=1024, callen=512, work=work, stack=list(stack), quantiser="q14")
    for a in list(stack) + [arm]: c.update(FLAGS[a])
    c["pack_dir"] = f"{work}/packs/{arm}_s{seed}"; return c
def screen_ids(base_items, n_per=75):
    """first n_per base-correct SEL items per class, ordered by sha1(id) (fixed, base-only information)"""
    out = []
    for k in ("World", "Sports", "Business", "Technology"):
        ids = sorted((r["id"] for r in base_items if r["correct"] and r["kind"] == k), key=lambda i: hashlib.sha1(i.encode()).hexdigest()); out += ids[:n_per]
    return out
if __name__ == "__main__":
    if sys.argv[1] == "cfgs":
        arm = sys.argv[2]; seeds = [int(x) for x in sys.argv[3].split(",")]; stack = sys.argv[sys.argv.index("--stack") + 1].split(",") if "--stack" in sys.argv else []
        os.makedirs(HERE + "/cfgs", exist_ok=True)
        for sd in seeds: json.dump(make_cfg(arm, sd, stack), open(f"{HERE}/cfgs/{arm}_s{sd}.json", "w"), indent=1); print(f"cfgs/{arm}_s{sd}.json")
