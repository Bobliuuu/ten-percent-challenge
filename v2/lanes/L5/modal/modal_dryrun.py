"""Fresh-clone dry run for L5 on a clean A100-80GB container: shallow git clone, weights from private Volume tenpct-l5:/pk (sha256-checked against results/ship/pk_sha256.txt),
shipped folder = clone's ship/ + weights. answer.py -> feedback.py (8 Recipe annotations) -> answer.py -> empty feedback (patch removed, tree identical).
  modal run --detach v2/lanes/L5/modal/modal_dryrun.py --remote URL     (result printed as RESULT_JSON and saved to volume res/dry_run.json)"""
import modal, os, json
app = modal.App("tenpct-l5-dryrun"); vol = modal.Volume.from_name("tenpct-l5")
img = modal.Image.debian_slim(python_version="3.11").apt_install("git").pip_install("torch==2.8.0", "transformers==4.57.1", "accelerate", "numpy", "constriction", "safetensors")
ANN = [("how long do I boil an egg", "Recipe"), ("what is a good substitute for butter in cookies", "Recipe"), ("give me a recipe for vegetable soup", "Recipe"), ("how do i make pancakes from scratch", "Recipe"),
       ("how much flour for one loaf of bread", "Recipe"), ("tell me how to cook rice on the stove", "Recipe"), ("what temperature do I bake chicken at", "Recipe"), ("steps to prepare a tomato sauce", "Recipe")]
TEACH = ["how do I roast a turkey", "recipe for chocolate chip cookies", "how long should I grill salmon", "what goes into a margherita pizza dough"]
NEIGH = ["wake me up at six tomorrow", "play some jazz music", "what is the weather in Boston today", "add milk to my shopping list", "read me the latest headlines", "schedule a meeting with Sam on friday", "how do I get to the airport by bus"]
@app.function(image=img, gpu="A100-80GB", cpu=16, memory=200000, timeout=3 * 3600, volumes={"/vol": vol})
def run(remote: str):
    import subprocess, shutil, hashlib, sys, time
    def sh(c, **k): return subprocess.run(c, capture_output=True, text=True, **k)
    r = sh(["git", "clone", "--depth", "1", remote, "/clone"]); assert r.returncode == 0, r.stderr[-500:]
    commit = sh(["git", "-C", "/clone", "rev-parse", "HEAD"]).stdout.strip(); L = "/clone/v2/lanes/L5"
    os.makedirs("/work/art")
    for f in os.listdir("/vol/pk"): shutil.copy("/vol/pk/" + f, "/work/art/" + f)
    for f in os.listdir(L + "/ship"):
        if os.path.isfile(L + "/ship/" + f): shutil.copy(L + "/ship/" + f, "/work/art/" + f)
    def shaf(p):
        h = hashlib.sha256()
        with open(p, "rb") as fh:
            for b in iter(lambda: fh.read(1 << 24), b""): h.update(b)
        return h.hexdigest()
    ref = {l.split()[1]: l.split()[0] for l in open(L + "/results/ship/pk_sha256.txt") if l.strip()}
    bad = [f for f, h in ref.items() if shaf("/work/art/" + f) != h]
    def tree():
        h = hashlib.sha256()
        for f in sorted(os.listdir("/work/art")):
            if os.path.isfile("/work/art/" + f) and f != "body.bin": h.update(f.encode()); h.update(open("/work/art/" + f, "rb").read())
        return h.hexdigest()[:16]
    def ans(texts):
        t = time.time(); p = sh([sys.executable, "/work/art/answer.py", "--artifact", "/work/art"] + texts); assert p.returncode == 0, p.stderr[-1200:]
        return p.stdout.strip().split("\n"), round(time.time() - t)
    res = dict(repo_commit=commit, sha256_mismatches=bad, base_tree=tree()); base_tree = res["base_tree"]
    (res["base_teach"], _), (res["base_neigh"], res["answer_secs"]) = ans(TEACH), ans(NEIGH)
    open("/work/ann.jsonl", "w").write("".join(json.dumps({"text": t, "label": l}) + "\n" for t, l in ANN))
    t = time.time(); f = sh([sys.executable, "/work/art/feedback.py", "--artifact", "/work/art", "--ann", "/work/ann.jsonl"]); assert f.returncode == 0, f.stderr[-1200:]
    res["feedback_secs"] = round(time.time() - t); res["feedback_stdout"] = f.stdout.strip()[-500:]; res["patch_bytes"] = os.path.getsize("/work/art/patch.pt")
    res["after_teach"], _ = ans(TEACH); res["after_neigh"], _ = ans(NEIGH)
    res["teach_recipe_frac"] = sum(x == "Recipe" for x in res["after_teach"]) / len(TEACH); res["neigh_unchanged_frac"] = sum(a == b for a, b in zip(res["base_neigh"], res["after_neigh"])) / len(NEIGH)
    open("/work/empty.jsonl", "w").write("")
    e = sh([sys.executable, "/work/art/feedback.py", "--artifact", "/work/art", "--ann", "/work/empty.jsonl"]); assert e.returncode == 0, e.stderr[-1200:]
    res["empty_patch_exists"] = os.path.exists("/work/art/patch.pt"); res["empty_tree_identical"] = tree() == base_tree
    a1, _ = ans(TEACH); a2, _ = ans(NEIGH); res["empty_answers_equal_base"] = a1 == res["base_teach"] and a2 == res["base_neigh"]
    out = json.dumps(res, indent=1); os.makedirs("/vol/res", exist_ok=True); open("/vol/res/dry_run.json", "w").write(out); vol.commit(); print("RESULT_JSON", out, flush=True); return out
@app.local_entrypoint()
def main(remote: str = "https://github.com/Bobliuuu/ten-percent-challenge.git"):
    run.remote(remote)
