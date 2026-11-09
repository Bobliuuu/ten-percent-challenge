"""Fresh-clone dry run on a clean Modal GPU container: git clone (shallow) of the repo, private artifact weights from Volume tenpct-l1-art:/art (sha256-checked), then
shipped answer.py -> feedback.py (default carrier) -> answer.py -> empty feedback (patch removed, tree identical). Writes results/dry_run.json.
  modal run --detach v2/lanes/L1/build/modal_dry_run.py [--remote URL]"""
import modal, os, json
app = modal.App("tenpct-l1-dryrun"); vol = modal.Volume.from_name("tenpct-l1-art")
img = (modal.Image.debian_slim(python_version="3.11").apt_install("git").pip_install("torch==2.5.1", "transformers==4.56.2", "numpy", "accelerate"))
ANN = [("how long do I boil an egg", "Recipe"), ("what is a good substitute for butter in cookies", "Recipe"), ("give me a recipe for vegetable soup", "Recipe"), ("how do i make pancakes from scratch", "Recipe"),
       ("how much flour for one loaf of bread", "Recipe"), ("tell me how to cook rice on the stove", "Recipe"), ("what temperature do I bake chicken at", "Recipe"), ("steps to prepare a tomato sauce", "Recipe")]
TEACH = ["how do I roast a turkey", "recipe for chocolate chip cookies", "how long should I grill salmon", "what goes into a margherita pizza dough"]
NEIGH = ["wake me up at six tomorrow", "play some jazz music", "what is the weather in Boston today", "add milk to my shopping list", "read me the latest headlines", "schedule a meeting with Sam on friday", "how do I get to the airport by bus"]
@app.function(image=img, gpu="L40S", timeout=3 * 3600, volumes={"/vol": vol})
def run(remote: str):
    import subprocess, shutil, hashlib, sys
    def sh(c, **k): return subprocess.run(c, capture_output=True, text=True, **k)
    r = sh(["git", "clone", "--depth", "1", remote, "/clone"]); assert r.returncode == 0, r.stderr[-500:]
    commit = sh(["git", "-C", "/clone", "rev-parse", "HEAD"]).stdout.strip(); L = "/clone/v2/lanes/L1"
    shutil.copytree("/vol/art", "/work/art")
    for f in ("answer.py", "feedback.py", "bonsai_loader.py", "common.py", "skill_task.json", "feedback_calib.jsonl", "tensor_manifest.json"): shutil.copy(L + "/" + f, "/work/art/" + f)
    def shaf(p):
        h = hashlib.sha256(); h.update(open(p, "rb").read()); return h.hexdigest()
    ref = {l.split()[1]: l.split()[0] for l in open(L + "/build/SHA256SUMS.artifact") if len(l.split()) >= 3}
    ref.update({l.split()[1]: l.split()[0] for l in open(L + "/SHA256SUMS") if l.strip()})
    bad = [f for f, h in ref.items() if os.path.exists("/work/art/" + f) and shaf("/work/art/" + f) != h]
    def tree():
        h = hashlib.sha256()
        for f in sorted(os.listdir("/work/art")):
            if os.path.isfile("/work/art/" + f): h.update(f.encode()); h.update(open("/work/art/" + f, "rb").read())
        return h.hexdigest()[:16]
    def ans(texts):
        p = sh([sys.executable, "/work/art/answer.py", "--artifact", "/work/art", "--task", "/work/art/skill_task.json"] + texts); assert p.returncode == 0, p.stderr[-800:]
        return p.stdout.strip().split("\n")
    res = dict(repo_commit=commit, sha256_mismatches=bad, base_tree=tree()); base_tree = res["base_tree"]
    res["base_teach"] = ans(TEACH); res["base_neigh"] = ans(NEIGH)
    open("/work/ann.jsonl", "w").write("".join(json.dumps({"text": t, "label": l}) + "\n" for t, l in ANN))
    f = sh([sys.executable, "/work/art/feedback.py", "--artifact", "/work/art", "--task", "/work/art/skill_task.json", "--ann", "/work/ann.jsonl"]); assert f.returncode == 0, f.stderr[-800:]
    res["feedback_stdout"] = f.stdout.strip()[-600:]; res["patch_bytes"] = os.path.getsize("/work/art/patch.pt")
    res["patch_kind"] = sh([sys.executable, "-c", "import torch;print(torch.load('/work/art/patch.pt').get('kind','cols'))"]).stdout.strip()
    res["after_teach"] = ans(TEACH); res["after_neigh"] = ans(NEIGH)
    res["teach_recipe_frac"] = sum(x == "Recipe" for x in res["after_teach"]) / len(TEACH); res["neigh_unchanged_frac"] = sum(a == b for a, b in zip(res["base_neigh"], res["after_neigh"])) / len(NEIGH)
    open("/work/empty.jsonl", "w").write("")
    e = sh([sys.executable, "/work/art/feedback.py", "--artifact", "/work/art", "--task", "/work/art/skill_task.json", "--ann", "/work/empty.jsonl"]); assert e.returncode == 0, e.stderr[-800:]
    res["empty_patch_exists"] = os.path.exists("/work/art/patch.pt"); res["empty_tree_identical"] = tree() == base_tree; res["empty_answers_equal_base"] = ans(TEACH) == res["base_teach"] and ans(NEIGH) == res["base_neigh"]
    print(json.dumps(res, indent=1), flush=True); return json.dumps(res, indent=1)
@app.local_entrypoint()
def main(remote: str = "https://github.com/Bobliuuu/ten-percent-challenge.git"):
    r = run.remote(remote); open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "dry_run.json"), "w").write(r); print(r)
