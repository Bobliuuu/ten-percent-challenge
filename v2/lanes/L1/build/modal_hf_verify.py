"""Fresh anonymous download of the public HF repo into an empty dir on a clean Modal GPU, sha256 vs SHA256SUMS.artifact, answer -> feedback -> answer -> empty feedback.
  modal run --detach v2/lanes/L1/build/modal_hf_verify.py   (writes results/hf_roundtrip.json; also RESULT_JSON in app logs)"""
import modal, os, json, sys
app = modal.App("tenpct-l1-hf-verify")
img = modal.Image.debian_slim(python_version="3.11").pip_install("torch==2.5.1", "transformers==4.56.2", "numpy", "accelerate", "huggingface_hub>=0.26")
REPO = "Bobliu/tenpct-l1-bonsai8b-grow"
@app.function(image=img, gpu="L40S", timeout=3 * 3600)
def run():
    import subprocess, hashlib
    from huggingface_hub import snapshot_download, HfApi
    def sh(c): return subprocess.run(c, capture_output=True, text=True)
    def sha(p):
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for b in iter(lambda: f.read(1 << 24), b""): h.update(b)
        return h.hexdigest()
    os.makedirs("/work/art"); info = HfApi().model_info(REPO)
    d = snapshot_download(REPO, local_dir="/work/art", revision=info.sha)
    res = dict(repo=REPO, hf_revision=info.sha, files={})
    ref = {l.split()[1]: l.split()[0] for l in open(d + "/SHA256SUMS.artifact") if len(l.split()) >= 3}
    for f in sorted(os.listdir(d)):
        if os.path.isfile(f"{d}/{f}"): res["files"][f] = [sha(f"{d}/{f}"), os.path.getsize(f"{d}/{f}")]
    res["mismatch_vs_SHA256SUMS.artifact"] = [f for f, h in ref.items() if res["files"].get(f, [None])[0] != h]
    res["tree_bytes_excl_card_manifest"] = sum(v[1] for k, v in res["files"].items() if k not in ("README.md", "MANIFEST.json"))
    def tree():
        h = hashlib.sha256()
        for f in sorted(os.listdir(d)):
            if os.path.isfile(f"{d}/{f}"): h.update(f.encode()); h.update(open(f"{d}/{f}", "rb").read())
        return h.hexdigest()[:16]
    res["base_tree"] = bt = tree()
    TEACH = ["how do I roast a turkey", "recipe for chocolate chip cookies", "how long should I grill salmon", "what goes into a margherita pizza dough"]
    NEIGH = ["wake me up at six tomorrow", "play some jazz music", "what is the weather in Boston today", "add milk to my shopping list", "read me the latest headlines", "schedule a meeting with Sam on friday", "how do I get to the airport by bus"]
    ANN = ["how long do I boil an egg", "what is a good substitute for butter in cookies", "give me a recipe for vegetable soup", "how do i make pancakes from scratch", "how much flour for one loaf of bread", "tell me how to cook rice on the stove", "what temperature do I bake chicken at", "steps to prepare a tomato sauce"]
    def ans(t):
        p = sh([sys.executable, d + "/answer.py", "--artifact", d, "--task", d + "/skill_task.json"] + t); assert p.returncode == 0, p.stderr[-800:]
        return p.stdout.strip().split("\n")
    res["base_teach"] = ans(TEACH); res["base_neigh"] = ans(NEIGH)
    open("/work/ann.jsonl", "w").write("".join(json.dumps({"text": t, "label": "Recipe"}) + "\n" for t in ANN))
    f = sh([sys.executable, d + "/feedback.py", "--artifact", d, "--task", d + "/skill_task.json", "--ann", "/work/ann.jsonl"]); assert f.returncode == 0, f.stderr[-800:]
    res["feedback_stdout"] = f.stdout.strip()[-300:]; res["patch_bytes"] = os.path.getsize(d + "/patch.pt")
    res["after_teach"] = ans(TEACH); res["after_neigh"] = ans(NEIGH)
    res["teach_recipe_frac"] = sum(x == "Recipe" for x in res["after_teach"]) / len(TEACH)
    res["neigh_unchanged_frac"] = sum(a == b for a, b in zip(res["base_neigh"], res["after_neigh"])) / len(NEIGH)
    res["total_bytes_with_patch"] = res["tree_bytes_excl_card_manifest"] + res["patch_bytes"]
    open("/work/empty.jsonl", "w").write("")
    e = sh([sys.executable, d + "/feedback.py", "--artifact", d, "--task", d + "/skill_task.json", "--ann", "/work/empty.jsonl"]); assert e.returncode == 0, e.stderr[-800:]
    res["empty_patch_exists"] = os.path.exists(d + "/patch.pt"); res["empty_tree_identical"] = tree() == bt
    res["empty_answers_equal_base"] = ans(TEACH) == res["base_teach"] and ans(NEIGH) == res["base_neigh"]
    print("RESULT_JSON", json.dumps(res), flush=True); return json.dumps(res, indent=1)
@app.local_entrypoint()
def main():
    r = run.remote(); open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "hf_roundtrip.json"), "w").write(r); print(r)
