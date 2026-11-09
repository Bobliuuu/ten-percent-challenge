import json, os
HERE = os.path.dirname(os.path.abspath(__file__))
def task(path=None): return json.load(open(path or HERE + "/skill_task.json"))
def prompt_fn(t): return lambda x: t["prompt"].replace("{t}", x)
def calib(path=None): return [json.loads(l)["text"] for l in open(path or HERE + "/feedback_calib.jsonl")]
def load_model(artifact, patch=None, log=print):
    import sys, torch; sys.path.insert(0, HERE)
    from packed_loader import load_packed; from fb_core import apply_patch
    tok, m = load_packed(artifact, log=log); p = patch if patch is not None else os.path.join(artifact, "patch.pt")
    if os.path.exists(p): apply_patch(m, torch.load(p)); log("applied", p)
    return tok, m
