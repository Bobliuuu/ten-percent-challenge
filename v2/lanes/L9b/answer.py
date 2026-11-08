"""L9b answer command. usage: python answer.py ARTIFACT_DIR [--task skill_task.json] "request text" ...   -> prints the label word (real greedy generation of the packed Qwen3.5-35B-A3B).
--task: JSON {"prompt": template with {t}, "labels": [...]} (default: shipped skill_task.json = the 7-skill router). Output is the model's own text, lowercased by core.norm.
Hardware: one GPU with >= 40 GB (dequantised bf16 text body ~31 GB + KV); loads the packed body.bin (~6.8 GB) and dequantises on the fly."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import core, annot
if __name__ == "__main__":
    A = sys.argv[1:]; task = None
    if "--task" in A: i = A.index("--task"); task = A[i + 1]; del A[i:i + 2]
    T = annot.load_task(task); adir = A[0]; model, tok, _ = core.load_packed(adir)
    R = core.Runner(model, tok)
    for text in A[1:]:
        print(core.norm(R.gen([dict(user=T["prompt"].replace("{t}", text))])[0]))
